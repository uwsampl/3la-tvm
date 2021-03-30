#include "ilavta_helpers.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class ILAVTARuntime : public JSONRuntimeBase {
 public:
  ILAVTARuntime(const std::string& symbol_name, const std::string& graph_json,
                const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "ilavta"; }  // namespace contrib

  void Init(const Array<NDArray>& consts) override {
    CHECK(consts.size() == 0) << "matmul should have no consts";
  }

  void Run() override {
    CHECK(symbol_name_.substr(0, 6) == "ilavta");
    LOG(INFO) << "[Runtime] enter " << symbol_name_ << " runtime";

    auto dump_toggle_fn = runtime::Registry::Get("vta.simulator.profiler_dump_mode");
    CHECK(dump_toggle_fn != nullptr) << "Cannot get profiler_dump_mode toggle";
    std::vector<TVMValue> values(10);
    std::vector<int> codes(10);
    runtime::TVMArgsSetter setter(values.data(), codes.data());
    setter(0, 1L);
    TVMRetValue rv;
    TVMArgs arg(values.data(), codes.data(), 5);
    dump_toggle_fn->CallPacked(arg, &rv);

    auto op_name = nodes_[outputs_[0].id_].GetOpName();
    if (op_name != "ilavta.dense" && op_name != "ilavta.bias_add" && op_name != "ilavta.relu") {
      LOG(FATAL) << "Unknown pattern " << symbol_name_;
    }

    if (outputs_.size() == 1 && nodes_[outputs_[0].id_].GetOpName() == "ilavta.dense") {
      LOG(INFO) << "[Runtime] off-loading ilavta.dense";
      // assume there're only two inputs for now
      auto input_eid = EntryID(input_nodes_[0], 0);
      auto& input_node_data = data_entry_[input_eid];

      auto wgt_eid = EntryID(input_nodes_[1], 0);
      auto& wgt_node_data = data_entry_[wgt_eid];

# if 0
      for (int i = 0; i < input_nodes_.size(); ++i) {
        auto eid = EntryID(input_nodes_[0], 0);
        for (int dim = 0; dim < data_entry_[eid]->ndim; ++dim) {
          LOG(INFO) << "Idx: " << i << " shape: " << data_entry_[eid]->shape[dim];
        }
      }
#endif
      int n_inp_rows = input_node_data->shape[0];
      int n_inp_cols = input_node_data->shape[1];
      int n_wgt_cols = n_inp_cols;
      int n_wgt_rows = wgt_node_data->shape[0];

      int batch_size = n_inp_rows;
      int batch = batch_size * VTA_BATCH;
      int in_dim = n_inp_cols % VTA_BLOCK_IN != 0 ? n_inp_cols / VTA_BLOCK_IN + 1 : n_inp_cols / VTA_BLOCK_IN;
      int out_dim = n_wgt_rows % VTA_BLOCK_OUT != 0 ? n_wgt_rows / VTA_BLOCK_OUT + 1 : n_wgt_rows / VTA_BLOCK_OUT;
      int in_channels = in_dim * VTA_BLOCK_IN;
      int out_channels = out_dim * VTA_BLOCK_OUT;

      int ptr = 0;
      int num_instr = 64;
      int uop_size = batch / VTA_BATCH * in_channels / VTA_BLOCK_IN * out_channels / VTA_BLOCK_OUT;

      int input_size = batch / VTA_BATCH * in_channels / VTA_BLOCK_IN;
      int output_size = batch / VTA_BATCH * out_channels / VTA_BLOCK_OUT;
      int wgt_size = in_channels / VTA_BLOCK_IN * out_channels / VTA_BLOCK_OUT;

      int8_t* input = reinterpret_cast<int8_t*>(input_node_data->data);
      int8_t* weight = reinterpret_cast<int8_t*>(wgt_node_data->data);

      VTAGenericInsn *instrs = static_cast<VTAGenericInsn *>(VTAMemAlloc(sizeof(VTAGenericInsn) * num_instr, 0));

      int8_t* input_buf = reinterpret_cast<int8_t *>(VTAMemAlloc(sizeof(int8_t) * batch * in_channels, 0));
      int8_t* wgt_buf   = reinterpret_cast<int8_t *>(VTAMemAlloc(sizeof(int8_t) * out_channels * in_channels, 0));
      int32_t* acc_buf  = reinterpret_cast<int32_t *>(VTAMemAlloc(sizeof(int32_t) * batch * out_channels, 0));
      VTAUop* uop_buf   = getGEMMUops(batch / VTA_BATCH, in_channels / VTA_BLOCK_IN, out_channels / VTA_BLOCK_OUT);
      int8_t* out_buf   = reinterpret_cast<int8_t *>(VTAMemAlloc(sizeof(int8_t) * out_channels * batch, 0));
      VTADeviceHandle device = VTADeviceAlloc();

      for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < in_channels; ++j) {
          if (i >= n_inp_rows || j >= n_inp_cols) {
            // zero padding
            input_buf[i * in_channels + j] = 0;
          } else {
            input_buf[i * in_channels + j] = input[i * n_inp_cols + j];
          }
        }
      }

      int wgt_ptr_x = 0;
      int wgt_ptr_y = 0;

      /*
      * Split the weight according submatrices with the dimension
      * VTA_BLOCK_OUT* VTA_BLOCK_IN and flatten each block by rows
      * For instance a 4 by 4 matrix
      *         1 2 3 4
      *         5 6 7 8
      *         9 A B C
      *         D E F G
      * will become
      *         1 2 5 6
      *         3 4 7 8
      *         9 A D E
      *         B C F G
      * if VTA_BLOCK_OUT and VTA_BLOCK_IN are equal to 2.
      * Zero-padding applies when there are not enough elements in a block
      * that fills up VTA_BLOCK_OUT * VTA_BLOCK_IN elements.
      * For example, using the above matrix, if VTA_BLOCK_OUT and VTA_BLOCK_IN are 3, the
      * resulting weight buffer will be
      *         1 2 3 5 6 7 9 A B
      *         4 0 0 8 0 0 C 0 0
      *         D E F 0 0 0 0 0 0
      *         G 0 0 0 0 0 0 0 0
      * */
      for (int i = 0; i < n_wgt_rows; i += VTA_BLOCK_OUT) {
        for (int j = 0; j < n_wgt_cols; j += VTA_BLOCK_IN) {
          // Flatten a block into weight buffer
          for (int x = i; x < i + VTA_BLOCK_OUT; ++x) {
            for (int y = j; y < j + VTA_BLOCK_IN; ++y) {
              if (x >= n_wgt_rows || y >= n_wgt_cols) {
                // zero padding
                wgt_buf[wgt_ptr_x * in_channels + wgt_ptr_y] = 0;
              } else {
                wgt_buf[wgt_ptr_x * in_channels + wgt_ptr_y] = weight[x * n_wgt_cols + y];
              }
              wgt_ptr_y++;
              if (wgt_ptr_y == in_channels) {
                wgt_ptr_y = 0;
                wgt_ptr_x++;
              }
            }
          }
        }
      }

#if 0
      for (int i = 0; i < out_channels; ++i) {
        for (int j = 0; j < in_channels; ++j) {
          std::cerr << (int)(wgt_buf[i * in_channels + j]) << " ";
        }
        std::cerr << std::endl;
      }
#endif

      for (int i = 0; i < batch * out_channels; ++i) {
        acc_buf[i] = 0;
      }

      instrs[ptr++] = get1DLoadStoreInsn(
        VTA_OPCODE_LOAD,
        VTA_MEM_ID_UOP,
        0, VTAMemGetPhyAddr(uop_buf) / VTA_UOP_ELEM_BYTES, uop_size, 0, 0, 0, 0);
      
      instrs[ptr++] = get1DLoadStoreInsn(
        VTA_OPCODE_LOAD,
        VTA_MEM_ID_ACC,
        0, VTAMemGetPhyAddr(acc_buf) / VTA_ACC_ELEM_BYTES, output_size, 0, 0, 1, 0
      );

      instrs[ptr++] = get1DLoadStoreInsn(
        VTA_OPCODE_LOAD,
        VTA_MEM_ID_WGT,
        0, VTAMemGetPhyAddr(wgt_buf) / VTA_WGT_ELEM_BYTES, wgt_size, 0, 1, 0, 0
      );

      instrs[ptr++] = get1DLoadStoreInsn(
        VTA_OPCODE_LOAD,
        VTA_MEM_ID_INP,
        0, VTAMemGetPhyAddr(input_buf) / VTA_INP_ELEM_BYTES, input_size, 0, 0, 0, 1
      );

      instrs[ptr++] = getGEMMInsn(
        0,
        batch / VTA_BATCH,
        in_channels / VTA_BLOCK_IN,
        out_channels / VTA_BLOCK_OUT,
        0,
        1, 0, 0, 1
      );

      instrs[ptr++] = get1DLoadStoreInsn(
        VTA_OPCODE_STORE,
        VTA_MEM_ID_OUT,
        0, VTAMemGetPhyAddr(out_buf) / VTA_OUT_ELEM_BYTES,
        output_size, 1, 0, 1, 0
      );

      instrs[ptr++] = getFinishInsn(0, 1);

      VTADeviceRun(device, VTAMemGetPhyAddr(instrs), ptr, 1000);

      auto output_file = runILASimulator("ilavta_dense");
      
      auto output_node_id = outputs_[0].id_;
      auto output_data = data_entry_[output_node_id];

      CHECK(output_data->ndim == 2) << "Output dimension error: " << "expected 2, actual " << output_data->ndim;

      tvm::runtime::contrib::ila_output_data out_values;
      auto buf_size = GetDataSize(*output_data);
      int8_t* buffer = new int8_t[buf_size];
      readILAOutput(output_file, out_values);
      CHECK(out_values.size() == static_cast<size_t>(output_size * VTA_BLOCK_OUT)) << "Output element size mismatch: " << output_size * VTA_BLOCK_OUT << " v.s. " << buf_size;
      
      auto& out_shape = output_data->shape;
      size_t out_h = out_shape[0];
      size_t out_w = out_shape[1];

      CHECK(out_h == static_cast<size_t>(n_inp_rows));
      CHECK(out_w == static_cast<size_t>(n_wgt_rows)) << "Dimension mismatch: " << out_w << "; expected " << n_wgt_rows;

      size_t bufsize_read = loadILAOutput(out_values, buffer, out_h, out_w);

      CHECK(bufsize_read == buf_size) << "Number read differs from expected buffer size: " << bufsize_read << " v.s. " << buf_size;
      memcpy(reinterpret_cast<int8_t*>(output_data->data), buffer, sizeof(int8_t) * buf_size);
    } else if (outputs_.size() == 1 && nodes_[outputs_[0].id_].GetOpName() == "ilavta.bias_add") {
      auto input_eid = EntryID(input_nodes_[0], 0);
      auto bias_eid = EntryID(input_nodes_[1], 0);
      auto output_eid = outputs_[0].id_;

      auto input_data = data_entry_[input_eid];
      auto bias_data = data_entry_[bias_eid];
      auto output_data = data_entry_[output_eid];
      auto output_buffer_size = GetDataSize(*output_data);

      CHECK(input_data != nullptr);
      CHECK(bias_data != nullptr);

      auto n_inp_rows = input_data->shape[0];
      auto n_inp_cols = input_data->shape[1];
      auto n_bias_cols = bias_data->shape[0];

      CHECK(n_bias_cols == n_inp_cols) << "Dimension mismatch between input and bias";
      CHECK(n_inp_rows == output_data->shape[0]);
      CHECK(n_inp_cols == output_data->shape[1]);

      const int num_instr = 64;

      int batch = n_inp_rows * VTA_BATCH;
      int in_feat = n_inp_cols % VTA_BLOCK_OUT == 0 ? n_inp_cols / VTA_BLOCK_OUT : n_inp_cols / VTA_BLOCK_IN + 1;
      int bias_feat = in_feat;
      int in_channels = in_feat * VTA_BLOCK_OUT;
      int bias_channels = bias_feat * VTA_BLOCK_OUT;

      int uop_size = batch / VTA_BATCH * in_feat;
      int input_size = batch / VTA_BATCH * in_feat;
      int output_size = input_size;

      VTAGenericInsn *instrs = static_cast<VTAGenericInsn *>(VTAMemAlloc(sizeof(VTAGenericInsn) * num_instr, 0));

      int32_t* input_buf =  reinterpret_cast<int32_t *>(VTAMemAlloc(sizeof(int32_t) * batch * in_channels, 0));
      // TVM does array broadcasting over the matrix in bias_add
      int32_t* bias_buf  = reinterpret_cast<int32_t *>(VTAMemAlloc(sizeof(int32_t) * 1 * bias_channels, 0));
      int8_t* out_buf   = reinterpret_cast<int8_t *>(VTAMemAlloc(sizeof(int8_t) * batch * in_channels, 0));
      VTADeviceHandle device = VTADeviceAlloc();

      auto input = reinterpret_cast<int8_t*>(input_data->data);
      auto bias  = reinterpret_cast<int8_t*>(bias_data->data);
      for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < in_channels; ++j) {
          if (i >= n_inp_rows || j >= n_inp_cols) {
            // zero padding
            input_buf[i * in_channels + j] = 0;
            bias_buf[i * in_channels + j] = 0;
          } else {
            input_buf[i * in_channels + j] = input[i * n_inp_cols + j];
          }
        }
      }

      for (int i = 0; i < in_channels; ++i) {
        if (i < n_inp_cols) {
          bias_buf[i] = bias[i];
        } else {
          bias_buf[i] = 0;
        }
      }

      VTAUop* uop_buf   = getBiasAddUops(batch / VTA_BATCH, in_channels / VTA_BLOCK_IN);

      int ptr = 0;
      instrs[ptr++] = get1DLoadStoreInsn(
        VTA_OPCODE_LOAD,
        VTA_MEM_ID_UOP,
        0, VTAMemGetPhyAddr(uop_buf) / VTA_UOP_ELEM_BYTES, uop_size, 0, 0, 0, 0);
      
      instrs[ptr++] = get1DLoadStoreInsn(
        VTA_OPCODE_LOAD,
        VTA_MEM_ID_ACC,
        input_size, VTAMemGetPhyAddr(bias_buf) / VTA_ACC_ELEM_BYTES, output_size, 0, 0, 0, 0
      );

      instrs[ptr++] = get1DLoadStoreInsn(
        VTA_OPCODE_LOAD,
        VTA_MEM_ID_ACC,
        0, VTAMemGetPhyAddr(input_buf) / VTA_ACC_ELEM_BYTES, input_size, 0, 0, 0, 0
      );

      instrs[ptr++] = getAluInsn(
        VTA_ALU_OPCODE_ADD,
        0, uop_size, false, 0, 0, 0, 0, 1);
      
      instrs[ptr++] = get1DLoadStoreInsn(
        VTA_OPCODE_STORE,
        VTA_MEM_ID_OUT,
        0, VTAMemGetPhyAddr(out_buf) / VTA_OUT_ELEM_BYTES, output_size, 1, 0, 1, 0
      );

      instrs[ptr++] = getFinishInsn(0, 1);

      VTADeviceRun(device, VTAMemGetPhyAddr(instrs), ptr, 1000);

      VTAMemFree(input_buf);
      VTAMemFree(bias_buf);
      VTAMemFree(out_buf);
      VTADeviceFree(device);

      runSimGetData("ilavta_bias_add", output_buffer_size, n_inp_rows, n_inp_cols, output_data->data);
    } else if (outputs_.size() == 1 && nodes_[outputs_[0].id_].GetOpName() == "ilavta.relu") {
      auto input_eid = EntryID(input_nodes_[0], 0);
      auto output_eid = outputs_[0].id_;

      auto input_data = data_entry_[input_eid];
      auto output_data = data_entry_[output_eid];

      auto n_inp_rows = input_data->shape[0];
      auto n_inp_cols = input_data->shape[1];
      auto output_buffer_size = GetDataSize(*output_data);

      int batch = n_inp_rows * VTA_BATCH;
      int in_feat = n_inp_cols % VTA_BLOCK_OUT == 0 ? n_inp_cols / VTA_BLOCK_OUT : n_inp_cols / VTA_BLOCK_IN + 1;
      int in_channels = in_feat * VTA_BLOCK_OUT;

      int uop_size = batch / VTA_BATCH * in_feat;
      int input_size = batch / VTA_BATCH * in_feat;
      int sim_output_size = input_size;

      int num_instrs = 64;
      VTADeviceHandle device = VTADeviceAlloc();
      VTAGenericInsn *instrs = static_cast<VTAGenericInsn *>(VTAMemAlloc(sizeof(VTAGenericInsn) * num_instrs, 0));
      int32_t* input_buf = reinterpret_cast<int32_t *>(VTAMemAlloc(sizeof(int32_t) * batch * in_channels, 0));

      VTAUop *uop_buf = getReluUops(batch, in_feat);

      int8_t* inputs = reinterpret_cast<int8_t*>(input_data->data);
      for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < in_channels; ++j) {
          if (i >= n_inp_rows || j >= n_inp_cols) {
            // zero padding
            input_buf[i * in_channels + j] = 0;
          } else {
            input_buf[i * in_channels + j] = inputs[i * n_inp_cols + j];
          }
        }
      }

      int ptr = 0;
      instrs[ptr++] = get1DLoadStoreInsn(
        VTA_OPCODE_LOAD,
        VTA_MEM_ID_UOP,
        0, VTAMemGetPhyAddr(uop_buf) / VTA_UOP_ELEM_BYTES, uop_size, 0, 0, 0, 0);

      instrs[ptr++] = get1DLoadStoreInsn(
        VTA_OPCODE_LOAD,
        VTA_MEM_ID_ACC,
        0, VTAMemGetPhyAddr(input_buf) / VTA_ACC_ELEM_BYTES, input_size, 0, 0, 0, 0
      );

      instrs[ptr++] = getAluInsn(VTA_ALU_OPCODE_MAX, 0, uop_size, true, 0, 0, 0, 0, 1);
      
      instrs[ptr++] = get1DLoadStoreInsn(
        VTA_OPCODE_STORE,
        VTA_MEM_ID_OUT,
        0, VTAMemGetPhyAddr(input_buf) / VTA_OUT_ELEM_BYTES, sim_output_size, 1, 0, 1, 0
      );

      instrs[ptr++] = getFinishInsn(0, 1);

      VTADeviceRun(device, VTAMemGetPhyAddr(instrs), ptr, 1000);

      VTAMemFree(input_buf);
      VTAMemFree(uop_buf);
      VTAMemFree(instrs);
      VTADeviceFree(device);

      runSimGetData("ilavta_relu", output_buffer_size, n_inp_rows, n_inp_cols, output_data->data);
    }
  }

 protected:
 private:
};  // namespace runtime

runtime::Module ILAVTARuntimeCreate(String symbol_name, String graph_json,
                                    const Array<String>& const_names) {
  auto n = make_object<ILAVTARuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.ILAVTARuntimeCreate")
    .set_body_typed(ILAVTARuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_ilavta")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<ILAVTARuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
