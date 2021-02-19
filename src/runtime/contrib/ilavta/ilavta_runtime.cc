#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <vta/driver.h>
#include <vta/hw_spec.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

const int64_t SIM_DUMP = 1;
const std::string RAW_DUMP = "vta_sim_dump.json";
const std::string OUTPUT_DUMP = "vta_output_dump.json";

/*
* Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
* */
VTAUop * getGEMMUops(int batch, int in_feat, int out_feat) {
  // Derive the total uop size
  int uop_size = batch * in_feat * out_feat;

  // Allocate buffer
  VTAUop *uop_buf = static_cast<VTAUop *>(VTAMemAlloc(sizeof(VTAUop) * uop_size, 0));

  int uop_idx = 0;
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < in_feat; j++) {
      for (int k = 0; k < out_feat; k++) {
        uop_buf[uop_idx].dst_idx = i * out_feat + k;
        uop_buf[uop_idx].src_idx = i * in_feat + j;
        uop_buf[uop_idx].wgt_idx = k * in_feat + j;
        uop_idx++;
      }
    }
  }
  return uop_buf;
}

/*
* Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
* */
VTAGenericInsn getGEMMInsn(int uop_offset, int batch, int in_feat, int out_feat,
                           bool uop_compression, int pop_prev_dep, int pop_next_dep,
                           int push_prev_dep, int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // GEMM instruction initialization
  VTAGemInsn insn;
  insn.opcode = VTA_OPCODE_GEMM;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.reset_reg = false;
  insn.uop_bgn = uop_offset;
  insn.uop_end = uop_offset + batch * in_feat * out_feat;;
  insn.iter_out = 1;
  insn.iter_in = 1;
  insn.dst_factor_out = 0;
  insn.src_factor_out = 0;
  insn.wgt_factor_out = 0;
  insn.dst_factor_in = 0;
  insn.src_factor_in = 0;
  insn.wgt_factor_in = 0;
  converter.gemm = insn;
  return converter.generic;
}

/*
* Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
* */
VTAGenericInsn getFinishInsn(bool pop_prev, bool pop_next) {
  // Converter
  union VTAInsn converter;
  // GEMM instruction initialization
  VTAGemInsn insn;
  insn.opcode = VTA_OPCODE_FINISH;
  insn.pop_prev_dep = pop_prev;
  insn.pop_next_dep = pop_next;
  insn.push_prev_dep = 0;
  insn.push_next_dep = 0;
  insn.reset_reg = false;
  insn.uop_bgn = 0;
  insn.uop_end = 0;
  insn.iter_out = 0;
  insn.iter_in = 0;
  insn.dst_factor_out = 0;
  insn.src_factor_out = 0;
  insn.wgt_factor_out = 0;
  insn.dst_factor_in = 0;
  insn.src_factor_in = 0;
  insn.wgt_factor_in = 0;
  converter.gemm = insn;
  return converter.generic;
}

/*
* Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
* */
VTAGenericInsn get1DLoadStoreInsn(int opcode, int type, int sram_offset, int dram_offset, int size,
                                  int pop_prev_dep, int pop_next_dep, int push_prev_dep,
                                  int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // Memory instruction initialization
  VTAMemInsn insn = {};
  insn.opcode = opcode;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.memory_type = type;
  insn.sram_base = sram_offset;
  insn.dram_base = dram_offset;
  insn.y_size = 1;
  insn.x_size = size;
  insn.x_stride = size;
  insn.y_pad_0 = 0;
  insn.y_pad_1 = 0;
  insn.x_pad_0 = 0;
  insn.x_pad_1 = 0;
  converter.mem = insn;
  return converter.generic;
}

/*
* Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
* */
VTAGenericInsn get2DLoadStoreInsn(int opcode, int type, int sram_offset, int dram_offset,
                                  int y_size, int x_size, int x_stride, int y_pad, int x_pad,
                                  int pop_prev_dep, int pop_next_dep, int push_prev_dep,
                                  int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // Memory instruction initialization
  VTAMemInsn insn = {};
  insn.opcode = opcode;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.memory_type = type;
  insn.sram_base = sram_offset;
  insn.dram_base = dram_offset;
  insn.y_size = y_size;
  insn.x_size = x_size;
  insn.x_stride = x_stride;
  insn.y_pad_0 = y_pad;
  insn.y_pad_1 = y_pad;
  insn.x_pad_0 = x_pad;
  insn.x_pad_1 = x_pad;
  converter.mem = insn;
  return converter.generic;
}

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

      // for (int i = 0; i < (input_size * VTA_INP_ELEM_BYTES) / 4; i++) {
      //   printf("i[%02d]:%08x\n", i,
      //          reinterpret_cast<int32_t*>(input_buf)[i]);
      // }

      VTADeviceRun(device, VTAMemGetPhyAddr(instrs), ptr, 1000);

      // Check dump file
      auto ret = std::system("stat vta_sim_dump.json > /dev/null 2> /dev/null");
      CHECK(ret == 0) << "vta_sim_dump.json does not exists";

      ret = std::system("python3 produce_ila_fragment.py vta_sim_dump.json ./prog_frag/ilavta_dense_input.json");
      CHECK(ret == 0) << "Failed to produce program fragment";
      
      std::system("vta_ila_sim ilavta_dense");
      ret = std::system("stat ./result/ilavta_dense_out.json > /dev/null 2> /dev/null");
      CHECK(ret == 0) << "Not output result found";
      
      auto output_node_id = outputs_[0].id_;
      auto output_data = data_entry_[output_node_id];

      CHECK(output_data->ndim == 2) << "Output dimension error: " << "expected 2, actual " << output_data->ndim;

      LOG(INFO) << "[Runtime] Copy back simulation result";
      std::ifstream input_stream("./result/ilavta_dense_out.json");
      dmlc::JSONReader reader(&input_stream);

      auto buf_size = GetDataSize(*output_data);
      int8_t* buffer = new int8_t[buf_size];
      std::vector<std::unordered_map<std::string, std::string> > out_values;
      std::unordered_map<std::string, std::string> value;
      std::string key;
      std::string addr;
      std::string data;
      
      reader.BeginArray();
      while (reader.NextArrayItem()) {
        reader.Read(&value);
        out_values.push_back(value);
      }

      CHECK(out_values.size() == output_size * VTA_BLOCK_OUT) << "Output element size mismatch: " << output_size * VTA_BLOCK_OUT << " v.s. " << buf_size;
      
      auto& out_shape = output_data->shape;
      size_t out_h = out_shape[0];
      size_t out_w = out_shape[1];

      CHECK(out_h == n_inp_rows);
      CHECK(out_w == n_wgt_rows) << "Dimension mismatch: " << out_w << "; expected " << n_wgt_rows;

      size_t data_cur = 0;
      size_t buf_cur = 0;
      uint32_t temp;

      LOG(INFO) << "[Copying] from output json to byte buffer"; 
      for (size_t i = 0; i < out_h; ++i) {
        if (data_cur % VTA_BLOCK_OUT != 0) {
          data_cur = (data_cur / VTA_BLOCK_OUT + 1) * VTA_BLOCK_OUT;
        }
        for (size_t j = 0; j < out_w; ++j) {
          auto val = out_values[data_cur++]["data"];
          std::stringstream ss;
          ss << std::hex << val;
          ss >> temp;
          // LOG(INFO) << "Copying " << out_values[data_cur - 1]["addr"] << ": " << val << " == " << temp;
          buffer[buf_cur++] = static_cast<int8_t>(temp);
        }
      }

      CHECK(buf_cur == buf_size) << "Number read differs from expected buffer size: " << buf_cur << " v.s. " << buf_size;
      memcpy(reinterpret_cast<int8_t*>(output_data->data), buffer, sizeof(int8_t) * buf_size);
    } else {
      LOG(FATAL) << "Unknown pattern " << symbol_name_;
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
