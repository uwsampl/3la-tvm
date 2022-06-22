#include <tvm/runtime/data_type.h>
#include <tvm/support/json.hpp>

#include <chrono>
#include <iostream>

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

    // auto dump_toggle_fn = runtime::Registry::Get("vta.simulator.profiler_dump_mode");
    // CHECK(dump_toggle_fn != nullptr) << "Cannot get profiler_dump_mode toggle";
    // std::vector<TVMValue> values(10);
    // std::vector<int> codes(10);
    // runtime::TVMArgsSetter setter(values.data(), codes.data());
    // setter(0, 1L);
    // TVMRetValue rv;
    // TVMArgs arg(values.data(), codes.data(), 5);
    // dump_toggle_fn->CallPacked(arg, &rv);

    const std::string wall_clock_file = "ilavta_wallclock.json";
    std::string driver_dir = getenv("PY_3LA_DRIVER");
    driver_dir += "/vta";
    auto call_node = nodes_[outputs_[0].id_];
    auto op_name = call_node.GetOpName();
    int64_t sim_time = -1;
    if (op_name != "ilavta.dense" && op_name != "ilavta.bias_add"
     && op_name != "ilavta.relu" && op_name != "ilavta.conv1d" && op_name != "ilavta.linear") {
      LOG(FATAL) << "Unknown pattern " << symbol_name_ << " " << op_name;
    }

    if (outputs_.size() == 1 && nodes_[outputs_[0].id_].GetOpName() == "ilavta.dense") {
      LOG(INFO) << "[Runtime] off-loading ilavta.dense";
      // assume there're only two inputs for now
      auto input_eid = EntryID(input_nodes_[0], 0);
      auto& input_node_data = data_entry_[input_eid];

      auto wgt_eid = EntryID(input_nodes_[1], 0);
      auto& wgt_node_data = data_entry_[wgt_eid];

      // auto bias_eid = EntryID(input_nodes_[2], 0);
      // auto& bias_node_data = data_entry_[bias_eid];

      auto s_data_eid = EntryID(input_nodes_[2], 0);
      auto& s_data = data_entry_[s_data_eid];
      auto s_w_eid = EntryID(input_nodes_[3], 0);
      auto s_w = data_entry_[s_w_eid];
      auto s_act_eid = EntryID(input_nodes_[4], 0);
      auto s_act = data_entry_[s_act_eid];
      auto block = VTA_BLOCK_OUT;

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
      int batch = batch_size;
      int in_feat = n_inp_cols;
      int out_feat = n_wgt_rows;

      // int bias_dim = bias_node_data->ndim;
      // assert(bias_dim == 1);
      // int bias_size = bias_node_data->shape[0];
      int uop_size = 16 * 16 * 16 + block;

      int8_t* input = reinterpret_cast<int8_t*>(input_node_data->data);
      int8_t* weight = reinterpret_cast<int8_t*>(wgt_node_data->data);
      // int32_t* bias = reinterpret_cast<int32_t*>(bias_node_data->data);
      // int32_t* bias_buf = new int32_t[out_feat * in_feat];
      int8_t* t_weight = new int8_t[out_feat * in_feat];
      int32_t* bias_buf = new int32_t[batch * out_feat];
      memset(bias_buf, 0, sizeof(int32_t) * batch * out_feat);

      // LOG(INFO) << "Input buffer:";
      // for (int i = 0; i < batch; ++i) {
      //   for (int j = 0; j < in_feat; ++j) {
      //     std::cerr << std::hex << (int32_t)(input[i * in_feat + j]) << ' ';
      //   }
      //   std::cerr << "\n";
      // }

      // Process weights layout
      int ptr = 0;
      for (int i = 0; i < out_feat / block; ++i) {
        // int init_row = i * block;
        for (int j = 0; j < in_feat / block; ++j) {
          // int init_col = j * block;
          for (int block_in = 0; block_in < block; ++block_in) {
            // int row = init_row + block_in;
            for (int block_out = 0; block_out < block; ++block_out) {
              t_weight[ptr++] = weight[(i * block + block_in) * in_feat + (j * block + block_out)];
            }
          }
        }
      }

      // LOG(INFO) << "weights:";
      // for (int i = 0; i < out_feat; ++i) {
      //   for (int j = 0; j < in_feat; ++j) {
      //     std::cerr << (int32_t)weight[i * in_feat + j] << " ";
      //     if (j % block == block - 1) std::cerr << "| ";
      //   }
      //   std::cerr << "\n";
      //   if (i % block == block - 1) std::cerr << "\n";
      // }

      // LOG(INFO) << "Transformed weights:";
      // for (int i = 0; i < out_feat; ++i) {
      //   for (int j = 0; j < in_feat; ++j) {
      //     std::cerr << (int32_t)t_weight[i * in_feat + j] << " ";
      //     if (j % block == block - 1) std::cerr << "| ";
      //   }
      //   std::cerr << "\n";
      //   if (i % block == block - 1) std::cerr << "\n";
      // }

      // Process bias layout (pad 0)
      // for (int i = 0; i < out_feat; ++i) {
      //   for (int j = 0; j < in_feat; ++j) {
      //     if (i * in_feat + j < bias_size) {
      //       bias_buf[i * in_feat + j] = bias[i * in_feat + j];
      //     } else {
      //       bias_buf[i * in_feat + j] = 0;
      //     }
      //   }
      // }
      auto output_data = data_entry_[outputs_[0].id_];
      auto output_node = nodes_[outputs_[0].id_];
      int8_t* out_buf = reinterpret_cast<int8_t*>(output_data->data);
      float* scale_data = reinterpret_cast<float*>(s_data->data);
      float* scale_w = reinterpret_cast<float*>(s_w->data);
      float* scale_act = reinterpret_cast<float*>(s_act->data);
      // LOG(INFO) << "S_data: " << (*scale_data) << " " << "S_w: " << (*scale_w) << " " << "S_act: " << (*scale_act);
      double s_act_val = (*scale_data) * (*scale_w) / (*scale_act);
      auto imm = approximate_scale(s_act_val);
      int factor = imm[0];
      int nbits = imm[1];
      LOG(INFO) << "factor = " << factor << " " << "nbits = " << nbits << " approx = " << (double)factor / (double)(1 << nbits);

      int32_t* acc_buf = new int32_t[batch * n_wgt_rows];
      memset(acc_buf, 0, sizeof(int) * batch * n_wgt_rows);

      VTAUop* uop_buf   = getGEMMUops(16 / VTA_BATCH, 1, 1, block);
      std::string data_file = dump_datafile(input, batch * in_feat,
                t_weight, out_feat * in_feat,
                bias_buf, batch * out_feat,
                uop_buf, uop_size,
                "ilavta_dense_" + std::to_string(batch) + "x" + std::to_string(in_feat) + "_" + std::to_string(out_feat));
      std::string filename = call_node.GetAttr<std::vector<std::string>>("asm_file")[0];
      std::string ila_asm = CompileGEMM(batch, in_feat, out_feat, factor, nbits, filename);
      // nlohmann::json asm_data = get_gemm(batch, in_feat, out_feat, factor, nbits);
      // std::ofstream fout(ila_asm);
      // fout << asm_data;
      // fout.close();
      LOG(INFO) << "Size " << GetDataSize(*output_data) << " " << batch * n_wgt_rows;
      sim_time = runSimGetData("ilavta_dense", driver_dir, ila_asm, data_file,
                  batch * out_feat, batch, out_feat, acc_buf, "int8_t");
      ptr = 0;
      memcpy(out_buf, acc_buf, sizeof(int8_t) * GetDataSize(*output_data));
      delete[] acc_buf;
      delete[] t_weight;
      delete[] bias_buf;
      // for (int i = 0; i < batch; ++i) {
      //   for (int j = 0; j < out_feat; ++j) {
      //     out_buf[i * out_feat + j] = (acc_buf[i * out_feat + j]);
      //   }
      // }
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

      int batch = n_inp_rows * VTA_BATCH;
      int in_feat = n_inp_cols % VTA_BLOCK_OUT == 0 ? n_inp_cols / VTA_BLOCK_OUT : n_inp_cols / VTA_BLOCK_IN + 1;
      int bias_feat = in_feat;
      int in_channels = in_feat * VTA_BLOCK_OUT;
      int bias_channels = bias_feat * VTA_BLOCK_OUT;

      int uop_size = batch / VTA_BATCH * in_feat;

      // int32_t* input_buf =  reinterpret_cast<int32_t *>(VTAMemAlloc(sizeof(int32_t) * batch * in_channels, 0));
      // TVM does array broadcasting over the matrix in bias_add
      // int32_t* bias_buf  = reinterpret_cast<int32_t *>(VTAMemAlloc(sizeof(int32_t) * 1 * bias_channels, 0));
      // VTADeviceHandle device = VTADeviceAlloc();
      int32_t* combined_acc = reinterpret_cast<int32_t*>(VTAMemAlloc(sizeof(uint32_t) * (bias_channels + batch * in_channels), 0));
      size_t acc_ptr = 0;

      auto input = reinterpret_cast<int8_t*>(input_data->data);
      auto bias  = reinterpret_cast<int8_t*>(bias_data->data);
      for (int i = 0; i < batch; ++i) {
        for (int j = 0; j < in_channels; ++j) {
          if (i >= n_inp_rows || j >= n_inp_cols) {
            // zero padding
            combined_acc[acc_ptr++] = 0;
            // input_buf[i * in_channels + j] = 0;
            // bias_buf[i * in_channels + j] = 0;
          } else {
            // input_buf[i * in_channels + j] = input[i * n_inp_cols + j];
            combined_acc[acc_ptr++] = input[i * n_inp_cols + j];
          }
        }
      }

      for (int i = 0; i < in_channels; ++i) {
        if (i < n_inp_cols) {
          // bias_buf[i] = bias[i];
          combined_acc[acc_ptr++] = bias[i];
        } else {
          // bias_buf[i] = 0;
          combined_acc[acc_ptr++] = 0;
        }
      }

      VTAUop* uop_buf   = getBiasAddUops(batch / VTA_BATCH, in_channels / VTA_BLOCK_IN);
      
      std::string data_dump = dump_datafile(nullptr, 0, nullptr, 0, combined_acc, acc_ptr, uop_buf, uop_size, "ilavta_bias_add");
      std::string ila_asm   = call_node.GetAttr<std::vector<std::string>>("asm_file")[0];
      auto dtype            = DLDataType2String(output_data->dtype);

      sim_time = runSimGetData("ilavta_bias_add", driver_dir, ila_asm, data_dump, output_buffer_size, n_inp_rows, n_inp_cols, output_data->data, dtype);
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
      
      std::string data_dump = dump_datafile(nullptr, 0,
                                            nullptr, 0,
                                            input_buf, batch * in_channels,
                                            uop_buf, uop_size,
                                            "ilavta_relu");
      std::string ila_asm   = call_node.GetAttr<std::vector<std::string>>("asm_file")[0];
      auto dtype            = DLDataType2String(output_data->dtype);

      VTAMemFree(input_buf);
      VTAMemFree(uop_buf);

      sim_time = runSimGetData("ilavta_relu", driver_dir, ila_asm, data_dump, output_buffer_size, n_inp_rows, n_inp_cols, output_data->data, dtype);
    } else if (outputs_.size() == 1 && nodes_[outputs_[0].id_].GetOpName() == "ilavta.conv1d") {
      auto input_eid = EntryID(input_nodes_[0], 0);
      auto& input_node_data = data_entry_[input_eid];

      auto wgt_eid = EntryID(input_nodes_[1], 0);
      auto& wgt_node_data = data_entry_[wgt_eid];

      auto output_data = data_entry_[outputs_[0].id_];

      int N = input_node_data->shape[0];
      int C = input_node_data->shape[1];
      int W = input_node_data->shape[2];

      int O = wgt_node_data->shape[0];
      int I = wgt_node_data->shape[1];
      int wgtW = wgt_node_data->shape[2];
      int vec_width = I * wgtW;

      CHECK(I == C) << "C != I: this should not be type checked";

      int8_t* input_data = reinterpret_cast<int8_t*>(input_node_data->data);
      int8_t* wgt_data   = reinterpret_cast<int8_t*>(wgt_node_data->data);

      int8_t* data_col = reinterpret_cast<int8_t*>(malloc(sizeof(int8_t) * N * I * wgtW * (W - wgtW + 1)));
      int ptr = 0;
      int vec_cnt = 0;
      for (int batch = 0; batch < N; ++batch) {
        int start_offset = batch * C * W;
        for (int i = start_offset; i < start_offset + W - wgtW; ++i) {
          vec_cnt += 1;
          // flatten the current window in input to a vector
          for (int row = 0; row < I; ++row) {
            for (int col = 0; col < wgtW; ++col) {
              CHECK(ptr < N * I * wgtW * (W - wgtW + 1));
              data_col[ptr++] = input_data[i + row * W + col];
            }
          }
        }
      }
      VTAUop* uop_buf   = getGEMMUops(vec_cnt / VTA_BATCH, vec_width / VTA_BLOCK_IN, O / VTA_BLOCK_OUT, VTA_BLOCK_OUT);

      std::string data_file = dump_datafile(data_col, N * I * wgtW * (W - wgtW + 1),
                                            wgt_data, O * vec_width,
                                            nullptr, 0, uop_buf,
                                            vec_cnt / VTA_BATCH * vec_width / VTA_BLOCK_IN * O / VTA_BLOCK_OUT,
                                            "ilavta_conv1d");
      std::string ila_asm   = call_node.GetAttr<std::vector<std::string>>("asm_file")[0];
      auto dtype            = DLDataType2String(output_data->dtype);
      sim_time = runSimGetData("ilavta_conv1d", driver_dir, ila_asm, data_file, GetDataSize(*output_data), vec_cnt, O, output_data->data, dtype);
      // uint8_t* out_data = reinterpret_cast<uint8_t*>(malloc(sizeof(uint8_t) * GetDataSize(*output_data)));
      // uint8_t* raw_data = reinterpret_cast<uint8_t*>(output_data->data);
      // ptr = 0;
      // for (int batch = 0; batch < N; ++batch) {
      //   int start_offset = batch * O * (W - wgtW + 1);
      //   for (int n_kernel = 0; n_kernel < O; ++n_kernel) {
      //     for (int ncol = 0; ncol < W  - wgtW + 1; ++ncol) {
      //       out_data[ptr++] = raw_data[start_offset + n_kernel + ncol * O];
      //     }
      //   }
      // }
      // for (int i = 0; i < ptr; ++i) {
      //   raw_data[i] = out_data[i];
      // }
    }
    std::ifstream fin(wall_clock_file);
    nlohmann::json wall_clock_data = nlohmann::json::parse(fin);
    fin.close();
    if (wall_clock_data.find(op_name) == wall_clock_data.end()) {
      wall_clock_data[op_name] = nlohmann::json::array();
    }
    wall_clock_data[op_name].push_back(sim_time);
    std::ofstream fout(wall_clock_file);
    fout << wall_clock_data;
    fout.close();
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
