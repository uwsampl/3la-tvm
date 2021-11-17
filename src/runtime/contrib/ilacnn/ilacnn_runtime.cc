#include <tvm/node/reflection.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/json.hpp>
#include <tvm/tir/op.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class IlaCNNRuntime : public JSONRuntimeBase {
 public:
  IlaCNNRuntime(const std::string& symbol_name, const std::string& graph_json,
                 const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "ilacnn"; }

  void Init(const Array<NDArray>& consts) override {
    // CHECK(consts.size() == 0);
  }

  void Run() override {
    CHECK(symbol_name_.substr(0, 6) == "ilacnn") << symbol_name_;
    LOG(INFO) << "[Runtime] entering " << symbol_name_ << " runtime";

    const std::string wall_clock_file = "ilacnn_wallclock.json";
    auto op_name = nodes_[outputs_[0].id_].GetOpName();
    std::chrono::_V2::system_clock::time_point start_time;
    std::chrono::_V2::system_clock::time_point end_time;

    if (outputs_.size() == 1 && input_nodes_.size() == 2 &&
        nodes_[outputs_[0].id_].GetOpName() == "ilacnn.conv2d") {
      auto call_node = nodes_[outputs_[0].id_];

      // data
      auto eid_data = EntryID(input_nodes_[0], 0);
      auto& data_info = data_entry_[eid_data];
      CHECK(data_info->ndim == 4);
      std::cout << "Data shape: ("
                << data_info->shape[0] << ", "
                << data_info->shape[1] << ", "
                << data_info->shape[2] << ", "
                << data_info->shape[3]
                << ")" << std::endl;
      if (data_info->shape[0] > 1) {
        LOG(FATAL) << "HLSCNN conv only support batch num 1 for now";
        return;
      }
      auto inp_data_size = GetDataSize(*data_info)/sizeof(float);
      CHECK(inp_data_size == data_info->shape[1] * data_info->shape[2] * data_info->shape[3]);
      float* inp_data_ptr = new float[inp_data_size];
      std::copy(reinterpret_cast<float*>(data_info->data),
                reinterpret_cast<float*>(data_info->data) + inp_data_size,
                inp_data_ptr);
      dump_data(inp_data_ptr, inp_data_size, "./data/inp.txt");

      // weight
      auto eid_weight = EntryID(input_nodes_[1], 0);
      auto& weight_info = data_entry_[eid_weight];
      CHECK(weight_info->ndim == 4);
      std::cout << "Weight shape: ("
                << weight_info->shape[0] << ", "
                << weight_info->shape[1] << ", "
                << weight_info->shape[2] << ", "
                << weight_info->shape[3]
                << ")" << std::endl;
      auto wgt_data_size = GetDataSize(*weight_info)/sizeof(float);
      CHECK(wgt_data_size == weight_info->shape[0] * weight_info->shape[1] *
                             weight_info->shape[2] * weight_info->shape[3]);
      float* wgt_data_ptr = new float[wgt_data_size];
      std::copy(reinterpret_cast<float*>(weight_info->data),
                reinterpret_cast<float*>(weight_info->data) + wgt_data_size,
                wgt_data_ptr);
      dump_data(wgt_data_ptr, wgt_data_size, "./data/wgt.txt");

      // output
      auto eid_o = outputs_[0].id_;
      auto out_info = data_entry_[eid_o];
      // auto out_info = data_entry_[EntryID(outputs_[0])];
      CHECK(out_info->ndim == 4);
      std::cout << "Output shape: ("
                << out_info->shape[0] << ", "
                << out_info->shape[1] << ", "
                << out_info->shape[2] << ", "
                << out_info->shape[3]
                << ")" << std::endl;
      auto o_data_size = GetDataSize(*out_info)/sizeof(float);
      CHECK(o_data_size == out_info->shape[0] * out_info->shape[1] * 
                           out_info->shape[2] * out_info->shape[3]);
      float* o_data_ptr = new float[o_data_size];
      
      // attributes
      auto strides = call_node.GetAttr<std::vector<std::string>>("strides");
      auto padding = call_node.GetAttr<std::vector<std::string>>("padding");
      auto data_layout = call_node.GetAttr<std::vector<std::string>>("data_layout");
      auto kernel_layout = call_node.GetAttr<std::vector<std::string>>("kernel_layout");
      // etc

      std::cout << "Strides: " << "(";
      for (const auto dim : strides) {
        std::cout << dim << ",";
      }
      std::cout << ")" << std::endl;
      std::cout << "Padding: " << "(";
      for (const auto dim : padding) {
        std::cout << dim << ",";
      }
      std::cout << ")" << std::endl;
      std::cout << "Data layout: " << data_layout[0] << std::endl;
      std::cout << "Kernel layout: " << kernel_layout[0] << std::endl;

      // Instantiate and call driver
      std::string driver_dir = getenv("PY_3LA_DRIVER");
      driver_dir += "/hlscnn";
      std::stringstream call_builder;
      call_builder << "python3 " << driver_dir << "/conv_layer_driver.py "
        << "--in_size " << data_info->shape[1] << " " << data_info->shape[2] << " " << data_info->shape[3] << " "
        << "--out_size " << out_info->shape[1] << " " << out_info->shape[2] << " " << out_info->shape[3] << " "
        << "--kernel_size " << weight_info->shape[0] << " " << weight_info->shape[1] << " "
                            << weight_info->shape[2] << " " << weight_info->shape[3] << " "
        << "--stride " << strides[0] << " " << strides[1] << " "
        << "--op_name " << symbol_name_;

      if (getenv("TVM_3LA_REF_RUN")) {
        LOG(INFO) << "Differential debugging enabled. Getting SW ref results!";
        call_builder << " " << "--ref_run " << "True";
      }

      std::string call_cmd = call_builder.str();


      LOG(INFO) << "calling hlscnn driver\n" << "command: " << call_cmd;
      start_time = std::chrono::high_resolution_clock::now();
      auto res = std::system(call_cmd.c_str());
      end_time = std::chrono::high_resolution_clock::now();
      CHECK(res == 0) << "Error executing simulator " << call_cmd;

      // retrieve the results
      retrieve_result(o_data_ptr, o_data_size, "./data/conv_result.txt");
      std::copy(o_data_ptr, o_data_ptr + o_data_size,
                reinterpret_cast<float*>(out_info->data));
      free(inp_data_ptr);
      free(wgt_data_ptr);
      free(o_data_ptr);

    } else {
      LOG(FATAL) << "Unknown pattern " << symbol_name_;
    }
    std::ifstream fin(wall_clock_file);
    nlohmann::json wall_clock_data;
    fin >> wall_clock_data;
    fin.close();
    if (wall_clock_data.find(op_name) == wall_clock_data.end()) {
      wall_clock_data[op_name] = nlohmann::json::array({});
    }
    wall_clock_data[op_name].push_back(
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count()
    );
    std::ofstream fout(wall_clock_file);
    fout << wall_clock_data;
    fout.close();
    LOG(INFO) << "[Runtime] exit " << symbol_name_ << " runtime, resume host";
  }

  void dump_data(float* data_ptr, unsigned long size, std::string path) {
    std::ofstream fout;
    std::stringstream ss;
    fout.open(path, std::ios::out | std::ios::trunc);
    for (auto i = 0; i < size; ++i) {
      ss << data_ptr[i] << '\n';
    }
    fout << ss.rdbuf();
    fout.close();
  }

  void retrieve_result(float* data_ptr, unsigned long& size, std::string path) {
    // retrieve flexnlp results
    std::ifstream fin;
    fin.open(path, std::ios::in);
    std::string float_str;
    unsigned long cntr = 0;

    while(std::getline(fin, float_str)) {
      if (cntr >= size) {
        LOG(FATAL) << "wrong number of elements in the result tensor";
      }
      data_ptr[cntr] = std::stof(float_str);
      ++cntr;
    }
  }

 protected:
 private:
};  // namespace runtime

runtime::Module IlaCNNRuntimeCreate(String symbol_name, String graph_json,
                                     const Array<String>& const_names) {
  auto n = make_object<IlaCNNRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.IlaCNNRuntimeCreate")
    .set_body_typed(IlaCNNRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_ilacnn")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<IlaCNNRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
