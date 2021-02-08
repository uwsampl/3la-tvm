#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

class ILAFlexRuntime : public JSONRuntimeBase {
 public:
  ILAFlexRuntime(const std::string& symbol_name, const std::string& graph_json,
                 const Array<String> const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  const char* type_key() const { return "ilaflex"; }

  void Init(const Array<NDArray>& consts) override {
    // CHECK(consts.size() == 0);
  }

  void Run() override {
    CHECK(symbol_name_.substr(0, 7) == "ilaflex") << symbol_name_;
    LOG(INFO) << "[Runtime] enter " << symbol_name_ << " runtime";

    if (outputs_.size() == 1 && input_nodes_.size() == 3 &&
        nodes_[outputs_[0].id_].GetOpName() == "ilaflex.linear") {
      /*
       * out = bias_add(batch_matmul(x, y), z)
       *
       * input x:
       *  - dimension: (x_dim_0, x_dim_1)
       *  - data: x_data_ptr, x_data_size
       *
       * input y:
       *  - dimension: (y_dim_0, y_dim_1)
       *  - data: y_data_ptr, y_data_size
       *
       * input z:
       *  - dimension: (z_dim_0)
       *  - data: z_data_ptr, z_data_size
       *
       * output:
       *  - dimension: (o_dim_0, o_dim_1)
       *  - data: o_data_ptr, o_data_size
       */

      // x
      auto eid_x = EntryID(input_nodes_[0], 0);
      auto& node_data_x = data_entry_[eid_x];
      CHECK(node_data_x->ndim == 2);
      auto x_dim_0 = node_data_x->shape[0];
      auto x_dim_1 = node_data_x->shape[1];
      auto x_data_size = GetDataSize(*node_data_x)/sizeof(float);
      float* x_data_ptr = new float[x_data_size];
      std::copy(reinterpret_cast<float*>(node_data_x->data),
                reinterpret_cast<float*>(node_data_x->data) + x_data_size,
                x_data_ptr);

      // y
      auto eid_y = EntryID(input_nodes_[1], 0);
      auto& node_data_y = data_entry_[eid_y];
      CHECK(node_data_y->ndim == 2);
      auto y_dim_0 = node_data_y->shape[0];
      auto y_dim_1 = node_data_y->shape[1];
      auto y_data_size = GetDataSize(*node_data_y)/sizeof(float);
      float* y_data_ptr = new float[y_data_size];
      std::copy(reinterpret_cast<float*>(node_data_y->data),
                reinterpret_cast<float*>(node_data_y->data) + y_data_size,
                y_data_ptr);

      // z
      auto eid_z = EntryID(input_nodes_[2], 0);
      auto& node_data_z = data_entry_[eid_z];
      CHECK(node_data_z->ndim == 1);
      auto z_dim_0 = node_data_z->shape[0];
      auto z_data_size = GetDataSize(*node_data_z)/sizeof(float);
      float* z_data_ptr = new float[z_data_size];
      std::copy(reinterpret_cast<float*>(node_data_z->data),
                reinterpret_cast<float*>(node_data_z->data) + z_data_size,
                z_data_ptr);

      // output
      auto eid_o = outputs_[0].id_;
      auto node_data_o = data_entry_[eid_o];
      CHECK(node_data_o->ndim == 2);
      auto o_dim_0 = node_data_o->shape[0];
      auto o_dim_1 = node_data_o->shape[1];
      auto o_data_size = GetDataSize(*node_data_o)/sizeof(float);
      float* o_data_ptr = new float[o_data_size];

      /* TODO
       *  - FlexNLP ILA simulator is available in $PATH as "flexnlp_ila_sim"
       *  - generate tensor-level assembly
       *  - generate data library
       *  - translate to ILA instruction program fragment
       *  - invoke the ILA simulator
       *  - read back the result and store to o_data_ptr
       */
      // dump data to files
      dump_data(x_data_ptr, x_data_size, "./data/inp.txt");
      dump_data(y_data_ptr, y_data_size, "./data/wgt.txt");
      dump_data(z_data_ptr, z_data_size, "./data/bias.txt");

      // calculate flexnlp tensor assembly parameters;
      int num_vector_in = x_dim_1/16;
      int num_vector_out = y_dim_0/16;
      int num_timestep = x_dim_0;
      int is_bias = 1;

      // call ILAng-generated simulator
      std::string call_cmd = "python3 linear_layer_driver.py " + 
                             std::to_string(num_vector_in) + " " +
                             std::to_string(num_vector_out) + " " +
                             std::to_string(num_timestep) + " " +
                             std::to_string(is_bias);
      // std::string command = "echo \"call assembly helper\"";
      std::system("echo \"calling flexnlp linear layer driver\"");
      auto res = std::system(call_cmd.c_str());
      CHECK(res == 0) << "Error executing simulator " << call_cmd;

      // retrieve the results
      retrieve_result(o_data_ptr, o_data_size, "./data/result.txt");
#if 1
      LOG(INFO) << "x_dimension: " << x_dim_0 << ", " << x_dim_1;
      LOG(INFO) << "x_data_size: " << x_data_size;
      LOG(INFO) << "y_dimension: " << y_dim_0 << ", " << y_dim_1;
      LOG(INFO) << "y_data_size: " << y_data_size;
      LOG(INFO) << "z_dimension: " << z_dim_0;
      LOG(INFO) << "z_data_size: " << z_data_size;
      LOG(INFO) << o_dim_0 << ", " << o_dim_1;
#endif

      // copy the result and resume
      std::copy(o_data_ptr, o_data_ptr + o_data_size,
                reinterpret_cast<float*>(node_data_o->data));

    } else {
      LOG(FATAL) << "Unknown pattern " << symbol_name_;
    }
    LOG(INFO) << "[Runtime] exit " << symbol_name_ << " runtime, resume host";
  }

  void dump_data(float* data_ptr, unsigned long& size, std::string path) {
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
    fin.open("./data/result.txt", std::ios::in);
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

runtime::Module ILAFlexRuntimeCreate(String symbol_name, String graph_json,
                                     const Array<String>& const_names) {
  auto n = make_object<ILAFlexRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.ILAFlexRuntimeCreate")
    .set_body_typed(ILAFlexRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_ilaflex")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<ILAFlexRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
