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
    LOG(INFO) << "[Runtime] entering " << symbol_name_ << " runtime";

    // TODO: we should probably package up all the files inside TVM
    // to avoid having to refer to other directories
    std::string driver_dir = getenv("PY_3LA_DRIVER");

    driver_dir += "flexnlp"; 

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
      LOG(INFO) << "[Runtime] operator name is " << nodes_[outputs_[0].id_].GetOpName();
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
      std::system("mkdir -p data");
      dump_data(x_data_ptr, x_data_size, "./data/inp.txt");
      dump_data(y_data_ptr, y_data_size, "./data/wgt.txt");
      dump_data(z_data_ptr, z_data_size, "./data/bias.txt");

      // calculate flexnlp tensor assembly parameters;
      CHECK(x_dim_1 % 16 == 0) "linear_layer input timestep size is: " << x_dim_1;
      CHECK(y_dim_0 % 16 == 0) "linear_layer output timestep size is: " << y_dim_0;
      CHECK(z_dim_0 % 16 == 0) "linear_layer bias size is: " << z_dim_0;
      CHECK(y_dim_0 == z_dim_0) "linear_layer bias size different from output timestep size";
      int num_vector_in = x_dim_1/16;
      int num_vector_out = y_dim_0/16;
      int num_timestep = x_dim_0;
      int is_bias = 1;
      CHECK(num_vector_out % 4 == 0) "linear_layer output vector number is : " << num_vector_out;

      // call ILAng-generated simulator
      std::stringstream call_builder;
      call_builder << "python3 " << driver_dir << "/linear_layer_driver.py "
                   << num_vector_in << " " << num_vector_out << " "
                   << num_timestep << " " << is_bias << " " << symbol_name_;
      std::string call_cmd = call_builder.str();

      LOG(INFO) << "calling flexnlp linear layer driver";
      auto res = std::system(call_cmd.c_str());
      CHECK(res == 0) << "Error executing simulator " << call_cmd;

      // retrieve the results
      retrieve_result(o_data_ptr, o_data_size, "./data/result.txt");
#if 0
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

    } 
    else if (outputs_.size() == 1 && nodes_[outputs_[0].id_].GetOpName() == "ilaflex.lstm") {
    // else if (nodes_[outputs_[0].id_].GetOpName() == "ilaflex.lstm") {
      LOG(INFO) << "[Runtime] operator name is " << nodes_[outputs_[0].id_].GetOpName();
      LOG(INFO) << "LSTM input nodes size: " << input_nodes_.size();
      // TODO: why initial state only has single vector?
      for (auto it : input_nodes_) {
        auto data_node_ptr = data_entry_[EntryID(it, 0)];
        LOG(INFO) << it << '\t' << (data_node_ptr->ndim) << '\t' << data_node_ptr->dtype << '\t' << GetDataSize(*data_node_ptr);
        for (auto i = 0; i < data_node_ptr->ndim; i++) {
          std::cout << "dim_" << i << ": " << data_node_ptr->shape[i] << '\t';
        }
        std::cout << std::endl;
      }
      // check output data dimension
      auto out_data_ptr = data_entry_[EntryID(outputs_[0])];
      LOG(INFO) << "LSTM output dimension: " << out_data_ptr->ndim;
      for (auto i = 0; i < out_data_ptr->ndim; i++) {
        std::cout << "dim_" << i << ": " << out_data_ptr->shape[i] << '\t';
      }
      std::cout << std::endl;


      // lstm input
      auto eid_inp = EntryID(input_nodes_[0], 0);
      auto& node_data_inp = data_entry_[eid_inp];
      CHECK(node_data_inp->ndim == 3);
      auto num_ts = node_data_inp->shape[1];
      CHECK(node_data_inp->shape[2] % 16 == 0);
      auto num_v_in = (int)(node_data_inp->shape[2])/16;
      auto inp_data_size = GetDataSize(*node_data_inp)/sizeof(float);
      float* inp_data_ptr = new float[inp_data_size];
      std::copy(reinterpret_cast<float*>(node_data_inp->data),
                reinterpret_cast<float*>(node_data_inp->data) + inp_data_size,
                inp_data_ptr);
      
      // lstm initial state
      // TODO: support non-zero lstm initial state in the future

      // lstm i2h_wgt
      auto eid_i2h_wgt = EntryID(input_nodes_[2], 0);
      auto& node_data_i2h_wgt = data_entry_[eid_i2h_wgt];
      CHECK(node_data_i2h_wgt->ndim == 2);
      auto i2h_wgt_data_size = GetDataSize(*node_data_i2h_wgt)/sizeof(float);

      auto num_v_out = (int)(node_data_i2h_wgt->shape[0])/64;
      CHECK(i2h_wgt_data_size == 16*num_v_in * 4 * 16*num_v_out);
      float* i2h_wgt_data_ptr = new float[i2h_wgt_data_size];
      std::copy(reinterpret_cast<float*>(node_data_i2h_wgt->data),
                reinterpret_cast<float*>(node_data_i2h_wgt->data) + i2h_wgt_data_size,
                i2h_wgt_data_ptr);
      
      // lstm h2h_wgt
      auto eid_h2h_wgt = EntryID(input_nodes_[3], 0);
      auto& node_data_h2h_wgt = data_entry_[eid_h2h_wgt];
      CHECK(node_data_h2h_wgt->ndim == 2);
      auto h2h_wgt_data_size = GetDataSize(*node_data_h2h_wgt)/sizeof(float);
      CHECK(h2h_wgt_data_size == 16*num_v_out * 4 * 16*num_v_out);
      float* h2h_wgt_data_ptr = new float[h2h_wgt_data_size];
      std::copy(reinterpret_cast<float*>(node_data_h2h_wgt->data),
                reinterpret_cast<float*>(node_data_h2h_wgt->data) + h2h_wgt_data_size,
                h2h_wgt_data_ptr);
      
      // lstm bias
      auto eid_bias = EntryID(input_nodes_[4], 0);
      auto& node_data_bias = data_entry_[eid_bias];
      CHECK(node_data_bias->ndim == 1);
      auto bias_data_size = GetDataSize(*node_data_bias)/sizeof(float);
      CHECK(bias_data_size == 4 * 16*num_v_out);
      float* bias_data_ptr = new float[bias_data_size];
      std::copy(reinterpret_cast<float*>(node_data_bias->data),
                reinterpret_cast<float*>(node_data_bias->data) + bias_data_size,
                bias_data_ptr);

      // output
      // LSTM output is flatten?
      // auto eid_o = outputs_[0].id_;
      auto node_data_o = data_entry_[EntryID(outputs_[0])];
      CHECK(node_data_o->ndim == 3);
      auto o_data_size = GetDataSize(*node_data_o)/sizeof(float);
      CHECK(o_data_size == 16*num_v_out*num_ts);
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
      std::system("mkdir -p data");
      dump_data(inp_data_ptr, inp_data_size, "./data/lstm_inp.txt");
      dump_data(i2h_wgt_data_ptr, i2h_wgt_data_size, "./data/lstm_i2h_wgt.txt");
      dump_data(h2h_wgt_data_ptr, h2h_wgt_data_size, "./data/lstm_h2h_wgt.txt");
      dump_data(bias_data_ptr, bias_data_size, "./data/lstm_bias.txt");

      // set flexnlp tensor assembly parameters;
      int is_bias = 1;
      int is_zero_first = 1;

      // call ILAng-generated simulator
      std::stringstream call_builder;
      call_builder << "python3 " << driver_dir << "/lstm_driver.py "
                   << num_v_in << " " << num_v_out << " "
                   << num_ts << " " << is_bias << " " << is_zero_first << " "
                   << symbol_name_;
      std::string call_cmd = call_builder.str();

      LOG(INFO) << "calling flexnlp lstm driver";
      auto res = std::system(call_cmd.c_str());
      CHECK(res == 0) << "Error executing simulator " << call_cmd;

      // retrieve the results
      retrieve_result(o_data_ptr, o_data_size, "./data/lstm_out.txt");
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
