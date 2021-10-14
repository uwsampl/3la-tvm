#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <tvm/support/json.hpp>

#include <boost/numeric/conversion/cast.hpp>
#include <boost/lexical_cast.hpp>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <chrono>
#include <math.h>

#include "../json/json_node.h"
#include "../json/json_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;
using namespace tvm::runtime::json;

using boost::lexical_cast;
using boost::numeric_cast;

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

    // TODO: we should probably package up all the files inside TVM
    // to avoid having to refer to other directories
    std::string driver_dir = getenv("PY_3LA_DRIVER");
    driver_dir += "/flexnlp"; 
    // LOG(INFO) << "[Runtime] operator name is " << nodes_[outputs_[0].id_].GetOpName();
    // LOG(INFO) << "outputs size: " << outputs_.size() << '\t' << "input_size: " << input_nodes_.size();

    auto op_name = nodes_[outputs_[0].id_].GetOpName();
    const std::string wall_clock_file = "ilaflex_wallclock.json";
    std::chrono::_V2::system_clock::time_point start_time;
    std::chrono::_V2::system_clock::time_point end_time;

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
      int64_t x_dim_0 = node_data_x->shape[0];
      int64_t x_dim_1 = node_data_x->shape[1];
      auto x_data_size = GetDataSize(*node_data_x)/(node_data_x->dtype.bits/8);
      CHECK(x_data_size == x_dim_0 * x_dim_1) "wrong input data size";
      // y
      auto eid_y = EntryID(input_nodes_[1], 0);
      auto& node_data_y = data_entry_[eid_y];
      CHECK(node_data_y->ndim == 2);
      int64_t y_dim_0 = node_data_y->shape[0];
      int64_t y_dim_1 = node_data_y->shape[1];
      auto y_data_size = GetDataSize(*node_data_y)/(node_data_y->dtype.bits/8);
      CHECK(y_data_size == y_dim_0 * y_dim_1) "wrong weights data size";
      // z
      auto eid_z = EntryID(input_nodes_[2], 0);
      auto& node_data_z = data_entry_[eid_z];
      CHECK(node_data_z->ndim == 1);
      int64_t z_dim_0 = node_data_z->shape[0];
      auto z_data_size = GetDataSize(*node_data_z)/(node_data_z->dtype.bits/8);
      CHECK(z_data_size == z_dim_0) "wrong bias data size";
      // output
      auto eid_o = outputs_[0].id_;
      auto node_data_o = data_entry_[eid_o];
      CHECK(node_data_o->ndim == 2);
      int64_t o_dim_0 = node_data_o->shape[0];
      int64_t o_dim_1 = node_data_o->shape[1];
      auto o_data_size = GetDataSize(*node_data_o)/(node_data_o->dtype.bits/8);
      CHECK(o_dim_0 == x_dim_0);
      CHECK(o_dim_1 == y_dim_0);
      CHECK(o_data_size == o_dim_0 * o_dim_1);


      /* TODO
       *  - FlexNLP ILA simulator is available in $PATH as "flexnlp_ila_sim"
       *  - generate tensor-level assembly
       *  - generate data library
       *  - translate to ILA instruction program fragment
       *  - invoke the ILA simulator
       *  - read back the result and store to o_data_ptr
       */
      // pad the data if shape doesn't align
      void* x_data_ext_ptr;
      void* y_data_ext_ptr;
      void* z_data_ext_ptr;
      int64_t x_dim_1_ext = ceil(x_dim_1 / 16) * 16;
      int64_t y_dim_0_ext = (ceil(y_dim_0 / 64) * 64);
      int64_t y_dim_1_ext = (ceil(y_dim_1 / 16) * 16);
      int64_t z_dim_0_ext = ceil(z_dim_0 / 64) * 64;

      int64_t x_data_ext_size = x_dim_0 * x_dim_1_ext;
      int64_t y_data_ext_size = y_dim_0_ext * y_dim_1_ext; 
      int64_t z_data_ext_size = z_dim_0_ext;
      x_data_ext_ptr = extend_data_2d(node_data_x->data, x_data_ext_ptr, x_dim_0, x_dim_1_ext, 
                                      x_dim_0, x_dim_1, node_data_x->dtype.bits/8);
      y_data_ext_ptr = extend_data_2d(node_data_y->data, y_data_ext_ptr, y_dim_0_ext, y_dim_1_ext, 
                                      y_dim_0, y_dim_1, node_data_y->dtype.bits/8);
      z_data_ext_ptr = extend_data_2d(node_data_z->data, z_data_ext_ptr, z_dim_0_ext, 1,
                                      z_dim_0, 1, node_data_z->dtype.bits/8);
      // declare pointer and space for the output
      auto o_data_ext_size = x_dim_0 * y_dim_0_ext;
      void* o_data_ext_ptr = std::calloc(o_data_ext_size, node_data_o->dtype.bits/8);
    
      // dump data to files
      std::system("mkdir -p data");
      std::cout << "dumping data" << std::endl;
      dump_data(x_data_ext_ptr, x_data_ext_size, "./data/inp.txt", node_data_x->dtype.code);
      dump_data(y_data_ext_ptr, y_data_ext_size, "./data/wgt.txt", node_data_y->dtype.code);
      dump_data(z_data_ext_ptr, z_data_ext_size, "./data/bias.txt", node_data_z->dtype.code);

      // int num_vector_in = x_dim_1/16;
      // int num_vector_out = y_dim_0/16;
      int num_vector_in = x_dim_1_ext / 16;
      int num_vector_out = y_dim_0_ext / 16;
      int num_timestep = x_dim_0;
      int is_bias = 1;
      CHECK(num_vector_out % 4 == 0) "linear_layer output vector number is : " << num_vector_out;

      // call ILAng-generated simulator
      std::stringstream call_builder;
      // call_builder << "python3 " << driver_dir << "/linear_layer_driver.py "
      //              << num_vector_in << " " << num_vector_out << " "
      //              << num_timestep << " " << is_bias << " " << symbol_name_;
      CHECK(node_data_x->dtype.code == 0 || node_data_x->dtype.code == 2) << "Unsupported datatype!";
      std::string dtype = (node_data_x->dtype.code == 0) ? "int8" : "float32";
      call_builder << "python3 " << driver_dir << "/linear_layer_driver.py" << " "
                   << "--num_v_in " << num_vector_in << " "
                   << "--num_v_out " << num_vector_out << " "
                   << "--num_timestep " << num_timestep << " "
                   << "--is_bias " << "True" << " "
                   << "--dtype " << dtype << " "
                   << "--op_name " << symbol_name_;
      std::string call_cmd = call_builder.str();

      LOG(INFO) << "calling flexnlp linear layer driver";
      start_time = std::chrono::high_resolution_clock::now();
      auto res = std::system(call_cmd.c_str());
      end_time = std::chrono::high_resolution_clock::now();
      CHECK(res == 0) << "Error executing simulator " << call_cmd;

      // retrieve the results, retrieve to the extended buffer first
      // retrieve_result(node_data_o->data, o_data_size, "./data/result.txt", node_data_o->dtype.code);
      retrieve_result(o_data_ext_ptr, o_data_ext_size, "./data/result.txt", node_data_o->dtype.code);
      unpad_result_2d(node_data_o->data, o_data_ext_ptr, x_dim_0, y_dim_0_ext, 
                      o_dim_0, o_dim_1, node_data_o->dtype.bits/8);
#if 0
      LOG(INFO) << "x_dimension: " << x_dim_0 << ", " << x_dim_1;
      LOG(INFO) << "x_data_size: " << x_data_size;
      LOG(INFO) << "y_dimension: " << y_dim_0 << ", " << y_dim_1;
      LOG(INFO) << "y_data_size: " << y_data_size;
      LOG(INFO) << "z_dimension: " << z_dim_0;
      LOG(INFO) << "z_data_size: " << z_data_size;
      LOG(INFO) << o_dim_0 << ", " << o_dim_1;
#endif

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
      start_time = std::chrono::high_resolution_clock::now();
      auto res = std::system(call_cmd.c_str());
      end_time = std::chrono::high_resolution_clock::now();
      CHECK(res == 0) << "Error executing simulator " << call_cmd;

      // retrieve the results
      retrieve_result(o_data_ptr, o_data_size, "./data/lstm_out.txt");
      // copy the result and resume
      std::copy(o_data_ptr, o_data_ptr + o_data_size,
                reinterpret_cast<float*>(node_data_o->data));
    } else if (outputs_.size() == 1 && nodes_[outputs_[0].id_].GetOpName() == "ilaflex.attention") {
      LOG(INFO) << "[Runtime] operator name is " << nodes_[outputs_[0].id_].GetOpName();
      // dec_data
      auto eid_dec = EntryID(input_nodes_[0], 0);
      auto& node_data_dec = data_entry_[eid_dec];
      // CHECK(node_data_inp->ndim == 3);
      // auto num_ts = node_data_inp->shape[1];
      // CHECK(node_data_inp->shape[2] % 16 == 0);
      auto num_v_in = (int)(node_data_dec->shape[2])/16;
      auto dec_data_size = GetDataSize(*node_data_dec)/sizeof(float);
      float* inp_data_ptr = new float[dec_data_size];
      std::copy(reinterpret_cast<float*>(node_data_dec->data),
                reinterpret_cast<float*>(node_data_dec->data) + dec_data_size,
                inp_data_ptr);

      // attention enc_data
      auto eid_enc_data = EntryID(input_nodes_[1], 0);
      auto& enc_data = data_entry_[eid_enc_data];
      auto num_ts = enc_data->shape[1];
      auto enc_data_size = GetDataSize(*enc_data)/sizeof(float);
      float* enc_data_ptr = new float[enc_data_size];
      std::copy(reinterpret_cast<float*>(enc_data->data),
                reinterpret_cast<float*>(enc_data->data) + enc_data_size,
                enc_data_ptr);

      auto node_data_o = data_entry_[EntryID(outputs_[0])];
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
      dump_data(inp_data_ptr, dec_data_size, "./data/dec.txt");
      dump_data(enc_data_ptr, enc_data_size, "./data/enc.txt");

      // set flexnlp tensor assembly parameters;
      int mem_idx_enc = 0;
      int mem_idx_dec = 0;

      std::cerr << "dec shape: (" << node_data_dec->shape[0] << ", " << node_data_dec->shape[1] << ", " << node_data_dec->shape[2] << ")\n";
      std::cerr << "num_v_in: " << num_v_in << "\n";
      std::cerr << "enc shape: (" << enc_data->shape[0] << ", " << enc_data->shape[1] << ", " << enc_data->shape[2] << ")\n";
      std::cerr << "num_ts: " << num_ts << "\n";

      // call ILAng-generated simulator
      std::stringstream call_builder;
      call_builder << "python3 " << driver_dir << "/attention_driver.py "
                   << "--num_ts " << num_ts << " --num_v " << num_v_in << " --mem_idx_enc "
                   << mem_idx_enc << " --mem_idx_dec " << mem_idx_dec;
      std::string call_cmd = call_builder.str();
      // std::cerr << "calling " << call_cmd << "\n";

      LOG(INFO) << "calling flexnlp attention driver";
      start_time = std::chrono::high_resolution_clock::now();
      auto res = std::system(call_cmd.c_str());
      end_time = std::chrono::high_resolution_clock::now();
      CHECK(res == 0) << "Error executing simulator " << call_cmd;

      // retrieve the results
      retrieve_result(o_data_ptr, o_data_size, "./data/result_attention_ila.txt");
      // copy the result and resume
      std::copy(o_data_ptr, o_data_ptr + o_data_size,
                reinterpret_cast<float*>(node_data_o->data));
    } else {
      LOG(FATAL) << "Unknown pattern " << outputs_.size() << " " << nodes_[outputs_[0].id_].GetOpName();
    }
    std::ifstream fin(wall_clock_file);
    nlohmann::json wall_clock_data = nlohmann::json::parse(fin);
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

  void* extend_data_2d(void *ori_data_ptr, void *ext_data_ptr, int64_t ext_dim_0,
    int64_t ext_dim_1, const int64_t ori_dim_0, const int64_t ori_dim_1, uint8_t dbyte) {
    // This function pad the 2D tensor data in the given dimension
    auto ext_size = ext_dim_0 * ext_dim_1;
    ext_data_ptr = std::calloc(ext_size, dbyte);
    for (auto i = 0; i < ori_dim_0; i++) {
      std::memcpy(
        ext_data_ptr + i * ext_dim_1 * dbyte,
        ori_data_ptr + i * ori_dim_1 * dbyte,
        ori_dim_1 * dbyte
      );
    }
    return ext_data_ptr;
  }

  void unpad_result_2d(void *ori_data_ptr, void *ext_data_ptr, int64_t ext_dim_0,
    int64_t ext_dim_1, const int64_t ori_dim_0, const int64_t ori_dim_1, uint8_t dbyte) {
    // This function unpads the final results
    std::cout << "unpad the result 2D data (ext_dim_0, ext_dim_1) - " 
              << ext_dim_0 << '\t' << ext_dim_1 << std::endl;
    for (auto i = 0; i < ori_dim_0; i++) {
      std::memcpy(
        ori_data_ptr + i * ori_dim_1 * dbyte,
        ext_data_ptr + i * ext_dim_1 * dbyte,
        ori_dim_1 * dbyte
      );
    }
  }

  void dump_data(void* data_ptr, unsigned long size, std::string path, uint8_t type_code = 2) {
    std::ofstream fout;
    std::stringstream ss;
    fout.open(path, std::ios::out | std::ios::trunc);
    // cast the data ptr to the correct datatype according to the type_code
    for (auto i = 0; i < size; ++i) {
      if (type_code == 0) {
        ss << (signed)*((int8_t*)data_ptr + i) << '\n';
      } else {
        ss << *((float*)data_ptr + i) << '\n';
      }
    }
    fout << ss.rdbuf();
    fout.close();
  }

  void retrieve_result(void* data_ptr, unsigned long size, std::string path, uint8_t type_code = 2) {
    // retrieve flexnlp results
    std::ifstream fin;
    fin.open(path, std::ios::in);
    std::string result_str;
    unsigned long cntr = 0;
    auto data_ptr_int8 = reinterpret_cast<int8_t*>(data_ptr);
    auto data_ptr_f32 = reinterpret_cast<float*>(data_ptr);

    while(std::getline(fin, result_str)) {
      if (cntr >= size) {
        LOG(FATAL) << "wrong number of elements in the result tensor";
      }
      if (type_code == 0) {
        int8_t tmp = lexical_cast<int>(result_str);
        data_ptr_int8[cntr] = tmp;
      } else {
        data_ptr_f32[cntr] = lexical_cast<float>(result_str);
      }
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
