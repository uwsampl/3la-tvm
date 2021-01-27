#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

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

  const char* type_key() const { return "ilaflex"; }  // namespace contrib

  void Init(const Array<NDArray>& consts) override {
    CHECK(consts.size() == 0) << "matmul should have no consts";
  }

  void Run() override {
    CHECK(symbol_name_.substr(0, 7) == "ilaflex") << symbol_name_;
    LOG(INFO) << "[Runtime] enter " << symbol_name_ << " runtime";

    if (outputs_.size() == 1 &&
        nodes_[outputs_[0].id_].GetOpName() == "ilaflex.linear") {
      LOG(INFO) << "[Runtime] off-loading ilaflex.linear";

      // get input node data
      for (size_t i = 0; i < input_nodes_.size(); ++i) {
        auto eid = EntryID(input_nodes_[i], 0);
        auto& node_data = data_entry_[eid];

        auto ndim = node_data->ndim;
        // CHECK(ndim == 3) << "batch_matmul input dimension: " << ndim;
        LOG(INFO) << "Input " << eid << " dim: " << ndim;

        LOG(INFO) << "[Runtime-TODO] virtual store node " << eid << " data:";
        auto buffer_size = GetDataSize(*node_data);
        char* dst = new char[buffer_size];
        std::copy(reinterpret_cast<char*>(node_data->data),
                  reinterpret_cast<char*>(node_data->data) + buffer_size, dst);

        std::string data_str = "";
        for (size_t j = 0; j < buffer_size; j++) {
          data_str = data_str + " " + std::to_string((uint8_t)(dst[j]));
        }
        LOG(INFO) << "[Runtime-TODO]   <" << data_str << ">";

#if 0
        for (auto dim = 0; dim < data_entry_[eid]->ndim; dim++) {
          LOG(INFO) << "shape: " << data_entry_[eid]->shape[dim];
        }
#endif
      }

      // call ILAng-generated simulator
      std::string simulator = "/root/ilasim/flex";
      std::string command = "echo \"call assembly helper\"";
      auto res = std::system(command.c_str());
      CHECK(res == 0) << "Error executing simulator " << command;

      // reads back the output
      auto output_node_id = outputs_[0].id_;
      auto output_node_data = data_entry_[output_node_id];

      {
        LOG(INFO) << "[Runtime-TODO] read back simulation results (fake):";
        auto buffer_size = GetDataSize(*output_node_data);
        char* src = new char[buffer_size];
        std::copy(src, src + buffer_size,
                  reinterpret_cast<char*>(output_node_data->data));
        std::string data_str = "";
        for (size_t j = 0; j < buffer_size; j++) {
          data_str = data_str + " " + std::to_string((uint8_t)(src[j]));
        }
        LOG(INFO) << "[Runtime-TODO]  <" << data_str << ">";
      }

      LOG(INFO) << "[Runtime] resume execution";

    } else {
      LOG(FATAL) << "Unknown pattern " << symbol_name_;
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
