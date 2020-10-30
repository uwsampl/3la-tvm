#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <cstddef>
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
    LOG(INFO) << "enter " << symbol_name_ << " runtime";

    if (outputs_.size() == 1 &&
        nodes_[outputs_[0].id_].GetOpName() == "ilavta.batch_matmul") {
      LOG(INFO) << "off-loading ilavta.batch_matmul";

      // get input node data
      for (size_t i = 0; i < input_nodes_.size(); ++i) {
        auto eid = EntryID(input_nodes_[i], 0);
        auto& node_data = data_entry_[eid];

        auto buffer_size = GetDataSize(*node_data);
        auto ndim = node_data->ndim;
        CHECK(ndim == 3);

        auto node = nodes_[eid];

        LOG(INFO) << "data entry: " << data_entry_[eid]->data;
        LOG(INFO) << "data type: " << data_entry_[eid]->dtype;

#if 0
        auto handle = data_entry_[eid]->data;
        char* dst = new char[buffer_size];
        std::copy(reinterpret_cast<char*>(handle),
                  reinterpret_cast<char*>(handle) + buffer_size, dst);
#endif

        for (auto dim = 0; dim < data_entry_[eid]->ndim; dim++) {
          LOG(INFO) << "shape: " << data_entry_[eid]->shape[dim];
        }
      }
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
