#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/transform.h>
#include <tvm/relay/type.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>

#include "../../utils.h"

#include "../../../../runtime/contrib/json/json_node.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

class ILAVTAJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  ILAVTAJSONSerializer(const std::string& symbol, const Expr& expr)
      : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    Expr expr = GetRef<Expr>(cn);
    std::string name;

    if (const auto* op_node = cn->op.as<OpNode>()) {
      name = op_node->name;
    } else if (const auto* fn = cn->op.as<FunctionNode>()) {
      auto comp = fn->GetAttr<String>(attr::kComposite);
      CHECK(comp.defined())
          << "JSON runtime only supports composite functions.";
      name = comp.value();
      if (!(name == "ilavta.conv2d" || name == "ilavta.batch_matmul" || name == "ilavta.dense")) {
        LOG(FATAL) << "Unrecognized pattern: " << name;
      }
      if (name == "ilavta.dense") {
        //
      } 
    } else {
      LOG(FATAL) << "ILAVTA runtime does not support calls to "
                 << cn->op->GetTypeKey();
    }
    LOG(INFO) << "[Pattern Matching] Find annotated: " << name;

    std::vector<JSONGraphNodeEntry> inputs;
    for (const auto& arg : cn->args) {
      auto res = VisitExpr(arg);
      inputs.insert(inputs.end(), res.begin(), res.end());
    }
    auto node = std::make_shared<JSONGraphNode>(name,     /* name_ */
                                                "kernel", /* op_type_ */
                                                inputs, 1 /* num_outputs_ */);
    return AddNode(node, GetRef<Expr>(cn));
  }

};  // class ILAVTAJSONSerializer

runtime::Module ILAVTACompiler(const ObjectRef& ref) {
  CHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);

  ILAVTAJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.GetParams();

  const auto* pf = runtime::Registry::Get("runtime.ILAVTARuntimeCreate");
  CHECK(pf != nullptr) << "Cannot find ILAVTA runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  LOG(INFO) << "Module created";
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.ilavta").set_body_typed(ILAVTACompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
