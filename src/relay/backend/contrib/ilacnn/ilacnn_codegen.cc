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
#include <chrono>

#include "../../utils.h"

#include "../../../../runtime/contrib/json/json_node.h"
#include "../codegen_json/codegen_json.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

class IlaCNNJSONSerializer : public backend::contrib::JSONSerializer {
  using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
  using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;

 public:
  IlaCNNJSONSerializer(const std::string& symbol, const Expr& expr) : JSONSerializer(symbol, expr) {}

  std::vector<JSONGraphNodeEntry> VisitExpr_(const CallNode* cn) override {
    std::string name;

    if (const auto* op_node = cn->op.as<OpNode>()) {
      name = op_node->name;
    } else if (const auto* fn = cn->op.as<FunctionNode>()) {
      auto comp = fn->GetAttr<String>(attr::kComposite);
      CHECK(comp.defined())
          << "JSON runtime only supports composite functions.";
      name = comp.value();

      if (name != "ilacnn.conv2d") {
        LOG(FATAL) << "Unrecognized pattern: " << name;
      }
    } else {
      LOG(FATAL) << "IlaCNN runtime does not support calls to "
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

    // Note: conv2d has a lot of attrs that are relevant for codegen,
    // especially the stride size.
    // However, the pattern matcher will produce patterns in the form of
    // fn(Compiler="ilacnn") {
    //   fn(Composuite="ilacnn.conv2d") { nn.conv2d(...) }
    // }
    // so we need to reach inside the inner function to get the conv2d attrs (weird, yeah);
    // see codegen_json.h:SetCallNodeAttribute

    tvm::relay::backend::contrib::OpAttrExtractor extractor(node);
    auto inner_func = Downcast<Function>(cn->op);
    auto inner_call = Downcast<Call>(inner_func->body);
    const Object* inner_call_attr = inner_call->attrs.get();
    extractor.Extract(const_cast<Object*>(inner_call_attr));
    return AddNode(node, GetRef<Expr>(cn));
  }
};  // class IlaCNNJSONSerializer

runtime::Module IlaCNNCompiler(const ObjectRef& ref) {
  LOG(INFO) << "Begin HLSCNN codegen";
  const std::string wall_clock_file = "./ilacnn_compile_time.json";
  auto start_time = std::chrono::high_resolution_clock::now();
  CHECK(ref->IsInstance<FunctionNode>());
  auto func = Downcast<Function>(ref);
  auto func_name = GetExtSymbol(func);

  IlaCNNJSONSerializer serializer(func_name, func);
  serializer.serialize();
  std::string graph_json = serializer.GetJSON();
  auto params = serializer.GetParams();

  const auto* pf = runtime::Registry::Get("runtime.IlaCNNRuntimeCreate");
  CHECK(pf != nullptr) << "Cannot find IlaCNN runtime module to create";
  auto mod = (*pf)(func_name, graph_json, params);
  auto end_time = std::chrono::high_resolution_clock::now();
  record_compile_time(end_time - start_time, wall_clock_file);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.ilacnn").set_body_typed(IlaCNNCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
