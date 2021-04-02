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
#include <set>

#include "ilavta_codegen_utils.h"
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
    std::string filename;

    if (const auto* op_node = cn->op.as<OpNode>()) {
      name = op_node->name;
      LOG(INFO) << "Annotated Operator: " << name;
      if (name != "qnn.dense") {
        LOG(FATAL) << "Unrecognized operator: " << name; 
      }
      if (name == "qnn.dense") {
        auto input_shape = GetShape(cn->args[0]->checked_type());
        auto weight_shape = GetShape(cn->args[1]->checked_type());
        int batch = input_shape[0];
        int n_inp_cols = input_shape[1];
        int n_wgt_rows = weight_shape[0];
        int info[] = {batch, n_inp_cols, n_wgt_rows};
        filename = GetCompiledFilename("qnn_dense", info, 3);
        if (this->compiled_func.find(filename) == this->compiled_func.end()) {
          filename = CompileGEMM(batch, n_inp_cols, n_wgt_rows, "./prog_frag/" + filename);
        }
      } 
    } else if (const auto* fn = cn->op.as<FunctionNode>()) {
      auto comp = fn->GetAttr<String>(attr::kComposite);
      CHECK(comp.defined())
          << "JSON runtime only supports composite functions.";
      name = comp.value();
      if (!(name == "ilavta.conv2d" || name == "ilavta.bias_add" || name == "ilavta.qnn_dense" || name == "ilavta.dense" || name == "ilavta.relu")) {
        LOG(FATAL) << "Unrecognized pattern: " << name;
      }
      if (name == "ilavta.dense" || name == "ilavta.qnn_dense") {
        LOG(INFO) << "ilavta.dense pattern";
        auto input_shape = GetShape(cn->args[0]->checked_type());
        auto weight_shape = GetShape(cn->args[1]->checked_type());
        int batch = input_shape[0];
        int n_inp_cols = input_shape[1];
        int n_wgt_rows = weight_shape[0];
        int info[] = {batch, n_inp_cols, n_wgt_rows};
        filename = GetCompiledFilename(name, info, 3);
        if (this->compiled_func.find(filename) == this->compiled_func.end()) {
          filename = CompileGEMM(batch, n_inp_cols, n_wgt_rows, "./prog_frag/" + filename);
        }
      }  else if (name == "ilavta.bias_add") {
        LOG(INFO) << "ilavta.bias_add pattern";
        auto input_shape = GetShape(cn->args[0]->checked_type());
        int batch = input_shape[0];
        int n_feat = input_shape[1];
        int info[] = {batch, n_feat};
        filename = GetCompiledFilename("bias_add", info, 2);
        if (this->compiled_func.find(filename) == this->compiled_func.end()) {
          filename = CompilBiasAdd(batch, n_feat, "./prog_frag/" + filename);
        }
      } else if (name == "ilavta.relu") {
        LOG(INFO) << "ilavta.relu pattern";
        auto input_shape = GetShape(cn->args[0]->checked_type());
        int batch = input_shape[0];
        int n_feat = input_shape[1];
        int info[] = {batch, n_feat};
        filename = GetCompiledFilename("relu", info, 2);
        if (this->compiled_func.find(filename) == this->compiled_func.end()) {
          filename = CompileRelu(batch, n_feat, "./prog_frag/" + filename);
        }
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
    std::vector<std::string> vec;
    std::vector<dmlc::any> compiler_attr;
    vec.push_back(filename);
    compiler_attr.emplace_back(vec);
    node->SetAttr("asm_file", compiler_attr);
    return AddNode(node, GetRef<Expr>(cn));
  }
private:
  std::set<std::string> compiled_func;

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
