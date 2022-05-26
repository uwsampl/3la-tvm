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
#include <chrono>

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
    } else if (const auto* fn = cn->op.as<FunctionNode>()) {
      auto comp = fn->GetAttr<String>(attr::kComposite);
      CHECK(comp.defined())
          << "JSON runtime only supports composite functions.";
      name = comp.value();
      if (!(name == "ilavta.conv2d" || name == "ilavta.bias_add"
            || name == "ilavta.dense" || name == "ilavta.relu" || name == "ilavta.conv1d" || name == "ilavta.linear")) {
        LOG(FATAL) << "Unrecognized pattern: " << name;
      }
      if (name == "ilavta.dense") {
        // Linear layer pattern of VTA
        // required arguments:
        //  - inputs
        //  - weights
        //  - bias
        //  - multiplicative factor & # of bits to right shift
        //  -- these two approximates the calibrated scaling factor of re-quantized (actual) activations 
        //     (i.e. using multiplication and right shifts to approximate a division)
        // NOTE: 
        LOG(INFO) << "ilavta.dense pattern";
        auto input_shape = GetShape(cn->args[0]->checked_type());
        auto weight_shape = GetShape(cn->args[1]->checked_type());
        // these casts must success
        // auto factor = cn->args[2].as<IntImmNode>()->value;
        // auto nbits = cn->args[3].as<IntImmNode>()->value;
        int info[] = {input_shape[0], input_shape[1], weight_shape[0]};
        filename = GetCompiledFilename("linear", info, 3);
        // if (this->compiled_func.find(filename) == this->compiled_func.end()) {
          // this->compiled_func.insert(filename);
          // filename = CompileGEMM(input_shape[0], input_shape[1], weight_shape[0], factor, nbits, "./prog_frag/" + filename);
        // }
      }else if (name == "ilavta.dense") {
        LOG(INFO) << "ilavta.dense pattern";
        auto input_shape = GetShape(cn->args[0]->checked_type());
        auto weight_shape = GetShape(cn->args[1]->checked_type());
        int batch = input_shape[0];
        int n_inp_cols = input_shape[1];
        int n_wgt_rows = weight_shape[0];
        int info[] = {batch, n_inp_cols, n_wgt_rows};
        filename = GetCompiledFilename("dense", info, 3);
        if (this->compiled_func.find(filename) == this->compiled_func.end()) {
          this->compiled_func.insert(filename);
          // `factor` and `nbits` are placeholders for "partial evaluation"
          filename = CompileGEMM(batch, n_inp_cols, n_wgt_rows, 1, 0, "./prog_frag/" + filename);
        }
      }  else if (name == "ilavta.bias_add") {
        LOG(INFO) << "ilavta.bias_add pattern";
        auto input_shape = GetShape(cn->args[0]->checked_type());
        int batch = input_shape[0];
        int n_feat = input_shape[1];
        int info[] = {batch, n_feat};
        filename = GetCompiledFilename("bias_add", info, 2);
        if (this->compiled_func.find(filename) == this->compiled_func.end()) {
          this->compiled_func.insert(filename);
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
          this->compiled_func.insert(filename);
          filename = CompileRelu(batch, n_feat, "./prog_frag/" + filename);
        }
      } else if (name == "ilavta.conv1d") {
        auto input_shape = GetShape(cn->args[0]->checked_type());
        auto weight_shape = GetShape(cn->args[1]->checked_type());
        int N = input_shape[0];
        int C = input_shape[1];
        int W = input_shape[2];

        int O = weight_shape[0];
        int I = C;
        int wgtW = weight_shape[2];

        int vec_width = I * wgtW;
        int vec_cnt = N * (W - wgtW + 1);
        int input_info[5] = {N, C, W, O, wgtW};
        filename = GetCompiledFilename("conv1d", input_info, 5);
        if (this->compiled_func.find(filename) == this->compiled_func.end()) {
          this->compiled_func.insert(filename);
          filename = CompileGEMM(vec_cnt, vec_width, O, 1, 0, "./prog_frag/" + filename);
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
  LOG(INFO) << "Begin ILAVTA Codegen";
  const std::string wall_clock_file = "./ilavta_wallclock.json";
  auto start_time = std::chrono::high_resolution_clock::now();
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
  auto end_time = std::chrono::high_resolution_clock::now();
  record_compile_time(end_time - start_time, wall_clock_file);
  return mod;
}

TVM_REGISTER_GLOBAL("relay.ext.ilavta").set_body_typed(ILAVTACompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm
