#include <tvm/relay/attrs/nn.h>
#include <tvm/relay/expr_functor.h>
#include <tvm/relay/type.h>
#include <tvm/relay/transform.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <numeric>

#include "../../utils.h"
#include "../codegen_c/codegen_c.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace backend;

inline size_t GetShape1DSize(const Type& type) {
  const auto shape = GetShape(type);
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

std::vector<std::string> MatMul(const CallNode* call) {
  auto data_shape = GetShape(call->args[0]->checked_type());
  auto weight_shape = GetShape(call->args[1]->checked_type());

  std::vector<std::string> args;
  args.push_back(std::to_string(data_shape[0]));
  args.push_back(std::to_string(data_shape[1]));
  args.push_back(std::to_string(weight_shape[0]));

  return args;
}

class CodegenVTA : public MemoizedExprTranslator<std::vector<Output>>, public CodegenCBase {

  public:
    CodegenVTA(const std::string& id) {
      this->ext_func_id_ = id;
    }

    std::vector<Output> VisitExpr_(const VarNode* node) final {
    ext_func_args_.push_back(GetRef<Var>(node));
    Output output;
    output.name = node->name_hint();
    return {output};
  }

    std::vector<Output> VisitExpr_(const TupleNode* node) final {
      std::vector<Output> outs;
      for (auto field : node->fields) {
        auto res = VisitExpr(field);
        CHECK_EQ(res.size(), 1U) << "Do not support tuple nest";
        outs.push_back(res[0]);
      }
      return outs;
    }

    std::vector<Output> VisitExpr_(const TupleGetItemNode* op) final {
      auto res = VisitExpr(op->tuple);
      CHECK_GT(res.size(), static_cast<size_t>(op->index));

      // Only keep the item we want for the child node.
      // FIXME(@comaniac): The other items should still be requried for the primary outputs.
      return {res[op->index]};
    }

    std::vector<Output> VisitExpr_(const ConstantNode* cn) final {
      std::ostringstream decl_stream;
      std::ostringstream buf_stream;

      Output output;
      // Get const: static_cast<float*>(gcc_0_consts[0]->data)
      output.name = CreateDataReference(ext_func_id_, const_idx_);
      const auto* type_node = cn->checked_type().as<TensorTypeNode>();
      CHECK(type_node);
      const auto& dtype = GetDtypeString(type_node);

      // Generate the global variable for needed ndarrays
      if (const_array_name_.empty()) {
        const_array_name_ = CreateNDArrayPool(ext_func_id_);
        std::string checker = CreateInitChecker(ext_func_id_);
        ext_func_body_.insert(ext_func_body_.begin(), checker);
      }

      CHECK(dtype == "float" || dtype == "int") << "Only float and int are supported for now.";
      output.dtype = dtype;

      std::string const_var_name = CreateConstVar(ext_func_id_, const_idx_);
      const_vars_.push_back(const_var_name);
      const_idx_++;

      return {output};
    }

    std::vector<Output> VisitExpr_(const CallNode* call) {
      std::ostringstream decl_stream;
      std::ostringstream buf_stream;
      std::ostringstream call_stream;

      std::string func_name = ext_func_id_ + "_";

      if (IsOp(call, "nn.dense")) {
        auto ret = GenerateBody(call, "vta_matmul_ila", GetArgumentNames(call), MatMul(call));
        buf_decl_.insert(buf_decl_.end(), ret.buffers.begin(), ret.buffers.end());
        ext_func_body_.push_back(ret.decl);
        return ret.outputs;
      } else {
        LOG(FATAL) << "Only support dense currently";
        return {};
      }
    }

    std::string JIT(const std::vector<Output>& out) {
      return JitImpl(ext_func_id_, ext_func_args_, buf_decl_, ext_func_body_, const_array_name_, out);
    }
  
  private:
    struct GenerateBodyOutput {
      std::string decl;
      std::vector<std::string> buffers;
      std::vector<Output> outputs;
    };

    // from dnnl codegen
    GenerateBodyOutput GenerateBody(const CallNode* root_call, const std::string& func_name,
                                  const std::vector<std::string>& func_args,
                                  const std::vector<std::string>& attribute_args) {
    // Make function call with input buffers when visiting arguments
    CHECK_GT(func_args.size(), 0);
    std::ostringstream decl_stream;
    decl_stream << "(" << func_args[0];
    for (size_t i = 1; i < func_args.size(); ++i) {
      decl_stream << ", " << func_args[i];
    }

    // Analyze the output buffers
    std::vector<Type> out_types;
    if (root_call->checked_type()->IsInstance<TupleTypeNode>()) {
      auto type_node = root_call->checked_type().as<TupleTypeNode>();
      for (auto field : type_node->fields) {
        CHECK(field->IsInstance<TensorTypeNode>());
        out_types.push_back(field);
      }
    } else if (root_call->checked_type()->IsInstance<TensorTypeNode>()) {
      CHECK(root_call->checked_type()->IsInstance<TensorTypeNode>());
      out_types.push_back(root_call->checked_type());
    } else {
      LOG(FATAL) << "Unrecognized type node: " << AsText(root_call->checked_type(), false);
    }

    GenerateBodyOutput ret;
    for (const auto& out_type : out_types) {
      this->PrintIndents();
      const std::string out = "buf_" + std::to_string(buf_idx_++);
      const auto out_size = GetShape1DSize(out_type);
      decl_stream << ", " << out;

      Output output;
      output.name = out;
      output.size = out_size;
      output.dtype = GetDtypeString(out_type.as<TensorTypeNode>());
      output.need_copy = true;
      ret.buffers.push_back("float* " + out + " = (float*)std::malloc(4 * " +
                            std::to_string(out_size) + ");");
      ret.outputs.push_back(output);
    }

    // Attach attribute arguments
    for (size_t i = 0; i < attribute_args.size(); ++i) {
      decl_stream << ", " << attribute_args[i];
    }
    decl_stream << ");";
    ret.decl = func_name + decl_stream.str();
    return ret;
  }

    std::vector<std::string> GetArgumentNames(const CallNode* call) {
      std::vector<std::string> arg_names;
      std::cerr << "Arg Size: " << call->args.size() << std::endl;
      for (size_t i = 0; i < call->args.size(); ++i) {
        auto res = VisitExpr(call->args[i]);
        for (const auto& out : res) {
          arg_names.push_back(out.name);
          // std::cerr << out.name << " ";
        }
        // std::cerr << std::endl;
      }
      return arg_names;
    }

    /*! \brief The id of the external dnnl ext_func. */
    std::string ext_func_id_{""};
    /*!
    * \brief The index to track the output buffer. Each kernel will redirect the
    * output to a buffer that may be consumed by other kernels.
    */
    int buf_idx_{0};
    /*! \brief The index of global constants. */
    int const_idx_{0};
    int func_id_{0};
    /*! \brief The arguments used by a wrapped function that calls DNNL kernels. */
    Array<Var> ext_func_args_;
    /*! \brief Statement of the function that will be compiled using DNNL kernels. */
    std::vector<std::string> ext_func_body_;
    /*! \brief The array declared to store the constant values. */
    std::string const_array_name_;
    /*! \brief The declaration of intermeidate buffers. */
    std::vector<std::string> buf_decl_;
    /*! \brief The variable name to constant mapping. */
    Array<String> const_vars_;

    friend class VTAMatMulModuleCodegen;
};

class VTAMatMulModuleCodegen : public CSourceModuleCodegenBase {
  public:
    std::pair<std::string, Array<String>> GenVTAFunc(const Function& func) {
      CHECK(func.defined()) << "Input error: expect a Relay function.";

      // Record the external symbol for runtime lookup.
      auto sid = GetExtSymbol(func);

      CodegenVTA builder(sid);
      auto out = builder.VisitExpr(func->body);
      code_stream_ << builder.JIT(out);

      return {sid, builder.const_vars_};
    }

    runtime::Module CreateCSourceModule(const ObjectRef& ref) override {
      code_stream_ << "#include <iostream>\n";
      code_stream_ << "#include <cstring>\n";
      code_stream_ << "#include <vector>\n";
      code_stream_ << "#include <tvm/runtime/c_runtime_api.h>\n";
      code_stream_ << "#include <tvm/runtime/container.h>\n";
      code_stream_ << "#include <tvm/runtime/packed_func.h>\n";
      code_stream_ << "#include <vta_matmul/vta_matmul_runtime.h>\n";
      code_stream_ << "#include <dlpack/dlpack.h>\n";
      code_stream_ << "using namespace tvm::runtime;\n";
      code_stream_ << "using namespace tvm::runtime::contrib;\n";
      code_stream_ << "\n";

      const char* ila_code = R"op_macro(
        extern "C" void vta_matmul_ila(float* inp, float* weight, float* out, int in_0, int in_1, int w_0) {
          std::cerr << "Called vta_matmul_ila\n";
          std::cerr << "DEBUG: " << inp[0] << std::endl;
          run_vta_simulator(inp, weight, in_0, in_1, w_0, out);
        })op_macro";
      // code_stream_ << "using namespace tvm::runtime::contrib;\n";

      code_stream_ << ila_code << "\n";
      auto func = Downcast<Function>(ref);
      auto res = GenVTAFunc(func);
      auto code = code_stream_.str();
      String symbol = std::get<0>(res);
      Array<String> vars = std::get<1>(res);
      const auto* pf = runtime::Registry::Get("runtime.CSourceModuleCreate");
      CHECK(pf != nullptr) << "Cannot find csource module to create the external runtime module";
      return (*pf)(code, "c", symbol, vars);
    }
  private:
    std::ostringstream code_stream_;
};

runtime::Module VTAMatMulCompiler(const ObjectRef& ref) {
  VTAMatMulModuleCodegen codegen;
  return codegen.CreateCSourceModule(ref);
}

TVM_REGISTER_GLOBAL("relay.ext.vta_matmul").set_body_typed(VTAMatMulCompiler);

}  // namespace contrib
}  // namespace relay
}  // namespace tvm