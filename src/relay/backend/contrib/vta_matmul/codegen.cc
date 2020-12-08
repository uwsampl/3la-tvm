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
      code_stream_ << "#include <cmath>\n";
      code_stream_ << "using namespace tvm::runtime;\n";
      code_stream_ << "using namespace tvm::runtime::contrib;\n";
      code_stream_ << "\n";

      const char* ila_code = R"op_macro(
        typedef uint32_t uop_T;
typedef int8_t wgt_T;
typedef int8_t inp_T;
typedef int8_t out_T;
typedef int32_t acc_T;

void* allocBuffer(size_t num_bytes) { return VTAMemAlloc(num_bytes, 0); }

template <typename T>
T ** alloc2dArray(int rows, int cols) {
  T **array = static_cast<T **>(malloc(sizeof(T *) * rows));
  for (int i = 0; i < rows; i++) {
    array[i] = static_cast<T *>(malloc(sizeof(T) * cols));
  }
  return array;
}

VTAUop* getGEMMUops(int batch, int in_channels, int out_channels) {
  VTAUop *uop_buf = static_cast<VTAUop *>(VTAMemAlloc(sizeof(VTAUop) * batch, 0));
  for (int i = 0; i < batch; ++i) {
    uop_buf[i].dst_idx = i * out_channels;
    uop_buf[i].src_idx = i * in_channels;
    uop_buf[i].wgt_idx = 0;
  }
  return uop_buf;
}

template <typename DST_T, int DST_T_WIDTH, typename SRC_T, int SRC_T_WIDTH>
void packBuffer(DST_T* dst, SRC_T** src, int y_size, int x_size, int y_block, int x_block) {
  assert((SRC_T_WIDTH * x_block * y_block) % DST_T_WIDTH == 0);
  assert(DST_T_WIDTH <= 64);
  int buffer_idx = 0;
  int ratio = DST_T_WIDTH / SRC_T_WIDTH;
  long long int mask = (1ULL << SRC_T_WIDTH) - 1;
  DST_T tmp = 0;
  for (int i = 0; i < y_size / y_block; i++) {
    for (int j = 0; j < x_size / x_block; j++) {
      for (int k = 0; k < y_block; k++) {
        for (int l = 0; l < x_block; l++) {
          int block_idx = l + k * x_block;
          tmp |= (src[i * y_block + k][j * x_block + l] & mask)
                 << ((block_idx % ratio) * SRC_T_WIDTH);
          // When tmp is packed, write to destination array
          if (block_idx % ratio == ratio - 1) {
            dst[buffer_idx++] = tmp;
            tmp = 0;
          }
        }
      }
    }
  }
}

template <typename T>
T** allocInit2dArray(int rows, int cols, bool init_data, float* from, T data = 0) {
  // Allocate
  T** array = static_cast<T**>(malloc(sizeof(T*) * rows));
  for (int i = 0; i < rows; i++) {
    array[i] = static_cast<T*>(malloc(sizeof(T) * cols));
  }
  // Init
  if (init_data) {
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (from == nullptr) {
          array[i][j] = static_cast<T>(data);
        } else {
          array[i][j] = static_cast<T>(from[i * cols + j]);
        }
      }
    }
  }
  return array;
}

template <typename DST_T, int DST_T_WIDTH, typename SRC_T, int SRC_T_WIDTH>
void unpackBuffer(DST_T **dst, SRC_T *src, int y_size, int x_size, int y_block, int x_block) {
  assert((DST_T_WIDTH * x_block * y_block) % SRC_T_WIDTH == 0);
  int buffer_idx = 0;
  long long int mask = (1ULL << DST_T_WIDTH) - 1;
  int ratio = SRC_T_WIDTH / DST_T_WIDTH;
  for (int i = 0; i < y_size / y_block; i++) {
    for (int j = 0; j < x_size / x_block; j++) {
      for (int k = 0; k < y_block; k++) {
        for (int l = 0; l < x_block; l++) {
          int block_idx = l + k * x_block;
          dst[i * y_block + k][j * x_block + l] = (src[buffer_idx] >> ((block_idx % ratio) * DST_T_WIDTH)) & mask;
          if (block_idx % ratio == ratio - 1) {
            buffer_idx++;
          }
        }
      }
    }
  }
}


VTAGenericInsn get2DLoadStoreInsn(int opcode, int type, int sram_offset, int dram_offset,
                                  int y_size, int x_size, int x_stride, int y_pad, int x_pad,
                                  int pop_prev_dep, int pop_next_dep, int push_prev_dep,
                                  int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // Memory instruction initialization
  VTAMemInsn insn = {};
  insn.opcode = opcode;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.memory_type = type;
  insn.sram_base = sram_offset;
  insn.dram_base = dram_offset;
  insn.y_size = y_size;
  insn.x_size = x_size;
  insn.x_stride = x_stride;
  insn.y_pad_0 = y_pad;
  insn.y_pad_1 = y_pad;
  insn.x_pad_0 = x_pad;
  insn.x_pad_1 = x_pad;
  converter.mem = insn;
  return converter.generic;
}

VTAGenericInsn get1DLoadStoreInsn(int opcode, int type, int sram_offset, int dram_offset, int size,
                                  int pop_prev_dep, int pop_next_dep, int push_prev_dep,
                                  int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // Memory instruction initialization
  VTAMemInsn insn = {};
  insn.opcode = opcode;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.memory_type = type;
  insn.sram_base = sram_offset;
  insn.dram_base = dram_offset;
  insn.y_size = 1;
  insn.x_size = size;
  insn.x_stride = size;
  insn.y_pad_0 = 0;
  insn.y_pad_1 = 0;
  insn.x_pad_0 = 0;
  insn.x_pad_1 = 0;
  converter.mem = insn;
  return converter.generic;
}

VTAGenericInsn getGEMMInsn(int uop_offset, int batch, int in_feat, int out_feat,
                           bool uop_compression, int pop_prev_dep, int pop_next_dep,
                           int push_prev_dep, int push_next_dep) {
  // Converter
  union VTAInsn converter;
  // GEMM instruction initialization
  VTAGemInsn insn;
  insn.opcode = VTA_OPCODE_GEMM;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.reset_reg = false;
  insn.uop_bgn = uop_offset;
  insn.uop_end = uop_offset + 1;
  insn.iter_out = 1;
  insn.iter_in = 1;
  insn.dst_factor_out = 0;
  insn.src_factor_out = 0;
  insn.wgt_factor_out = 0;
  insn.dst_factor_in = 0;
  insn.src_factor_in = 0;
  insn.wgt_factor_in = 0;
  converter.gemm = insn;
  return converter.generic;
}

VTAGenericInsn getFinishInsn(bool pop_prev, bool pop_next) {
  // Converter
  union VTAInsn converter;
  // GEMM instruction initialization
  VTAGemInsn insn;
  insn.opcode = VTA_OPCODE_FINISH;
  insn.pop_prev_dep = pop_prev;
  insn.pop_next_dep = pop_next;
  insn.push_prev_dep = 0;
  insn.push_next_dep = 0;
  insn.reset_reg = false;
  insn.uop_bgn = 0;
  insn.uop_end = 0;
  insn.iter_out = 0;
  insn.iter_in = 0;
  insn.dst_factor_out = 0;
  insn.src_factor_out = 0;
  insn.wgt_factor_out = 0;
  insn.dst_factor_in = 0;
  insn.src_factor_in = 0;
  insn.wgt_factor_in = 0;
  converter.gemm = insn;
  return converter.generic;
}

template<typename T>
void copyData(T* dst, float* src, int h, int w) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      dst[i * w + j] = src[i * w + j];
    }
  }
}

template<typename T>
void resetVal(T* dst, T v, int h, int w) {
  for (int i = 0; i < h; ++i) {
    for (int j = 0; j < w; ++j) {
      dst[i * w + j] = v;
    }
  }
}

template<typename T>
/**
 * Copy the `r_src`th row of `src` to `r_dst`th row of `dst`
 * */
void copyRow(T* dst, float* src, int r_dst, int w_dst, int r_src, int w_src) {
  assert(w_src <= w_dst);
  for (int i = 0; i < w_src; ++i) {
    dst[r_dst * w_dst + i] = (int)round(src[r_src * w_src + i]);
  }
}

extern "C" TVM_DLL void run_vta_simulator(float* input, float* weight, int batch, int in_channels,
                                          int out_channels, float* out_buf) {
  const uint32_t num_instr = 256;
  memset(out_buf, 0, sizeof(float) * out_channels * in_channels);
  // LOAD UOP
  // LOAD INP
  // LOAD WGT
  // GEMM
  // ALU
  // STORE
  // FINISH
  VTADeviceHandle device_handle = VTADeviceAlloc();
  VTAGenericInsn *instr_buf =
    static_cast<VTAGenericInsn *>(allocBuffer(sizeof(VTAGenericInsn) * num_instr));

  int8_t* wgt_buf = static_cast<int8_t*>(allocBuffer(VTA_WGT_ELEM_BYTES * out_channels * in_channels));
  copyData<int8_t>(wgt_buf, weight, out_channels, in_channels);
  // packBuffer<uint32_t, 32, out_T, VTA_WGT_WIDTH>(wgt_buf, weights, out_channels, in_channels,
                                                //  VTA_BLOCK_OUT, VTA_BLOCK_IN);

  uint32_t* bias_buf = static_cast<uint32_t*>(allocBuffer(VTA_ACC_ELEM_BYTES * batch * out_channels));
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < out_channels; ++j) {
      bias_buf[i * out_channels + j] = 0;
    }
  }
  // packBuffer<uint32_t, 32, acc_T, VTA_OUT_WIDTH>(bias_buf, bias, batch, out_channels, VTA_BATCH,
                                                //  VTA_BLOCK_OUT);
  int8_t* output_buf = static_cast<int8_t*>(allocBuffer(VTA_OUT_ELEM_BYTES * batch * out_channels));
  // memset(output_buf, 0, VTA_OUT_ELEM_BYTES * out_size * out_size);
  resetVal<int8_t>(output_buf, 0, batch, out_channels);

  // std::cerr << "Define instructions" << std::endl;

  int ptr = 0;
  // We do not write 0s to the ACC buffer since it is initally 0
  // This will reduce the length of translated ILA program fragment
  /* instr_buf[ptr++] = get2DLoadStoreInsn(
      VTA_OPCODE_LOAD, // op_code
      VTA_MEM_ID_ACC,  // type
      0,               // sram base
      VTAMemGetPhyAddr(bias_buf) / VTA_ACC_ELEM_BYTES, // dram base
      batch, // y_size
      out_channels, // x_size
      out_channels, // x_stride
      0,        // y pad
      0,        // x_pad
      0, 0, 1, 0
    ); */

  instr_buf[ptr++] = get2DLoadStoreInsn(
    VTA_OPCODE_LOAD,
    VTA_MEM_ID_WGT,
    0,
    VTAMemGetPhyAddr(wgt_buf) / VTA_WGT_ELEM_BYTES,
    out_channels,
    in_channels,
    in_channels,
    0,
    0,
    0, 0, 0, 0
  );
  std::vector<void*> allocated_inp;
  for (int i = 0; i < batch; ++i) {
    int8_t* ibuf = static_cast<int8_t*>(allocBuffer(VTA_INP_ELEM_BYTES * batch * in_channels));
    resetVal<int8_t>(ibuf, 0, batch, in_channels);
    copyRow<int8_t>(ibuf, input, 0, in_channels, i, in_channels);
    allocated_inp.push_back(ibuf);

    VTAUop* uop = reinterpret_cast<VTAUop*>(VTAMemAlloc(sizeof(VTAUop), 0));
    uop->dst_idx = i;
    uop->src_idx = 0;
    uop->wgt_idx = 0;
    allocated_inp.push_back(uop);

    instr_buf[ptr++] = get2DLoadStoreInsn(
      VTA_OPCODE_LOAD,
      VTA_MEM_ID_INP,
      0,
      VTAMemGetPhyAddr(ibuf) / VTA_INP_ELEM_BYTES,
      batch,
      in_channels,
      in_channels,
      0,
      0,
      0, i > 0, 0, 1 // pop prev | pop next | push prev | push next
    );
    
    instr_buf[ptr++] = get1DLoadStoreInsn(
      VTA_OPCODE_LOAD,
      VTA_MEM_ID_UOP,
      0,
      VTAMemGetPhyAddr(uop) / VTA_UOP_ELEM_BYTES,
      sizeof(VTAUop),
      1, 0, 0, 0
    );

    instr_buf[ptr++] = getGEMMInsn(
      0,
      batch / VTA_BATCH,
      in_channels / VTA_BLOCK_IN,
      out_channels / VTA_BLOCK_OUT,
      1,
      0, 0, 1, i + 1 == out_channels
    );
  }

  instr_buf[ptr++] = get2DLoadStoreInsn(
    VTA_OPCODE_STORE,
    VTA_MEM_ID_OUT,
    0,
    VTAMemGetPhyAddr(output_buf) / VTA_OUT_ELEM_BYTES,
    1,
    out_channels,
    out_channels,
    0,
    0,
    1, 0, 1, 0
  );

  instr_buf[ptr++] = getFinishInsn(0, 1);

  VTADeviceRun(device_handle, VTAMemGetPhyAddr(instr_buf), ptr, 1000);

  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < out_channels; ++j) {
      out_buf[i * out_channels + j] = output_buf[i * out_channels + j];
    }
  }

  std::cerr << "finished\n"; 

  VTAMemFree(output_buf);
  for (auto x : allocated_inp) {
    VTAMemFree(x);
  }
  VTAMemFree(wgt_buf);
  VTAMemFree(instr_buf);
  VTADeviceFree(device_handle);
}
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
