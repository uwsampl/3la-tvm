#include "ilavta_helpers.h"

#include <iomanip>

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;

using addr_byte_pairs = std::vector<std::pair<vta_phy_addr_t, uint8_t>>;

const int64_t SIM_DUMP = 1;
const std::string RAW_DUMP = "vta_sim_dump.json";
const std::string OUTPUT_DUMP = "vta_output_dump.json";

/*
 * Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
 * */
VTAUop* getGEMMUops(int batch, int in_feat, int out_feat, int block_size) {
  // Derive the total uop size
  int uop_size = batch * in_feat * out_feat + block_size;

  // Allocate buffer
  VTAUop* uop_buf = static_cast<VTAUop*>(VTAMemAlloc(sizeof(VTAUop) * uop_size, 0));

  int uop_idx = 0;
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < in_feat; j++) {
      for (int k = 0; k < out_feat; k++) {
        uop_buf[uop_idx].dst_idx = i * out_feat + k;
        uop_buf[uop_idx].src_idx = i * in_feat + j;
        uop_buf[uop_idx].wgt_idx = k * in_feat + j;
        uop_idx++;
      }
    }
  }
  for (int  i = 0; i < block_size; ++i) {
    uop_buf[uop_idx++].dst_idx = i;
  }
  return uop_buf;
}

VTAUop* getBiasAddUops(int batch, int in_feat) {
  int uop_size = batch * in_feat;
  VTAUop* uop_buf = static_cast<VTAUop*>(VTAMemAlloc(sizeof(VTAUop) * uop_size, 0));

  int uop_idx = 0;
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < in_feat; j++) {
      uop_buf[uop_idx].dst_idx = i * in_feat + j;
      // Bias is stored one block next to the input data
      uop_buf[uop_idx].src_idx = batch * in_feat + j;
      uop_idx++;
    }
  }
  return uop_buf;
}

VTAUop* getReluUops(int batch, int in_feat) {
  int uop_size = batch * in_feat;
  VTAUop* uop_buf = static_cast<VTAUop*>(VTAMemAlloc(sizeof(VTAUop) * uop_size, 0));

  int uop_idx = 0;
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < in_feat; j++) {
      uop_buf[uop_idx].dst_idx = i * in_feat + j;
      uop_idx++;
    }
  }
  return uop_buf;
}

json getGEMMAsm(int uop_offset, int uop_end) {
  return {{"name", "gemm"},        {"reset_f", 0},
          {"uop_bgn", uop_offset}, {"uop_end", uop_end},
          {"iter_o", 1},           {"iter_i", 1},
          {"dst_fo", 0},           {"dst_fi", 0},
          {"src_fo", 0},           {"src_fi", 0},
          {"dst_fo", 0},           {"dst_fi", 0},
          {"wgt_fo", 0},           {"wgt_fi", 0}};
}

/*
 * Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
 * */
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
  insn.uop_end = uop_offset + batch * in_feat * out_feat;
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

/*
 * Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
 * */
VTAGenericInsn getAluInsn(int alu_opcode, int uop_begin, int uop_end, bool use_imm, int imm,
                          int pop_prev_dep, int pop_next_dep, int push_prev_dep,
                          int push_next_dep) {
  VTAInsn converter;
  VTAAluInsn insn;
  insn.opcode = VTA_OPCODE_ALU;
  insn.alu_opcode = alu_opcode;
  insn.uop_bgn = uop_begin;
  insn.uop_end = uop_end;
  insn.use_imm = use_imm;
  insn.imm = imm;
  insn.pop_prev_dep = pop_prev_dep;
  insn.pop_next_dep = pop_next_dep;
  insn.push_prev_dep = push_prev_dep;
  insn.push_next_dep = push_next_dep;
  insn.iter_in = 1;
  insn.iter_out = 1;
  insn.reset_reg = false;
  insn.dst_factor_out = 0;
  insn.src_factor_out = 0;
  insn.dst_factor_in = 0;
  insn.dst_factor_out = 0;

  converter.alu = insn;
  return converter.generic;
}

/*
 * Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
 * */
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

/*
 * Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
 * */
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

/*
 * Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
 * */
template <typename DST_T, int DST_T_WIDTH, typename SRC_T, int SRC_T_WIDTH>
void unpackBuffer(DST_T** dst, SRC_T* src, int y_size, int x_size, int y_block, int x_block) {
  assert((DST_T_WIDTH * x_block * y_block) % SRC_T_WIDTH == 0);
  int buffer_idx = 0;
  long long int mask = (1ULL << DST_T_WIDTH) - 1;
  int ratio = SRC_T_WIDTH / DST_T_WIDTH;
  for (int i = 0; i < y_size / y_block; i++) {
    for (int j = 0; j < x_size / x_block; j++) {
      for (int k = 0; k < y_block; k++) {
        for (int l = 0; l < x_block; l++) {
          int block_idx = l + k * x_block;
          dst[i * y_block + k][j * x_block + l] =
              (src[buffer_idx] >> ((block_idx % ratio) * DST_T_WIDTH)) & mask;
          if (block_idx % ratio == ratio - 1) {
            buffer_idx++;
          }
        }
      }
    }
  }
}

json get2DLoadStoreAsm(int opcode, int mem_type, int sram_id, int dram_id, int y_size, int x_size) {
  std::string cmd_type;
  switch (opcode) {
    case VTA_OPCODE_LOAD:
      cmd_type = "load_";
      break;
    case VTA_OPCODE_STORE:
      cmd_type = "store_";
      break;
    default:
      fprintf(stderr, "Unknown load / store: %d", opcode); 
      exit(-1);
  }
  switch (mem_type) {
    case VTA_MEM_ID_INP:
      cmd_type += "inp";
      break;
    case VTA_MEM_ID_WGT:
      cmd_type += "wgt";
      break;
    case VTA_MEM_ID_UOP:
      cmd_type += "uop";
      break;
    case VTA_MEM_ID_ACC:
      cmd_type += "bias";
      break;
    case VTA_MEM_ID_OUT:
      cmd_type += "acc";
      break;
  }
  if (cmd_type == "load_uop") {
    return {
        {"name", cmd_type},
        {"sram_id", sram_id},
        {"dram_id", dram_id},
        {"x_size", x_size}
    };
  } else if (cmd_type == "load_wgt" || cmd_type == "load_bias" || opcode == VTA_OPCODE_STORE){
    return {
      {"name", cmd_type},
      {"sram_id", sram_id},
      {"dram_id", dram_id},
      {"y_size", y_size},
      {"x_size", x_size},
      {"x_stride", 1}
    };
  } else if (cmd_type == "load_inp") {
    return {
      {"name", cmd_type},
      {"sram_id", sram_id},
      {"dram_id", dram_id},
      {"y_size", y_size},
      {"x_size", x_size},
      {"x_stride", 1},
      {"y_pad0", 0},
      {"x_pad0", 0},
      {"y_pad1", 0},
      {"x_pad1", 0}
    };
  } else {
    fprintf(stderr, "Command %s not supported by ASM", cmd_type.c_str());
    exit(-1);
  }
}

json getAluAsm(int alu_opcode, int uop_bgn, int uop_end, bool use_imm, uint16_t imm) {
  int asm_opcode = -1;
  std::string op_name = "";
  switch (alu_opcode) {
    case VTA_ALU_OPCODE_MIN:
      asm_opcode = 0;
      op_name = "min";
      break;
    case VTA_ALU_OPCODE_MAX:
      asm_opcode = 1;
      op_name = "max";
      break;
    case VTA_ALU_OPCODE_ADD:
      asm_opcode = 2;
      op_name = "add";
      break;
    case VTA_ALU_OPCODE_SHR:
      asm_opcode = 3;
      op_name = "shr";
      break;
    case 4:
      asm_opcode = 4;
      op_name = "mul";
      break;
    default:
      fprintf(stderr, "ALU Opcode %d is not valid", alu_opcode);
      exit(-1);
  }
  return {{"name", "alu_" + op_name},
          {"reset_f", 0},
          {"uop_bgn", uop_bgn},
          {"uop_end", uop_end},
          {"iter_o", 1},
          {"iter_i", 1},
          {"dst_fo", 0},
          {"dst_fi", 0},
          {"src_fo", 0},
          {"src_fi", 0},
          {"alu_op", asm_opcode},
          {"use_imm", use_imm},
          {"imm", imm}};
}

/*
 * Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
 * */
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

/*
 * Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
 * */
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

template <class T>
std::string to_hex(T x) {
  std::stringstream ss;
  ss << "0x" << std::setfill('0') << std::setw(sizeof(T) * 2) << std::hex
     << static_cast<uint64_t>(x);
  return ss.str();
}

std::string dump_datafile(int8_t* input_buf, size_t input_size, int8_t* weight_buf,
                          size_t weight_size, int32_t* acc_buf, size_t acc_size, VTAUop* uop_buf,
                          size_t uop_size, std::string filename) {
  json data_file;
  std::string out_filename = filename + "_data.json";
  std::ofstream out_file(out_filename);
  data_file["data_dump"] = json::array({});
  auto& data = data_file["data_dump"];
  for (int i = 0; i < input_size; ++i) {
    data.push_back(
        {{"idx", i}, {"name", "input_buffer"}, {"value", to_hex<int8_t>(input_buf[i])}});
  }
  for (int i = 0; i < weight_size; ++i) {
    data.push_back(
        {{"idx", i}, {"name", "weight_buffer"}, {"value", to_hex<int8_t>(weight_buf[i])}});
  }
  for (int i = 0; i < acc_size; ++i) {
    data.push_back({{"idx", i}, {"name", "bias_buffer"}, {"value", to_hex<int32_t>(acc_buf[i])}});
  }
  for (int i = 0; i < uop_size; ++i) {
    data.push_back({{"idx", i},
                    {"name", "uop_buffer"},
                    {"value", to_hex<uint64_t>(*(reinterpret_cast<uint64_t*>(&uop_buf[i])))}});
  }
  out_file << std::setw(4) << data_file << "\n";
  return out_filename;
}

// Calculates GCD (greatest common divisor) of `x` and `y`
int gcd(int x, int y) {
  while (x ^= y ^= x ^= y %= x)
    ;
  return y;
}

// approximate the scale of activation (scale of input * scale of weights)
// the denominator is
std::vector<int> approximate_scale(double x) {
  int n = 1;
  int d = 1;
  double eps = 1e-7;
  double fract_value = (double)n / (double)d;
  while (fabs(fract_value - x) > eps) {
    if (fract_value < x) {
      n += 1;
    } else {
      d += 1;
      n = int(round(x * (double)d));
    }
    fract_value = (double)n / (double)d;
  }
  int nbits_r = (int)(ceil(log2((double)d)));
  int nbits_l = (int)(floor(log2((double)d)));
  int round_down_d = 1 << nbits_l;
  int round_up_d = 1 << nbits_r;
  int nbits = nbits_l;
  if (round_up_d - d < d - round_down_d) {
    nbits = nbits_r;
  } else {
    nbits = nbits_l;
    round_up_d = round_down_d;
  }
  double fact = (double)round_up_d / (double)d;
  double n_scaled = (double)n * fact;
  int round_up_n = round(n_scaled);
  // int div = gcd(round_up_n, round_up_d);
  std::vector<int> result = {round_up_n, nbits};
  return result;
}

std::string runILASimulator(const std::string exp_name, const std::string driver_dir,
                            int64_t& out_compile_time, const std::string ila_asm,
                            const std::string data_dump, const bool use_trace) {
  // Check dump file
  std::string input_filename = exp_name + "_input.json";
  std::string output_filename = exp_name + "_out.json";
  if (use_trace) {
    auto ret = std::system("stat vta_sim_dump.json > /dev/null 2> /dev/null");
    CHECK(ret == 0) << "vta_sim_dump.json does not exists";

    ret = std::system(("python3 " + driver_dir +
                       "/produce_ila_fragment.py vta_sim_dump.json ./prog_frag/" + input_filename)
                          .c_str());
    CHECK(ret == 0) << "Failed to produce program fragment";
  } else {
    auto start_time = std::chrono::high_resolution_clock::now();
    CHECK(std::system(("python3 " + driver_dir + "/produce_prog_frag.py " + ila_asm + " " +
                       data_dump + " " + "./prog_frag/" + input_filename)
                          .c_str()) == 0)
        << "Failed to convert to program fragment";
    auto end_time = std::chrono::high_resolution_clock::now();
    out_compile_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
  }
  int ret = std::system(("vta_ila_sim " + exp_name).c_str());
  CHECK(ret == 0) << "Failed to run ILA simulator";

  ret = std::system(("stat ./result/" + output_filename + " > /dev/null 2> /dev/null").c_str());
  CHECK(ret == 0) << "Not output result found";

  return "./result/" + output_filename;
}

void readILAOutput(const std::string filename, ila_output_data& out_values) {
  LOG(INFO) << "[Runtime] Reading results from ILA Simulator";

  std::unordered_map<std::string, std::string> value;
  std::string key;
  std::string data;
  std::ifstream input_stream(filename);
  dmlc::JSONReader reader(&input_stream);

  reader.BeginArray();
  while (reader.NextArrayItem()) {
    reader.Read(&value);
    out_values.push_back(value);
  }
}

json get_gemm(int batch, size_t n_inp_cols, size_t n_wgt_rows, int factor, int nbits) {
    size_t in_dim = n_inp_cols % VTA_BLOCK_IN != 0 ? n_inp_cols / VTA_BLOCK_IN + 1 : n_inp_cols / VTA_BLOCK_IN;
    size_t out_dim = n_wgt_rows % VTA_BLOCK_OUT != 0 ? n_wgt_rows / VTA_BLOCK_OUT + 1 : n_wgt_rows / VTA_BLOCK_OUT;
    size_t uop_size = batch * in_dim * out_dim;
    json prog_frag = {};
    prog_frag["asm"] = json::array({});
    auto& prog = prog_frag["asm"];
    // prog.push_back(get2DLoadStoreAsm(VTA_OPCODE_LOAD, VTA_MEM_ID_UOP, 0, 0, 1, uop_size));
    // prog.push_back(get2DLoadStoreAsm(VTA_OPCODE_LOAD, VTA_MEM_ID_WGT, 0, 0, out_dim * in_dim, 1));
    // prog.push_back(get2DLoadStoreAsm(VTA_OPCODE_LOAD, VTA_MEM_ID_INP, 0, 0, batch * in_dim, 1));
    // prog.push_back(getGEMMAsm(0, uop_size));
    // prog.push_back(getAluAsm(4 /* VTA_ALU_OPCODE_MUL */, 0, uop_size, 1, factor));
    // prog.push_back(getAluAsm(VTA_ALU_OPCODE_SHR, 0, uop_size, 1, nbits));
    // prog.push_back(getAluAsm(VTA_ALU_OPCODE_MAX, 0, uop_size, 1, -127));
    // prog.push_back(getAluAsm(VTA_ALU_OPCODE_MIN, 0, uop_size, 1, 127));
    // prog.push_back(get2DLoadStoreAsm(VTA_OPCODE_STORE, VTA_MEM_ID_OUT, 0, 0, batch * out_dim, 1));
    // Input/output channels
    const int virtual_threads = 1;
    const int uop_compression = false;
    const int block = VTA_BLOCK_OUT;
    int in_feat = n_inp_cols;
    int out_feat = n_wgt_rows;
    // Derive number of elements that need to be loaded/stored
    int ins_size = batch / block * out_feat / block * (2 + in_feat / block * 3) + 2;
    int uop_size = uop_compression ?
        block / VTA_BATCH * virtual_threads :
        block / VTA_BATCH * block / VTA_BLOCK_IN * block / VTA_BLOCK_OUT * virtual_threads;
    int inp_size = batch / VTA_BATCH * in_feat / VTA_BLOCK_IN;
    int wgt_size = in_feat / VTA_BLOCK_IN * out_feat / VTA_BLOCK_OUT;
    int out_size = batch / VTA_BATCH * out_feat / VTA_BLOCK_OUT;
    // Blocked buffer sizes (in terms of elements)
    int inp_block_size = block / VTA_BATCH * block / VTA_BLOCK_IN;
    int wgt_block_size = block / VTA_BLOCK_IN * block / VTA_BLOCK_OUT;
    int out_block_size = block / VTA_BATCH * block / VTA_BLOCK_OUT;
    // Make sure we don't exceed buffer bounds
    assert(uop_size <= VTA_UOP_BUFF_DEPTH);
    assert(inp_block_size <= VTA_INP_BUFF_DEPTH);
    assert(wgt_block_size <= VTA_WGT_BUFF_DEPTH);
    assert(out_block_size <= VTA_ACC_BUFF_DEPTH);

    std::cerr << VTA_INP_ELEM_BYTES << " " << VTA_WGT_ELEM_BYTES << " " << VTA_ACC_ELEM_BYTES << " " << VTA_OUT_ELEM_BYTES << " " << VTA_UOP_ELEM_BYTES << "\n";

    // Initialize instruction buffer
    // VTAGenericInsn *insn_buf =
    //     static_cast<VTAGenericInsn *>(allocBuffer(sizeof(VTAGenericInsn) * ins_size));
    int insn_idx = 0;

    // Initialize inputs
    // int8_t **inputs = allocInit2dArray<int8_t>(batch, in_feat);
    // Initialize weights
    // int8_t **weights = allocInit2dArray<int8_t>(out_feat, in_feat);
    // Initialize biases
    // int32_t **biases = allocInit2dArray<int32_t>(batch, out_feat);
    // for (int i = 0; i < batch; ++i) {
    //   memset(biases[i], 0, sizeof(int32_t) * out_feat);
    // }

    // fprintf(stdout, "Packed buffer: \n");
    int8_t **outputs = alloc2dArray<int8_t>(batch, out_feat);

    uint32_t *input_buf = static_cast<uint32_t *>(
        allocBuffer(VTA_INP_ELEM_BYTES * inp_size));
    packBuffer<uint32_t, 32, int8_t, VTA_INP_WIDTH>(input_buf,
                                                  inputs,
                                                  batch,
                                                  in_feat,
                                                  VTA_BATCH,
                                                  VTA_BLOCK_IN);
    
    // for (int i = 0; i < batch; ++i) {
    //   for (int j = 0; j < in_feat; ++j) {
    //     fprintf(stdout, "%d ", input_buf[i * in_feat + j]);
    //   }
    //   fprintf(stdout, "\n");
    // }

    // Prepare the weight buffer
    uint32_t *weight_buf = static_cast<uint32_t *>(
        allocBuffer(VTA_WGT_ELEM_BYTES * wgt_size));
    packBuffer<uint32_t, 32, int8_t, VTA_WGT_WIDTH>(weight_buf,
                                                  weights,
                                                  out_feat,
                                                  in_feat,
                                                  VTA_BLOCK_OUT,
                                                  VTA_BLOCK_IN);
    // Prepare the bias buffer
    uint32_t *bias_buf = static_cast<uint32_t *>(
        allocBuffer(VTA_ACC_ELEM_BYTES * out_size));
    packBuffer<uint32_t, 32, int32_t, VTA_ACC_WIDTH>(bias_buf,
                                                  biases,
                                                  batch,
                                                  out_feat,
                                                  VTA_BATCH,
                                                  VTA_BLOCK_OUT);
    // Prepare the output buffer
    uint32_t *output_buf = static_cast<uint32_t *>(
        allocBuffer(VTA_INP_ELEM_BYTES * out_size));
    
    // Load uops
    // Prepare the uop buffer
    VTAUop * uop_buf = getGEMMUops(
        block / VTA_BATCH,
        block / VTA_BLOCK_IN,
        block / VTA_BLOCK_OUT);
    insn_buf[insn_idx++] = get1DLoadStoreInsn(VTA_OPCODE_LOAD,
                                              VTA_MEM_ID_UOP,
                                              0,
                                              VTAMemGetPhyAddr(uop_buf) / VTA_UOP_ELEM_BYTES,
                                              uop_size,
                                              0,
                                              0,
                                              0,
                                              0);
    // Iterate over batch blocks
    for (int i = 0; i < batch; i += block) {
      // Iterate over output channel blocks
      for (int j = 0; j < out_feat; j += block) {
        // std::cerr << "i (batch): " << i  << "\tj (out_feat): " << j << std::endl;
        // Load bias block (pop next if not first, push prev)
        insn_buf[insn_idx++] = get2DLoadStoreInsn(
            VTA_OPCODE_LOAD,                                    // opcode
            VTA_MEM_ID_ACC,                                     // type
            0,                                                  // sram offset
            (VTAMemGetPhyAddr(bias_buf) + (i / VTA_BATCH * out_feat + j) / block * VTA_ACC_ELEM_BYTES) / VTA_ACC_ELEM_BYTES,     // dram offset
            block / VTA_BATCH,                                  // y size
            block / VTA_BLOCK_OUT,                              // x size
            out_feat / VTA_BLOCK_OUT,                           // x stride
            0,                                                  // y pad
            0,                                                  // x pad
            0,                                                  // pop prev dep
            (i > 0 || j > 0),                                   // pop next dep
            (virtual_threads == 1),                             // push prev dep
            0);                                                 // push next dep
        // Iterate over input channel blocks
        for (int k = 0; k < in_feat; k += block * virtual_threads) {
          for (int l = 0; l < block * virtual_threads; l += block) {
            // std::cerr << "k (in_feat): " << k << "\tl (block): " << l << std::endl;
            // Derive dependence flags
            bool pop = (virtual_threads == 1) ?
                1 :
                (i > 0 || j > 0 || k > 0 || l > 0) && (k + l != block * virtual_threads - block);
            bool push_prev = (virtual_threads == 1) ?
                ((k + l) != in_feat - block) :
                ((k + l) != in_feat - virtual_threads * block) &&
                (
                    (k + l != in_feat - block) ||
                    (j != out_feat - block) ||
                    (i != batch - block));
            bool push_next = (k + l == in_feat - block);
            // Load weight block (pop next)
            // std::cerr << "LOAD WGT: " << std::endl;
            // std::cerr << "\tSRAM offset: " << l / VTA_BLOCK_IN * block / VTA_BLOCK_OUT << "\n";
            // std::cerr << "\tDRAM offset: " << (j / VTA_BLOCK_OUT * in_feat + k + l) << "\n";
            // std::cerr << "\ty size: " << block / VTA_BLOCK_OUT << "\n";
            // std::cerr << "\tx size: " << block / VTA_BLOCK_IN << "\n";
            // std::cerr << "\tx stride: " << in_feat / VTA_BLOCK_IN << "\n";
            insn_buf[insn_idx++] = get2DLoadStoreInsn(
                VTA_OPCODE_LOAD,                                // opcode
                VTA_MEM_ID_WGT,                                 // type
                l / VTA_BLOCK_IN * block,       // sram offset
                (VTAMemGetPhyAddr(weight_buf) + (j / VTA_BLOCK_OUT * in_feat + k + l) / block * VTA_WGT_ELEM_BYTES) / VTA_WGT_ELEM_BYTES ,  // dram offset
                block / VTA_BLOCK_OUT,                          // y size
                block / VTA_BLOCK_IN,                           // x size
                in_feat / VTA_BLOCK_IN,                         // x stride
                0,                                              // y pad
                0,                                              // x pad
                0,                                              // pop prev dep
                pop,                                            // pop next dep
                0,                                              // push prev dep
                0);                                             // push next dep
            // Load input block (push next)
            // 
            // std::cerr << "LOAD INP: " << std::endl;
            // std::cerr << "\tSRAM offset: " << l / VTA_BLOCK_IN * block / VTA_BLOCK_OUT << "\n";
            // std::cerr << "\tDRAM offset: " << (j / VTA_BLOCK_OUT * in_feat + k + l) << "\n";
            // std::cerr << "\ty size: " << block / VTA_BATCH << "\n";
            // std::cerr << "\tx size: " << block / VTA_BLOCK_IN << "\n";
            // std::cerr << "\tx stride: " << in_feat / VTA_BLOCK_IN << "\n";
            insn_buf[insn_idx++] = get2DLoadStoreInsn(
                VTA_OPCODE_LOAD,                                // opcode
                VTA_MEM_ID_INP,                                 // type
                l / VTA_BLOCK_IN * block,           // sram offset
                (VTAMemGetPhyAddr(input_buf) + (i / VTA_BATCH * in_feat + k + l) / block * VTA_INP_ELEM_BYTES) / VTA_INP_ELEM_BYTES ,  // dram offset
                block / VTA_BATCH,                              // y size
                block / VTA_BLOCK_IN,                           // x size
                in_feat / VTA_BLOCK_IN,                         // x stride
                0,                                              // y pad
                0,                                              // x pad
                0,                                              // pop prev dep
                0,                                              // pop next dep
                0,                                              // push prev dep
                1);                                             // push next dep
            // Perform GEMM (pop prev, push prev if not last, push next if last)
            // std::cerr << "GEMM" << std::endl;
            int uop_start = l / block * uop_size / virtual_threads;
            // std::cerr << "UOP Offset " << l / block * uop_size / virtual_threads << std::endl;
            // for (int i = uop_start; i < uop_start + block / VTA_BATCH * block / VTA_BLOCK_IN * block / VTA_BLOCK_OUT; ++i) {
            //   std::cerr << "\t" << "uop[" << i << "]: " << "dst = "
            //             << uop_buf[i].dst_idx
            //             << "; src_idx = " << uop_buf[i].src_idx
            //             << "; wgt_idx = " << uop_buf[i].wgt_idx << "\n";
            // }
            insn_buf[insn_idx++] = getGEMMInsn(
                l / block * uop_size / virtual_threads,         // uop offset
                block / VTA_BATCH,                              // batch
                block / VTA_BLOCK_IN,                           // in_feat
                block / VTA_BLOCK_OUT,                          // out_feat
                uop_compression,                                // uop_compression
                1,                                              // pop_prev_dep
                0,                                              // pop_next_dep
                push_prev,                                      // push prev dep
                push_next);                                     // push_next_dep
          }
        }

        // Store output block (pop prev, push prev if not last)
        // std::cerr << "STORE: " << std::endl;
        // std::cerr << "\tSRAM offset: " << 0 << "\n";
        // std::cerr << "\tDRAM offset: " << (i / VTA_BATCH * out_feat + j) << "\n";
        // std::cerr << "\ty size: " << block / VTA_BATCH << "\n";
        // std::cerr << "\tx size: " << block / VTA_BLOCK_OUT << "\n";
        // std::cerr << "\tx stride: " << out_feat / VTA_BLOCK_OUT << "\n";
        insn_buf[insn_idx++] = get2DLoadStoreInsn(
            VTA_OPCODE_STORE,                                   // opcode
            VTA_MEM_ID_OUT,                                     // type
            0,                                                  // sram offset
            (VTAMemGetPhyAddr(output_buf)
              + (i / VTA_BATCH * out_feat + j)
              / block * VTA_OUT_ELEM_BYTES) / VTA_OUT_ELEM_BYTES,     // dram offset
            block / VTA_BATCH,                                  // y size
            block / VTA_BLOCK_OUT,                              // x size
            out_feat / VTA_BLOCK_OUT,                           // x stride
            0,                                                  // y pad
            0,                                                  // x pad
            1,                                                  // pop prev dep
            0,                                                  // pop next dep
            1,                                                  // pop prev dep
            0);                                                 // push next dep
      }
    }
    // Finish
    insn_buf[insn_idx++] = getFinishInsn(0, 1);
    return prog_frag;
}

size_t loadILAOutput(const ila_output_data& out_values, int8_t* buffer, size_t out_h,
                     size_t out_w) {
  LOG(INFO) << "[Runtime] Copying from output json to byte buffer";

  size_t data_cur = 0;
  size_t buf_cur = 0;
  int32_t temp;
  for (size_t i = 0; i < out_h; ++i) {
    if (data_cur % VTA_BLOCK_OUT != 0) {
      data_cur = (data_cur / VTA_BLOCK_OUT + 1) * VTA_BLOCK_OUT;
    }
    for (size_t j = 0; j < out_w; ++j) {
      auto val = out_values[data_cur++].at("data");
      std::stringstream ss;
      ss << std::hex << val;
      ss >> temp;
      buffer[buf_cur++] = static_cast<int8_t>(temp);
    }
  }
  return buf_cur;
}

void copy_data(int8_t* from_, int8_t* out_data, size_t size) {
  std::cerr << "Read back\n";
  for (size_t i = 0; i < size; ++i) {
    std::cerr << (int)from_[i] << " ";
    out_data[i] = from_[i];
  }
  std::cerr << "\n";
}

int64_t runSimGetData(std::string pattern_name, std::string driver_dir, std::string ila_asm,
                      std::string data_dump, size_t output_size, int n_output_rows,
                      int n_output_cols, void* output_data, std::string output_dtype) {
  auto start_time = std::chrono::high_resolution_clock::now();
  int64_t compile_time;
  std::string output_file =
      runILASimulator(pattern_name, driver_dir, compile_time, ila_asm, data_dump, false);
  auto end_time = std::chrono::high_resolution_clock::now();

  ila_output_data out_data;
  readILAOutput(output_file, out_data);

  int8_t* buffer = new int8_t[output_size];
  auto buf_read = loadILAOutput(out_data, buffer, n_output_rows, n_output_cols);
  // CHECK(buf_read == output_size) << "Output size mismatch: " << buf_read << " v.s. " <<
  // output_size;
  copy_data(buffer, reinterpret_cast<int8_t*>(output_data), buf_read);
  return compile_time;
  // return std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
