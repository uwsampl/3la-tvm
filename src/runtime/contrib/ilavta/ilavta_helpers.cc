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
