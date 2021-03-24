#include "ilavta_helpers.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime;

const int64_t SIM_DUMP = 1;
const std::string RAW_DUMP = "vta_sim_dump.json";
const std::string OUTPUT_DUMP = "vta_output_dump.json";

/*
* Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
* */
VTAUop * getGEMMUops(int batch, int in_feat, int out_feat) {
  // Derive the total uop size
  int uop_size = batch * in_feat * out_feat;

  // Allocate buffer
  VTAUop *uop_buf = static_cast<VTAUop *>(VTAMemAlloc(sizeof(VTAUop) * uop_size, 0));

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
  return uop_buf;
}

VTAUop * getBiasAddUops(int batch, int in_feat) {
  int uop_size = batch * in_feat;
  VTAUop *uop_buf = static_cast<VTAUop *>(VTAMemAlloc(sizeof(VTAUop) * uop_size, 0));

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

VTAUop * getReluUops(int batch, int in_feat) {
  int uop_size = batch * in_feat;
  VTAUop *uop_buf = static_cast<VTAUop *>(VTAMemAlloc(sizeof(VTAUop) * uop_size, 0));

  int uop_idx = 0;
  for (int i = 0; i < batch; i++) {
    for (int j = 0; j < in_feat; j++) {
      uop_buf[uop_idx].dst_idx = i * in_feat + j;
      uop_idx++;
    }
  }
  return uop_buf;
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
  insn.uop_end = uop_offset + batch * in_feat * out_feat;;
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
                          int pop_prev_dep, int pop_next_dep, int push_prev_dep, int push_next_dep) {
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

std::string runILASimulator(const std::string exp_name,
                            const std::string ila_asm = "",
                            const std::string data_dump = "", bool use_trace = true) {
  // Check dump file
  std::string input_filename = exp_name + "_input.json";
  std::string output_filename = exp_name + "_out.json";
  if (use_trace) {
    auto ret = std::system("stat vta_sim_dump.json > /dev/null 2> /dev/null");
    CHECK(ret == 0) << "vta_sim_dump.json does not exists";

    ret = std::system(("python3 produce_ila_fragment.py vta_sim_dump.json ./prog_frag/" + input_filename).c_str());
    CHECK(ret == 0) << "Failed to produce program fragment";
  } else {
    CHECK(std::system(("python3 produce_prog_frag.py "
                      + ila_asm + " "
                      + data_dump + " "
                      + "./prog_frag/" + input_filename).c_str()) == 0) << "Failed to convert to program fragment";
  }
  int ret = std::system(("vta_ila_sim " + exp_name).c_str());
  CHECK(ret == 0) << "Failed to run ILA simulator";

  ret = std::system(("stat ./result/" + output_filename + " > /dev/null 2> /dev/null").c_str());
  CHECK(ret == 0) << "Not output result found";

  return "./result/" + output_filename;
}

void readILAOutput(const std::string filename, ila_output_data &out_values) {
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

size_t loadILAOutput(const ila_output_data &out_values, uint8_t* buffer, size_t out_h, size_t out_w) {
  LOG(INFO) << "[Runtime] Copying from output json to byte buffer"; 

  size_t data_cur = 0;
  size_t buf_cur = 0;
  uint32_t temp;
  for (size_t i = 0; i < out_h; ++i) {
    if (data_cur % VTA_BLOCK_OUT != 0) {
      data_cur = (data_cur / VTA_BLOCK_OUT + 1) * VTA_BLOCK_OUT;
    }
    for (size_t j = 0; j < out_w; ++j) {
      auto val = out_values[data_cur++].at("data");
      std::stringstream ss;
      ss << std::hex << val;
      ss >> temp;
      buffer[buf_cur++] = static_cast<uint8_t>(temp);
    }
  }
  return buf_cur;
}

void runSimGetData(std::string pattern_name, size_t output_size, int n_output_rows, int n_output_cols, void *output_data) {
  std::string output_file = runILASimulator(pattern_name);

  ila_output_data out_data;
  readILAOutput(output_file, out_data);

  uint8_t* buffer = new uint8_t[output_size];

  auto buf_read = loadILAOutput(out_data, buffer, n_output_rows, n_output_cols);
  CHECK(buf_read == output_size) << "Output size mismatch: " << buf_read << " v.s. " << output_size;
  uint8_t* o_data = reinterpret_cast<uint8_t*>(output_data);
  for (size_t i = 0; i < buf_read; ++i) {
    o_data[i] = buffer[i];
  }
}

}
}
}
