#include <iomanip>
#include "ilavta_codegen_utils.h"

namespace tvm {
namespace relay {
namespace contrib {

using namespace nlohmann;
using addr_byte_pairs = std::vector<std::pair<vta_phy_addr_t, uint8_t>>;

json byte_pairs_to_json(const addr_byte_pairs& byte_pairs) {
  std::vector<json> pair_list;

  for (const auto& pair : byte_pairs) {
    std::stringstream addr_stream;
    addr_stream << "0x" << std::setfill('0') << std::setw(sizeof(vta_phy_addr_t)*2)
                << std::hex << pair.first;

    std::stringstream byte_stream;
    // casting to uint32_t because uint8_t's are treated as char literals, not ints
    byte_stream << "0x" << std::setfill('0') << std::setw(2)
                << std::hex << static_cast<uint32_t>(pair.second);

    pair_list.push_back({
        {"addr", addr_stream.str()},
        {"value", byte_stream.str()}
      });
  }

  return pair_list;
}

json getGEMMAsm(int uop_offset, int batch, int in_feat, int out_feat) {
  return {
    {"name", "gemm"},
    {"reset_f", 0},
    {"uop_bgn", uop_offset},
    {"uop_end", uop_offset + batch * in_feat * out_feat},
    {"iter_o", 1},
    {"iter_i", 1},
    {"dst_fo", 0},
    {"dst_fi", 0},
    {"src_fo", 0},
    {"src_fi", 0},
    {"dst_fo", 0},
    {"dst_fi", 0},
    {"wgt_fo", 0},
    {"wgt_fi", 0}
  };
}

std::string getMemOpInsnStr(int opcode, int mem_type) {
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
  return cmd_type;
}

json get2DLoadStoreAsm(int opcode, int mem_type, int sram_id, int dram_id, int y_size, int x_size, int x_stride) {
  std::string cmd_type = getMemOpInsnStr(opcode, mem_type);
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
      {"x_stride", x_stride}
    };
  } else if (cmd_type == "load_inp") {
    return {
      {"name", cmd_type},
      {"sram_id", sram_id},
      {"dram_id", dram_id},
      {"y_size", y_size},
      {"x_size", x_size},
      {"x_stride", x_stride},
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

json get1DLoadStoreAsm(int opcode, int mem_type, int sram_id, int dram_id, int size) {
  std::string cmd_type = getMemOpInsnStr(opcode, mem_type);
  if (cmd_type == "load_uop") {
    return {
      {"name", cmd_type},
      {"sram_id", sram_id},
      {"dram_id", dram_id},
      {"size", size},
    };
  } else {
    return get2DLoadStoreAsm(opcode, mem_type, sram_id, dram_id, 1, size, size);
  }
}

json getAluAsm(int alu_opcode, int uop_bgn, int uop_end, bool use_imm, uint16_t imm) {
  int asm_opcode = -1;
  std::string op_name = "";
  switch (alu_opcode) {
    case VTA_ALU_OPCODE_MIN: asm_opcode = 0; op_name = "min"; break;
    case VTA_ALU_OPCODE_MAX: asm_opcode = 1; op_name = "max"; break;
    case VTA_ALU_OPCODE_ADD: asm_opcode = 2; op_name = "add"; break;
    case VTA_ALU_OPCODE_SHR: asm_opcode = 3; op_name = "shr"; break;
    case VTA_ALU_OPCODE_MUL: asm_opcode = 4; op_name = "mul"; break;
    default:
      fprintf(stderr, "ALU Opcode %d is not valid", alu_opcode);
      exit(-1);
  }
  return {
    {"name", "alu_" + op_name},
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
    {"imm", imm}
  };
}

std::string write_to_file(const std::string& filename, const json& data) {
  std::ofstream out_file(filename + "_prog_frag.json");
  out_file << std::setw(4) << data << "\n";
  return filename + "_prog_frag.json";
}

std::string CompileGEMM(int batch, size_t in_channels, size_t out_channels, int factor, int nbits, std::string filename) {
  // Input/output channels
  const int block = VTA_BLOCK_OUT; // should be 16 by default
  assert(block == 16);
  int in_feat = in_channels;
  int out_feat = out_channels;
  const int virtual_threads = 1;
  // Derive number of elements that need to be loaded/stored
  int ins_size = batch / block * out_feat / block * (2 + in_feat / block * 3) + 2;
  int uop_size = block / VTA_BATCH * block / VTA_BLOCK_IN * block / VTA_BLOCK_OUT * virtual_threads;
  uop_size += batch;
  int inp_size = batch / VTA_BATCH * in_feat / VTA_BLOCK_IN;
  int wgt_size = in_feat / VTA_BLOCK_IN * out_feat / VTA_BLOCK_OUT;
  int out_size = batch / VTA_BATCH * out_feat / VTA_BLOCK_OUT;
  // Blocked buffer sizes (in terms of elements)
  int inp_block_size = block / VTA_BATCH * block / VTA_BLOCK_IN;
  int wgt_block_size = block / VTA_BLOCK_IN * block / VTA_BLOCK_OUT;
  int out_block_size = block / VTA_BATCH * block / VTA_BLOCK_OUT;
  json prog_frag = {};
  prog_frag["asm"] = json::array({});
  auto& prog = prog_frag["asm"];
  // load uop
  prog.push_back(get1DLoadStoreAsm(VTA_OPCODE_LOAD, VTA_MEM_ID_UOP, 0, 0, uop_size));
  // Iterate over batch blocks
  for (int i = 0; i < batch; i += block) {
    // Iterate over output channel blocks
    for (int j = 0; j < out_feat; j += block) {
      // std::cerr << "i (batch): " << i  << "\tj (out_feat): " << j << std::endl;
      // Load bias block (pop next if not first, push prev)
      prog.push_back(get2DLoadStoreAsm(
          VTA_OPCODE_LOAD,                                    // opcode
          VTA_MEM_ID_ACC,                                     // type
          0,                                                  // sram offset
          i / VTA_BATCH * out_feat + j,                       // dram offset
          block / VTA_BATCH,                                  // y size
          block / VTA_BLOCK_OUT,                              // x size
          out_feat / VTA_BLOCK_OUT                            // x stride
        ));
      // Iterate over input channel blocks
      for (int k = 0; k < in_feat; k += block * virtual_threads) {
        for (int l = 0; l < block * virtual_threads; l += block) {
          prog.push_back(get2DLoadStoreAsm(
              VTA_OPCODE_LOAD,                                // opcode
              VTA_MEM_ID_WGT,                                 // type
              l / VTA_BLOCK_IN * block,                       // sram offset
              j / VTA_BLOCK_OUT * in_feat + k + l,            // dram offset
              block / VTA_BLOCK_OUT,                          // y size
              block / VTA_BLOCK_IN,                           // x size
              in_feat / VTA_BLOCK_IN                          // x stride
          ));
          // Load input block (push next)
          prog.push_back(get2DLoadStoreAsm(
              VTA_OPCODE_LOAD,                                // opcode
              VTA_MEM_ID_INP,                                 // type
              l / VTA_BLOCK_IN * block,                       // sram offset
              (i / VTA_BATCH * in_feat + k + l),              // dram offset
              block / VTA_BATCH,                              // y size
              block / VTA_BLOCK_IN,                           // x size
              in_feat / VTA_BLOCK_IN                          // x stride
          ));
          // Perform GEMM (pop prev, push prev if not last, push next if last)
          prog.push_back(getGEMMAsm(
              l / block * uop_size / virtual_threads,         // uop offset
              block / VTA_BATCH,                              // batch
              block / VTA_BLOCK_IN,                           // in_feat
              block / VTA_BLOCK_OUT                           // out_feat
          ));
        }
      }
      prog.push_back(getAluAsm(
        VTA_ALU_OPCODE_MUL,
        block / VTA_BATCH * block / VTA_BLOCK_IN * block / VTA_BLOCK_OUT,
        block / VTA_BATCH * block / VTA_BLOCK_IN * block / VTA_BLOCK_OUT + block,
        true,
        factor
      ));
      prog.push_back(getAluAsm(
        VTA_ALU_OPCODE_SHR,
        block / VTA_BATCH * block / VTA_BLOCK_IN * block / VTA_BLOCK_OUT,
        block / VTA_BATCH * block / VTA_BLOCK_IN * block / VTA_BLOCK_OUT + block,
        true,
        nbits
      ));
      prog.push_back(get2DLoadStoreAsm(
          VTA_OPCODE_STORE,                                         // opcode
          VTA_MEM_ID_OUT,                                           // type
          0,                                                        // sram offset
          i / VTA_BATCH * out_feat + j,                             // dram offset
          block / VTA_BATCH,                                        // y size
          block / VTA_BLOCK_OUT,                                    // x size
          out_feat / VTA_BLOCK_OUT                                  // x stride
      ));
    }
  }
  return write_to_file(filename, prog_frag);
}

std::string CompilBiasAdd(int batch, size_t n_feat, std::string filename) {
  size_t in_dim = n_feat % VTA_BLOCK_IN != 0 ? n_feat / VTA_BLOCK_IN + 1 : n_feat / VTA_BLOCK_IN;
  size_t uop_size = batch * in_dim;
  json prog_frag = {
    {"asm", json::array({})}
  };
  auto& prog = prog_frag["asm"];
  prog.push_back(get2DLoadStoreAsm(VTA_OPCODE_LOAD, VTA_MEM_ID_UOP, 0, 0, 1, uop_size));
  prog.push_back(get2DLoadStoreAsm(VTA_OPCODE_LOAD, VTA_MEM_ID_ACC, 0, 0, batch * in_dim, 1));
  prog.push_back(get2DLoadStoreAsm(VTA_OPCODE_LOAD, VTA_MEM_ID_ACC, batch * in_dim, batch * in_dim, in_dim, 1));
  prog.push_back(getAluAsm(VTA_ALU_OPCODE_ADD, 0, uop_size, 0, 0));
  prog.push_back(get2DLoadStoreAsm(VTA_OPCODE_STORE, VTA_MEM_ID_OUT, 0, 0, batch * in_dim, 1));
  return write_to_file(filename, prog_frag);
}

std::string CompileRelu(int batch, size_t n_feat, std::string filename) {
  size_t in_dim = n_feat % VTA_BLOCK_IN != 0 ? n_feat / VTA_BLOCK_IN + 1 : n_feat / VTA_BLOCK_IN;
  size_t uop_size = batch * in_dim;
  json prog_frag = {
    {"asm", json::array({})}
  };
  auto& prog = prog_frag["asm"];
  prog.push_back(get2DLoadStoreAsm(VTA_OPCODE_LOAD, VTA_MEM_ID_UOP, 0, 0, 1, uop_size));
  prog.push_back(get2DLoadStoreAsm(VTA_OPCODE_LOAD, VTA_MEM_ID_ACC, 0, 0, batch * in_dim, 1));
  prog.push_back(getAluAsm(VTA_ALU_OPCODE_MAX, 0, uop_size, 1, 0));
  prog.push_back(get2DLoadStoreAsm(VTA_OPCODE_STORE, VTA_MEM_ID_OUT, 0, 0, batch * in_dim, 1));
  return write_to_file(filename, prog_frag);
}

std::string GetCompiledFilename(const std::string op_name, const int* input_info, const int num_info) {
  std::stringstream ss;
  ss << op_name + "_";
  for (int i = 0; i < num_info; ++i) {
    ss << input_info[i] << "_";
  }
  return ss.str();
}

}
}
}

