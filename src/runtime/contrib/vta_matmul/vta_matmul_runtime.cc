#include "vta_matmul_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {

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

VTAUop* getGEMMUops() {
  VTAUop *uop_buf = static_cast<VTAUop *>(VTAMemAlloc(sizeof(VTAUop), 0));
  uop_buf->dst_idx = 0;
  uop_buf->src_idx = 0;
  uop_buf->wgt_idx = 0;
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

extern "C" TVM_DLL void run_vta_simulator(float* input, float* weight, int batch, int in_channels,
                                          int out_channels, float* out_buf) {
  const uint32_t num_instr = 7;
  std::cerr << "in_W: " << in_channels << " "
            << "in_H: " << batch << "\n";
  std::cerr << "w_H: " << out_channels << std::endl;
  // std::cerr << "Input:\n";
  // for (int i = 0; i < batch; ++i) {
  //   for (int j = 0; j < in_channels; ++j) {
  //     std::cerr << input[i * in_channels + j] << " ";
  //   }
  //   std::cerr << "\n";
  // }
  // std::cerr << "\n";
  // for (int i = 0; i < out_channels; ++i) {
  //   for (int j = 0; j < in_channels; ++j) {
  //     std::cerr << weight[i * out_channels + j] << " ";
  //   }
  //   std::cerr << "\n";
  // }
  int uop_size = sizeof(VTAUop);
  int inp_size = 16;
  int wgt_size = 16;
  int out_size = 16;
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

  // inp_T** inputs = allocInit2dArray<inp_T>(batch, in_channels, 1, input);
  // wgt_T** weights = allocInit2dArray<wgt_T>(out_channels, in_channels, 1, weight);
  // acc_T** bias = allocInit2dArray<acc_T>(batch, out_channels, 0, nullptr, 0);
  VTAUop* uops = getGEMMUops();

  std::cerr << "Inp Size: " << inp_size << " " << "ELEM BYTES: " << VTA_INP_ELEM_BYTES << std::endl;
  std::cerr << "Wgt Size: " << wgt_size << "\n";

  int8_t* input_buf = static_cast<int8_t*>(allocBuffer(VTA_INP_ELEM_BYTES * inp_size * inp_size));
  // copyData<int8_t>(input_buf, input, batch, in_channels);
  // packBuffer<uint32_t, 32, inp_T, VTA_INP_WIDTH>(input_buf, inputs, batch, in_channels, VTA_BATCH,
                                                //  VTA_BLOCK_IN);
  std::cerr << "Input 1:\n";
  // for (int i = 0; i < batch; ++i) {
  //   for (int j = 0; j < in_channels; ++j) {
  //     std::cerr << input[i * in_channels + j] << " ";
  //   }
  //   std::cerr << "\n";
  // }
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < in_channels; ++j) {
      input_buf[i * in_channels + j] = 1;
      // std::cerr << static_cast<int>(input[i * in_channels + j]) << " ";
    }
    std::cerr << "\n";
  }
  int8_t* wgt_buf = static_cast<int8_t*>(allocBuffer(VTA_WGT_ELEM_BYTES * wgt_size * wgt_size));
  copyData<int8_t>(wgt_buf, weight, out_channels, in_channels);
  // packBuffer<uint32_t, 32, out_T, VTA_WGT_WIDTH>(wgt_buf, weights, out_channels, in_channels,
                                                //  VTA_BLOCK_OUT, VTA_BLOCK_IN);

  uint32_t* bias_buf = static_cast<uint32_t*>(allocBuffer(VTA_ACC_ELEM_BYTES * out_size * out_size));
  for (int i = 0; i < out_size; ++i) {
    for (int j = 0; j < out_size; ++j) {
      bias_buf[i * out_size + j] = 0;
    }
  }
  // packBuffer<uint32_t, 32, acc_T, VTA_OUT_WIDTH>(bias_buf, bias, batch, out_channels, VTA_BATCH,
                                                //  VTA_BLOCK_OUT);
  int8_t* output_buf = static_cast<int8_t*>(allocBuffer(VTA_OUT_ELEM_BYTES * out_size * out_size));
  memset(output_buf, 0, VTA_OUT_ELEM_BYTES * out_size * out_size);

  std::cerr << "Define instructions" << std::endl;

  int ptr = 0;

  instr_buf[ptr++] = get1DLoadStoreInsn(
    VTA_OPCODE_LOAD,
    VTA_MEM_ID_UOP,
    0,
    VTAMemGetPhyAddr(uops) / VTA_UOP_ELEM_BYTES,
    uop_size,
    0, 0, 0, 0
  );

  instr_buf[ptr++] = get2DLoadStoreInsn(
    VTA_OPCODE_LOAD, // op_code
    VTA_MEM_ID_ACC,  // type
    0,               // sram base
    VTAMemGetPhyAddr(bias_buf) / VTA_ACC_ELEM_BYTES, // dram base
    out_size, // y_size
    out_size, // x_size
    out_size, // x_stride
    0,        // y pad
    0,        // x_pad
    0, 0, 1, 0
  );

  instr_buf[ptr++] = get2DLoadStoreInsn(
    VTA_OPCODE_LOAD,
    VTA_MEM_ID_WGT,
    0,
    VTAMemGetPhyAddr(wgt_buf) / VTA_WGT_ELEM_BYTES,
    16,
    16,
    16,
    0,
    0,
    0, 1, 0, 0
  );

  instr_buf[ptr++] = get2DLoadStoreInsn(
    VTA_OPCODE_LOAD,
    VTA_MEM_ID_INP,
    0,
    VTAMemGetPhyAddr(input_buf) / VTA_INP_ELEM_BYTES,
    inp_size,
    inp_size,
    inp_size,
    0,
    0,
    0, 0, 0, 1
  );

  instr_buf[ptr++] = getGEMMInsn(
    0,
    batch / VTA_BATCH,
    in_channels / VTA_BLOCK_IN,
    out_channels / VTA_BLOCK_OUT,
    1,
    1, 0, 0, 1
  );

  instr_buf[ptr++] = get2DLoadStoreInsn(
    VTA_OPCODE_STORE,
    VTA_MEM_ID_OUT,
    0,
    VTAMemGetPhyAddr(output_buf) / VTA_OUT_ELEM_BYTES,
    out_size,
    out_size,
    out_size,
    0,
    0,
    1, 0, 1, 0
  );

  instr_buf[ptr++] = getFinishInsn(0, 1);

  std::cerr << "Run" << std::endl;

  VTADeviceRun(device_handle, VTAMemGetPhyAddr(instr_buf), num_instr, 1000);

  // out_T **outputs = alloc2dArray<out_T>(batch, out_channels);
  // unpackBuffer<out_T, VTA_OUT_WIDTH, uint32_t, 32>(outputs,
  //                                                  output_buf,
  //                                                  batch,
  //                                                  out_channels,
  //                                                  VTA_BATCH,
  //                                                  VTA_BLOCK_OUT);

  uint8_t* packed_out = reinterpret_cast<uint8_t*>(output_buf);

  // for (int i = 0; i < VTA_OUT_ELEM_BYTES; ++i) {
  //   printf("%d: %d\n", i, packed_out[i]);
  // }
  for (int i = 0; i < batch; ++i) {
    for (int j = 0; j < out_channels; ++j) {
      std::cerr << (int)output_buf[i * out_channels + j] << " ";
    }
    std::cerr << "\n";
  }
  VTAMemFree(input_buf);
  VTAMemFree(output_buf);
  VTAMemFree(wgt_buf);
  VTAMemFree(instr_buf);
  VTADeviceFree(device_handle);
}
}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
