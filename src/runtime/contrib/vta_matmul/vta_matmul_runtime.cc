#include "vta_matmul_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {
    void run_vta_simulator(float *acc, float *weight, int in_H, int in_W,
                        int w_W, float *out_buf) {
        const uint32_t num_instr = 5;
        void *instr_buf;
        void *mem_buf;
        void *wgt_buf;
        memset(out_buf, 0, sizeof(float) * in_W * w_W);
        VTADeviceHandle device_handle = VTADeviceAlloc();
        instr_buf = VTAMemAlloc(sizeof(VTAGenericInsn) * num_instr, 0);
        mem_buf = VTAMemAlloc((in_H * in_W + 64) * sizeof(float), 0);
        wgt_buf = VTAMemAlloc((in_W * w_W + 64) * sizeof(float), 0);

        float *mem_float = reinterpret_cast<float*>(mem_buf);
        float *wgt_float = reinterpret_cast<float*>(wgt_buf);
        for (int i = 0; i < in_H; ++i) {
            for (int j = 0; j < in_W; ++j) {
                mem_float[i * in_W + j] = acc[i * in_W + j];
            }
        }

        for (int i = 0; i < in_W; ++i) {
            for (int j = 0; j < w_W; ++j) {
                wgt_float[i * w_W + j] = weight[i * w_W + j];
            }
        }
        
        VTAMemInsn *instr_mem = reinterpret_cast<VTAMemInsn*>(instr_buf);
        VTAGemInsn *instr_gem = reinterpret_cast<VTAGemInsn*>(instr_buf);

        instr_mem[0].opcode = VTA_OPCODE_LOAD;
        instr_mem[0].memory_type = VTA_MEM_ID_INP;
        instr_mem[0].sram_base = 0;
        instr_mem[0].dram_base = VTAMemGetPhyAddr(mem_buf) / VTA_INP_ELEM_BYTES;
        instr_mem[0].x_size = in_W;
        instr_mem[0].y_size = in_H;
        instr_mem[0].y_pad_0 = 0;
        instr_mem[0].y_pad_1 = 0;
        instr_mem[0].x_pad_0 = 0;
        instr_mem[0].x_pad_1 = 0;
        instr_mem[0].push_prev_dep = 0;
        instr_mem[0].pop_prev_dep = 0;
        instr_mem[0].push_next_dep = 0;
        instr_mem[0].pop_prev_dep = 0;

        instr_mem[0].opcode = VTA_OPCODE_LOAD;
        instr_mem[0].memory_type = VTA_MEM_ID_WGT;
        instr_mem[0].sram_base = 0;
        instr_mem[0].dram_base = VTAMemGetPhyAddr(wgt_float) / VTA_WGT_ELEM_BYTES;
        instr_mem[0].x_size = w_W;
        instr_mem[0].y_size = in_W;
        instr_mem[0].y_pad_0 = 0;
        instr_mem[0].y_pad_1 = 0;
        instr_mem[0].x_pad_0 = 0;
        instr_mem[0].x_pad_1 = 0;
        instr_mem[0].push_prev_dep = 0;
        instr_mem[0].pop_prev_dep = 0;
        instr_mem[0].push_next_dep = 1;
        instr_mem[0].pop_prev_dep = 0;

    }
}
}
}