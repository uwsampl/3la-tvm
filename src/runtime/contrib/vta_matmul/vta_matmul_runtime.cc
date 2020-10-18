#include "vta_matmul_runtime.h"

namespace tvm {
namespace runtime {
namespace contrib {
    extern "C" TVM_DLL void run_vta_simulator(float *acc, float *weight, int in_H, int in_W,
                        int w_W, float *out_buf) {
        const uint32_t num_instr = 7;
        // LOAD UOP
        // LOAD INP
        // LOAD WGT
        // GEMM
        // ALU
        // STORE
        // FINISH
        void *instr_buf;
        void *mem_buf;
        void *wgt_buf;
        void *sram_out;
        memset(out_buf, 0, sizeof(float) * in_W * w_W);
        VTADeviceHandle device_handle = VTADeviceAlloc();
        instr_buf = VTAMemAlloc(sizeof(VTAGenericInsn) * num_instr, 0);
        mem_buf = VTAMemAlloc((in_H * in_W) * sizeof(float), 0);
        wgt_buf = VTAMemAlloc((in_W * w_W) * sizeof(float), 0);
        sram_out = VTAMemAlloc((in_H * w_W) * sizeof(uint8_t), 0);

        float *mem_float = reinterpret_cast<float*>(mem_buf);
        float *wgt_float = reinterpret_cast<float*>(wgt_buf);
        uint8_t *sram_obuf = reinterpret_cast<uint8_t*>(sram_out);
        std::cerr << "Initializing data" << std::endl;
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
        VTAAluInsn *instr_alu = reinterpret_cast<VTAAluInsn*>(instr_buf);

        std::cerr << "Define instructions" << std::endl;

        VTAUop *uop = new VTAUop;
        uop->dst_idx = 0;
        uop->src_idx = 0;
        uop->wgt_idx = 0;

        instr_mem[0].opcode = VTA_OPCODE_LOAD;
        instr_mem[0].memory_type = VTA_MEM_ID_UOP;
        instr_mem[0].x_size = sizeof(VTAUop);
        instr_mem[0].y_size = 0;
        instr_mem[0].sram_base = 0;
        instr_mem[0].dram_base = VTAMemGetPhyAddr(uop) / VTA_UOP_ELEM_BYTES;
        instr_mem[1].y_pad_0 = 0;
        instr_mem[1].y_pad_1 = 0;
        instr_mem[1].x_pad_0 = 0;
        instr_mem[1].x_pad_1 = 0;
        instr_mem[1].push_prev_dep = 0;
        instr_mem[1].pop_prev_dep = 0;
        instr_mem[1].push_next_dep = 0;
        instr_mem[1].pop_prev_dep = 0;

        instr_mem[1].opcode = VTA_OPCODE_LOAD;
        instr_mem[1].memory_type = VTA_MEM_ID_INP;
        instr_mem[1].sram_base = 0;
        instr_mem[1].dram_base = VTAMemGetPhyAddr(mem_buf) / VTA_INP_ELEM_BYTES;
        instr_mem[1].x_size = in_W;
        instr_mem[1].y_size = in_H;
        instr_mem[1].y_pad_0 = 0;
        instr_mem[1].y_pad_1 = 0;
        instr_mem[1].x_pad_0 = 0;
        instr_mem[1].x_pad_1 = 0;
        instr_mem[1].push_prev_dep = 0;
        instr_mem[1].pop_prev_dep = 0;
        instr_mem[1].push_next_dep = 0;
        instr_mem[1].pop_prev_dep = 0;

        instr_mem[2].memory_type = VTA_MEM_ID_WGT;
        instr_mem[2].opcode = VTA_OPCODE_LOAD;
        instr_mem[2].sram_base = 0;
        instr_mem[2].dram_base = VTAMemGetPhyAddr(wgt_float) / VTA_WGT_ELEM_BYTES;
        instr_mem[2].x_size = w_W;
        instr_mem[2].y_size = in_W;
        instr_mem[2].y_pad_0 = 0;
        instr_mem[2].y_pad_1 = 0;
        instr_mem[2].x_pad_0 = 0;
        instr_mem[2].x_pad_1 = 0;
        instr_mem[2].push_prev_dep = 0;
        instr_mem[2].pop_prev_dep = 0;
        instr_mem[2].push_next_dep = 1;
        instr_mem[2].pop_prev_dep = 0;

        instr_gem[3].opcode = VTA_OPCODE_GEMM;
        instr_gem[3].pop_prev_dep = 1;
        instr_gem[3].pop_next_dep = 0;
        instr_gem[3].push_next_dep = 1;
        instr_gem[3].push_prev_dep = 0;
        instr_gem[3].uop_bgn = 0;
        instr_gem[3].uop_end = 1;
        instr_gem[3].dst_factor_in = 0;
        instr_gem[3].dst_factor_out = 0;
        instr_gem[3].src_factor_in = 0;
        instr_gem[3].src_factor_out = 0;
        instr_gem[3].wgt_factor_in = 0;
        instr_gem[3].wgt_factor_out = 0;
        instr_gem[3].iter_in = 1;
        instr_gem[3].iter_out = 1;

        instr_alu[4].opcode = VTA_OPCODE_ALU;
        instr_alu[4].alu_opcode = VTA_ALU_OPCODE_SHR;
        instr_alu[4].uop_bgn = 0;
        instr_alu[4].uop_end = 1;
        instr_alu[4].use_imm = true;
        instr_alu[4].imm = 0;
        instr_alu[4].pop_prev_dep = 0;
        instr_alu[4].pop_next_dep = 0;
        instr_alu[4].push_prev_dep = 0;
        instr_alu[4].push_next_dep = 1;
        instr_alu[4].iter_in = 1;
        instr_alu[4].iter_out = 1;
        instr_alu[4].reset_reg = false;
        instr_alu[4].dst_factor_out = 0;
        instr_alu[4].src_factor_out = 0;
        instr_alu[4].dst_factor_in = 0;
        instr_alu[4].src_factor_in = 0;

        instr_mem[5].opcode = VTA_OPCODE_STORE;
        instr_mem[5].memory_type = VTA_MEM_ID_OUT;
        instr_mem[5].sram_base = 0;
        instr_mem[5].dram_base = VTAMemGetPhyAddr(sram_obuf) / VTA_OUT_ELEM_BYTES;
        instr_mem[5].pop_prev_dep = 1;
        instr_mem[5].push_prev_dep = 1;
        instr_mem[5].pop_next_dep = 0;
        instr_mem[5].push_next_dep = 0;
        instr_mem[5].x_stride = 0;
        instr_mem[5].y_pad_0 = 0;
        instr_mem[5].y_pad_1 = 0;
        instr_mem[5].x_pad_0 = 0;
        instr_mem[5].x_pad_1 = 0;

        instr_gem[6].opcode = VTA_OPCODE_FINISH;
        instr_gem[6].pop_prev_dep = 0;
        instr_gem[6].pop_next_dep = 1;
        instr_gem[6].push_prev_dep = 0;
        instr_gem[6].push_next_dep = 0;

        std::cerr << "Run" << std::endl;

        VTADeviceRun(device_handle, VTAMemGetPhyAddr(instr_buf), num_instr, 1000);
        for (int i = 0; i < in_H * w_W; ++i) {
            out_buf[i] = static_cast<float>(sram_obuf[i]);
            std::cerr << i << ": " << sram_obuf[i] << std::endl;
        }

        VTAMemFree(mem_buf);
        VTAMemFree(sram_out);
        VTAMemFree(wgt_buf);
        VTAMemFree(instr_buf);
        VTADeviceFree(device_handle);
    }
}
}
}