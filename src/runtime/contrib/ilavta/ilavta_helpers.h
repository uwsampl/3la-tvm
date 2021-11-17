#ifndef ILAVTA_HELPERS_H__
#define ILAVTA_HELPERS_H__
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>
#include <dmlc/json.h>
#include <vta/driver.h>
#include <vta/hw_spec.h>

#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <chrono>

namespace tvm {
namespace runtime {
namespace contrib {

using ila_output_data = std::vector<std::unordered_map<std::string, std::string> >;

/*
* Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
* Given `batch`, `in_feat` (number of rows [16 bytes] used to store a row in the input matrix)
* and `out_feat` (number of `in_feat`s), return UOps that perform a dense operator over
* INPUT buffer and ACC buffer
* */
VTAUop * getGEMMUops(int batch, int in_feat, int out_feat); 

/*
 * Given `batch` and `in_feat`, return UOps that perform ADD operator
 * over each row in the ACC buffer
 * */
VTAUop * getBiasAddUops(int batch, int in_feat);

/*
 * Given `batch` and `in_feat`, return UOps that perform max(x, 0) on each row
 * in the ACC buffer.
 * */
VTAUop * getReluUops(int batch, int in_feat); 
 
/*
* Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
* */
VTAGenericInsn getGEMMInsn(int uop_offset, int batch, int in_feat, int out_feat,
                           bool uop_compression, int pop_prev_dep, int pop_next_dep,
                           int push_prev_dep, int push_next_dep); 

/*
* Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
* */
VTAGenericInsn getAluInsn(int alu_opcode, int uop_begin, int uop_end, bool use_imm, int imm,
                          int pop_prev_dep, int pop_next_dep, int push_prev_dep, int push_next_dep);

/*
* Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
* */
VTAGenericInsn getFinishInsn(bool pop_prev, bool pop_next);

/*
* Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
* */
VTAGenericInsn get1DLoadStoreInsn(int opcode, int type, int sram_offset, int dram_offset, int size,
                                  int pop_prev_dep, int pop_next_dep, int push_prev_dep,
                                  int push_next_dep); 

/*
* Code adopted from https://github.com/apache/tvm-vta/blob/main/tests/hardware/common/test_lib.cc
* */
VTAGenericInsn get2DLoadStoreInsn(int opcode, int type, int sram_offset, int dram_offset,
                                  int y_size, int x_size, int x_stride, int y_pad, int x_pad,
                                  int pop_prev_dep, int pop_next_dep, int push_prev_dep,
                                  int push_next_dep); 

// Calls the ILA simularor; The result will be stored
// in `./result`
// Users should not call this directly
std::string runILASimulator(const std::string exp_name,
                            const std::string driver_dir,
                            int64_t& compile_time_out,
                            const std::string ila_asm = "",
                            const std::string data_dump = "",
                            const bool use_trace = true);

// Read back the result produced by the ILA simulator.
// The results will be stored in `out_values`.
// This function will not throw away data at addresses that
// are not touched during computation
// Users should not call this directly
void readILAOutput(const std::string filename, ila_output_data &out_values); 

// Load the data produced by the ILA simulator to `buffer`
// Addresses that are not touched during the computation
// will be thrown away in this process
// returns actual number of bytes read
// Users should not call this directly
size_t loadILAOutput(const ila_output_data &out_values, uint8_t* buffer, size_t out_h, size_t out_w); 

// Run `pattern_name` on ILA simulator and then copy back
// data produced by the ILA simulator and store into `output_data`
// returns time spent on running the simulator
// This is the interface provided to users
int64_t runSimGetData(std::string pattern_name, std::string driver_dir, std::string ila_asm, std::string data_dump,
                   size_t output_size, int n_output_rows, int n_output_cols, void *output_data, std::string output_dtype);

// Create a data dump which could be used paired with an ILA ASM to produce
// the ILA program fragment
std::string dump_datafile(int8_t* input_buf, size_t input_size,
                   int8_t* weight_buf, size_t weight_size,
                   int32_t* acc_buf, size_t acc_size,
                   VTAUop* uop_buf, size_t uop_size,
                   std::string filename);

std::vector<int> approximate_scale(double x);
}
}
}
#endif  // ILAVTA_HELPERS_H__
