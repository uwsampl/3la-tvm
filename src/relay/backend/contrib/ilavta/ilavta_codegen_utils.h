#ifndef ILAVTA_CODEGEN_UTILS_H__
#define ILAVTA_CODEGEN_UTILS_H__
#include <vta/hw_spec.h>
#include <vta/driver.h>
#include <tvm/support/json.hpp>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>

namespace tvm {
namespace relay {
namespace contrib {

std::string CompileGEMM(int batch, size_t n_inp_cols, size_t n_wgt_rows, int factor, int nbits, std::string filename);
std::string CompilBiasAdd(int batch, size_t n_feat, std::string filename);
std::string CompileRelu(int batch, size_t n_feat, std::string filename);
std::string GetCompiledFilename(const std::string op_name, const int* input_info, const int num_info);
std::vector<int> approximate_scale(double x);

}
}
}

#endif  // ILAVTA_CODEGEN_UTILS_H__