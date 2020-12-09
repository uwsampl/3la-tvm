
#ifndef TVM_RUNTIME_CONTRIB_VTA_MATMUL_VTA_MATMUL_RUNTIME_H_
#define TVM_RUNTIME_CONTRIB_VTA_MATMUL_VTA_MATMUL_RUNTIME_H_
#include <vta/driver.h>
#include <vta/hw_spec.h>
#include <tvm/runtime/ndarray.h>

namespace tvm {
namespace runtime {
namespace contrib {

    extern "C" TVM_DLL void run_vta_simulator(float *acc, float *weight, int in_H, int in_W,
                        int w_W, float *out_buf);

}  //  namespace contrib
}  //  namespace runtime
}  //  namespace tvm

#endif  //  TVM_RUNTIME_CONTRIB_VAT_MATMUL_VTA_MATMUL_RUNTIME_H_
