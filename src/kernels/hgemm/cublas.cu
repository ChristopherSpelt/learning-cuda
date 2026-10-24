#include "cublas_utils.cuh"
#include "kernels/hgemm.h"

#include <cublas_v2.h>

namespace cul {
namespace {

void cublas_bf16_gemm_row_major(cublasComputeType_t compute_type, int M, int N, int K, float alpha,
                                const __nv_bfloat16 *A, const __nv_bfloat16 *B, float beta,
                                float *C) {
  auto h = cublas_utils::handle();
  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, N, A,
                            CUDA_R_16BF, K, &beta, C, CUDA_R_32F, N, compute_type,
                            CUBLAS_GEMM_DEFAULT));
}

} // namespace

void kernels::hgemm::cublas_pedantic(const HgemmArgs &a) {
  cublas_bf16_gemm_row_major(CUBLAS_COMPUTE_32F_PEDANTIC, a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta,
                             a.C);
}

void kernels::hgemm::cublas_default(const HgemmArgs &a) {
  cublas_bf16_gemm_row_major(CUBLAS_COMPUTE_32F, a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
}

} // namespace cul
