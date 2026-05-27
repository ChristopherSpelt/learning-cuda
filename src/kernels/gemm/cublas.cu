#include "cublas_utils.cuh"
#include "kernels/gemm.h"

#include <cublas_v2.h>

namespace cul {
namespace {

void cublas_sgemm_row_major(cublasMath_t mode, int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  auto h = cublas_utils::handle();
  CUBLAS_CHECK(cublasSetMathMode(h, mode));
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

void cublas_sgemm_ex_row_major(cublasComputeType_t compute_type, int M, int N, int K, float alpha,
                               const float *A, const float *B, float beta, float *C) {

  auto h = cublas_utils::handle();
  CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_32F, N, A,
                            CUDA_R_32F, K, &beta, C, CUDA_R_32F, N, compute_type,
                            CUBLAS_GEMM_DEFAULT));
}

} // namespace

void kernels::gemm::cublas_pedantic(const GemmArgs &a) {
  cublas_sgemm_row_major(CUBLAS_PEDANTIC_MATH, a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
}

void kernels::gemm::cublas_default(const GemmArgs &a) {
  cublas_sgemm_row_major(CUBLAS_DEFAULT_MATH, a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
}

void kernels::gemm::cublas_tf32(const GemmArgs &a) {
  cublas_sgemm_ex_row_major(CUBLAS_COMPUTE_32F_FAST_TF32, a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta,
                            a.C);
}
} // namespace cul
