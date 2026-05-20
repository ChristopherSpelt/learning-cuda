#include "cublas_utils.cuh"
#include "gemm_cublas.h"

#include <cublas_v2.h>

namespace {
class CublasHandle {
public:
  CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }
  ~CublasHandle() noexcept { cublasDestroy(handle_); }
  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;

  cublasHandle_t get() const { return handle_; }

private:
  cublasHandle_t handle_{};
};

cublasHandle_t cublas_handle() {
  static CublasHandle h;
  return h.get();
}

void cublas_sgemm_row_major(cublasMath_t mode, int M, int N,
                            int K, float alpha, const float *A,
                            const float *B, float beta, float *C) {
  auto h = cublas_handle();
  CUBLAS_CHECK(cublasSetMathMode(h, mode));
  CUBLAS_CHECK(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A,
                           K, &beta, C, N));
}

void cublas_sgemm_ex_row_major(cublasComputeType_t compute_type,
                               int M, int N,
                               int K, float alpha, const float *A,
                               const float *B, float beta, float *C) {

  auto h = cublas_handle();
  CUBLAS_CHECK(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH));

  CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B,
                            CUDA_R_32F, N, A, CUDA_R_32F, K, &beta, C,
                            CUDA_R_32F, N, compute_type, CUBLAS_GEMM_DEFAULT));
}

} // namespace

void kernels::cublas_pedantic(const GemmArgs &a) {
  cublas_sgemm_row_major(CUBLAS_PEDANTIC_MATH, a.M, a.N, a.K, a.alpha, a.A, a.B,
                         a.beta, a.C);
}

void kernels::cublas_default(const GemmArgs &a) {
  cublas_sgemm_row_major(CUBLAS_DEFAULT_MATH, a.M, a.N, a.K, a.alpha, a.A, a.B,
                         a.beta, a.C);
}

void kernels::cublas_tf32(const GemmArgs &a) {
  cublas_sgemm_ex_row_major(CUBLAS_COMPUTE_32F_FAST_TF32, a.M, a.N, a.K,
                            a.alpha, a.A, a.B, a.beta, a.C);
}
