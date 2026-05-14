#include "gemm_cpu.h"

void gemm_cpu(std::uint32_t M, std::uint32_t N, std::uint32_t K, float alpha,
              const float *A, const float *B, float beta, float *C) {
  for (std::uint32_t i = 0; i < M; ++i) {
    for (std::uint32_t j = 0; j < N; ++j) {
      C[i * N + j] *= beta;
    }
  }

  for (std::uint32_t i = 0; i < M; ++i) {
    for (std::uint32_t k = 0; k < K; ++k) {
      const float a = alpha * A[i * K + k];
      for (std::uint32_t j = 0; j < N; ++j) {
        C[i * N + j] += a * B[k * N + j];
      }
    }
  }
}
