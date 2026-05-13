#include "gemm_cpu.h"

void gemm_cpu(std::uint32_t M, std::uint32_t N, std::uint32_t K, float alpha, float* A, float* B, float beta, float* C) {
    for (std::uint32_t i = 0; i < M; ++i) {
        for (std::uint32_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (std::uint32_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = alpha * sum + beta * C[i * N +j];
        }
    }
}
