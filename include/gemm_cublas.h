#pragma once
#include <cstdint>

namespace kernels {

void cublas_pedantic(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                     float alpha, const float *A, const float *B, float beta,
                     float *C);

void cublas_default(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                    float alpha, const float *A, const float *B, float beta,
                    float *C);

void cublas_tf32(std::uint32_t M, std::uint32_t N, std::uint32_t K, float alpha,
                 const float *A, const float *B, float beta, float *C);
} // namespace kernels
