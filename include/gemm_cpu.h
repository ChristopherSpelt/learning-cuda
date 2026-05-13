#pragma once

#include <cstdint>

void gemm_cpu(std::uint32_t M, std::uint32_t N, std::uint32_t K, float alpha, float* A, float* B, float beta, float* C);
