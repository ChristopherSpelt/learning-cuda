#pragma once

#include <cstdint>

namespace gemm_naive {

template <int TILE>
void launch(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                       float alpha, float const *A, float const *B, float beta,
                       float *C);
} // namespace gemm_naive
