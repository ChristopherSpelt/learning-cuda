#pragma once

#include <cstdint>
#include <string_view>

using GemmLaunch = void (*)(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                            float alpha, const float *A, const float *B,
                            float beta, float *C);

struct GemmKernel {
  std::string_view name;
  GemmLaunch launch;
};

namespace kernels {

void naive(std::uint32_t M, std::uint32_t N, std::uint32_t K, float alpha,
           const float *A, const float *B, float beta, float *C);

void shared_mem(std::uint32_t M, std::uint32_t N, std::uint32_t K, float alpha,
                const float *A, const float *B, float beta, float *C);

void shared_mem_1d_block(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                         float alpha, const float *A, const float *B,
                         float beta, float *C);

void shared_mem_2d_block(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                         float alpha, const float *A, const float *B,
                         float beta, float *C);

void shared_mem_vec(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                    float alpha, const float *A, const float *B, float beta,
                    float *C);

void shared_mem_vec_warp(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                         float alpha, const float *A, const float *B,
                         float beta, float *C);

} // namespace kernels
