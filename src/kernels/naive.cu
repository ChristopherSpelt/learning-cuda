#include "cuda_utils.cuh"
#include "kernels.h"

template <std::uint32_t BLOCKSIZE>
__global__ void naive_kernel(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                             float alpha, const float *__restrict__ A,
                             const float *__restrict__ B, float beta,
                             float* __restrict__ C) {

  const std::uint32_t row = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const std::uint32_t col = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE); 

  if (row >= M || col >= N)
    return;

  float sum = 0.0f;
  for (std::uint32_t k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
  }

  C[row * N + col] = alpha * sum + beta * C[row * N + col];
}

void kernels::naive(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                    float alpha, const float *A, const float *B, float beta,
                    float *C) {
  constexpr int BLOCKSIZE = 32;

  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid(ceil_div(N, BLOCKSIZE), ceil_div(M, BLOCKSIZE));

  naive_kernel<BLOCKSIZE><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK_LAUNCH();
}
