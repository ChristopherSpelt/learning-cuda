#include "cuda_utils.cuh"
#include "kernels.h"

template <std::uint32_t BLOCKSIZE>
__global__ void shared_mem_kernel(std::uint32_t M, std::uint32_t N,
                                  std::uint32_t K, float alpha,
                                  const float *__restrict__ A,
                                  const float *__restrict__ B, float beta,
                                  float *__restrict__ C) {

  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  auto block_row = blockIdx.y;
  auto block_col = blockIdx.x;

  auto thread_row = threadIdx.x / BLOCKSIZE;
  auto thread_col = threadIdx.x % BLOCKSIZE;

  auto global_row = block_row * BLOCKSIZE + thread_row;
  auto global_col = block_col * BLOCKSIZE + thread_col;

  A += block_row * BLOCKSIZE * K;       // BLOCK: jump to my row stripe of A
  B += block_col * BLOCKSIZE;           // BLOCK: jump to my col stripe of B
  C += block_row * BLOCKSIZE * N +
       block_col * BLOCKSIZE;           // BLOCK: jump to my tile in C

  float sum = 0.0f;

  for (std::uint32_t tile = 0; tile < K; tile += BLOCKSIZE) {
    const std::uint32_t a_col_global = tile + thread_col;
    const std::uint32_t b_row_global = tile + thread_row;

    As[thread_row * BLOCKSIZE + thread_col] =
        (global_row < M && a_col_global < K) ? A[thread_row * K + thread_col]
                                             : 0.0f;
    Bs[thread_row * BLOCKSIZE + thread_col] =
        (global_col < N && b_row_global < K) ? B[thread_row * N + thread_col]
                                             : 0.0f;

    __syncthreads();
    A += BLOCKSIZE;     // BLOCK: advance by one tile to the right
    B += BLOCKSIZE * N; // BLOCK: advance by one tile down.

    for (std::uint32_t k = 0; k < BLOCKSIZE; ++k) {
      sum += As[thread_row * BLOCKSIZE + k] * Bs[k * BLOCKSIZE + thread_col];
    }
    __syncthreads();
  }
  C[thread_row * N + thread_col] =
      alpha * sum + beta * C[thread_row * N + thread_col];
}

void kernels::shared_mem(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                         float alpha, const float *A, const float *B,
                         float beta, float *C) {
  constexpr int BLOCKSIZE = 32;

  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid(ceil_div(N, BLOCKSIZE), ceil_div(M, BLOCKSIZE));

  shared_mem_kernel<BLOCKSIZE><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK_LAUNCH();
}
