#include "cuda_utils.cuh"
#include "kernels/asum.h"

namespace cul {
namespace {
template <int BLOCKSIZE>
__global__ void grid_stride_kernel(int n, const float *__restrict__ x, float *__restrict__ result) {

  static_assert(BLOCKSIZE > 0 && ((BLOCKSIZE & (BLOCKSIZE - 1)) == 0),
                "BLOCKSIZE must be a power of 2");

  // ---- Prologue: thread coordinates, shared-memory cooperative load --------
  const int thread_idx = threadIdx.x;
  const int global_idx = BLOCKSIZE * blockIdx.x + thread_idx;
  const int total_threads = BLOCKSIZE * gridDim.x;

  __shared__ float partial_sum_s[BLOCKSIZE];

  float sum = 0.0f;
  for (int i = global_idx; i < n; i += total_threads) {
    sum += fabsf(x[i]);
  }
  partial_sum_s[thread_idx] = sum;
  __syncthreads();

  // ---- Main loop: in-block tree reduction ----------------------------------
  for (int stride = BLOCKSIZE / 2; stride > 0; stride >>= 1) {
    if (thread_idx < stride) {
      partial_sum_s[thread_idx] += partial_sum_s[thread_idx + stride];
    }
    __syncthreads();
  }

  // ---- Epilogue: one atomicAdd per block to the global scalar --------------
  if (thread_idx == 0) {
    atomicAdd(result, partial_sum_s[0]);
  }
}
} // namespace

void kernels::asum::grid_stride(const AsumArgs &a) {

  // Clear any garbage value that still possibly sits in result.
  CUDA_CHECK(cudaMemset(a.result, 0, sizeof(float)));

  constexpr int BLOCKSIZE = 256;
  constexpr int SM_COUNT = 82;
  constexpr int BLOCKS_PER_SM = 1536 / BLOCKSIZE;

  dim3 block(BLOCKSIZE);
  dim3 grid(SM_COUNT * BLOCKS_PER_SM * 2);

  grid_stride_kernel<BLOCKSIZE><<<grid, block>>>(a.n, a.x, a.result);
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
