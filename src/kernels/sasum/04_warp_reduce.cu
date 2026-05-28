#include "cuda_utils.cuh"
#include "kernels/sasum.h"

namespace cul {
namespace {
template <int BLOCKSIZE>
__global__ void warp_reduce_kernel(int n, const float *__restrict__ x, float *__restrict__ result) {

  static_assert(BLOCKSIZE > 0 && ((BLOCKSIZE & (BLOCKSIZE - 1)) == 0),
                "BLOCKSIZE must be a power of 2");

  // ---- Prologue: thread coordinates, shared-memory cooperative load --------
  const int thread_idx = threadIdx.x;
  const int global_idx = BLOCKSIZE * blockIdx.x + thread_idx;
  const int total_threads = BLOCKSIZE * gridDim.x;
  const int warp_id = threadIdx.x / warpSize;
  const int lane_id = threadIdx.x % warpSize;

  __shared__ float partial_sum_s[BLOCKSIZE / 32];

  float sum = 0.0f;
  for (int i = global_idx; i < n; i += total_threads) {
    sum += fabsf(x[i]);
  }

  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
  }
  if (lane_id == 0) {
    partial_sum_s[warp_id] = sum;
  }
  __syncthreads();

  if (warp_id == 0) {
    sum = (lane_id < BLOCKSIZE / warpSize) ? partial_sum_s[lane_id] : 0.0f;

    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
      sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }

    // ---- Epilogue: one atomicAdd per block to the global scalar --------------
    if (thread_idx == 0) {
      atomicAdd(result, sum);
    }
  }
}
} // namespace

void kernels::sasum::warp_reduce(const SasumArgs &a) {

  // Clear any garbage value that still possibly sits in result.
  CUDA_CHECK(cudaMemset(a.result, 0, sizeof(float)));

  constexpr int BLOCKSIZE = 256;
  constexpr int SM_COUNT = 82;
  constexpr int BLOCKS_PER_SM = 1536 / BLOCKSIZE;

  dim3 block(BLOCKSIZE);
  dim3 grid(SM_COUNT * BLOCKS_PER_SM * 2);

  warp_reduce_kernel<BLOCKSIZE><<<grid, block>>>(a.n, a.x, a.result);
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
