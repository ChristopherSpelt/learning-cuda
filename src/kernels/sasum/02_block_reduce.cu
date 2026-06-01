#include "cuda_utils.cuh"
#include "kernels/sasum.h"

namespace cul {
namespace {
template <int BLOCKSIZE>
__global__ void block_reduce_kernel(int n, const float *__restrict__ x,
                                    float *__restrict__ result) {

  static_assert(BLOCKSIZE > 0 && ((BLOCKSIZE & (BLOCKSIZE - 1)) == 0),
                "BLOCKSIZE must be a power of 2");

  // ---- Prologue: thread coordinates, shared-memory cooperative load --------
  const int thread_idx = threadIdx.x;
  const int global_idx = BLOCKSIZE * blockIdx.x + thread_idx;

  __shared__ float partial_sum_s[BLOCKSIZE];

  partial_sum_s[thread_idx] = global_idx < n ? fabsf(x[global_idx]) : 0.0f;
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

void kernels::sasum::block_reduce(const SasumArgs &a) {

  // Clear any garbage value that still possibly sits in result.
  CUDA_CHECK(cudaMemset(a.result, 0, sizeof(float)));

  // 256 = 100% occupancy on cc 8.6: 1536 threads/SM / 256 = 6 blocks/SM (1024 caps at 1 block = 66.7%).
  constexpr int BLOCKSIZE = 256;

  dim3 block(BLOCKSIZE);
  dim3 grid(cuda_utils::ceil_div(a.n, BLOCKSIZE));

  block_reduce_kernel<BLOCKSIZE><<<grid, block>>>(a.n, a.x, a.result);
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
