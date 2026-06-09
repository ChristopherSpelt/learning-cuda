#include "cuda_utils.cuh"
#include "kernels/sasum.h"

namespace cul {
namespace {
__global__ void naive_kernel(int n, const float *__restrict__ x, float *__restrict__ result) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n)
    // Expected to FAIL for n >~ 2^24: the single-float accumulator's half-ulp
    // grows past typical |x|, so most addends are absorbed (rounded away).
    atomicAdd(result, fabsf(x[idx]));
}
} // namespace

void kernels::sasum::naive(const SasumArgs &a) {
  CUDA_CHECK(cudaMemset(a.result, 0, sizeof(float)));

  constexpr int BLOCKSIZE = 256;

  dim3 block(BLOCKSIZE);
  dim3 grid(cuda_utils::ceil_div(a.n, BLOCKSIZE));

  naive_kernel<<<grid, block>>>(a.n, a.x, a.result);
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
