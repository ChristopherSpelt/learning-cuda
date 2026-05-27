#include "cuda_utils.cuh"
#include "kernels/asum.h"

namespace cul {
namespace {
__global__ void naive_kernel(int n, const float *__restrict__ x, float *__restrict__ result) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n)
    atomicAdd(result, fabsf(x[idx]));
}
} // namespace

void kernels::asum::naive(const AsumArgs &a) {
  CUDA_CHECK(cudaMemset(a.result, 0, sizeof(float)));

  constexpr int NUM_THREADS = 1024;

  dim3 block(NUM_THREADS);
  dim3 grid(cuda_utils::ceil_div(a.n, NUM_THREADS));

  naive_kernel<<<grid, block>>>(a.n, a.x, a.result);
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
