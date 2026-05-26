#include "cuda_utils.cuh"
#include "kernels/saxpy.h"

namespace cul {
namespace {
__global__ void naive_kernel(int n, float alpha, const float *__restrict__ x,
                             float *__restrict__ y) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n)
    y[idx] = alpha * x[idx] + y[idx];
}
} // namespace

void kernels::saxpy::naive(const SaxpyArgs &a) {
  constexpr int NUM_THREADS = 1024;

  dim3 block(NUM_THREADS);
  dim3 grid(cuda_utils::ceil_div(a.n, NUM_THREADS));

  naive_kernel<<<grid, block>>>(a.n, a.alpha, a.x, a.y);
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
