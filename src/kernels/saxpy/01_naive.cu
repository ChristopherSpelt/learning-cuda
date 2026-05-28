#include "cuda_utils.cuh"
#include "kernels/saxpy.h"

namespace cul {
namespace {
// clang-format off
// Work:        Per element: 2 loads, 1 FMA, 1 store.
// Coverage:    Each thread computes exactly one element of y. For this we need
//              gridDim.x * BLOCKSIZE >= n; our gridDim.x depends on the input size n.
// clang-format on
__global__ void naive_kernel(int n, float alpha, const float *__restrict__ x,
                             float *__restrict__ y) {

  const int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n)
    y[idx] = alpha * x[idx] + y[idx];
}
} // namespace

void kernels::saxpy::naive(const SaxpyArgs &a) {
  constexpr int BLOCKSIZE = 1024;

  dim3 block(BLOCKSIZE);
  dim3 grid(cuda_utils::ceil_div(a.n, BLOCKSIZE));

  naive_kernel<<<grid, block>>>(a.n, a.alpha, a.x, a.y);
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
