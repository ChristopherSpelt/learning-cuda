#include "cuda_utils.cuh"
#include "kernels/saxpy.h"

namespace cul {
namespace {
// clang-format off
// Work:        Per element: 2 loads, 1 FMA, 1 store.
// Coverage:    Main loop visits element indices {i : i ≡ global_idx (mod total_threads), i < n};
//              gridDim is hardware-derived, independent of n.
// clang-format on
__global__ void grid_stride_kernel(int n, float alpha, const float *__restrict__ x,
                                   float *__restrict__ y) {

  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_threads = gridDim.x * blockDim.x;

  for (int i = global_idx; i < n; i += total_threads) {
    y[i] = alpha * x[i] + y[i];
  }
}
} // namespace

void kernels::saxpy::grid_stride(const SaxpyArgs &a) {
  constexpr int BLOCKSIZE = 256;
  constexpr int OVERSUB = 2;
  static const int grid_size =
      cuda_utils::sm_count() * cuda_utils::blocks_per_sm(grid_stride_kernel, BLOCKSIZE) * OVERSUB;

  dim3 block(BLOCKSIZE);
  dim3 grid(grid_size);

  grid_stride_kernel<<<grid, block>>>(a.n, a.alpha, a.x, a.y);
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
