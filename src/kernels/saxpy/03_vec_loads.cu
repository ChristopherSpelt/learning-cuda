#include "cuda_utils.cuh"
#include "kernels/saxpy.h"

namespace cul {
namespace {
// clang-format off
// Work:        Per element: 2 loads, 1 FMA, 1 store. Memory transactions are at
//              float4 granularity (4 elements per load/store).
// Coverage:    Main loop visits float4 indices {i : i ≡ global_idx (mod total_threads), i < n / 4},
//              each i spanning 4 consecutive elements [4i, 4i+3].
//              Tail (n % 4 residual elements) is handled by the first tail_size threads of block 0.
//              gridDim is hardware-derived, independent of n.
// clang-format on
__global__ void vec_loads_kernel(int n, float alpha, const float *__restrict__ x,
                                 float *__restrict__ y) {

  const int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_threads = gridDim.x * blockDim.x;

  for (int i = global_idx; i < n / 4; i += total_threads) {
    float4 x_vec = reinterpret_cast<const float4 *>(x)[i];
    float4 y_vec = reinterpret_cast<float4 *>(y)[i];
    y_vec.x = alpha * x_vec.x + y_vec.x;
    y_vec.y = alpha * x_vec.y + y_vec.y;
    y_vec.z = alpha * x_vec.z + y_vec.z;
    y_vec.w = alpha * x_vec.w + y_vec.w;
    reinterpret_cast<float4 *>(y)[i] = y_vec;
  }

  const int tail_start = (n / 4) * 4;
  const int tail_size = n - tail_start;
  if (blockIdx.x == 0 && threadIdx.x < tail_size) {
    const int i = tail_start + threadIdx.x;
    y[i] = alpha * x[i] + y[i];
  }
}
} // namespace

void kernels::saxpy::vec_loads(const SaxpyArgs &a) {
  constexpr int BLOCKSIZE = 256;
  constexpr int SM_COUNT = 82;
  constexpr int BLOCKS_PER_SM = 1536 / BLOCKSIZE;

  dim3 block(BLOCKSIZE);
  dim3 grid(SM_COUNT * BLOCKS_PER_SM * 2);

  vec_loads_kernel<<<grid, block>>>(a.n, a.alpha, a.x, a.y);
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
