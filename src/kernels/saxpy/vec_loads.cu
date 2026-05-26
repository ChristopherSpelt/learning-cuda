#include "cuda_utils.cuh"
#include "kernels/saxpy.h"

namespace cul {
namespace {
__global__ void vec_loads_kernel(int n, float alpha, const float *__restrict__ x,
                                 float *__restrict__ y) {

  // assume n % 4 == 0

  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (idx < n / 4) {
    float4 x_vec = reinterpret_cast<const float4 *>(x)[idx];
    float4 y_vec = reinterpret_cast<float4 *>(y)[idx];
    y_vec.x = alpha * x_vec.x + y_vec.x;
    y_vec.y = alpha * x_vec.y + y_vec.y;
    y_vec.z = alpha * x_vec.z + y_vec.z;
    y_vec.w = alpha * x_vec.w + y_vec.w;
    reinterpret_cast<float4 *>(y)[idx] = y_vec;
  }
}
} // namespace

void kernels::saxpy::vec_loads(const SaxpyArgs &a) {
  constexpr int NUM_THREADS = 1024;

  dim3 block(NUM_THREADS / 4);
  dim3 grid(cuda_utils::ceil_div(a.n, NUM_THREADS));

  vec_loads_kernel<<<grid, block>>>(a.n, a.alpha, a.x, a.y);
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
