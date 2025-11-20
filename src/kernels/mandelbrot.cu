#include "cuda_utils.cuh"
#include <cstdint>

__global__ void mandelbrot(uint32_t img_size, uint32_t max_iters,
                           uint32_t *out) {

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= img_size || col >= img_size)
    return;

  float cy = (float(row) / float(img_size)) * 2.5f - 1.25f;
  float cx = (float(col) / float(img_size)) * 2.5f - 2.0f;

  float x2 = 0.0f;
  float y2 = 0.0f;
  float w = 0.0f;
  uint32_t iters = 0;
  while (x2 + y2 <= 4.0f && iters < max_iters) {
    float x = x2 - y2 + cx;
    float y = w - (x2 + y2) + cy;
    x2 = x * x;
    y2 = y * y;
    w = (x + y) * (x + y);
    ++iters;
  }
  out[row * img_size + col] = iters;
}

void launch_mandelbrot(uint32_t img_size, uint32_t max_iters, uint32_t *out) {

  int tile_size = 32;
  dim3 grid_dim(CEIL_DIV(img_size, tile_size), CEIL_DIV(img_size, tile_size));
  dim3 block_dim(tile_size, tile_size);

  mandelbrot<<<grid_dim, block_dim>>>(img_size, max_iters, out);
}
