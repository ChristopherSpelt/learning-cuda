#include "cuda_utils.cuh"
#include <cstdint>

__global__ void matmul(uint32_t size_i, uint32_t size_j, uint32_t size_k,
                       float const *a, float const *b, float *c) {}

void launch_matmul(uint32_t size_i, uint32_t size_j, uint32_t size_k,
                   float const *a, float const *b, float *c) {

  int tile_size = 32;
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(tile_size, tile_size);

  matmul<<<grid_dim, block_dim>>>(size_i, size_j, size_k, a, b, c);
}
