#include "cuda_utils.cuh"
#include "mandelbrot.cuh"
#include "io.h"

#include <cstdlib>
#include <vector>

void compute_mandelbrot(uint32_t img_size, uint32_t max_iters,
                        std::string_view filename) {
  // Allocate on host
  std::vector<uint32_t> result(img_size * img_size);
  uint32_t *resultd;

  // Allocate on device
  CUDA_CHECK(
      cudaMalloc((void **)&resultd, sizeof(uint32_t) * img_size * img_size));

  launch_mandelbrot(img_size, max_iters, resultd);

  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy device to host
  CUDA_CHECK(cudaMemcpy(result.data(), resultd,
                        sizeof(uint32_t) * img_size * img_size,
                        cudaMemcpyDeviceToHost));

  // Write output to binary file
  write_binary(filename, result.data(), img_size);

  CUDA_CHECK(cudaFree(resultd));
}

int main() {
  compute_mandelbrot(1024, 500, "gpu_mandelbrot.bin");

  return EXIT_SUCCESS;
}
