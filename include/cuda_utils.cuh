#pragma once

#include <cuda_runtime.h>
#include <iostream>

inline void cuda_check(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA error " << cudaGetErrorName(code) << ": "
              << cudaGetErrorString(code) << " at " << file << ": " << line
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cuda_check((x), __FILE__, __LINE__);                                       \
  } while (0)

#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
