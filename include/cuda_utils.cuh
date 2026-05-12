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

template<typename A, typename B>
constexpr auto ceil_div(A a, B b) {
    return (a + b - 1) / b;
}
