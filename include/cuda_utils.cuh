#pragma once

#include <cstdlib>
#include <cuda_runtime.h>
#include <format>
#include <iostream>

inline void cuda_check(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << std::format("CUDA error {}: {} at {}:{}\n",
                            cudaGetErrorName(code), cudaGetErrorString(code),
                            file, line);
    exit(EXIT_FAILURE);
  }
}

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    cuda_check((x), __FILE__, __LINE__);                                       \
  } while (0)

#define CUDA_CHECK_LAUNCH()                                                    \
  do {                                                                         \
    cuda_check(cudaGetLastError(), __FILE__, __LINE__);                        \
  } while (0)

template <typename A, typename B> constexpr auto ceil_div(A a, B b) {
  return (a + b - 1) / b;
}
