#pragma once

#include <cstdlib>
#include <cuda_runtime.h>
#include <format>
#include <iostream>

namespace cul::cuda_utils {

inline void check(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << std::format("CUDA error {}: {} at {}:{}\n",
                             cudaGetErrorName(code), cudaGetErrorString(code),
                             file, line);
    exit(EXIT_FAILURE);
  }
}

template <typename A, typename B> constexpr auto ceil_div(A a, B b) {
  return (a + b - 1) / b;
}
} // namespace cul::cuda_utils

#define CUDA_CHECK(x)                                                          \
  do {                                                                         \
    ::cul::cuda_utils::check((x), __FILE__, __LINE__);                              \
  } while (0)

#define CUDA_CHECK_LAUNCH()                                                    \
  do {                                                                         \
    ::cul::cuda_utils::check(cudaGetLastError(), __FILE__, __LINE__);               \
  } while (0)
