#pragma once

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>

namespace cul::cuda_utils {

inline void check(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    std::cerr << "CUDA error " << cudaGetErrorName(code) << ": "
              << cudaGetErrorString(code) << " at " << file << ":" << line
              << "\n";
    exit(EXIT_FAILURE);
  }
}

template <typename A, typename B> constexpr auto ceil_div(A a, B b) {
  return (a + b - 1) / b;
}

template <typename F> inline void dispatch_bool(bool b, F &&f) {
  if (b)
    f(std::true_type{});
  else
    f(std::false_type{});
}
} // namespace cul::cuda_utils

#define CUDA_CHECK(x)                                                                              \
  do {                                                                                             \
    ::cul::cuda_utils::check((x), __FILE__, __LINE__);                                             \
  } while (0)

#define CUDA_CHECK_LAUNCH()                                                                        \
  do {                                                                                             \
    ::cul::cuda_utils::check(cudaGetLastError(), __FILE__, __LINE__);                              \
  } while (0)
