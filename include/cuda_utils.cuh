#pragma once

#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <type_traits>

namespace cul::cuda_utils {

[[noreturn]] inline void fail(cudaError_t code, const char *file, int line) {
  std::cerr << "CUDA error " << cudaGetErrorName(code) << ": " << cudaGetErrorString(code) << " at "
            << file << ":" << line << "\n";
  std::exit(EXIT_FAILURE);
}

inline void check(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess)
    fail(code, file, line);
}

[[noreturn]] inline void require_failed(const char *cond, const char *msg, const char *file,
                                        int line) {
  std::cerr << "Requirement failed: " << cond << " (" << msg << ") at " << file << ":" << line
            << "\n";
  std::exit(EXIT_FAILURE);
}

template <typename A, typename B> constexpr auto ceil_div(A a, B b) { return (a + b - 1) / b; }

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

#define CUL_REQUIRE(cond, msg)                                                                     \
  do {                                                                                             \
    if (!(cond)) {                                                                                 \
      ::cul::cuda_utils::require_failed(#cond, msg, __FILE__, __LINE__);                           \
    }                                                                                              \
  } while (0)

#define CUDA_CHECK_LAUNCH()                                                                        \
  do {                                                                                             \
    ::cul::cuda_utils::check(cudaGetLastError(), __FILE__, __LINE__);                              \
  } while (0)
