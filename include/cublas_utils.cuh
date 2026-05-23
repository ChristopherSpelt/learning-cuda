#pragma once

#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>

namespace cul::cublas_utils {
inline void check(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error " << cublasGetStatusName(status) << ": "
              << cublasGetStatusString(status) << " at " << file << ":" << line
              << "\n";
    exit(EXIT_FAILURE);
  }
}
} // namespace cul::cublas_utils

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    ::cul::cublas_utils::check((x), __FILE__, __LINE__);                            \
  } while (0)
