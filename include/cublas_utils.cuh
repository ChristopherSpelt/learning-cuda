#pragma once

#include <cstdlib>
#include <cublas_v2.h>
#include <format>
#include <iostream>

inline void cublas_check(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << std::format("cuBLAS error {}: {} at {}:{}\n",
                             cublasGetStatusName(status),
                             cublasGetStatusString(status), file, line);
    exit(EXIT_FAILURE);
  }
}

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    cublas_check((x), __FILE__, __LINE__);                                     \
  } while (0)
