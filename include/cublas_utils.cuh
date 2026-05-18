#pragma once

#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>

inline void cublas_check(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error " << cublasGetStatusName(status) << ": "
              << cublasGetStatusString(status) << " at " << file << ": " << line
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

#define CUBLAS_CHECK(x)                                                        \
  do {                                                                         \
    cublas_check((x), __FILE__, __LINE__);                                     \
  } while (0)
