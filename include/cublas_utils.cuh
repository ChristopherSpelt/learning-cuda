#pragma once

#include <cstdlib>
#include <cublas_v2.h>
#include <iostream>

#define CUBLAS_CHECK(x)                                                                            \
  do {                                                                                             \
    ::cul::cublas_utils::check((x), __FILE__, __LINE__);                                           \
  } while (0)

namespace cul::cublas_utils {

inline void check(cublasStatus_t status, const char *file, int line) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::cerr << "cuBLAS error " << cublasGetStatusName(status) << ": "
              << cublasGetStatusString(status) << " at " << file << ":" << line << "\n";
    exit(EXIT_FAILURE);
  }
}

class CublasHandle {
public:
  CublasHandle() { CUBLAS_CHECK(cublasCreate(&handle_)); }
  ~CublasHandle() noexcept { cublasDestroy(handle_); }
  CublasHandle(const CublasHandle &) = delete;
  CublasHandle &operator=(const CublasHandle &) = delete;

  [[nodiscard]] cublasHandle_t get() const { return handle_; }

private:
  cublasHandle_t handle_{};
};

inline cublasHandle_t handle() {
  static CublasHandle h;
  return h.get();
}

class ScopedPointerMode {
public:
  ScopedPointerMode(cublasHandle_t h, cublasPointerMode_t mode) : h_(h) {
    CUBLAS_CHECK(cublasGetPointerMode(h_, &prev_));
    CUBLAS_CHECK(cublasSetPointerMode(h_, mode));
  }
  ~ScopedPointerMode() { cublasSetPointerMode(h_, prev_); }
  ScopedPointerMode(const ScopedPointerMode &) = delete;
  ScopedPointerMode &operator=(const ScopedPointerMode &) = delete;

private:
  cublasHandle_t h_;
  cublasPointerMode_t prev_;
};

} // namespace cul::cublas_utils
