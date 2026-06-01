#pragma once

#include "bench.h"
#include "cuda_utils.cuh"
#include "kernels/saxpy.h"
#include "numerics.h"
#include "saxpy_types.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <vector>

namespace cul {

class SaxpyHarness {
public:
  SaxpyHarness(int n)
      : n_(n), x_(numerics::random_vector(n, 1, 1)), y_init_(numerics::random_vector(n, 1, 2)),
        y_ref_(std::size_t(n)), y_scratch_(std::size_t(n)), xd_(x_.begin(), x_.end()),
        yd_(y_init_.begin(), y_init_.end()) {

    auto *xd_ptr = thrust::raw_pointer_cast(xd_.data());
    auto *yd_ptr = thrust::raw_pointer_cast(yd_.data());

    SaxpyArgs ref_args{
        .n = n_,
        .alpha = kAlpha,
        .x = xd_ptr,
        .y = yd_ptr,
    };

    kernels::saxpy::cublas(ref_args);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::copy(yd_.begin(), yd_.end(), y_ref_.begin());

    std::cerr << "[reference: cuBLAS saxpy, n=" << n_ << "]\n\n";
  }

  SaxpyHarness(const SaxpyHarness &) = delete;
  SaxpyHarness &operator=(const SaxpyHarness &) = delete;

  bench::Record run(const SaxpyKernel &kernel) {
    auto *xd_ptr = thrust::raw_pointer_cast(xd_.data());
    auto *yd_ptr = thrust::raw_pointer_cast(yd_.data());

    // Correctness: full αx + y against the cuBLAS reference.
    thrust::copy(y_init_.begin(), y_init_.end(), yd_.begin());

    SaxpyArgs args{
        .n = n_,
        .alpha = kAlpha,
        .x = xd_ptr,
        .y = yd_ptr,
    };

    kernel.launch(args);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::copy(yd_.begin(), yd_.end(), y_scratch_.begin());

    const float rel_err = numerics::relative_rmse(y_scratch_, y_ref_);

    auto launch = [&] { kernel.launch(args); };

    return bench::evaluate(kernel.name, n_, rel_err, kTolerance, launch,
                           bench::gbs(bench::saxpy_bytes(n_)));
  }

private:
  static constexpr float kAlpha = 0.5f;
  static constexpr float kTolerance = 1e-3f;

  int n_;
  std::vector<float> x_, y_init_, y_ref_, y_scratch_;
  thrust::device_vector<float> xd_, yd_;
};

} // namespace cul
