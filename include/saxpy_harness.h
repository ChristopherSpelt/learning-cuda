#pragma once

#include "bench.h"
#include "cuda_utils.cuh"
#include "kernels/saxpy.h"
#include "numerics.h"
#include "saxpy_types.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <iomanip>
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

    std::cout << "[reference: cuBLAS saxpy, n=" << n_ << "]\n\n";
  }

  SaxpyHarness(const SaxpyHarness &) = delete;
  SaxpyHarness &operator=(const SaxpyHarness &) = delete;

  void run(const SaxpyKernel &kernel) {
    auto *xd_ptr = thrust::raw_pointer_cast(xd_.data());
    auto *yd_ptr = thrust::raw_pointer_cast(yd_.data());

    // Correctness: full αx + y against the cuBLAS reference.
    thrust::copy(y_init_.begin(), y_init_.end(), yd_.begin());

    SaxpyArgs check_args{
        .n = n_,
        .alpha = kAlpha,
        .x = xd_ptr,
        .y = yd_ptr,
    };

    kernel.launch(check_args);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::copy(yd_.begin(), yd_.end(), y_scratch_.begin());

    const float rmse = numerics::relative_rmse(y_scratch_, y_ref_);
    if (rmse > kTolerance) {
      std::cout << std::left << std::setw(22) << kernel.name << std::right << "  rel RMSE "
                << std::scientific << std::setprecision(2) << rmse << "  FAILED\n";
      return;
    }

    auto launch = [&] { kernel.launch(check_args); };
    const auto metric = bench::gbs(bench::saxpy_bytes(n_));
    const auto cold = bench::run_cold(launch, metric);
    const auto batch = bench::run_batch(launch, metric);

    std::cout << std::left << std::setw(22) << kernel.name << std::right << "  rel RMSE "
              << std::scientific << std::setprecision(2) << rmse << "  " << std::fixed
              << "cold " << std::setprecision(3) << std::setw(8) << cold.best_ms << " ms  "
              << std::setprecision(2) << std::setw(7) << cold.rate << " " << cold.units
              << "  batch  " << std::setprecision(3) << std::setw(8) << batch.best_ms << " ms  "
              << std::endl;
  }

private:
  static constexpr float kAlpha = 0.5f;
  static constexpr float kTolerance = 1e-3f;

  int n_;
  std::vector<float> x_, y_init_, y_ref_, y_scratch_;
  thrust::device_vector<float> xd_, yd_;
};

} // namespace cul
