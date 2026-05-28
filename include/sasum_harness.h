#pragma once

#include "sasum_types.h"
#include "bench.h"
#include "cuda_utils.cuh"
#include "kernels/sasum.h"
#include "numerics.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

namespace cul {

class SasumHarness {
public:
  SasumHarness(int n)
      : n_(n), x_(numerics::random_vector(n, 1, 1)), result_ref_(0), xd_(x_.begin(), x_.end()),
        resultd_(1) {

    auto *xd_ptr = thrust::raw_pointer_cast(xd_.data());
    auto *resultd_ptr = thrust::raw_pointer_cast(resultd_.data());

    SasumArgs ref_args{
        .n = n_,
        .x = xd_ptr,
        .result = resultd_ptr,
    };

    kernels::sasum::cublas(ref_args);
    CUDA_CHECK(cudaDeviceSynchronize());
    result_ref_ = resultd_[0];

    std::cout << "[reference: cuBLAS sasum, n=" << n_ << "]\n\n";
  }

  SasumHarness(const SasumHarness &) = delete;
  SasumHarness &operator=(const SasumHarness &) = delete;

  void run(const SasumKernel &kernel) {
    auto *xd_ptr = thrust::raw_pointer_cast(xd_.data());
    auto *resultd_ptr = thrust::raw_pointer_cast(resultd_.data());

    SasumArgs check_args{
        .n = n_,
        .x = xd_ptr,
        .result = resultd_ptr,
    };

    // Correctness
    kernel.launch(check_args);
    CUDA_CHECK(cudaDeviceSynchronize());

    const float reldiff = std::fabs(resultd_[0] - result_ref_) / std::fabs(result_ref_);
    if (reldiff > kTolerance) {
      std::cout << std::left << std::setw(22) << kernel.name << std::right << "  rel diff "
                << std::scientific << std::setprecision(2) << reldiff << "  FAILED\n";
      return;
    }

    auto launch = [&] { kernel.launch(check_args); };
    const auto result = bench::run(launch, bench::gbs(bench::sasum_bytes(n_)));

    std::cout << std::left << std::setw(22) << kernel.name << std::right << "  rel diff "
              << std::scientific << std::setprecision(2) << reldiff << "  " << std::fixed
              << std::setprecision(3) << std::setw(8) << result.best_ms << " ms  "
              << std::setprecision(2) << std::setw(7) << result.rate << " " << result.units
              << std::endl;
  }

private:
  static constexpr float kTolerance = 1e-3f;

  int n_;
  std::vector<float> x_;
  float result_ref_;
  thrust::device_vector<float> xd_, resultd_;
};

} // namespace cul
