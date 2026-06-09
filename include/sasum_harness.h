#pragma once

#include "bench.h"
#include "cuda_utils.cuh"
#include "kernels/sasum.h"
#include "numerics.h"
#include "sasum_types.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cmath>
#include <iostream>
#include <limits>
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

    std::cerr << "[reference: cuBLAS sasum, n=" << n_ << "]\n\n";
  }

  SasumHarness(const SasumHarness &) = delete;
  SasumHarness &operator=(const SasumHarness &) = delete;

  bench::Record run(const SasumKernel &kernel) {
    auto *xd_ptr = thrust::raw_pointer_cast(xd_.data());
    auto *resultd_ptr = thrust::raw_pointer_cast(resultd_.data());

    SasumArgs args{
        .n = n_,
        .x = xd_ptr,
        .result = resultd_ptr,
    };

    // Poison the accumulator so a no-op kernel cannot inherit the previous
    // correct value.
    resultd_[0] = std::numeric_limits<float>::quiet_NaN();

    kernel.launch(args);
    CUDA_CHECK(cudaDeviceSynchronize());
    const float rel_err = std::fabs(resultd_[0] - result_ref_) / std::fabs(result_ref_);

    auto launch = [&] { kernel.launch(args); };
    return bench::evaluate(kernel.name, n_, rel_err, kTolerance, launch,
                           bench::gbs(bench::sasum_bytes(n_)));
  }

private:
  static constexpr float kTolerance = 1e-3f;

  int n_;
  std::vector<float> x_;
  float result_ref_;
  thrust::device_vector<float> xd_, resultd_;
};

} // namespace cul
