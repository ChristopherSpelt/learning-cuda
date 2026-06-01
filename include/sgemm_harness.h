#pragma once

#include "bench.h"
#include "cuda_utils.cuh"
#include "kernels/sgemm.h"
#include "numerics.h"
#include "sgemm_types.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <limits>
#include <vector>

namespace cul {

class SgemmHarness {
public:
  SgemmHarness(int M, int N, int K)
      : M_(M), N_(N), K_(K), A_(numerics::random_vector(M, K, 0)),
        B_(numerics::random_vector(K, N, 1)), C_init_(numerics::random_vector(M, N, 2)),
        C_ref_(std::size_t(M) * N), C_scratch_(std::size_t(M) * N), Ad_(A_.begin(), A_.end()),
        Bd_(B_.begin(), B_.end()), Cd_(C_init_.begin(), C_init_.end()) {
    const auto fits = [](std::size_t a, std::size_t b) {
      return a * b <= static_cast<std::size_t>(std::numeric_limits<int>::max());
    };
    CUL_REQUIRE(fits(M, K) && fits(K, N) && fits(M, N),
                "M*K, K*N and M*N must each fit in int (kernel index math is 32-bit)");

    auto *Ad_ptr = thrust::raw_pointer_cast(Ad_.data());
    auto *Bd_ptr = thrust::raw_pointer_cast(Bd_.data());
    auto *Cd_ptr = thrust::raw_pointer_cast(Cd_.data());

    SgemmArgs ref_args{
        .M = M_,
        .N = N_,
        .K = K_,
        .alpha = kAlpha,
        .beta = kBeta,
        .A = Ad_ptr,
        .B = Bd_ptr,
        .C = Cd_ptr,
    };

    kernels::sgemm::cublas_pedantic(ref_args);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::copy(Cd_.begin(), Cd_.end(), C_ref_.begin());

    std::cerr << "[reference: cuBLAS pedantic, " << M_ << " x " << N_ << " x " << K_ << "]\n\n";
  }

  SgemmHarness(const SgemmHarness &) = delete;
  SgemmHarness &operator=(const SgemmHarness &) = delete;

  bench::Record run(const SgemmKernel &kernel) {
    auto *Ad_ptr = thrust::raw_pointer_cast(Ad_.data());
    auto *Bd_ptr = thrust::raw_pointer_cast(Bd_.data());
    auto *Cd_ptr = thrust::raw_pointer_cast(Cd_.data());

    // Correctness: full αAB + βC against the cuBLAS reference.
    thrust::copy(C_init_.begin(), C_init_.end(), Cd_.begin());

    SgemmArgs check_args{
        .M = M_,
        .N = N_,
        .K = K_,
        .alpha = kAlpha,
        .beta = kBeta,
        .A = Ad_ptr,
        .B = Bd_ptr,
        .C = Cd_ptr,
    };

    kernel.launch(check_args);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::copy(Cd_.begin(), Cd_.end(), C_scratch_.begin());

    const float rel_err = numerics::relative_rmse(C_scratch_, C_ref_);

    // Benchmark with β=0: each launch overwrites C, no contraction map.
    // FLOP count is identical.
    SgemmArgs bench_args = check_args;
    bench_args.beta = 0.0f;
    auto launch = [&] { kernel.launch(bench_args); };

    return bench::evaluate(kernel.name, M_, rel_err, kTolerance, launch, // n = M (square sweep)
                           bench::tflops(bench::sgemm_flops(M_, N_, K_)));
  }

private:
  static constexpr float kAlpha = 1.0f;
  static constexpr float kBeta = 0.5f;
  static constexpr float kTolerance = 1e-3f;

  int M_, N_, K_;
  std::vector<float> A_, B_, C_init_, C_ref_, C_scratch_;
  thrust::device_vector<float> Ad_, Bd_, Cd_;
};

} // namespace cul
