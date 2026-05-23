#pragma once

#include "bench.h"
#include "cuda_utils.cuh"
#include "gemm_cublas.h"
#include "gemm_types.h"
#include "kernels.h"
#include "numerics.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <iomanip>
#include <iostream>
#include <vector>

namespace cul {

class GemmHarness {
public:
  GemmHarness(int M, int N, int K)
      : M_(M), N_(N), K_(K), A_(numerics::random_matrix(M, K, 0)),
        B_(numerics::random_matrix(K, N, 1)),
        C_init_(numerics::random_matrix(M, N, 2)), C_ref_(std::size_t(M) * N),
        C_scratch_(std::size_t(M) * N), Ad_(A_.begin(), A_.end()),
        Bd_(B_.begin(), B_.end()), Cd_(C_init_.begin(), C_init_.end()) {

    auto *Ad_ptr = thrust::raw_pointer_cast(Ad_.data());
    auto *Bd_ptr = thrust::raw_pointer_cast(Bd_.data());
    auto *Cd_ptr = thrust::raw_pointer_cast(Cd_.data());

    GemmArgs ref_args{
        .M = M_,
        .N = N_,
        .K = K_,
        .alpha = kAlpha,
        .beta = kBeta,
        .A = Ad_ptr,
        .B = Bd_ptr,
        .C = Cd_ptr,
    };

    kernels::cublas_pedantic(ref_args);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::copy(Cd_.begin(), Cd_.end(), C_ref_.begin());

    std::cout << "[reference: cuBLAS pedantic, " << M_ << " x " << N_ << " x "
              << K_ << "]\n\n";
  }

  GemmHarness(const GemmHarness &) = delete;
  GemmHarness &operator=(const GemmHarness &) = delete;

  void run(const GemmKernel &kernel) {
    auto *Ad_ptr = thrust::raw_pointer_cast(Ad_.data());
    auto *Bd_ptr = thrust::raw_pointer_cast(Bd_.data());
    auto *Cd_ptr = thrust::raw_pointer_cast(Cd_.data());

    // Correctness: full αAB + βC against the cuBLAS reference.
    thrust::copy(C_init_.begin(), C_init_.end(), Cd_.begin());

    GemmArgs check_args{
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

    const float rmse = numerics::relative_rmse(C_scratch_, C_ref_);
    if (rmse > kTolerance) {
      std::cout << std::left << std::setw(22) << kernel.name << std::right
                << "  rel RMSE " << std::scientific << std::setprecision(2)
                << rmse << "  FAILED\n";
      return;
    }

    // Benchmark with β=0: each launch overwrites C, no contraction map.
    // FLOP count is identical.
    GemmArgs bench_args = check_args;
    bench_args.beta = 0.0f;
    auto launch = [&] { kernel.launch(bench_args); };
    const auto result = bench::run(launch, bench::gemm_flops(M_, N_, K_));

    std::cout << std::left << std::setw(22) << kernel.name << std::right
              << "  rel RMSE " << std::scientific << std::setprecision(2)
              << rmse << "  " << std::fixed << std::setprecision(3)
              << std::setw(8) << result.best_ms << " ms  "
              << std::setprecision(2) << std::setw(7) << result.tflops
              << " TFLOPS\n";
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
