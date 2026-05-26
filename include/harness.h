#pragma once

#include "bench.h"
#include "cuda_utils.cuh"
#include "gemm_cublas.h"
#include "gemm_types.h"
#include "numerics.h"
#include "saxpy_cublas.h"
#include "saxpy_types.h"

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
        B_(numerics::random_matrix(K, N, 1)), C_init_(numerics::random_matrix(M, N, 2)),
        C_ref_(std::size_t(M) * N), C_scratch_(std::size_t(M) * N), Ad_(A_.begin(), A_.end()),
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

    kernels::gemm::cublas_pedantic(ref_args);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::copy(Cd_.begin(), Cd_.end(), C_ref_.begin());

    std::cout << "[reference: cuBLAS pedantic, " << M_ << " x " << N_ << " x " << K_ << "]\n\n";
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
      std::cout << std::left << std::setw(22) << kernel.name << std::right << "  rel RMSE "
                << std::scientific << std::setprecision(2) << rmse << "  FAILED\n";
      return;
    }

    // Benchmark with β=0: each launch overwrites C, no contraction map.
    // FLOP count is identical.
    GemmArgs bench_args = check_args;
    bench_args.beta = 0.0f;
    auto launch = [&] { kernel.launch(bench_args); };
    const auto result = bench::run(launch, bench::tflops(bench::gemm_flops(M_, N_, K_)));

    std::cout << std::left << std::setw(22) << kernel.name << std::right << "  rel RMSE "
              << std::scientific << std::setprecision(2) << rmse << "  " << std::fixed
              << std::setprecision(3) << std::setw(8) << result.best_ms << " ms  "
              << std::setprecision(2) << std::setw(7) << result.rate << " " << result.units
              << std::endl;
  }

private:
  static constexpr float kAlpha = 1.0f;
  static constexpr float kBeta = 0.5f;
  static constexpr float kTolerance = 1e-3f;

  int M_, N_, K_;
  std::vector<float> A_, B_, C_init_, C_ref_, C_scratch_;
  thrust::device_vector<float> Ad_, Bd_, Cd_;
};

class SaxpyHarness {
public:
  SaxpyHarness(int n)
      : n_(n), x_(numerics::random_matrix(n, 1, 1)), y_init_(numerics::random_matrix(n, 1, 2)),
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
    const auto result = bench::run(launch, bench::gbs(bench::saxpy_bytes(n_)));

    std::cout << std::left << std::setw(22) << kernel.name << std::right << "  rel RMSE "
              << std::scientific << std::setprecision(2) << rmse << "  " << std::fixed
              << std::setprecision(3) << std::setw(8) << result.best_ms << " ms  "
              << std::setprecision(2) << std::setw(7) << result.rate << " " << result.units
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
