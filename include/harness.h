#pragma once

#include "bench.h"
#include "cuda_utils.cuh"
#include "gemm_cublas.h"
#include "kernels.h"
#include "numerics.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <cstdint>
#include <cstdio>
#include <vector>

class GemmHarness {
public:
  GemmHarness(std::uint32_t M, std::uint32_t N, std::uint32_t K)
      : M_(M), N_(N), K_(K), A_(numerics::random_matrix(M, K, 0)),
        B_(numerics::random_matrix(K, N, 1)),
        C_init_(numerics::random_matrix(M, N, 2)), C_ref_(std::size_t(M) * N),
        C_scratch_(std::size_t(M) * N), Ad_(A_.begin(), A_.end()),
        Bd_(B_.begin(), B_.end()), Cd_(C_init_.begin(), C_init_.end()) {

    auto *Ad_ptr = thrust::raw_pointer_cast(Ad_.data());
    auto *Bd_ptr = thrust::raw_pointer_cast(Bd_.data());
    auto *Cd_ptr = thrust::raw_pointer_cast(Cd_.data());

    kernels::cublas_pedantic(M_, N_, K_, kAlpha, Ad_ptr, Bd_ptr, kBeta, Cd_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::copy(Cd_.begin(), Cd_.end(), C_ref_.begin());

    std::printf("[reference: cuBLAS pedantic, %u x %u x %u]\n\n", M_, N_, K_);
  }

  GemmHarness(const GemmHarness &) = delete;
  GemmHarness &operator=(const GemmHarness &) = delete;

  void run(const GemmKernel &kernel) {
    auto *Ad_ptr = thrust::raw_pointer_cast(Ad_.data());
    auto *Bd_ptr = thrust::raw_pointer_cast(Bd_.data());
    auto *Cd_ptr = thrust::raw_pointer_cast(Cd_.data());

    // Correctness: full αAB + βC against the cuBLAS reference.
    thrust::copy(C_init_.begin(), C_init_.end(), Cd_.begin());
    kernel.launch(M_, N_, K_, kAlpha, Ad_ptr, Bd_ptr, kBeta, Cd_ptr);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::copy(Cd_.begin(), Cd_.end(), C_scratch_.begin());

    const float rmse = numerics::relative_rmse(C_scratch_, C_ref_);
    if (rmse > kTolerance) {
      std::printf("%-22.*s  rel RMSE %.2e  FAILED\n", int(kernel.name.size()),
                  kernel.name.data(), rmse);
      return;
    }

    // Benchmark with β=0: each launch overwrites C, no contraction map.
    // FLOP count is identical.
    auto launch = [&] {
      kernel.launch(M_, N_, K_, kAlpha, Ad_ptr, Bd_ptr, 0.0f, Cd_ptr);
    };
    const auto result = bench::run(launch, bench::gemm_flops(M_, N_, K_));

    std::printf("%-22.*s  rel RMSE %.2e  %8.3f ms  %7.2f TFLOPS\n",
                int(kernel.name.size()), kernel.name.data(), rmse,
                result.best_ms, result.tflops);
  }

private:
  static constexpr float kAlpha = 1.0f;
  static constexpr float kBeta = 0.5f;
  static constexpr float kTolerance = 1e-3f;

  std::uint32_t M_, N_, K_;
  std::vector<float> A_, B_, C_init_, C_ref_, C_scratch_;
  thrust::device_vector<float> Ad_, Bd_, Cd_;
};
