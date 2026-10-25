#pragma once

#include "bench.h"
#include "cuda_utils.cuh"
#include "numerics.h"

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

namespace cul {

template <typename Args, typename Kernel, void (*Reference)(const Args &)> class GemmHarness {
  using T = std::remove_const_t<std::remove_pointer_t<decltype(Args::A)>>;

public:
  GemmHarness(int M, int N, int K)
      : M_(checked_m(M, N, K)), N_(N), K_(K), A_(make_input(M, K, 0)), B_(make_input(K, N, 1)),
        C_init_(numerics::random_vector(M, N, 2)), C_ref_(std::size_t(M) * N),
        C_ref_ab_(std::size_t(M) * N), C_scratch_(std::size_t(M) * N), Ad_(A_.begin(), A_.end()),
        Bd_(B_.begin(), B_.end()), Cd_(C_init_.begin(), C_init_.end()) {
    auto *Ad_ptr = thrust::raw_pointer_cast(Ad_.data());
    auto *Bd_ptr = thrust::raw_pointer_cast(Bd_.data());
    auto *Cd_ptr = thrust::raw_pointer_cast(Cd_.data());

    Args ref_args{
        .M = M_,
        .N = N_,
        .K = K_,
        .alpha = kAlpha,
        .beta = kBeta,
        .A = Ad_ptr,
        .B = Bd_ptr,
        .C = Cd_ptr,
    };

    Reference(ref_args);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::copy(Cd_.begin(), Cd_.end(), C_ref_.begin());

    Args ab_args = ref_args;
    // β=0 ignores Cd_'s current contents.
    ab_args.alpha = 1.0f;
    ab_args.beta = 0.0f;

    Reference(ab_args);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::copy(Cd_.begin(), Cd_.end(), C_ref_ab_.begin());

    ref_delta_ = numerics::relative_rmse(C_ref_, C_init_);
    CUL_REQUIRE(ref_delta_ > 2.0f, "degenerate reference data");
  }

  GemmHarness(const GemmHarness &) = delete;
  GemmHarness &operator=(const GemmHarness &) = delete;

  // Section header for this problem size; the blank line separates it from the previous block.
  void print_reference(std::ostream &os) const {
    os << "\n[reference: cuBLAS pedantic" << (std::is_same_v<T, float> ? "" : " BF16") << ", " << M_
       << " x " << N_ << " x " << K_ << ", ref_delta " << ref_delta_ << "]\n";
  }

  bench::Record run(const Kernel &kernel) {
    auto *Ad_ptr = thrust::raw_pointer_cast(Ad_.data());
    auto *Bd_ptr = thrust::raw_pointer_cast(Bd_.data());
    auto *Cd_ptr = thrust::raw_pointer_cast(Cd_.data());

    // Correctness: full αAB + βC against the cuBLAS reference.
    thrust::copy(C_init_.begin(), C_init_.end(), Cd_.begin());

    Args check_args{
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

    const float full_err = numerics::relative_rmse(C_scratch_, C_ref_);

    // Benchmark with β=0: each launch overwrites C, no contraction map.
    // FLOP count is identical.
    Args bench_args = check_args;
    bench_args.beta = 0.0f;
    bench_args.alpha = 1.0f;
    // β=0 picks a different kernel instantiation (BetaIsZero), so check it too.
    kernel.launch(bench_args);
    CUDA_CHECK(cudaDeviceSynchronize());
    thrust::copy(Cd_.begin(), Cd_.end(), C_scratch_.begin());

    const float ab_err = numerics::relative_rmse(C_scratch_, C_ref_ab_);
    const float rel_err = !(ab_err <= kTolerance) ? ab_err : full_err; // keeps NaN

    auto launch = [&] { kernel.launch(bench_args); };

    return bench::evaluate(kernel.name, M_, rel_err, kTolerance, launch, // n = M (square sweep)
                           bench::tflops(bench::sgemm_flops(M_, N_, K_)));
  }

private:
  static int checked_m(int M, int N, int K) {
    const auto fits = [](std::size_t a, std::size_t b) {
      return a * b <= static_cast<std::size_t>(std::numeric_limits<int>::max());
    };
    CUL_REQUIRE(fits(M, K) && fits(K, N) && fits(M, N),
                "M*K, K*N and M*N must each fit in int (kernel index math is 32-bit)");
    return M;
  }

  static std::vector<T> make_input(int rows, int cols, int seed) {
    auto v = numerics::random_vector(rows, cols, seed);
    if constexpr (std::is_same_v<T, float>)
      return v;
    else
      return numerics::to_bf16(v);
  }

  static constexpr float kAlpha = 1.5f;
  static constexpr float kBeta = 0.5f;
  static constexpr float kTolerance = 1e-3f;

  int M_, N_, K_;
  float ref_delta_ = 0.0f;
  std::vector<T> A_, B_;
  std::vector<float> C_init_, C_ref_, C_ref_ab_, C_scratch_;
  thrust::device_vector<T> Ad_, Bd_;
  thrust::device_vector<float> Cd_;
};

} // namespace cul
