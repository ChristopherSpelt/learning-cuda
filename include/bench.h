#pragma once

#include "cuda_utils.cuh"
#include <chrono>
#include <cstdint>
#include <limits>
#include <string_view>

namespace bench {

struct Result {
  double best_ms;
  double tflops;
};

template <typename Launch>
[[nodiscard]] Result run(Launch &&launch, double flop_count,
                         double target_ms = 1000.0, int inner_iters = 4) {

  double best_per_iter_ms = std::numeric_limits<double>::infinity();
  double elapsed_ms = 0;

  while (elapsed_ms < target_ms) {
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < inner_iters; ++i)
      launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto end = std::chrono::high_resolution_clock::now();

    const double window_ms =
        std::chrono::duration<double, std::milli>(end - start).count();
    elapsed_ms += window_ms;
    best_per_iter_ms = std::min(best_per_iter_ms, window_ms / inner_iters);
  }

  const double tflops = flop_count / (best_per_iter_ms * 1e-3) / 1e12;
  return {best_per_iter_ms, tflops};
}

[[nodiscard]] constexpr double gemm_flops(std::uint32_t M, std::uint32_t N,
                                          std::uint32_t K) {
  return 2.0 * double(M) * double(N) * double(K);
}

inline void print_result(std::string_view name, Result r) {
  std::printf("%-20.*s  %8.3f ms  %7.2f TFLOPS\n", int(name.size()),
              name.data(), r.best_ms, r.tflops);
}
} // namespace bench
