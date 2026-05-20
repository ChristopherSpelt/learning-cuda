#pragma once

#include "cuda_utils.cuh"
#include <limits>
#include <string_view>

namespace bench {

struct Result {
  double best_ms;
  double tflops;
};

class CudaTimer {
public:
  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~CudaTimer() {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }
  CudaTimer(const CudaTimer &) = delete;
  CudaTimer &operator=(const CudaTimer &) = delete;

  void start() { CUDA_CHECK(cudaEventRecord(start_)); }
  void stop() { CUDA_CHECK(cudaEventRecord(stop_)); }

  [[nodiscard]] double elapsed_ms() const {
    CUDA_CHECK(cudaEventSynchronize(stop_));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
    return double(ms);
  }

private:
  cudaEvent_t start_{}, stop_{};
};

template <typename Launch>
[[nodiscard]] Result run(Launch &&launch, double flop_count,
                         double target_ms = 1000.0, int inner_iters = 4) {

  launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  CudaTimer timer;
  double best_per_iter_ms = std::numeric_limits<double>::infinity();
  double elapsed_ms = 0;

  while (elapsed_ms < target_ms) {
    timer.start();
    for (int i = 0; i < inner_iters; ++i)
      launch();
    timer.stop();

    const double window_ms = timer.elapsed_ms();
    elapsed_ms += window_ms;
    best_per_iter_ms = std::min(best_per_iter_ms, window_ms / inner_iters);
  }

  const double tflops = flop_count / (best_per_iter_ms * 1e-3) / 1e12;
  return {best_per_iter_ms, tflops};
}

[[nodiscard]] constexpr double gemm_flops(int M, int N, int K) {
  return 2.0 * double(M) * double(N) * double(K);
}
} // namespace bench
