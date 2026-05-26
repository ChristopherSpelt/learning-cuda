#pragma once

#include "cuda_utils.cuh"
#include <limits>
#include <string_view>

namespace cul::bench {

// `work` MUST be in raw base units (FLOPs, bytes, elements — never
// pre-scaled). The SI prefix lives only in `scale` (1e12 = tera, 1e9 = giga).
// rate = work / time_in_seconds / scale.
struct Metric {
  double work;
  double scale;
  std::string_view units;
};

[[nodiscard]] constexpr Metric tflops(double flops) { return {flops, 1e12, "TFLOP/s"}; }
[[nodiscard]] constexpr Metric gbs(double bytes) { return {bytes, 1e9, "GB/s"}; }

struct Result {
  double best_ms;
  double rate;
  std::string_view units;
};

[[nodiscard]] constexpr double gemm_flops(int M, int N, int K) {
  return 2.0 * double(M) * double(N) * double(K);
}

[[nodiscard]] constexpr double saxpy_bytes(int n) { return 12.0 * double(n); }

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
[[nodiscard]] Result run(Launch &&launch, Metric metric, double target_ms = 1000.0,
                         int inner_iters = 4) {

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

  const double rate = metric.work / (best_per_iter_ms * 1e-3) / metric.scale;
  return {best_per_iter_ms, rate, metric.units};
}

} // namespace cul::bench
