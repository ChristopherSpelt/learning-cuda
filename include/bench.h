#pragma once

#include "cuda_utils.cuh"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

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

[[nodiscard]] constexpr double sgemm_flops(std::size_t M, std::size_t N, std::size_t K) {
  return 2.0 * double(M) * double(N) * double(K);
}
[[nodiscard]] constexpr double saxpy_bytes(std::size_t n) { return 12.0 * double(n); }
[[nodiscard]] constexpr double sasum_bytes(std::size_t n) { return 4.0 * double(n); }

class CudaTimer {
public:
  CudaTimer() {
    CUDA_CHECK(cudaEventCreate(&start_));
    CUDA_CHECK(cudaEventCreate(&stop_));
  }
  ~CudaTimer() noexcept {
    if (start_)
      cudaEventDestroy(start_);
    if (stop_)
      cudaEventDestroy(stop_);
  }
  CudaTimer(const CudaTimer &) = delete;
  CudaTimer &operator=(const CudaTimer &) = delete;

  CudaTimer(CudaTimer &&o) noexcept : start_(o.start_), stop_(o.stop_) {
    o.start_ = nullptr;
    o.stop_ = nullptr;
  }

  CudaTimer &operator=(CudaTimer &&o) noexcept {
    if (this != &o) {
      if (start_)
        cudaEventDestroy(start_);
      if (stop_)
        cudaEventDestroy(stop_);
      start_ = o.start_;
      stop_ = o.stop_;
      o.start_ = nullptr;
      o.stop_ = nullptr;
    }
    return *this;
  }

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

class L2Flusher {
public:
  L2Flusher() {
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int l2_byte_len = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&l2_byte_len, cudaDevAttrL2CacheSize, dev));
    byte_len_ = static_cast<std::size_t>(l2_byte_len);
    CUDA_CHECK(cudaMalloc(&scratch_, byte_len_));
  }

  ~L2Flusher() noexcept { cudaFree(scratch_); }
  L2Flusher(const L2Flusher &) = delete;
  L2Flusher &operator=(const L2Flusher &) = delete;

  void flush() const { CUDA_CHECK(cudaMemsetAsync(scratch_, 0, byte_len_)); }

private:
  void *scratch_{};
  std::size_t byte_len_{};
};

// Back to back launches under 1 timer; amortized launch overhead. Cache stays warm.
template <typename Launch>
[[nodiscard]] Result run_batch(Launch &&launch, Metric metric, double target_ms = 1000.0,
                               int inner_iters = 4) {
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  double best_per_iter_ms = std::numeric_limits<double>::infinity();
  double elapsed_ms = 0;
  CudaTimer timer;

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

// Flush L2 before each launch, time each launch in isolation. No cross launch cache reuse.
template <typename Launch>
[[nodiscard]] Result run_cold(Launch &&launch, Metric metric, double target_ms = 1000.0,
                              int inner_iters = 4) {

  launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  double best_per_iter_ms = std::numeric_limits<double>::infinity();
  double elapsed_ms = 0;

  L2Flusher flusher;
  std::vector<CudaTimer> timers(inner_iters);
  while (elapsed_ms < target_ms) {
    for (int i = 0; i < inner_iters; ++i) {
      flusher.flush();
      timers[i].start();
      launch();
      timers[i].stop();
    }
    for (int i = 0; i < inner_iters; ++i) {
      const double ms = timers[i].elapsed_ms();
      elapsed_ms += ms;
      best_per_iter_ms = std::min(best_per_iter_ms, ms);
    }
  }

  const double rate = metric.work / (best_per_iter_ms * 1e-3) / metric.scale;
  return {best_per_iter_ms, rate, metric.units};
}

} // namespace cul::bench
