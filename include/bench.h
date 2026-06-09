#pragma once

#include "cuda_utils.cuh"
#include <algorithm>
#include <charconv>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
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

struct Result {
  double best_ms;
  double rate;
  std::string_view units;
};

struct Samples {
  double best_ms = std::numeric_limits<double>::infinity();
  double elapsed_ms = 0;

  void add(double ms) {
    elapsed_ms += ms;
    best_ms = std::min(best_ms, ms);
  }

  [[nodiscard]] bool reached(double target_ms) const { return elapsed_ms >= target_ms; }
  [[nodiscard]] Result result(Metric m) const {
    return {best_ms, m.work / (best_ms * 1e-3) / m.scale, m.units};
  }
};

// Back to back launches under 1 timer; amortized launch overhead. Cache stays warm.
template <typename Launch>
[[nodiscard]] Result run_batch(Launch &&launch, Metric metric, double target_ms = 1000.0,
                               int inner_iters = 4) {
  launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  Samples s;
  CudaTimer timer;

  while (!s.reached(target_ms)) {
    timer.start();
    for (int i = 0; i < inner_iters; ++i)
      launch();
    timer.stop();

    const double per_launch = timer.elapsed_ms() / inner_iters;
    for (int i = 0; i < inner_iters; ++i) {
      s.add(per_launch);
    }
  }
  return s.result(metric);
}

// Flush L2 before each launch, time each launch in isolation. No cross launch cache reuse.
template <typename Launch>
[[nodiscard]] Result run_cold(Launch &&launch, Metric metric, double target_ms = 1000.0,
                              int inner_iters = 4) {

  launch();
  CUDA_CHECK(cudaDeviceSynchronize());

  Samples s;
  L2Flusher flusher;
  std::vector<CudaTimer> timers(inner_iters);
  while (!s.reached(target_ms)) {
    for (int i = 0; i < inner_iters; ++i) {
      flusher.flush();
      timers[i].start();
      launch();
      timers[i].stop();
    }
    for (auto &t : timers) {
      s.add(t.elapsed_ms());
    }
  }
  return s.result(metric);
}

struct Record {
  std::string_view name;
  int n;
  float rel_err;
  bool passed;
  Result cold;
  Result batch;
};

template <typename Launch>
[[nodiscard]] Record evaluate(std::string_view name, int n, float rel_err, float tol,
                              Launch &&launch, Metric metric) {

  // We need this expression (instead of rel_err > tol) to handle NaN properly.
  if (!(rel_err <= tol)) {
    return {name, n, rel_err, false, {}, {}};
  }
  const Result cold = run_cold(launch, metric);
  const Result batch = run_batch(launch, metric);
  return {name, n, rel_err, true, cold, batch};
}

inline void print_csv_header(std::ostream &os) {
  os << "n,kernel,rel_err,cold_ms,cold_rate,batch_ms,batch_rate\n";
}

inline void print_csv_row(std::ostream &os, const Record &r) {
  os << r.n << ',' << r.name << ',' << r.rel_err << ',';
  if (r.passed)
    os << r.cold.best_ms << ',' << r.cold.rate << ',' << r.batch.best_ms << ',' << r.batch.rate;
  else
    os << ",,,"; // failures are a gap, not a zero
  os << '\n';
}

inline void print_table_row(std::ostream &os, const Record &r) {
  os << std::left << std::setw(22) << r.name << std::right << "  rel_err " << std::scientific
     << std::setprecision(2) << r.rel_err;
  if (!r.passed) {
    os << "  FAILED\n";
    return;
  }
  const auto seg = [&os](std::string_view label, const Result &res) {
    os << "  " << label << ' ' << std::fixed << std::setprecision(3) << std::setw(8) << res.best_ms
       << " ms " << std::setprecision(2) << std::setw(7) << res.rate << ' ' << res.units;
  };
  seg("cold", r.cold);
  seg("batch", r.batch);
  os << '\n';
}

struct CliArgs {
  std::vector<int> sizes;
  bool csv = false;
};

[[noreturn]] inline void usage(std::string_view prog, int exit_code) {
  std::cerr << "Usage: " << prog << " [--csv] [size...]    (sizes are positive integers)\n";
  std::exit(exit_code);
}

inline CliArgs parse_args(int argc, char **argv, std::vector<int> default_sizes) {
  CliArgs a;

  for (int i = 1; i < argc; ++i) {

    const std::string_view arg = argv[i];

    if (arg == "--csv") {
      a.csv = true;
      continue;
    }
    if (arg == "--help") {
      usage(argv[0], EXIT_SUCCESS);
    }
    if (arg.starts_with("--")) {
      std::cerr << "Unknown flag: '" << arg << "'\n";
      usage(argv[0], EXIT_FAILURE);
    }

    int size = 0;
    const auto [end, ec] = std::from_chars(arg.data(), arg.data() + arg.size(), size);
    if (ec != std::errc{} || end != arg.data() + arg.size() || size <= 0) {
      std::cerr << "invalid size '" << arg << "'\n";
      usage(argv[0], EXIT_FAILURE);
    }
    a.sizes.push_back(size);
  }

  if (a.sizes.empty()) {
    a.sizes = std::move(default_sizes);
  }
  return a;
}

} // namespace cul::bench
