#pragma once

#include <cstdint>
#include <limits>
#include <random>
#include <vector>

namespace numerics {

inline std::vector<float> random_matrix(std::uint32_t rows, std::uint32_t cols,
                                        std::uint32_t seed) {

  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<float> out(std::size_t(rows) * cols);
  for (auto &x : out) {
    x = dist(gen);
  }

  return out;
}

inline float relative_rmse(const std::vector<float> &a,
                           const std::vector<float> &b) {
  if (a.size() != b.size())
    return std::numeric_limits<float>::infinity();

  double mse = 0.0;
  double ref_ms = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double diff = double(a[i]) - double(b[i]);
    mse += diff * diff;
    ref_ms += double(b[i]) * double(b[i]);
  }

  mse /= double(a.size());
  ref_ms /= double(a.size());

  return float(std::sqrt(mse) / std::sqrt(ref_ms));
}
} // namespace numerics
