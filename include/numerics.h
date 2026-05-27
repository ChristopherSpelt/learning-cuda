#pragma once

#include <cmath>
#include <limits>
#include <random>
#include <span>
#include <vector>

namespace cul::numerics {

inline std::vector<float> random_vector(int rows, int cols, int seed) {

  std::mt19937 gen(seed);
  std::normal_distribution<float> dist(0.0f, 1.0f);

  std::vector<float> out(std::size_t(rows) * cols);
  for (auto &x : out) {
    x = dist(gen);
  }

  return out;
}

inline float relative_rmse(std::span<const float> a, std::span<const float> b) {
  if (a.size() != b.size())
    return std::numeric_limits<float>::infinity();

  double mse = 0.0;
  double ref_sq_mean = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double diff = double(a[i]) - double(b[i]);
    mse += diff * diff;
    ref_sq_mean += double(b[i]) * double(b[i]);
  }

  mse /= double(a.size());
  ref_sq_mean /= double(a.size());

  if (ref_sq_mean == 0.0f) {
    return mse == 0.0f ? 0.0f : std::numeric_limits<float>::infinity();
  }

  return float(std::sqrt(mse) / std::sqrt(ref_sq_mean));
}
} // namespace cul::numerics
