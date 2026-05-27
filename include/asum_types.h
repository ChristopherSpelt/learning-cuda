#pragma once

#include <string_view>

namespace cul {

struct AsumArgs {
  int n;
  const float *x;
  float *result;
};

using AsumLaunch = void (*)(const AsumArgs &);

struct AsumKernel {
  std::string_view name;
  AsumLaunch launch;
};

} // namespace cul
