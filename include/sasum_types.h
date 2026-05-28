#pragma once

#include <string_view>

namespace cul {

struct SasumArgs {
  int n;
  const float *x;
  float *result;
};

using SasumLaunch = void (*)(const SasumArgs &);

struct SasumKernel {
  std::string_view name;
  SasumLaunch launch;
};

} // namespace cul
