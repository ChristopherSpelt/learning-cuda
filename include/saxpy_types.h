#pragma once

#include <string_view>

namespace cul {

struct SaxpyArgs {
  int n;
  float alpha;
  const float *x;
  float *y;
};

using SaxpyLaunch = void (*)(const SaxpyArgs &);

struct SaxpyKernel {
  std::string_view name;
  SaxpyLaunch launch;
};

} // namespace cul
