#pragma once

#include <string_view>

namespace cul {

struct SgemmArgs {
  int M, N, K;
  float alpha, beta;
  const float *A;
  const float *B;
  float *C;
};

using SgemmLaunch = void (*)(const SgemmArgs &);

struct SgemmKernel {
  std::string_view name;
  SgemmLaunch launch;
};

} // namespace cul
