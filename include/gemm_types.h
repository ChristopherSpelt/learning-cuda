#pragma once

#include <string_view>

struct GemmArgs {
  int M, N, K;
  float alpha, beta;
  const float *A;
  const float *B;
  float *C;
};

using GemmLaunch = void (*)(const GemmArgs &);

struct GemmKernel {
  std::string_view name;
  GemmLaunch launch;
};
