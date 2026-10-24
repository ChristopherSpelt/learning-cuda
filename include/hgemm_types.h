#pragma once

#include <cuda_bf16.h>

#include <string_view>

namespace cul {

struct HgemmArgs {
  int M, N, K;
  float alpha, beta;
  const __nv_bfloat16 *A;
  const __nv_bfloat16 *B;
  float *C;
};

using HgemmLaunch = void (*)(const HgemmArgs &);

struct HgemmKernel {
  std::string_view name;
  HgemmLaunch launch;
};

} // namespace cul
