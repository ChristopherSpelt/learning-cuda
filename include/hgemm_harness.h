#pragma once

#include "gemm_harness.h"
#include "kernels/hgemm.h"

namespace cul {
using HgemmHarness = GemmHarness<HgemmArgs, HgemmKernel, &kernels::hgemm::cublas_pedantic>;
} // namespace cul
