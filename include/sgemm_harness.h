#pragma once

#include "gemm_harness.h"
#include "kernels/sgemm.h"

namespace cul {
using SgemmHarness = GemmHarness<SgemmArgs, SgemmKernel, &kernels::sgemm::cublas_pedantic>;
} // namespace cul
