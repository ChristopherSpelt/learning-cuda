#pragma once

#include "gemm_types.h"

namespace kernels {

void cublas_pedantic(const GemmArgs &);

void cublas_default(const GemmArgs &);

void cublas_tf32(const GemmArgs &);
} // namespace kernels
