#pragma once

#include "gemm_types.h"

namespace cul::kernels {

void cublas_pedantic(const GemmArgs &);

void cublas_default(const GemmArgs &);

void cublas_tf32(const GemmArgs &);
} // namespace cul::kernels
