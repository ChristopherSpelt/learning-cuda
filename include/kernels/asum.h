#pragma once

#include "asum_types.h"

namespace cul::kernels::asum {

void cublas(const AsumArgs &a);

void naive(const AsumArgs &);

void shared_mem(const AsumArgs &a);

void shared_mem_grid_stride_shuffle(const AsumArgs &a);

} // namespace cul::kernels::asum
