#pragma once

#include "asum_types.h"

namespace cul::kernels::asum {

void cublas(const AsumArgs &a);

void naive(const AsumArgs &);

void block_reduce(const AsumArgs &a);

void grid_stride(const AsumArgs &a);

void warp_reduce(const AsumArgs &a);

} // namespace cul::kernels::asum
