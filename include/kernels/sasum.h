#pragma once

#include "sasum_types.h"

namespace cul::kernels::sasum {

void cublas(const SasumArgs &a);

void naive(const SasumArgs &);

void block_reduce(const SasumArgs &a);

void grid_stride(const SasumArgs &a);

void warp_reduce(const SasumArgs &a);

} // namespace cul::kernels::sasum
