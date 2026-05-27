#pragma once

#include "asum_types.h"

namespace cul::kernels::asum {

void cublas(const AsumArgs &a);

void naive(const AsumArgs &);

void shared_mem(const AsumArgs &a);

} // namespace cul::kernels::asum
