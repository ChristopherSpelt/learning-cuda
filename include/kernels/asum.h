#pragma once

#include "asum_types.h"

namespace cul::kernels::asum {

void cublas(const AsumArgs &a);

void naive(const AsumArgs &);

} // namespace cul::kernels::asum
