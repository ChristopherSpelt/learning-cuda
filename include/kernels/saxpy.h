#pragma once

#include "saxpy_types.h"

namespace cul::kernels::saxpy {

void naive(const SaxpyArgs &);

void vec_loads(const SaxpyArgs &);

} // namespace cul::kernels::saxpy
