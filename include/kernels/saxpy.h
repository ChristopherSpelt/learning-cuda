#pragma once

#include "saxpy_types.h"

namespace cul::kernels::saxpy {

void cublas(const SaxpyArgs &);

void naive(const SaxpyArgs &);

void grid_stride(const SaxpyArgs &);

void vec_loads(const SaxpyArgs &);

} // namespace cul::kernels::saxpy
