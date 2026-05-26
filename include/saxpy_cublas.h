#pragma once

#include "saxpy_types.h"

namespace cul::kernels::saxpy {

void cublas(const SaxpyArgs &);

} // namespace cul::kernels::saxpy
