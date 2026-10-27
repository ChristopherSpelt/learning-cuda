#pragma once

#include "hgemm_types.h"

namespace cul::kernels::hgemm {

void cublas_pedantic(const HgemmArgs &);

void cublas_default(const HgemmArgs &);

void naive(const HgemmArgs &);

} // namespace cul::kernels::hgemm
