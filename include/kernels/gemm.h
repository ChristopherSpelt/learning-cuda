#pragma once

#include "gemm_types.h"

namespace cul::kernels::gemm {

void cublas_pedantic(const GemmArgs &);

void cublas_default(const GemmArgs &);

void cublas_tf32(const GemmArgs &);

void naive(const GemmArgs &);

void block_tile(const GemmArgs &);

void thread_tile_1d(const GemmArgs &);

void thread_tile_2d(const GemmArgs &);

void vec_loads(const GemmArgs &);

void warp_tile(const GemmArgs &);

} // namespace cul::kernels::gemm
