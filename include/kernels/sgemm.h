#pragma once

#include "sgemm_types.h"

namespace cul::kernels::sgemm {

void cublas_pedantic(const SgemmArgs &);

void cublas_default(const SgemmArgs &);

void cublas_tf32(const SgemmArgs &);

void naive(const SgemmArgs &);

void block_tile(const SgemmArgs &);

void thread_tile_1d(const SgemmArgs &);

void thread_tile_2d(const SgemmArgs &);

void vec_loads(const SgemmArgs &);

void warp_tile(const SgemmArgs &);

void double_buffer(const SgemmArgs &);

} // namespace cul::kernels::sgemm
