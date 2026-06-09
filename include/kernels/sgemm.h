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

void resolve_bank(const SgemmArgs &);

void double_buffer(const SgemmArgs &);

void cp_async(const SgemmArgs &);

void inner_loop_prefetch(const SgemmArgs &);

void double_buffer_prefetch(const SgemmArgs &);

void cp_async_swizzle(const SgemmArgs &);

void double_buffer_prefetch_bank(const SgemmArgs &);

} // namespace cul::kernels::sgemm
