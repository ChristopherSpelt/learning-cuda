#pragma once

#include "gemm_types.h"

namespace cul::kernels::gemm {

void cublas_pedantic(const GemmArgs &);

void cublas_default(const GemmArgs &);

void cublas_tf32(const GemmArgs &);

void naive(const GemmArgs &);

void shared_mem(const GemmArgs &);

void shared_mem_1d_block(const GemmArgs &);

void shared_mem_2d_block(const GemmArgs &);

void shared_mem_vec(const GemmArgs &);

void shared_mem_vec_warp(const GemmArgs &);

} // namespace cul::kernels::gemm
