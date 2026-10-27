#include "cuda_utils.cuh"
#include "kernels/hgemm.h"

#include <mma.h>

using namespace nvcuda;

namespace cul {
namespace {
template <bool BetaIsZero>
__global__ void naive_kernel(int M, int N, int K, float alpha, const __nv_bfloat16 *__restrict__ A,
                             const __nv_bfloat16 *__restrict__ B, float beta,
                             float *__restrict__ C) {

  // ---- Prologue: warp coordinates -----------------------------------------
  const int row = blockIdx.y * 16;
  const int col = blockIdx.x * 16;

  if (row >= M || col >= N)
    return;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> A_frag;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> B_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> C_frag;

  // Initialize the output to zero
  wmma::fill_fragment(acc_frag, 0.0f);

  // ---- Main loop -----------------------------------------------------------
  for (int k = 0; k < K; k += 16) {
    // Load the inputs
    wmma::load_matrix_sync(A_frag, &A[row * K + k], K);
    wmma::load_matrix_sync(B_frag, &B[k * N + col], N);

    // Perform the matrix multiplication
    wmma::mma_sync(acc_frag, A_frag, B_frag, acc_frag);
  }

  if constexpr (BetaIsZero) {
    for (int i = 0; i < C_frag.num_elements; ++i) {
      C_frag.x[i] = alpha * acc_frag.x[i];
    }

  } else {
    wmma::load_matrix_sync(C_frag, &C[row * N + col], N, wmma::mem_row_major);
    for (int i = 0; i < C_frag.num_elements; ++i) {
      C_frag.x[i] = alpha * acc_frag.x[i] + beta * C_frag.x[i];
    }
  }

  // Store the output
  wmma::store_matrix_sync(&C[row * N + col], C_frag, N, wmma::mem_row_major);
}

} // namespace

void kernels::hgemm::naive(const HgemmArgs &a) {
  dim3 block(32);
  dim3 grid(cuda::ceil_div(a.N, 16), cuda::ceil_div(a.M, 16));

  if (a.beta == 0.0f) {
    naive_kernel<true><<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  } else {
    naive_kernel<false><<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  }
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
