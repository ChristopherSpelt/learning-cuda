#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels.h"

namespace cul {
namespace {
// clang-format off
// Tile shape:  none (no shared memory).
// Load:        direct global reads inside the inner-k loop; no cooperative load.
// Output:      each thread computes exactly one element of C.
// Symmetry:    no tile invariants; BLOCKSIZE has no restrictions.
// clang-format on
template <int BLOCKSIZE, bool BetaIsZero>
__global__ void
naive_kernel(int M, int N, int K, float alpha, const float *__restrict__ A,
             const float *__restrict__ B, float beta, float *__restrict__ C) {

  // ---- Prologue: thread coordinates -----------------------------------------
  const int row = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int col = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (row >= M || col >= N)
    return;

  float sum = 0.0f;

  // ---- Main loop -----------------------------------------------------------
  for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
  }

  // ---- Epilogue: αAB + βC store --------------------------------------------
  epilogue::store_result<BetaIsZero>(&C[row * N + col], alpha * sum, beta);
}
} // namespace

void kernels::naive(const GemmArgs &a) {
  constexpr int BLOCKSIZE = 32;

  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid(cuda_utils::ceil_div(a.N, BLOCKSIZE), cuda_utils::ceil_div(a.M, BLOCKSIZE));

  if (a.beta == 0.0f) {
    naive_kernel<BLOCKSIZE, true>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  } else {
    naive_kernel<BLOCKSIZE, false>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  }
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
