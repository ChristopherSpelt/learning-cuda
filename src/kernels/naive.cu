#include "cuda_utils.cuh"
#include "kernels.h"

// clang-format off
// Tile shape:  none (no shared memory).
// Load:        direct global reads inside the inner-k loop; no cooperative load.
// Output:      each thread computes exactly one element of C.
// Symmetry:    no tile invariants; BLOCKSIZE has no restrictions.
// clang-format on
template <int BLOCKSIZE>
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
  if (beta == 0.0f) {
    // β=0: pure write, no read of C
    C[row * N + col] = alpha * sum;
  } else {
    // β≠0: linear combination αAB + βC
    C[row * N + col] = alpha * sum + beta * C[row * N + col];
  }
}

void kernels::naive(const GemmArgs &a) {
  constexpr int BLOCKSIZE = 32;

  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid(ceil_div(a.N, BLOCKSIZE), ceil_div(a.M, BLOCKSIZE));

  naive_kernel<BLOCKSIZE>
      <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  CUDA_CHECK_LAUNCH();
}
