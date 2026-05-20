#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels.h"

// clang-format off
// Tile shape:  square block tile of BLOCKSIZE x BLOCKSIZE for As and Bs.
// Load:        one-shot; each thread fills one float slot of As and one of Bs.
//              NUM_THREADS = BLOCKSIZE * BLOCKSIZE.
// Output:      each thread computes exactly one element of C.
// Symmetry:    block tiles are square.
// Bounds:      handles non-aligned M/N/K — loads zero-fill out-of-range
//              slots; stores skip threads past the matrix edge.
// clang-format on
template <int BLOCKSIZE, bool BetaIsZero>
__global__ void shared_mem_kernel(int M, int N, int K, float alpha,
                                  const float *__restrict__ A,
                                  const float *__restrict__ B, float beta,
                                  float *__restrict__ C) {

  // ---- Prologue: tile coordinates, pointer offsets, register init -----------
  __shared__ float As[BLOCKSIZE * BLOCKSIZE];
  __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;

  const int thread_row = threadIdx.x / BLOCKSIZE;
  const int thread_col = threadIdx.x % BLOCKSIZE;

  const int global_row = block_row * BLOCKSIZE + thread_row;
  const int global_col = block_col * BLOCKSIZE + thread_col;

  A += block_row * BLOCKSIZE * K;
  B += block_col * BLOCKSIZE;
  C += block_row * BLOCKSIZE * N + block_col * BLOCKSIZE;

  float sum = 0.0f;

  // ---- Main loop: K-tile iteration ------------------------------------------
  for (int tile = 0; tile < K; tile += BLOCKSIZE) {
    // Load — out-of-range slots zero-fill; 0 is the GEMM identity.
    const int a_col_global = tile + thread_col;
    const int b_row_global = tile + thread_row;

    As[thread_row * BLOCKSIZE + thread_col] =
        (global_row < M && a_col_global < K) ? A[thread_row * K + thread_col]
                                             : 0.0f;
    Bs[thread_row * BLOCKSIZE + thread_col] =
        (global_col < N && b_row_global < K) ? B[thread_row * N + thread_col]
                                             : 0.0f;

    __syncthreads();
    A += BLOCKSIZE;
    B += BLOCKSIZE * N;

    // Compute loop
    for (int k = 0; k < BLOCKSIZE; ++k) {
      sum += As[thread_row * BLOCKSIZE + k] * Bs[k * BLOCKSIZE + thread_col];
    }
    __syncthreads();
  }

  // ---- Epilogue: αAB + βC store --------------------------------------------
  if (global_row < M && global_col < N) {
    store_result<BetaIsZero>(&C[thread_row * N + thread_col], alpha * sum,
                             beta);
  }
}

void kernels::shared_mem(const GemmArgs &a) {
  constexpr int BLOCKSIZE = 32;

  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid(ceil_div(a.N, BLOCKSIZE), ceil_div(a.M, BLOCKSIZE));

  if (a.beta == 0.0f) {
    shared_mem_kernel<BLOCKSIZE, true>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  } else {
    shared_mem_kernel<BLOCKSIZE, false>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  }
  CUDA_CHECK_LAUNCH();
}
