#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels.h"

// clang-format off
// Tile shape:  rectangular block tiles; As is BM x BK and Bs is BK x BN.
// Load:        strided cooperative; each thread fills (BM*BK)/NUM_THREADS slots
//              of As and (BK*BN)/NUM_THREADS slots of Bs per sweep, in stride_A
//              and stride_B row windows. NUM_THREADS = (BM * BN) / (TM * TN).
// Output:      each thread computes a TM x TN thread-tile of C.
// Symmetry:    strided load breaks the BM == BN constraint; per-tile slot
//              counts only need to be divisible by NUM_THREADS.
// Bounds:      handles non-aligned M/N/K — loads zero-fill out-of-range
//              slots; stores skip threads past the matrix edge.
// clang-format on
template <int BM, int BK, int BN, int TM, int TN, bool BetaIsZero>
__global__ void shared_mem_2d_block_kernel(int M, int N, int K, float alpha,
                                           const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float beta, float *__restrict__ C) {

  // ---- Compile-time invariants ----------------------------------------------
  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);
  constexpr int stride_A = NUM_THREADS / BK;
  constexpr int stride_B = NUM_THREADS / BN;

  static_assert(NUM_THREADS * (TM * TN) == BM * BN,
                "BM*BN must be divisible by TM*TN");
  static_assert(stride_A * BK == NUM_THREADS, "stride_A must be integer");
  static_assert(BM % stride_A == 0, "As load sweep must cover BM exactly.");
  static_assert(stride_B * BN == NUM_THREADS, "stride_B must be integer");
  static_assert(BK % stride_B == 0, "Bs load sweep must cover BK exactly.");

  // ---- Prologue: tile coordinates, pointer offsets, register init -----------
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;

  const int tile_row = threadIdx.x / (BN / TN);
  const int tile_col = threadIdx.x % (BN / TN);

  const int load_As_row = threadIdx.x / BK;
  const int load_As_col = threadIdx.x % BK;

  const int load_Bs_row = threadIdx.x / BN;
  const int load_Bs_col = threadIdx.x % BN;

  A += block_row * BM * K;
  B += block_col * BN;
  C += block_row * BM * N + block_col * BN;

  float thread_result[TM * TN] = {0.0f};

  float reg_M[TM] = {0.0f};
  float reg_N[TN] = {0.0f};

  // Block-invariant coord for the bounds-checked load below.
  const int B_col_global = block_col * BN + load_Bs_col;

  // ---- Main loop: K-tile iteration ------------------------------------------
  for (int block_idx = 0; block_idx < K; block_idx += BK) {
    const int A_col_global = block_idx + load_As_col;

    // Load — out-of-range slots zero-fill; 0 is the GEMM identity.
    for (int load_offset = 0; load_offset < BM; load_offset += stride_A) {
      const int A_row_global = block_row * BM + load_As_row + load_offset;

      As[(load_As_row + load_offset) * BK + load_As_col] =
          (A_row_global < M && A_col_global < K)
              ? A[(load_As_row + load_offset) * K + load_As_col]
              : 0.0f;
    }

    for (int load_offset = 0; load_offset < BK; load_offset += stride_B) {
      const int B_row_global = block_idx + load_Bs_row + load_offset;
      Bs[(load_Bs_row + load_offset) * BN + load_Bs_col] =
          (B_row_global < K && B_col_global < N)
              ? B[(load_Bs_row + load_offset) * N + load_Bs_col]
              : 0.0f;
    }

    __syncthreads();
    A += BK;
    B += BK * N;

    // Compute loop
    for (int k = 0; k < BK; ++k) {

      for (int i = 0; i < TM; ++i) {
        reg_M[i] = As[(tile_row * TM + i) * BK + k];
      }

      for (int i = 0; i < TN; ++i) {
        reg_N[i] = Bs[k * BN + (tile_col * TN + i)];
      }

      for (int result_idx_m = 0; result_idx_m < TM; ++result_idx_m) {
        for (int result_idx_n = 0; result_idx_n < TN; ++result_idx_n) {

          thread_result[result_idx_m * TN + result_idx_n] +=
              reg_M[result_idx_m] * reg_N[result_idx_n];
        }
      }
    }
    __syncthreads();
  }

  // ---- Epilogue: αAB + βC store --------------------------------------------
  for (int result_idx_m = 0; result_idx_m < TM; ++result_idx_m) {
    const int C_row_global = block_row * BM + tile_row * TM + result_idx_m;

    for (int result_idx_n = 0; result_idx_n < TN; ++result_idx_n) {
      const int C_col_global = block_col * BN + tile_col * TN + result_idx_n;

      if (C_row_global < M && C_col_global < N) {
        store_result<BetaIsZero>(
            &C[(tile_row * TM + result_idx_m) * N + tile_col * TN +
               result_idx_n],
            alpha * thread_result[result_idx_m * TN + result_idx_n], beta);
      }
    }
  }
}

void kernels::shared_mem_2d_block(const GemmArgs &a) {
  constexpr int BM = 64, BK = 8, BN = 64, TM = 8, TN = 8;
  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

  dim3 block(NUM_THREADS);
  dim3 grid(ceil_div(a.N, BN), ceil_div(a.M, BM));

  if (a.beta == 0.0f) {
    shared_mem_2d_block_kernel<BM, BK, BN, TM, TN, true>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  } else {
    shared_mem_2d_block_kernel<BM, BK, BN, TM, TN, false>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  }
  CUDA_CHECK_LAUNCH();
}
