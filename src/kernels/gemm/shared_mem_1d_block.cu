#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels/gemm.h"

namespace cul {
namespace {
// clang-format off
// Tile shape:  Rectangular block tiles; As is BM x BK and Bs is BK x BN.
// Load:        one-shot; each thread fills one float slot of As and one of Bs.
//              NUM_THREADS = BM * BN / TM.
// Output:      Each thread computes a vertical strip of TM outputs of C.
// Symmetry:    One-shot load on differently-shaped tiles forces
//              NUM_THREADS == BM*BK == BK*BN, hence BM == BN.
// Bounds:      Handles non-aligned M/N/K — loads zero-fill out-of-range
//              slots; stores skip threads past the matrix edge.
// clang-format on
template <int BM, int BK, int BN, int TM, bool BetaIsZero>
__global__ void shared_mem_1d_block_kernel(int M, int N, int K, float alpha,
                                           const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float beta, float *__restrict__ C) {
  // ---- Compile-time invariants ----------------------------------------------
  static_assert(BM == BN, "single-shot cooperative load requires BM == BN");
  static_assert(TM * BK == BM,
                "NUM_THREADS (BM*BN/TM) must equal BM*BK (As slot count)");

  // ---- Prologue: tile coordinates, pointer offsets, register init -----------
  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;

  const int strip_row = threadIdx.x / BN;
  const int strip_col = threadIdx.x % BN;

  const int load_As_row = threadIdx.x / BK;
  const int load_As_col = threadIdx.x % BK;

  const int load_Bs_row = threadIdx.x / BN;
  const int load_Bs_col = threadIdx.x % BN;

  A += block_row * BM * K;
  B += block_col * BN;
  C += block_row * BM * N + block_col * BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float thread_result[TM] = {0.0f};

  // Block-invariant coords for the bounds-checked load below.
  const int A_row_global = block_row * BM + load_As_row;
  const int B_col_global = block_col * BN + load_Bs_col;

  // ---- Main loop: K-tile iteration ------------------------------------------
  for (int k_tile = 0; k_tile < K; k_tile += BK) {
    // Load — out-of-range slots zero-fill; 0 is the GEMM identity.
    const int A_col_global = k_tile + load_As_col;
    const int B_row_global = k_tile + load_Bs_row;

    As[load_As_row * BK + load_As_col] = (A_row_global < M && A_col_global < K)
                                             ? A[load_As_row * K + load_As_col]
                                             : 0.0f;
    Bs[load_Bs_row * BN + load_Bs_col] = (B_row_global < K && B_col_global < N)
                                             ? B[load_Bs_row * N + load_Bs_col]
                                             : 0.0f;

    __syncthreads();
    A += BK;
    B += BK * N;

    // Compute loop
    for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
      for (int k = 0; k < BK; ++k) {
        thread_result[res_idx_m] +=
            As[(strip_row * TM + res_idx_m) * BK + k] * Bs[k * BN + strip_col];
      }
    }
    __syncthreads();
  }

  // ---- Epilogue: αAB + βC store --------------------------------------------
  for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
    const int C_row_global = block_row * BM + strip_row * TM + res_idx_m;
    const int C_col_global = block_col * BN + strip_col;

    if (C_row_global < M && C_col_global < N) {
      epilogue::store_result<BetaIsZero>(
          &C[(strip_row * TM + res_idx_m) * N + strip_col],
          alpha * thread_result[res_idx_m], beta);
    }
  }
}
} // namespace

void kernels::gemm::shared_mem_1d_block(const GemmArgs &a) {
  constexpr int BM = 64, BK = 8, BN = 64, TM = 8;
  constexpr int NUM_THREADS = BN * BM / TM;

  dim3 block(NUM_THREADS);
  dim3 grid(cuda_utils::ceil_div(a.N, BN), cuda_utils::ceil_div(a.M, BM));

  if (a.beta == 0.0f) {
    shared_mem_1d_block_kernel<BM, BK, BN, TM, true>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  } else {
    shared_mem_1d_block_kernel<BM, BK, BN, TM, false>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  }
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
