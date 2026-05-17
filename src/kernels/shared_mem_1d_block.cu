#include "cuda_utils.cuh"
#include "kernels.h"

template <const std::uint32_t BM, const std::uint32_t BK,
          const std::uint32_t BN, const std::uint32_t TM>
  requires(BM *BK == (BM * BN) / TM) && (BK * BN == (BM * BN) / TM) &&
          (BN * BM % TM == 0)
__global__ void shared_mem_1d_block_kernel(std::uint32_t M, std::uint32_t N,
                                           std::uint32_t K, float alpha,
                                           const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float beta, float *__restrict__ C) {

  // Shared tiles. Each thread fills exactly one slot in each tile per
  // iteration, then computes a strip of TM outputs in C. Consistency requires
  // that blockDim.x == BM*BK == BK*BN = (BM*BN)/TM which is enforced by the
  // `requires` clause above.
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Block coordinates: which BMxBN tile of C this block computes.
  const std::uint32_t block_row = blockIdx.y;
  const std::uint32_t block_col = blockIdx.x;

  // Where this thread's output strip sits within the BMxBN tile in C.
  // So strip_row ∈ [0, BM/TM) and strip_col ∈ [0, BN) and a single thread
  // writes to rows strip_row*TM, strip_row*TM + 1, ..., strip_row*TM + TM - 1.
  const std::uint32_t strip_row = threadIdx.x / BN;
  const std::uint32_t strip_col = threadIdx.x % BN;

  // This thread fills slot (load_As_row, load_As_col) of As, and
  // (load_Bs_row, load_Bs_col) of Bs.
  const std::uint32_t load_As_row = threadIdx.x / BK;
  const std::uint32_t load_As_col = threadIdx.x % BK;
  const std::uint32_t load_Bs_row = threadIdx.x / BN;
  const std::uint32_t load_Bs_col = threadIdx.x % BN;

  A += block_row * BM * K;                  // BLOCK: jump to my row stripe of A
  B += block_col * BN;                      // BLOCK: jump to my col stripe of B
  C += block_row * BM * N + block_col * BN; // BLOCK: jump to my tile in C

  float thread_result[TM] = {0.0f};

  // Global coordinates needed for bounds checking
  const std::uint32_t A_row_global = block_row * BM + load_As_row;
  const std::uint32_t B_col_global = block_col * BN + load_Bs_col;

  for (std::uint32_t tile = 0; tile < K; tile += BK) {
    // Global coordinates needed for bounds checking
    const std::uint32_t A_col_global = tile + load_As_col;
    const std::uint32_t B_row_global = tile + load_Bs_row;

    As[load_As_row * BK + load_As_col] = (A_row_global < M && A_col_global < K)
                                             ? A[load_As_row * K + load_As_col]
                                             : 0.0f;
    Bs[load_Bs_row * BN + load_Bs_col] = (B_row_global < K && B_col_global < N)
                                             ? B[load_Bs_row * N + load_Bs_col]
                                             : 0.0f;

    __syncthreads();
    A += BK;     // BLOCK: advance by one tile to the right
    B += BK * N; // BLOCK: advance by one tile down.

    for (std::uint32_t res_idx = 0; res_idx < TM; ++res_idx) {
      for (std::uint32_t k = 0; k < BK; ++k) {
        thread_result[res_idx] +=
            As[(strip_row * TM + res_idx) * BK + k] * Bs[k * BN + strip_col];
      }
    }
    __syncthreads();
  }

  for (std::uint32_t res_idx = 0; res_idx < TM; ++res_idx) {
    // Global coordinates needed for bounds checking
    const std::uint32_t C_row_global =
        block_row * BM + strip_row * TM + res_idx;
    const std::uint32_t C_col_global = block_col * BN + strip_col;

    if (C_row_global < M && C_col_global < N) {
      C[(strip_row * TM + res_idx) * N + strip_col] =
          alpha * thread_result[res_idx] +
          beta * C[(strip_row * TM + res_idx) * N + strip_col];
    }
  }
}

void kernels::shared_mem_1d_block(std::uint32_t M, std::uint32_t N,
                                  std::uint32_t K, float alpha, const float *A,
                                  const float *B, float beta, float *C) {
  constexpr int BM = 64, BK = 8, BN = 64, TM = 8;
  constexpr int THREADS = BN * BM / TM;

  dim3 block(THREADS);
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));

  shared_mem_1d_block_kernel<BM, BK, BN, TM>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK_LAUNCH();
}
