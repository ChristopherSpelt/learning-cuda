#include "cuda_utils.cuh"
#include "kernels.h"

template <const std::uint32_t BM, const std::uint32_t BK,
          const std::uint32_t BN, const std::uint32_t TM,
          const std::uint32_t TN>
__global__ void shared_mem_2d_block_kernel(std::uint32_t M, std::uint32_t N,
                                           std::uint32_t K, float alpha,
                                           const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float beta, float *__restrict__ C) {
  // A single thread is responsible for calculating a tile of TMxTN elements in
  // the BMxBN block of C.
  constexpr std::uint32_t threads_per_block = (BM * BN) / (TM * TN);

  // Shared tiles. Each thread loads (BM*BK)/threads_per_block slots of As and
  // (BK*BN)/threads_per_block slots of Bs per iteration, then computes a
  // TM x TN tile of outputs in C. The cooperative load fills stride_A complete
  // rows of As (and stride_B complete rows of Bs) per sweep, with each thread
  // contributing one slot per sweep.
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // Block coordinates: which BMxBN tile of C this block computes.
  const std::uint32_t block_row = blockIdx.y;
  const std::uint32_t block_col = blockIdx.x;

  // Where this thread's output tile sits within the BMxBN tile in C.
  // So tile_row ∈ [0, BM/TM) and tile_col ∈ [0, BN/TN) and a single thread
  // writes to rows tile_row*TM, tile_row*TM + 1, ..., tile_row * TM + TM - 1
  // and cols tile_col * TN, tile_col * TN + 1, ..., tile_col * TN + TN -1.
  const std::uint32_t tile_row = threadIdx.x / (BN / TN);
  const std::uint32_t tile_col = threadIdx.x % (BN / TN);

  // load_As_row, load_As_col: where this thread loads into As during one sweep.
  // Across sweeps, the thread strides by stride_A rows (same column).
  // stride_A = threads_per_block / BK because As has BK columns: that's how
  // many complete rows can be filled by one wave of threads_per_block threads.
  const std::uint32_t load_As_row = threadIdx.x / BK;
  const std::uint32_t load_As_col = threadIdx.x % BK;
  constexpr std::uint32_t stride_A = threads_per_block / BK;

  const std::uint32_t load_Bs_row = threadIdx.x / BN;
  const std::uint32_t load_Bs_col = threadIdx.x % BN;
  constexpr std::uint32_t stride_B = threads_per_block / BN;

  A += block_row * BM * K;                  // BLOCK: jump to my row stripe of A
  B += block_col * BN;                      // BLOCK: jump to my col stripe of B
  C += block_row * BM * N + block_col * BN; // BLOCK: jump to my tile in C

  float thread_result[TM * TN] = {0.0f};

  float reg_M[TM] = {0.0f};
  float reg_N[TN] = {0.0f};

  // Global coordinates needed for bounds checking
  const std::uint32_t A_row_global = block_row * BM + load_As_row;
  const std::uint32_t B_col_global = block_col * BN + load_Bs_col;

  for (std::uint32_t block_idx = 0; block_idx < K; block_idx += BK) {

    // Load into shared memory
    for (std::uint32_t load_offset = 0; load_offset < BM;
         load_offset += stride_A) {
      As[(load_As_row + load_offset) * BK + load_As_col] =
          A[(load_As_row + load_offset) * K + load_As_col];
    }

    for (std::uint32_t load_offset = 0; load_offset < BK;
         load_offset += stride_B) {
      Bs[(load_Bs_row + load_offset) * BN + load_Bs_col] =
          B[(load_Bs_row + load_offset) * N + load_Bs_col];
    }

    __syncthreads();
    A += BK;     // BLOCK: advance by one tile to the right
    B += BK * N; // BLOCK: advance by one tile down.

    // Compute loop
    for (std::uint32_t k = 0; k < BK; ++k) {

      for (std::uint32_t i = 0; i < TM; ++i) {
        reg_M[i] = As[(tile_row * TM + i) * BK + k];
      }

      for (std::uint32_t i = 0; i < TN; ++i) {
        reg_N[i] = Bs[k * BN + (tile_col * TN + i)];
      }

      for (std::uint32_t result_idx_m = 0; result_idx_m < TM; ++result_idx_m) {
        for (std::uint32_t result_idx_n = 0; result_idx_n < TN;
             ++result_idx_n) {

          thread_result[result_idx_m * TN + result_idx_n] +=
              reg_M[result_idx_m] * reg_N[result_idx_n];
        }
      }
    }
    __syncthreads();
  }

  if (beta == 0.0f) {
    for (std::uint32_t result_idx_m = 0; result_idx_m < TM; ++result_idx_m) {
      for (std::uint32_t result_idx_n = 0; result_idx_n < TN; ++result_idx_n) {

        C[(tile_row * TM + result_idx_m) * N + tile_col * TN + result_idx_n] =
            alpha * thread_result[result_idx_m * TN + result_idx_n];
      }
    }
  } else {
    for (std::uint32_t result_idx_m = 0; result_idx_m < TM; ++result_idx_m) {
      for (std::uint32_t result_idx_n = 0; result_idx_n < TN; ++result_idx_n) {

        C[(tile_row * TM + result_idx_m) * N + tile_col * TN + result_idx_n] =
            alpha * thread_result[result_idx_m * TN + result_idx_n] +
            beta * C[(tile_row * TM + result_idx_m) * N + tile_col * TN +
                     result_idx_n];
      }
    }
  }
}

void kernels::shared_mem_2d_block(std::uint32_t M, std::uint32_t N,
                                  std::uint32_t K, float alpha, const float *A,
                                  const float *B, float beta, float *C) {
  constexpr int BM = 64, BK = 8, BN = 64, TM = 8, TN = 8;
  constexpr int THREADS = (BN * BM) / (TM * TN);

  dim3 block(THREADS);
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));

  shared_mem_2d_block_kernel<BM, BK, BN, TM, TN>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK_LAUNCH();
}
