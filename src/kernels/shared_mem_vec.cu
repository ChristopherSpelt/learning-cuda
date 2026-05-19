#include "cuda_utils.cuh"
#include "kernels.h"

template <const std::uint32_t BM, const std::uint32_t BK,
          const std::uint32_t BN, const std::uint32_t TM,
          const std::uint32_t TN>
__global__ void shared_mem_vec_kernel(std::uint32_t M, std::uint32_t N,
                                      std::uint32_t K, float alpha,
                                      const float *__restrict__ A,
                                      const float *__restrict__ B, float beta,
                                      float *__restrict__ C) {

  // Shared tiles
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
  const std::uint32_t load_As_row = threadIdx.x / (BK / 4);
  const std::uint32_t load_As_col = threadIdx.x % (BK / 4);
  const std::uint32_t load_Bs_row = threadIdx.x / (BN / 4);
  const std::uint32_t load_Bs_col = threadIdx.x % (BN / 4);

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

    // Load A transpose into shared memory
    float4 tmp = reinterpret_cast<const float4 *>(
        &A[load_As_row * K + load_As_col * 4])[0];
    As[(load_As_col * 4 + 0) * BM + load_As_row] = tmp.x;
    As[(load_As_col * 4 + 1) * BM + load_As_row] = tmp.y;
    As[(load_As_col * 4 + 2) * BM + load_As_row] = tmp.z;
    As[(load_As_col * 4 + 3) * BM + load_As_row] = tmp.w;

    reinterpret_cast<float4 *>(&Bs[load_Bs_row * BN + load_Bs_col * 4])[0] =
        reinterpret_cast<const float4 *>(
            &B[load_Bs_row * N + load_Bs_col * 4])[0];

    __syncthreads();
    A += BK;     // BLOCK: advance by one tile to the right
    B += BK * N; // BLOCK: advance by one tile down.

    // Compute loop
    for (std::uint32_t k = 0; k < BK; ++k) {

      for (std::uint32_t i = 0; i < TM; ++i) {
        reg_M[i] = As[k * BM + (tile_row * TM + i)];
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

  } else {
    for (std::uint32_t result_idx_m = 0; result_idx_m < TM; ++result_idx_m) {
      for (std::uint32_t result_idx_n = 0; result_idx_n < TN;
           result_idx_n += 4) {

        float4 tmp =
            reinterpret_cast<float4 *>(&C[(tile_row * TM + result_idx_m) * N +
                                          tile_col * TN + result_idx_n])[0];

        tmp.x = alpha * thread_result[result_idx_m * TN + result_idx_n + 0] +
                beta * tmp.x;
        tmp.y = alpha * thread_result[result_idx_m * TN + result_idx_n + 1] +
                beta * tmp.y;
        tmp.z = alpha * thread_result[result_idx_m * TN + result_idx_n + 2] +
                beta * tmp.z;
        tmp.w = alpha * thread_result[result_idx_m * TN + result_idx_n + 3] +
                beta * tmp.w;

        reinterpret_cast<float4 *>(&C[(tile_row * TM + result_idx_m) * N +
                                      tile_col * TN + result_idx_n])[0] = tmp;
      }
    }
  }
}

void kernels::shared_mem_vec(std::uint32_t M, std::uint32_t N, std::uint32_t K,
                             float alpha, const float *A, const float *B,
                             float beta, float *C) {
  constexpr int BM = 128, BK = 8, BN = 128, TM = 8, TN = 8;
  constexpr int THREADS = (BN * BM) / (TM * TN);

  dim3 block(THREADS);
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));

  shared_mem_vec_kernel<BM, BK, BN, TM, TN>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK_LAUNCH();
}
