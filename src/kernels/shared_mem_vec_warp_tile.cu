#include "cuda_utils.cuh"
#include "kernels.h"

constexpr std::uint32_t WARPSIZE = 32;

template <const std::uint32_t BM, const std::uint32_t BK,
          const std::uint32_t BN, const std::uint32_t WM,
          const std::uint32_t WN, const std::uint32_t WNITER,
          const std::uint32_t TM, const std::uint32_t TN,
          const std::uint32_t NUM_THREADS>
__global__ void shared_mem_vec_warp_kernel(std::uint32_t M, std::uint32_t N,
                                           std::uint32_t K, float alpha,
                                           const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float beta, float *__restrict__ C) {

  // Block coordinates: which BMxBN tile of C this thread block computes.
  const std::uint32_t block_row = blockIdx.y;
  const std::uint32_t block_col = blockIdx.x;

  // Warp coordinates: where in the threadblock this warp block sits.
  const std::uint32_t warp_idx = threadIdx.x / WARPSIZE;
  const std::uint32_t warp_row = warp_idx / (BN / WN);
  const std::uint32_t warp_col = warp_idx % (BN / WN);

  // Given fixed WM, WN, WARPSIZE, TM, TN and WNITER we can deduce
  // WMITER, using the facts:
  // WSUBM = WM / WMITER
  // WSUBN = WN / WNITER
  // (WSUBM/TM)*(WSUBN/TN) = WARPSIZE
  constexpr std::uint32_t WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
  constexpr std::uint32_t WSUBM = WM / WMITER;
  constexpr std::uint32_t WSUBN = WN / WNITER;

  // Which thread tile inside the current subtile.
  const std::uint32_t thread_idx_in_warp = threadIdx.x % WARPSIZE;
  const std::uint32_t thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);
  const std::uint32_t thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);

  // Shared tiles
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  // load_As_row, load_As_col: where this thread loads into As during one sweep.
  const std::uint32_t load_As_row = threadIdx.x / (BK / 4);
  const std::uint32_t load_As_col = threadIdx.x % (BK / 4);
  constexpr std::uint32_t row_stride_A = (NUM_THREADS * 4) / BK;

  const std::uint32_t load_Bs_row = threadIdx.x / (BN / 4);
  const std::uint32_t load_Bs_col = threadIdx.x % (BN / 4);
  constexpr std::uint32_t row_stride_B = (NUM_THREADS * 4) / BN;

  A += block_row * BM * K; // BLOCK: jump to my row stripe of A
  B += block_col * BN;     // BLOCK: jump to my col stripe of B
  C += (block_row * BM + warp_row * WM) * N + block_col * BN +
       warp_col * WN; // BLOCK: jump to my tile in C

  float thread_result[WMITER * TM * WNITER * TN] = {0.0f};

  float reg_M[WMITER * TM] = {0.0f};
  float reg_N[WNITER * TN] = {0.0f};

  for (std::uint32_t block_idx = 0; block_idx < K; block_idx += BK) {

    for (std::uint32_t offset = 0; offset + row_stride_A <= BM;
         offset += row_stride_A) {
      // Load A transpose into shared memory
      const float4 tmp = reinterpret_cast<const float4 *>(
          &A[(load_As_row + offset) * K + load_As_col * 4])[0];
      As[(load_As_col * 4 + 0) * BM + load_As_row + offset] = tmp.x;
      As[(load_As_col * 4 + 1) * BM + load_As_row + offset] = tmp.y;
      As[(load_As_col * 4 + 2) * BM + load_As_row + offset] = tmp.z;
      As[(load_As_col * 4 + 3) * BM + load_As_row + offset] = tmp.w;
    }

    for (std::uint32_t offset = 0; offset + row_stride_B <= BK;
         offset += row_stride_B) {

      reinterpret_cast<float4 *>(
          &Bs[(load_Bs_row + offset) * BN + load_Bs_col * 4])[0] =
          reinterpret_cast<const float4 *>(
              &B[(load_Bs_row + offset) * N + load_Bs_col * 4])[0];
    }

    __syncthreads();

    A += BK;     // BLOCK: advance by one tile to the right
    B += BK * N; // BLOCK: advance by one tile down.

    // Compute loop
    for (std::uint32_t k = 0; k < BK; ++k) {

      // Populate registers for a warptile.
      for (std::uint32_t w_sub_row_idx = 0; w_sub_row_idx < WMITER;
           ++w_sub_row_idx) {
        for (std::uint32_t i = 0; i < TM; ++i) {
          reg_M[w_sub_row_idx * TM + i] =
              As[(k * BM) + (warp_row * WM + w_sub_row_idx * WSUBM +
                             thread_row_in_warp * TM + i)];
        }
      }

      for (std::uint32_t w_sub_col_idx = 0; w_sub_col_idx < WNITER;
           ++w_sub_col_idx) {
        for (std::uint32_t i = 0; i < TN; ++i) {
          reg_N[w_sub_col_idx * TN + i] =
              Bs[k * BN + (warp_col * WN + w_sub_col_idx * WSUBN +
                           thread_col_in_warp * TN + i)];
        }
      }

      // Warptile comutation
      for (std::uint32_t w_sub_row_idx = 0; w_sub_row_idx < WMITER;
           ++w_sub_row_idx) {
        for (std::uint32_t w_sub_col_idx = 0; w_sub_col_idx < WNITER;
             ++w_sub_col_idx) {

          for (std::uint32_t result_idx_m = 0; result_idx_m < TM;
               ++result_idx_m) {
            for (std::uint32_t result_idx_n = 0; result_idx_n < TN;
                 ++result_idx_n) {

              thread_result[(w_sub_row_idx * TM + result_idx_m) *
                                (TN * WNITER) +
                            (w_sub_col_idx * TN) + result_idx_n] +=
                  reg_M[w_sub_row_idx * TM + result_idx_m] *
                  reg_N[w_sub_col_idx * TN + result_idx_n];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  if (beta == 0.0f) {
    // TODO: add code here.
  } else {

    for (std::uint32_t w_sub_row_idx = 0; w_sub_row_idx < WMITER;
         ++w_sub_row_idx) {
      for (std::uint32_t w_sub_col_idx = 0; w_sub_col_idx < WNITER;
           ++w_sub_col_idx) {

        float *C_interim =
            C + (w_sub_row_idx * WSUBM) * N + (w_sub_col_idx * WSUBN);

        for (std::uint32_t result_idx_m = 0; result_idx_m < TM;
             ++result_idx_m) {
          for (std::uint32_t result_idx_n = 0; result_idx_n < TN;
               result_idx_n += 4) {

            float4 tmp = reinterpret_cast<float4 *>(
                &C_interim[(thread_row_in_warp * TM + result_idx_m) * N +
                           (thread_col_in_warp * TN + result_idx_n)])[0];

            const std::uint32_t i =
                (w_sub_row_idx * TM + result_idx_m) * (WNITER * TN) +
                w_sub_col_idx * TN + result_idx_n;

            tmp.x = alpha * thread_result[i + 0] + beta * tmp.x;
            tmp.y = alpha * thread_result[i + 1] + beta * tmp.y;
            tmp.z = alpha * thread_result[i + 2] + beta * tmp.z;
            tmp.w = alpha * thread_result[i + 3] + beta * tmp.w;

            reinterpret_cast<float4 *>(
                &C_interim[(thread_row_in_warp * TM + result_idx_m) * N +
                           thread_col_in_warp * TN + result_idx_n])[0] = tmp;
          }
        }
      }
    }
  }
}

void kernels::shared_mem_vec_warp(std::uint32_t M, std::uint32_t N,
                                  std::uint32_t K, float alpha, const float *A,
                                  const float *B, float beta, float *C) {
  constexpr int BM = 128, BK = 8, BN = 128, TM = 8, TN = 8;
  constexpr int THREADS = (BN * BM) / (TM * TN);

  dim3 block(THREADS);
  dim3 grid(ceil_div(N, BN), ceil_div(M, BM));

  shared_mem_vec_warp_kernel<BM, BK, BN, TM, TN>
      <<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
  CUDA_CHECK_LAUNCH();
}
