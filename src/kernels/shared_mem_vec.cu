#include "cuda_utils.cuh"
#include "kernels.h"

// clang-format off
// Tile shape:  rectangular block tiles; As is BM x BK and Bs is BK x BN. As is stored
//              transposed in shared memory so that the inner-k load is coalesced.
// Load:        one-shot at float4 granularity; each thread fills one float4 of As and
//              one of Bs per sweep.
//              NUM_THREADS = (BM * BN) / (TM * TN);
//              NUM_THREADS * 4 == BM * BK == BK * BN.
// Output:      each thread computes a TM x TN thread-tile of C; epilogue stores via float4.
// Symmetry:    one-shot float4 load re-imposes BM == BN. Adds float4 alignment requirements on
//              BK, BN and TN.
// clang-format on
template <int BM, int BK, int BN, int TM, int TN>
__global__ void shared_mem_vec_kernel(int M, int N, int K, float alpha,
                                      const float *__restrict__ A,
                                      const float *__restrict__ B, float beta,
                                      float *__restrict__ C) {

  // ---- Compile-time invariants ----------------------------------------------
  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

  static_assert(BM == BN, "single sweep float4 load requires BM == BN");
  static_assert(4 * NUM_THREADS == BM * BK,
                "NUM_THREADS must equal (BM*BK)/4 (As float4 slot count)");
  static_assert(BK % 4 == 0, "BK must be a multiple of 4 for float4 As load");
  static_assert(BN % 4 == 0, "BN must be a multiple of 4 for float4 Bs load");
  static_assert(TN % 4 == 0,
                "TN must be a multiple of 4 for float4 epilogue stores");

  // ---- Prologue: tile coordinates, pointer offsets, register init -----------
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;

  const int tile_row = threadIdx.x / (BN / TN);
  const int tile_col = threadIdx.x % (BN / TN);

  const int load_As_row = threadIdx.x / (BK / 4);
  const int load_As_col = threadIdx.x % (BK / 4);
  const int load_Bs_row = threadIdx.x / (BN / 4);
  const int load_Bs_col = threadIdx.x % (BN / 4);

  A += block_row * BM * K;
  B += block_col * BN;
  C += block_row * BM * N + block_col * BN;

  float thread_result[TM * TN] = {0.0f};

  float reg_M[TM] = {0.0f};
  float reg_N[TN] = {0.0f};

  // ---- Main loop: K-tile iteration ------------------------------------------
  for (int block_idx = 0; block_idx < K; block_idx += BK) {

    // Load (As transposed)
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
    A += BK;
    B += BK * N;

    // Compute loop
    for (int k = 0; k < BK; ++k) {

      for (int i = 0; i < TM; ++i) {
        reg_M[i] = As[k * BM + (tile_row * TM + i)];
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
  if (beta == 0.0f) {
    // β=0: pure write, no read of C
    for (int result_idx_m = 0; result_idx_m < TM; ++result_idx_m) {
      for (int result_idx_n = 0; result_idx_n < TN; result_idx_n += 4) {

        float4 tmp;
        tmp.x = alpha * thread_result[result_idx_m * TN + result_idx_n + 0];
        tmp.y = alpha * thread_result[result_idx_m * TN + result_idx_n + 1];
        tmp.z = alpha * thread_result[result_idx_m * TN + result_idx_n + 2];
        tmp.w = alpha * thread_result[result_idx_m * TN + result_idx_n + 3];

        reinterpret_cast<float4 *>(&C[(tile_row * TM + result_idx_m) * N +
                                      tile_col * TN + result_idx_n])[0] = tmp;
      }
    }

  } else {
    // β≠0: linear combination αAB + βC
    for (int result_idx_m = 0; result_idx_m < TM; ++result_idx_m) {
      for (int result_idx_n = 0; result_idx_n < TN; result_idx_n += 4) {

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

void kernels::shared_mem_vec(const GemmArgs &a) {
  constexpr int BM = 128, BK = 8, BN = 128, TM = 8, TN = 8;
  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

  dim3 block(NUM_THREADS);
  dim3 grid(ceil_div(a.N, BN), ceil_div(a.M, BM));

  shared_mem_vec_kernel<BM, BK, BN, TM, TN>
      <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  CUDA_CHECK_LAUNCH();
}
