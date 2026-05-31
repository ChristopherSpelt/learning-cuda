#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels/sgemm.h"
#include "loads.cuh"

namespace cul {
namespace {
// clang-format off
// Tile shape:  Rectangular block tiles; As is BM x BK and Bs is BK x BN. As is stored
//              transposed in shared memory so that the inner-k load is coalesced.
// Load:        One-shot at float4 granularity; each thread fills one float4 of As and
//              one of Bs per sweep.
//              NUM_THREADS = (BM * BN) / (TM * TN);
//              NUM_THREADS * 4 == BM * BK == BK * BN.
// Output:      Each thread computes a TM x TN thread-tile of C; epilogue stores via float4.
// Symmetry:    One-shot float4 load re-imposes BM == BN. Adds float4 alignment requirements on
//              BK, BN and TN.
// Bounds:      Handles non-aligned M/N/K via the BoundsCheck template parameter —
//              loads zero-fill out-of-range float4s; stores skip threads past the
//              matrix edge. Aligned inputs pay zero runtime cost.
// Requires:    K % 4 == 0 and N % 4 == 0 (asserted at launcher) for safe float4
//              alignment.
// clang-format on
template <int BM, int BK, int BN, int TM, int TN, bool BetaIsZero, bool BoundsCheck>
__global__ void vec_loads_kernel(int M, int N, int K, float alpha, const float *__restrict__ A,
                                 const float *__restrict__ B, float beta, float *__restrict__ C) {

  // ---- Compile-time invariants ----------------------------------------------
  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

  static_assert(BM == BN, "single sweep float4 load requires BM == BN");
  static_assert(4 * NUM_THREADS == BM * BK,
                "NUM_THREADS must equal (BM*BK)/4 (As float4 slot count)");
  static_assert(BK % 4 == 0, "BK must be a multiple of 4 for float4 As load");
  static_assert(BN % 4 == 0, "BN must be a multiple of 4 for float4 Bs load");
  static_assert(TN % 4 == 0, "TN must be a multiple of 4 for float4 epilogue stores");

  // ---- Prologue: tile coordinates, pointer offsets, register init -----------
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

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float thread_result[TM * TN] = {0.0f};

  float reg_M[TM] = {0.0f};
  float reg_N[TN] = {0.0f};

  const int A_row_global = block_row * BM + load_As_row;
  const int B_col_global = block_col * BN + load_Bs_col * 4;

  // ---- Main loop: K-tile iteration ------------------------------------------
  for (int k_tile = 0; k_tile < K; k_tile += BK) {
    const int A_col_global = k_tile + load_As_col * 4;
    const bool A_in = A_row_global < M && A_col_global < K;

    // Load As (transposed) — out-of-range float4s zero-fill; 0 is the GEMM identity.
    const float4 tmp =
        loads::masked_load_f4<BoundsCheck>(&A[load_As_row * K + load_As_col * 4], A_in);
    As[(load_As_col * 4 + 0) * BM + load_As_row] = tmp.x;
    As[(load_As_col * 4 + 1) * BM + load_As_row] = tmp.y;
    As[(load_As_col * 4 + 2) * BM + load_As_row] = tmp.z;
    As[(load_As_col * 4 + 3) * BM + load_As_row] = tmp.w;

    const int B_row_global = k_tile + load_Bs_row;
    const bool B_in = B_row_global < K && B_col_global < N;

    // Load Bs — out-of-range float4s zero-fill; 0 is the GEMM identity.
    reinterpret_cast<float4 *>(&Bs[load_Bs_row * BN + load_Bs_col * 4])[0] =
        loads::masked_load_f4<BoundsCheck>(&B[load_Bs_row * N + load_Bs_col * 4], B_in);

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

      for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
        for (int res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {

          thread_result[res_idx_m * TN + res_idx_n] += reg_M[res_idx_m] * reg_N[res_idx_n];
        }
      }
    }
    __syncthreads();
  }

  // ---- Epilogue: αAB + βC store --------------------------------------------
  for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
    const int C_row_global = block_row * BM + tile_row * TM + res_idx_m;

    for (int res_idx_n = 0; res_idx_n < TN; res_idx_n += 4) {

      const int C_col_global = block_col * BN + tile_col * TN + res_idx_n;

      if constexpr (BoundsCheck) {
        if (!(C_row_global < M && C_col_global < N))
          continue;
      }

      float4 product;
      product.x = alpha * thread_result[res_idx_m * TN + res_idx_n + 0];
      product.y = alpha * thread_result[res_idx_m * TN + res_idx_n + 1];
      product.z = alpha * thread_result[res_idx_m * TN + res_idx_n + 2];
      product.w = alpha * thread_result[res_idx_m * TN + res_idx_n + 3];

      auto *destination = reinterpret_cast<float4 *>(
          &C[(tile_row * TM + res_idx_m) * N + tile_col * TN + res_idx_n]);
      epilogue::store_result<BetaIsZero>(destination, product, beta);
    }
  }
}
} // namespace

void kernels::sgemm::vec_loads(const SgemmArgs &a) {
  CUL_REQUIRE(a.K % 4 == 0, "vec_loads requires K % 4 == 0 for float4 alignment");
  CUL_REQUIRE(a.N % 4 == 0, "vec_loads requires N % 4 == 0 for float4 alignment");

  constexpr int BM = 128, BK = 8, BN = 128, TM = 8, TN = 8;
  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

  const bool tile_aligned = (a.M % BM == 0) && (a.K % BK == 0) && (a.N % BN == 0);
  const bool beta_is_zero = a.beta == 0.0f;

  dim3 block(NUM_THREADS);
  dim3 grid(cuda_utils::ceil_div(a.N, BN), cuda_utils::ceil_div(a.M, BM));

  cuda_utils::dispatch_bool(beta_is_zero, [&](auto beta_zero) {
    cuda_utils::dispatch_bool(!tile_aligned, [&](auto bounds_check) {
      constexpr bool kBetaZero = decltype(beta_zero)::value;
      constexpr bool kBoundsCheck = decltype(bounds_check)::value;
      vec_loads_kernel<BM, BK, BN, TM, TN, kBetaZero, kBoundsCheck>
          <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
    });
  });
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
