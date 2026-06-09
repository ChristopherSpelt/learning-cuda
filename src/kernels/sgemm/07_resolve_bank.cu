#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels/sgemm.h"
#include "loads.cuh"

namespace cul {
namespace {

constexpr int WARPSIZE = 32;

// clang-format off
// Warp tiling: partition the block tile into per-warp tiles/sub-tiles. Tackles
// warp-level register reuse and ILP, the structural base the pipelines build on.
//
// Tile shape:  Block tile BM x BN partitioned into warp-tiles WM x WN, each
//              partitioned into WMITER x WNITER subtiles of WSUBM x WSUBN
//              (WSUBM = WM/WMITER, WSUBN = WN/WNITER). As stored transposed in
//              shared memory.
// Load:        Strided cooperative at float4 granularity; each thread loads
//              (BM*BK)/(NUM_THREADS*4) float4s of As and
//              (BK*BN)/(NUM_THREADS*4) float4s of Bs per sweep.
// Output:      Each thread computes WMITER x WNITER subtiles of TM x TN;
//              epilogue stores via float4.
// Symmetry:    Strided float4 load breaks BM == BN. WSUBM x WSUBN is sized so
//              (WSUBM/TM) * (WSUBN/TN) == WARPSIZE (one warp per subtile).
//              Adds float4 alignment requirements on BK, BN, and TN.
// clang-format on
template <int BM, int BK, int BN, int WM, int WN, int WMITER, int WNITER, int TM, int TN,
          bool BetaIsZero, bool BoundsCheck>
__global__ void resolve_bank_kernel(int M, int N, int K, float alpha, const float *__restrict__ A,
                                 const float *__restrict__ B, float beta, float *__restrict__ C) {

  // ---- Compile-time invariants ----------------------------------------------
  constexpr int NUM_THREADS = (BM * BN) / (WMITER * WNITER * TM * TN);
  constexpr int NUM_WARPS = NUM_THREADS / WARPSIZE;

  constexpr int WSUBM = WM / WMITER;
  constexpr int WSUBN = WN / WNITER;
  constexpr int row_stride_A = (NUM_THREADS * 4) / BK;
  constexpr int row_stride_B = (NUM_THREADS * 4) / BN;

  static_assert(NUM_THREADS % WARPSIZE == 0, "NUM_THREADS must be a whole-warp count");
  static_assert(BM % WM == 0, "BM must be a multiple of WM");
  static_assert(BN % WN == 0, "BN must be a multiple of WN");
  static_assert((BM / WM) * (BN / WN) == NUM_WARPS, "warps must tile the block exactly");
  static_assert(WM % WMITER == 0, "WM must be divisible by WMITER");
  static_assert(WN % WNITER == 0, "WN must be divisible by WNITER");
  static_assert(WSUBM % TM == 0, "WSUBM must be a multiple of TM");
  static_assert(WSUBN % TN == 0, "WSUBN must be a multiple of TN");
  static_assert((WSUBM / TM) * (WSUBN / TN) == WARPSIZE,
                "thread tile must cover exactly one warp (WARPSIZE lanes)");
  static_assert(BK % 4 == 0, "BK must be divisible by 4 for float4 As load");
  static_assert(BN % 4 == 0, "BN must be divisible by 4 for float4 Bs load");
  static_assert(TN % 4 == 0, "TN must be divisible by 4 for float4 epilogue stores");
  static_assert((NUM_THREADS * 4) % BK == 0, "row_stride_A must be integer");
  static_assert((NUM_THREADS * 4) % BN == 0, "row_stride_B must be integer");

  // ---- Prologue: tile coordinates, pointer offsets, register init -----------
  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;

  const int warp_idx = threadIdx.x / WARPSIZE;
  const int warp_row = warp_idx / (BN / WN);
  const int warp_col = warp_idx % (BN / WN);

  const int thread_idx_in_warp = threadIdx.x % WARPSIZE;
  const int thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);
  const int thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);

  const int load_As_k = threadIdx.x % BK;       // K-column this thread stages
  const int load_As_m = (threadIdx.x / BK) * 4; // Start of its 4-wide M-strip.

  const int load_Bs_row = threadIdx.x / (BN / 4);
  const int load_Bs_col = threadIdx.x % (BN / 4);

  A += block_row * BM * K;
  B += block_col * BN;
  C += (block_row * BM + warp_row * WM) * N + block_col * BN + warp_col * WN;

  __shared__ float As[BK * (BM + 4)];
  __shared__ float Bs[BK * BN];

  float thread_result[WMITER * TM * WNITER * TN] = {0.0f};

  float reg_M[WMITER * TM] = {0.0f};
  float reg_N[WNITER * TN] = {0.0f};

  const int B_col_global = block_col * BN + load_Bs_col * 4;

  // ---- Main loop: K-tile iteration ------------------------------------------
  for (int k_tile = 0; k_tile < K; k_tile += BK) {

    const int A_col_global = k_tile + load_As_k;

    // Load (As transposed)
    for (int offset = 0; offset + row_stride_A <= BM; offset += row_stride_A) {
      const int m = load_As_m + offset;
      float r[4];

#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int A_row_global = block_row * BM + m + i;
        const bool A_in = A_row_global < M && A_col_global < K;

        r[i] = loads::masked_load<BoundsCheck>(&A[(m + i) * K + load_As_k], A_in);
      }
      reinterpret_cast<float4 *>(&As[load_As_k * (BM + 4) + m])[0] =
          make_float4(r[0], r[1], r[2], r[3]);
    }

    for (int offset = 0; offset + row_stride_B <= BK; offset += row_stride_B) {

      const int B_row_global = k_tile + load_Bs_row + offset;
      const bool B_in = B_row_global < K && B_col_global < N;

      reinterpret_cast<float4 *>(&Bs[(load_Bs_row + offset) * BN + load_Bs_col * 4])[0] =
          loads::masked_load_f4<BoundsCheck>(&B[(load_Bs_row + offset) * N + load_Bs_col * 4],
                                             B_in);
    }

    __syncthreads();

    A += BK;
    B += BK * N;

    // Compute loop
    for (int k = 0; k < BK; ++k) {

      // Populate registers for a warptile.
      for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
        const int As_base =
            k * (BM + 4) + (warp_row * WM + w_sub_row_idx * WSUBM + thread_row_in_warp * TM);

#pragma unroll
        for (int i = 0; i < TM; i += 4) {
          reinterpret_cast<float4 *>(&reg_M[w_sub_row_idx * TM + i])[0] =
              reinterpret_cast<const float4 *>(&As[As_base + i])[0];
        }
      }

      for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {
        for (int i = 0; i < TN; ++i) {
          reg_N[w_sub_col_idx * TN + i] =
              Bs[k * BN + (warp_col * WN + w_sub_col_idx * WSUBN + thread_col_in_warp * TN + i)];
        }
      }

      // Warptile computation
      for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
        for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {

          for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
            for (int res_idx_n = 0; res_idx_n < TN; ++res_idx_n) {

              thread_result[(w_sub_row_idx * TM + res_idx_m) * (TN * WNITER) +
                            (w_sub_col_idx * TN) + res_idx_n] +=
                  reg_M[w_sub_row_idx * TM + res_idx_m] * reg_N[w_sub_col_idx * TN + res_idx_n];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // ---- Epilogue: αAB + βC store --------------------------------------------
  for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
    for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {

      float *C_interim = C + (w_sub_row_idx * WSUBM) * N + (w_sub_col_idx * WSUBN);

      for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m) {
        const int C_row_global = block_row * BM + warp_row * WM + w_sub_row_idx * WSUBM +
                                 thread_row_in_warp * TM + res_idx_m;

        for (int res_idx_n = 0; res_idx_n < TN; res_idx_n += 4) {
          const int C_col_global = block_col * BN + warp_col * WN + w_sub_col_idx * WSUBN +
                                   thread_col_in_warp * TN + res_idx_n;

          if constexpr (BoundsCheck) {
            if (!(C_row_global < M && C_col_global < N))
              continue;
          }

          const int flat_idx =
              (w_sub_row_idx * TM + res_idx_m) * (WNITER * TN) + w_sub_col_idx * TN + res_idx_n;

          float4 product;
          product.x = alpha * thread_result[flat_idx + 0];
          product.y = alpha * thread_result[flat_idx + 1];
          product.z = alpha * thread_result[flat_idx + 2];
          product.w = alpha * thread_result[flat_idx + 3];

          auto *destination =
              reinterpret_cast<float4 *>(&C_interim[(thread_row_in_warp * TM + res_idx_m) * N +
                                                    thread_col_in_warp * TN + res_idx_n]);

          epilogue::store_result<BetaIsZero>(destination, product, beta);
        }
      }
    }
  }
}
} // namespace

void kernels::sgemm::resolve_bank(const SgemmArgs &a) {
  CUL_REQUIRE(a.K % 4 == 0, "warp_tile requires K % 4 == 0 for float4 alignment");
  CUL_REQUIRE(a.N % 4 == 0, "warp_tile requires N % 4 == 0 for float4 alignment");
  constexpr int BM = 128, BK = 16, BN = 128, TM = 8, TN = 4;
  constexpr int WM = 64, WN = 64, WMITER = 1, WNITER = 4;
  constexpr int NUM_THREADS = (BM * BN) / (WMITER * WNITER * TM * TN);

  const bool tile_aligned = (a.M % BM == 0) && (a.K % BK == 0) && (a.N % BN == 0);
  const bool beta_is_zero = a.beta == 0.0f;

  dim3 block(NUM_THREADS);
  dim3 grid(cuda_utils::ceil_div(a.N, BN), cuda_utils::ceil_div(a.M, BM));

  cuda_utils::dispatch_bool(beta_is_zero, [&](auto beta_zero) {
    cuda_utils::dispatch_bool(!tile_aligned, [&](auto bounds_check) {
      constexpr bool kBetaZero = decltype(beta_zero)::value;
      constexpr bool kBoundsCheck = decltype(bounds_check)::value;
      resolve_bank_kernel<BM, BK, BN, WM, WN, WMITER, WNITER, TM, TN, kBetaZero, kBoundsCheck>
          <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
    });
  });
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
