#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels/sgemm.h"
#include "loads.cuh"

#include <cuda_pipeline.h>

namespace cul {
namespace {

constexpr int WARPSIZE = 32;

// clang-format off
// cp.async load: copy global->shared directly, no register staging. Tackles load
// latency, but LOST to double_buffer (the transposed-As scatter forces scalar copies).
//
// Tile shape:  Block tile BM x BN partitioned into warp-tiles WM x WN, each
//              partitioned into WMITER x WNITER subtiles of WSUBM x WSUBN
//              (WSUBM = WM/WMITER, WSUBN = WN/WNITER). As stored transposed in
//              shared memory.
// Load:        Strided cooperative global->shared via cp.async. Bs: one float4
//              (16 B) cp.async per slot. As: stored transposed, so it can't be
//              a single float4 — 16 scalar 4-byte cp.async per thread instead
//              (see Pipeline for why that cost it).
// Pipeline:    cp.async (__pipeline_memcpy_async) copies global->shared directly,
//              bypassing 07's register staging (no LDG->reg->STS); a 2-stage
//              pipeline (wait_prior/commit) overlaps the next tile's copy with
//              the current compute. VERDICT — this LOST to 07: cp.async copies
//              only CONTIGUOUS chunks, so the transposed As scatter becomes 16
//              scalar 4-byte cp.async/thread (the "transpose tax") vs Bs's clean
//              float4, and that load overhead outweighed the staging it saved.
// Output:      Each thread computes WMITER x WNITER subtiles of TM x TN;
//              epilogue stores via float4.
// Symmetry:    Strided float4 load breaks BM == BN. WSUBM x WSUBN is sized so
//              (WSUBM/TM) * (WSUBN/TN) == WARPSIZE (one warp per subtile).
//              Adds float4 alignment requirements on BK, BN, and TN.
// Bounds:      Aligned sizes ONLY. BoundsCheck is applied in the epilogue, but
//              the cp.async loads do no masking, so non-128-multiple sizes read
//              out of bounds. (Benchmark sizes are all multiples of 128.)
// clang-format on
template <int BM, int BK, int BN, int WM, int WN, int WMITER, int WNITER, int TM, int TN,
          bool BetaIsZero, bool BoundsCheck>
__global__ void cp_async_kernel(int M, int N, int K, float alpha, const float *__restrict__ A,
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

  const int load_As_row = threadIdx.x / (BK / 4);
  const int load_As_col = threadIdx.x % (BK / 4);

  const int load_Bs_row = threadIdx.x / (BN / 4);
  const int load_Bs_col = threadIdx.x % (BN / 4);

  A += block_row * BM * K;
  B += block_col * BN;
  C += (block_row * BM + warp_row * WM) * N + block_col * BN + warp_col * WN;

  __shared__ float As[2][BM * BK];
  __shared__ float Bs[2][BK * BN];

  float thread_result[WMITER * TM * WNITER * TN] = {0.0f};

  float reg_M[WMITER * TM] = {0.0f};
  float reg_N[WNITER * TN] = {0.0f};

  // Prologue: kick off tile 0 into buffer 0 via cp.async (no wait here — the
  // loop's first wait_prior handles it). As stored transposed.
  for (int offset = 0; offset + row_stride_A <= BM; offset += row_stride_A) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      __pipeline_memcpy_async(&As[0][(load_As_col * 4 + c) * BM + load_As_row + offset],
                              &A[(load_As_row + offset) * K + load_As_col * 4 + c], sizeof(float));
    }
  }

  for (int offset = 0; offset + row_stride_B <= BK; offset += row_stride_B) {
    __pipeline_memcpy_async(&Bs[0][(load_Bs_row + offset) * BN + load_Bs_col * 4],
                            &B[(load_Bs_row + offset) * N + load_Bs_col * 4], sizeof(float4));
  }

  __pipeline_commit();
  A += BK;
  B += BK * N;

  int cur = 0;
  // ---- Main loop: K-tile iteration ------------------------------------------
  for (int k_tile = 0; k_tile < K; k_tile += BK) {

    const int next = cur ^ 1;
    const bool has_next = (k_tile + BK) < K;

    __pipeline_wait_prior(0);
    __syncthreads();

    if (has_next) {
      // Prefetch the next tile into the other buffer (overlaps the compute below).
      // As stored transposed.
      for (int offset = 0; offset + row_stride_A <= BM; offset += row_stride_A) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          __pipeline_memcpy_async(&As[next][(load_As_col * 4 + c) * BM + load_As_row + offset],
                                  &A[(load_As_row + offset) * K + load_As_col * 4 + c],
                                  sizeof(float));
        }
      }

      for (int offset = 0; offset + row_stride_B <= BK; offset += row_stride_B) {
        __pipeline_memcpy_async(&Bs[next][(load_Bs_row + offset) * BN + load_Bs_col * 4],
                                &B[(load_Bs_row + offset) * N + load_Bs_col * 4], sizeof(float4));
      }

      __pipeline_commit();
      A += BK;
      B += BK * N;
    }

    // Compute loop
    for (int k = 0; k < BK; ++k) {

      // Populate registers for a warptile.
      for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
        for (int i = 0; i < TM; ++i) {
          reg_M[w_sub_row_idx * TM + i] =
              As[cur]
                [(k * BM) + (warp_row * WM + w_sub_row_idx * WSUBM + thread_row_in_warp * TM + i)];
        }
      }

      for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {
        for (int i = 0; i < TN; ++i) {
          reg_N[w_sub_col_idx * TN + i] =
              Bs[cur]
                [k * BN + (warp_col * WN + w_sub_col_idx * WSUBN + thread_col_in_warp * TN + i)];
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
    cur = next;
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

void kernels::sgemm::cp_async(const SgemmArgs &a) {
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
      cp_async_kernel<BM, BK, BN, WM, WN, WMITER, WNITER, TM, TN, kBetaZero, kBoundsCheck>
          <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
    });
  });
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
