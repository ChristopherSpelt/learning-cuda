#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels/sgemm.h"
#include "loads.cuh"

namespace cul {
namespace {

constexpr int WARPSIZE = 32;

// clang-format off
// Inner-loop (register) prefetch: register double-buffer the shared->register
// loads. Tackles LDS latency: hidden under the FMAs; single shared buffer.
//
// Tile shape:  Block tile BM x BN partitioned into warp-tiles WM x WN, each
//              partitioned into WMITER x WNITER subtiles of WSUBM x WSUBN
//              (WSUBM = WM/WMITER, WSUBN = WN/WNITER). As stored transposed in
//              shared memory.
// Load:        Strided cooperative at float4 granularity; each thread loads
//              (BM*BK)/(NUM_THREADS*4) float4s of As and
//              (BK*BN)/(NUM_THREADS*4) float4s of Bs per sweep.
// Pipeline:    Register double-buffer (software pipelined at the inner loop):
//              the shared->register loads (reg_M/reg_N) overlap with the FMAs.
//              Sibling of 07 one level down — 07 pipelines global->shared; this
//              pipelines shared->register. Shared memory stays SINGLE-buffered
//              (As/Bs, no [2]); only the registers are double-buffered, and no
//              staging registers are needed since the prefetch never leaves the
//              register file. Each k computes the current bank while prefetching
//              slice k+1 into the other. Uses two NAMED banks per operand
//              (reg_M_a/_b, reg_N_a/_b) + an unroll-by-2 role-swap, NOT a
//              reg[2][..] array indexed by k&1: registers aren't runtime-
//              addressable, so a dynamically-indexed register array is demoted
//              to local memory (a spill) — which made it 5x slower.
// Output:      Each thread computes WMITER x WNITER subtiles of TM x TN;
//              epilogue stores via float4.
// Symmetry:    Strided float4 load breaks BM == BN. WSUBM x WSUBN is sized so
//              (WSUBM/TM) * (WSUBN/TN) == WARPSIZE (one warp per subtile).
//              Adds float4 alignment requirements on BK, BN, and TN.
// clang-format on
template <int BM, int BK, int BN, int WM, int WN, int WMITER, int WNITER, int TM, int TN,
          bool BetaIsZero, bool BoundsCheck>
__global__ void inner_loop_prefetch_kernel(int M, int N, int K, float alpha,
                                           const float *__restrict__ A, const float *__restrict__ B,
                                           float beta, float *__restrict__ C) {

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

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float thread_result[WMITER * TM * WNITER * TN] = {0.0f};

  float reg_M_a[WMITER * TM]{};
  float reg_M_b[WMITER * TM]{};

  float reg_N_a[WNITER * TN]{};
  float reg_N_b[WNITER * TN]{};

  const int B_col_global = block_col * BN + load_Bs_col * 4;

  // ---- Main loop: K-tile iteration ------------------------------------------
  for (int k_tile = 0; k_tile < K; k_tile += BK) {
    const int A_col_global = k_tile + load_As_col * 4;

    // Load (As transposed)
    for (int offset = 0; offset + row_stride_A <= BM; offset += row_stride_A) {

      const int A_row_global = block_row * BM + load_As_row + offset;
      const bool A_in = A_row_global < M && A_col_global < K;

      const float4 tmp = loads::masked_load_f4<BoundsCheck>(
          &A[(load_As_row + offset) * K + load_As_col * 4], A_in);
      As[(load_As_col * 4 + 0) * BM + load_As_row + offset] = tmp.x;
      As[(load_As_col * 4 + 1) * BM + load_As_row + offset] = tmp.y;
      As[(load_As_col * 4 + 2) * BM + load_As_row + offset] = tmp.z;
      As[(load_As_col * 4 + 3) * BM + load_As_row + offset] = tmp.w;
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

    // Populate registers reg_M[0][...] and reg_N[0][...]
    for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
      for (int i = 0; i < TM; ++i) {
        reg_M_a[w_sub_row_idx * TM + i] =
            As[(warp_row * WM + w_sub_row_idx * WSUBM + thread_row_in_warp * TM + i)];
      }
    }
    for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {
      for (int i = 0; i < TN; ++i) {
        reg_N_a[w_sub_col_idx * TN + i] =
            Bs[(warp_col * WN + w_sub_col_idx * WSUBN + thread_col_in_warp * TN + i)];
      }
    }

    // Compute loop
    //   #pragma unroll
    for (int k = 0; k < BK; k += 2) {

      // ---- phase A: prefetch k+1 -> _b, compute k from _a ----
      if (k + 1 < BK) {
        for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx)
          for (int i = 0; i < TM; ++i)
            reg_M_b[w_sub_row_idx * TM + i] =
                As[((k + 1) * BM) +
                   (warp_row * WM + w_sub_row_idx * WSUBM + thread_row_in_warp * TM + i)];
        for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx)
          for (int i = 0; i < TN; ++i)
            reg_N_b[w_sub_col_idx * TN + i] =
                Bs[(k + 1) * BN +
                   (warp_col * WN + w_sub_col_idx * WSUBN + thread_col_in_warp * TN + i)];
      }
      for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx)
        for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx)
          for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
            for (int res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
              thread_result[(w_sub_row_idx * TM + res_idx_m) * (TN * WNITER) +
                            (w_sub_col_idx * TN) + res_idx_n] +=
                  reg_M_a[w_sub_row_idx * TM + res_idx_m] * reg_N_a[w_sub_col_idx * TN + res_idx_n];

      // ---- phase B: prefetch k+2 -> _a, compute k+1 from _b ----
      if (k + 2 < BK) {
        for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx)
          for (int i = 0; i < TM; ++i)
            reg_M_a[w_sub_row_idx * TM + i] =
                As[((k + 2) * BM) +
                   (warp_row * WM + w_sub_row_idx * WSUBM + thread_row_in_warp * TM + i)];
        for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx)
          for (int i = 0; i < TN; ++i)
            reg_N_a[w_sub_col_idx * TN + i] =
                Bs[(k + 2) * BN +
                   (warp_col * WN + w_sub_col_idx * WSUBN + thread_col_in_warp * TN + i)];
      }
      if (k + 1 < BK) {
        for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx)
          for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx)
            for (int res_idx_m = 0; res_idx_m < TM; ++res_idx_m)
              for (int res_idx_n = 0; res_idx_n < TN; ++res_idx_n)
                thread_result[(w_sub_row_idx * TM + res_idx_m) * (TN * WNITER) +
                              (w_sub_col_idx * TN) + res_idx_n] +=
                    reg_M_b[w_sub_row_idx * TM + res_idx_m] *
                    reg_N_b[w_sub_col_idx * TN + res_idx_n];
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

void kernels::sgemm::inner_loop_prefetch(const SgemmArgs &a) {
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
      inner_loop_prefetch_kernel<BM, BK, BN, WM, WN, WMITER, WNITER, TM, TN, kBetaZero,
                                 kBoundsCheck>
          <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
    });
  });
  CUDA_CHECK_LAUNCH();
}

} // namespace cul
