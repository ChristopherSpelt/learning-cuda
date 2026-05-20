# BetaIsZero Compile-Time Dispatch Implementation Plan

**Goal:** Compile-time specialize every GEMM kernel on whether β=0, sharing one device‑side epilogue helper. The β=0 specialization drops the `*C` load entirely (a real DRAM saving in the float4 kernels), and the codebase ends up with one place to change the epilogue math instead of six.

**Architecture:**
1. One new header `include/epilogue.cuh` with two `__device__ __forceinline__` `store_result` overloads — one for `float*`, one for `float4*`. Both `if constexpr (BetaIsZero)` between "pure write" and "α·prod + β·C[idx]".
2. Every kernel grows `bool BetaIsZero` as its last template parameter and replaces its in-kernel `if (beta == 0.0f) … else …` epilogue with calls to `store_result<BetaIsZero>(…)`.
3. Every `kernels::xxx` launcher does an explicit `if (a.beta == 0.0f) … else …` to pick the specialization. Six near‑identical if/elses, by design — each kernel stays self‑contained at the launcher level.

**Tech Stack:** CUDA (C++20), CMake. Built/run on a remote GPU box — the user has no local CUDA toolchain, so build/verify steps are commands *they* run, not ones the agent runs.

**Verification model:** The codebase has no unit tests. `include/harness.h` runs each kernel twice per registry entry:
- **Correctness pass** with `kBeta = 0.5f` (harness.h:96) — exercises the `BetaIsZero=false` specialization, RMSE'd against cuBLAS.
- **Benchmark pass** with `beta = 0.0f` (harness.h:85) — exercises the `BetaIsZero=true` specialization, but is *not* RMSE‑checked.

So a normal run after each task validates the false specialization. Task 8 flips `kBeta` to 0 temporarily so the true specialization gets the same RMSE treatment end‑to‑end.

---

## File Structure

```
include/
  cuda_utils.cuh             (unchanged)
  epilogue.cuh               (CREATE — store_result overloads)
src/kernels/
  naive.cu                   (MODIFY — Task 2)
  shared_mem.cu              (MODIFY — Task 3)
  shared_mem_1d_block.cu     (MODIFY — Task 4)
  shared_mem_2d_block.cu     (MODIFY — Task 5)
  shared_mem_vec.cu          (MODIFY — Task 6)
  shared_mem_vec_warp_tile.cu (MODIFY — Task 7)
```

Tasks 2–7 all have the same shape: add one template parameter, replace the epilogue body, add an if/else in the launcher, build, run, verify the RMSE line for that kernel still passes, commit.

**Prerequisite note for Task 7:** `shared_mem_vec_warp_tile.cu`'s launcher currently references `WM`, `WN`, `WNITER` without declaring them in the launcher scope (line 193 — they're only the kernel's template params, not constexpr in the launcher). This kernel likely does not build today. If it doesn't, fix that separately (declare the three values, e.g. `constexpr int WM = 64, WN = 32, WNITER = 1;` — pick values satisfying the kernel's `static_assert`s) **before** doing Task 7. The BetaIsZero refactor does not fix this; it just lives on top of a working file.

---

## Task 1: Create the shared epilogue helper

**Files:**
- Create: `include/epilogue.cuh`

This is the only entirely new file. It's standalone — no behavior change yet, so this commit is safe in isolation.

- [ ] **Step 1: Create `include/epilogue.cuh` with the following contents:**

```cpp
#pragma once

#include <cuda_runtime.h>

// Apply αAB+βC and store, specialized at compile time on whether β=0.
// When BetaIsZero=true the load of *dst is elided entirely; that matters in
// the float4 kernels where it removes a real DRAM read per 4 outputs.
template <bool BetaIsZero>
__device__ __forceinline__ void
store_result(float *dst, float prod, [[maybe_unused]] float beta) {
  if constexpr (BetaIsZero) {
    *dst = prod;
  } else {
    *dst = prod + beta * (*dst);
  }
}

template <bool BetaIsZero>
__device__ __forceinline__ void
store_result(float4 *dst, float4 prod, [[maybe_unused]] float beta) {
  if constexpr (BetaIsZero) {
    *dst = prod;
  } else {
    float4 c = *dst;
    c.x = prod.x + beta * c.x;
    c.y = prod.y + beta * c.y;
    c.z = prod.z + beta * c.z;
    c.w = prod.w + beta * c.w;
    *dst = c;
  }
}
```

Two overloads, not one template on `T` — the scalar and float4 paths are honestly different operations (scalar FMA vs vectorized read‑modify‑write). `[[maybe_unused]]` suppresses unused‑parameter warnings in the `BetaIsZero=true` instantiation under `-Wall -Wextra`.

- [ ] **Step 2: Build on your GPU box.**

```
cmake -B build -S .
cmake --build build -j
```

Expected: build succeeds. No file consumes `epilogue.cuh` yet, so the only thing being checked here is that the header itself compiles cleanly when the CMake configure step picks it up later. (If you want a stronger check, add a single `#include "epilogue.cuh"` to `src/main.cu` temporarily — but this is overkill; Task 2 will pull it in for real.)

- [ ] **Step 3: Commit.**

```
git add include/epilogue.cuh
git commit -m "add store_result device helper for αAB+βC epilogue"
```

---

## Task 2: Refactor `naive.cu`

**Files:**
- Modify: `src/kernels/naive.cu`

- [ ] **Step 1: Replace `src/kernels/naive.cu` with the following:**

```cpp
#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels.h"

// clang-format off
// Tile shape:  none (no shared memory).
// Load:        direct global reads inside the inner-k loop; no cooperative load.
// Output:      each thread computes exactly one element of C.
// Symmetry:    no tile invariants; BLOCKSIZE has no restrictions.
// clang-format on
template <int BLOCKSIZE, bool BetaIsZero>
__global__ void
naive_kernel(int M, int N, int K, float alpha, const float *__restrict__ A,
             const float *__restrict__ B, float beta, float *__restrict__ C) {

  // ---- Prologue: thread coordinates -----------------------------------------
  const int row = blockIdx.y * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
  const int col = blockIdx.x * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

  if (row >= M || col >= N)
    return;

  float sum = 0.0f;

  // ---- Main loop ------------------------------------------------------------
  for (int k = 0; k < K; ++k) {
    sum += A[row * K + k] * B[k * N + col];
  }

  // ---- Epilogue: αAB + βC store --------------------------------------------
  store_result<BetaIsZero>(&C[row * N + col], alpha * sum, beta);
}

void kernels::naive(const GemmArgs &a) {
  constexpr int BLOCKSIZE = 32;

  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid(ceil_div(a.N, BLOCKSIZE), ceil_div(a.M, BLOCKSIZE));

  if (a.beta == 0.0f) {
    naive_kernel<BLOCKSIZE, true>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  } else {
    naive_kernel<BLOCKSIZE, false>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  }
  CUDA_CHECK_LAUNCH();
}
```

What changed: `bool BetaIsZero` template param added; the in-kernel `if (beta == 0.0f) … else …` collapsed to a single `store_result<BetaIsZero>(…)` call; the launcher does the runtime→compile‑time dispatch via `if (a.beta == 0.0f)`.

- [ ] **Step 2: Build.**

```
cmake --build build -j
```

Expected: build succeeds.

- [ ] **Step 3: Run and verify.**

```
./build/cuda_learning
```

Look for the `naive` line in the output. Expected shape:

```
naive                   rel RMSE 1.xxe-05   xxx.xxx ms   x.xx TFLOPS
```

The RMSE should be similar to what you saw before this change (i.e. small — under `kTolerance = 1e-3`). TFLOPS should also be similar (this refactor doesn't change the inner loop). If RMSE printed `FAILED`, the `BetaIsZero=false` epilogue is wrong — re‑check the `else` branch in `store_result` and the call site.

- [ ] **Step 4: Commit.**

```
git add src/kernels/naive.cu
git commit -m "naive: compile-time dispatch on BetaIsZero via store_result"
```

---

## Task 3: Refactor `shared_mem.cu`

**Files:**
- Modify: `src/kernels/shared_mem.cu`

- [ ] **Step 1: Apply two edits to `src/kernels/shared_mem.cu`.**

**Edit 1** — at the top of the file, add the `epilogue.cuh` include and change the template signature:

```cpp
#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels.h"
```

And:

```cpp
template <int BLOCKSIZE, bool BetaIsZero>
__global__ void shared_mem_kernel(int M, int N, int K, float alpha,
                                  const float *__restrict__ A,
                                  const float *__restrict__ B, float beta,
                                  float *__restrict__ C) {
```

**Edit 2** — replace the entire `// ---- Epilogue` block (lines 62‑74 in the current file) with:

```cpp
  // ---- Epilogue: αAB + βC store --------------------------------------------
  if (global_row < M && global_col < N) {
    store_result<BetaIsZero>(&C[thread_row * N + thread_col],
                             alpha * sum, beta);
  }
```

**Edit 3** — replace the launcher function with:

```cpp
void kernels::shared_mem(const GemmArgs &a) {
  constexpr int BLOCKSIZE = 32;

  dim3 block(BLOCKSIZE * BLOCKSIZE);
  dim3 grid(ceil_div(a.N, BLOCKSIZE), ceil_div(a.M, BLOCKSIZE));

  if (a.beta == 0.0f) {
    shared_mem_kernel<BLOCKSIZE, true>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  } else {
    shared_mem_kernel<BLOCKSIZE, false>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  }
  CUDA_CHECK_LAUNCH();
}
```

Note: the bounds check `if (global_row < M && global_col < N)` lifts out of both old branches into a single guard around the `store_result` call. The body inside the guard is now one line.

- [ ] **Step 2: Build, run, verify the `shared mem` line.**

```
cmake --build build -j
./build/cuda_learning
```

Look for:

```
shared mem              rel RMSE 1.xxe-05   xxx.xxx ms   x.xx TFLOPS
```

Same expectation: RMSE small, TFLOPS unchanged vs. baseline.

- [ ] **Step 3: Commit.**

```
git add src/kernels/shared_mem.cu
git commit -m "shared_mem: compile-time dispatch on BetaIsZero via store_result"
```

---

## Task 4: Refactor `shared_mem_1d_block.cu`

**Files:**
- Modify: `src/kernels/shared_mem_1d_block.cu`

- [ ] **Step 1: Apply edits.**

**Edit 1** — add include + change template signature:

```cpp
#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels.h"
```

```cpp
template <int BM, int BK, int BN, int TM, bool BetaIsZero>
__global__ void shared_mem_1d_block_kernel(int M, int N, int K, float alpha,
                                           const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float beta, float *__restrict__ C) {
```

**Edit 2** — replace the `// ---- Epilogue` block (current lines 76‑100) with:

```cpp
  // ---- Epilogue: αAB + βC store --------------------------------------------
  for (int res_idx = 0; res_idx < TM; ++res_idx) {
    const int C_row_global = block_row * BM + strip_row * TM + res_idx;
    const int C_col_global = block_col * BN + strip_col;

    if (C_row_global < M && C_col_global < N) {
      store_result<BetaIsZero>(
          &C[(strip_row * TM + res_idx) * N + strip_col],
          alpha * thread_result[res_idx], beta);
    }
  }
```

The two old loops (β=0 and β≠0) collapse to one. The bounds check stays per‑iteration because each `res_idx` writes a different global row.

**Edit 3** — replace the launcher:

```cpp
void kernels::shared_mem_1d_block(const GemmArgs &a) {
  constexpr int BM = 64, BK = 8, BN = 64, TM = 8;
  constexpr int NUM_THREADS = BN * BM / TM;

  dim3 block(NUM_THREADS);
  dim3 grid(ceil_div(a.N, BN), ceil_div(a.M, BM));

  if (a.beta == 0.0f) {
    shared_mem_1d_block_kernel<BM, BK, BN, TM, true>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  } else {
    shared_mem_1d_block_kernel<BM, BK, BN, TM, false>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  }
  CUDA_CHECK_LAUNCH();
}
```

- [ ] **Step 2: Build, run, verify the `shared mem 1D block` line.**

```
cmake --build build -j
./build/cuda_learning
```

Look for:

```
shared mem 1D block     rel RMSE 1.xxe-05   xxx.xxx ms   x.xx TFLOPS
```

- [ ] **Step 3: Commit.**

```
git add src/kernels/shared_mem_1d_block.cu
git commit -m "shared_mem_1d_block: compile-time dispatch on BetaIsZero via store_result"
```

---

## Task 5: Refactor `shared_mem_2d_block.cu`

**Files:**
- Modify: `src/kernels/shared_mem_2d_block.cu`

- [ ] **Step 1: Apply edits.**

**Edit 1** — add include + change template signature:

```cpp
#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels.h"
```

```cpp
template <int BM, int BK, int BN, int TM, int TN, bool BetaIsZero>
__global__ void shared_mem_2d_block_kernel(int M, int N, int K, float alpha,
                                           const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float beta, float *__restrict__ C) {
```

**Edit 2** — replace the `// ---- Epilogue` block (current lines 109‑140) with:

```cpp
  // ---- Epilogue: αAB + βC store --------------------------------------------
  for (int result_idx_m = 0; result_idx_m < TM; ++result_idx_m) {
    const int C_row_global = block_row * BM + tile_row * TM + result_idx_m;

    for (int result_idx_n = 0; result_idx_n < TN; ++result_idx_n) {
      const int C_col_global = block_col * BN + tile_col * TN + result_idx_n;

      if (C_row_global < M && C_col_global < N) {
        const int idx =
            (tile_row * TM + result_idx_m) * N + tile_col * TN + result_idx_n;
        store_result<BetaIsZero>(
            &C[idx], alpha * thread_result[result_idx_m * TN + result_idx_n],
            beta);
      }
    }
  }
```

**Edit 3** — replace the launcher:

```cpp
void kernels::shared_mem_2d_block(const GemmArgs &a) {
  constexpr int BM = 64, BK = 8, BN = 64, TM = 8, TN = 8;
  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

  dim3 block(NUM_THREADS);
  dim3 grid(ceil_div(a.N, BN), ceil_div(a.M, BM));

  if (a.beta == 0.0f) {
    shared_mem_2d_block_kernel<BM, BK, BN, TM, TN, true>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  } else {
    shared_mem_2d_block_kernel<BM, BK, BN, TM, TN, false>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  }
  CUDA_CHECK_LAUNCH();
}
```

- [ ] **Step 2: Build, run, verify the `shared mem 2D block` line.**

```
cmake --build build -j
./build/cuda_learning
```

Look for:

```
shared mem 2D block     rel RMSE 1.xxe-05   xxx.xxx ms   x.xx TFLOPS
```

- [ ] **Step 3: Commit.**

```
git add src/kernels/shared_mem_2d_block.cu
git commit -m "shared_mem_2d_block: compile-time dispatch on BetaIsZero via store_result"
```

---

## Task 6: Refactor `shared_mem_vec.cu`

**Files:**
- Modify: `src/kernels/shared_mem_vec.cu`

This is the first kernel where the refactor pays measurable perf — the `BetaIsZero=true` specialization drops a `float4` *load* of C per 4 outputs.

- [ ] **Step 1: Apply edits.**

**Edit 1** — add include + change template signature:

```cpp
#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels.h"
```

```cpp
template <int BM, int BK, int BN, int TM, int TN, bool BetaIsZero>
__global__ void shared_mem_vec_kernel(int M, int N, int K, float alpha,
                                      const float *__restrict__ A,
                                      const float *__restrict__ B, float beta,
                                      float *__restrict__ C) {
```

**Edit 2** — replace the `// ---- Epilogue` block (current lines 97‑136) with:

```cpp
  // ---- Epilogue: αAB + βC store --------------------------------------------
  for (int result_idx_m = 0; result_idx_m < TM; ++result_idx_m) {
    for (int result_idx_n = 0; result_idx_n < TN; result_idx_n += 4) {

      float4 prod;
      prod.x = alpha * thread_result[result_idx_m * TN + result_idx_n + 0];
      prod.y = alpha * thread_result[result_idx_m * TN + result_idx_n + 1];
      prod.z = alpha * thread_result[result_idx_m * TN + result_idx_n + 2];
      prod.w = alpha * thread_result[result_idx_m * TN + result_idx_n + 3];

      auto *dst = reinterpret_cast<float4 *>(
          &C[(tile_row * TM + result_idx_m) * N + tile_col * TN +
             result_idx_n]);
      store_result<BetaIsZero>(dst, prod, beta);
    }
  }
```

The two old loops collapse to one. The `BetaIsZero=true` instantiation of `store_result(float4*, …)` will skip the `c = *dst` load and emit a pure store; the false instantiation emits the read‑modify‑write you had before.

**Edit 3** — replace the launcher:

```cpp
void kernels::shared_mem_vec(const GemmArgs &a) {
  constexpr int BM = 128, BK = 8, BN = 128, TM = 8, TN = 8;
  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

  dim3 block(NUM_THREADS);
  dim3 grid(ceil_div(a.N, BN), ceil_div(a.M, BM));

  if (a.beta == 0.0f) {
    shared_mem_vec_kernel<BM, BK, BN, TM, TN, true>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  } else {
    shared_mem_vec_kernel<BM, BK, BN, TM, TN, false>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  }
  CUDA_CHECK_LAUNCH();
}
```

- [ ] **Step 2: Build, run, verify the `shared mem vectorized` line.**

```
cmake --build build -j
./build/cuda_learning
```

Look for:

```
shared mem vectorized   rel RMSE 1.xxe-05   xxx.xxx ms   x.xx TFLOPS
```

This is where you may see a small but measurable TFLOPS bump in the benchmark line (β=0 path), because the `BetaIsZero=true` specialization drops the float4 load of C. Compare against the previous run; expect a few percent improvement in TFLOPS for this kernel specifically.

- [ ] **Step 3: Commit.**

```
git add src/kernels/shared_mem_vec.cu
git commit -m "shared_mem_vec: compile-time dispatch on BetaIsZero via store_result"
```

---

## Task 7: Refactor `shared_mem_vec_warp_tile.cu`

**Files:**
- Modify: `src/kernels/shared_mem_vec_warp_tile.cu`

**Prerequisite:** see the note at the top of this plan. If the launcher in this file doesn't currently compile (`WM`, `WN`, `WNITER` not declared in launcher scope), declare them as `constexpr int` in the launcher with values that satisfy the kernel's `static_assert`s — that's an existing concern, not part of this refactor. The edits below assume the launcher has those declarations.

This kernel currently has an empty `// TODO: add code here.` for the β=0 store (line 152). After this refactor that TODO disappears for free — `store_result<true>(float4*, …)` is the β=0 store.

- [ ] **Step 1: Apply edits.**

**Edit 1** — add include + change template signature:

```cpp
#include "cuda_utils.cuh"
#include "epilogue.cuh"
#include "kernels.h"
```

```cpp
template <int BM, int BK, int BN, int WM, int WN, int WNITER, int TM, int TN,
          int NUM_THREADS, bool BetaIsZero>
__global__ void shared_mem_vec_warp_kernel(int M, int N, int K, float alpha,
                                           const float *__restrict__ A,
                                           const float *__restrict__ B,
                                           float beta, float *__restrict__ C) {
```

**Edit 2** — replace the `// ---- Epilogue` block (current lines 149‑183, including the empty β=0 TODO) with:

```cpp
  // ---- Epilogue: αAB + βC store --------------------------------------------
  for (int w_sub_row_idx = 0; w_sub_row_idx < WMITER; ++w_sub_row_idx) {
    for (int w_sub_col_idx = 0; w_sub_col_idx < WNITER; ++w_sub_col_idx) {

      float *C_interim =
          C + (w_sub_row_idx * WSUBM) * N + (w_sub_col_idx * WSUBN);

      for (int result_idx_m = 0; result_idx_m < TM; ++result_idx_m) {
        for (int result_idx_n = 0; result_idx_n < TN; result_idx_n += 4) {

          const int i = (w_sub_row_idx * TM + result_idx_m) * (WNITER * TN) +
                        w_sub_col_idx * TN + result_idx_n;

          float4 prod;
          prod.x = alpha * thread_result[i + 0];
          prod.y = alpha * thread_result[i + 1];
          prod.z = alpha * thread_result[i + 2];
          prod.w = alpha * thread_result[i + 3];

          auto *dst = reinterpret_cast<float4 *>(
              &C_interim[(thread_row_in_warp * TM + result_idx_m) * N +
                         (thread_col_in_warp * TN + result_idx_n)]);
          store_result<BetaIsZero>(dst, prod, beta);
        }
      }
    }
  }
```

**Edit 3** — replace the launcher. You'll need `WM`, `WN`, `WNITER` declared as `constexpr int` here (see prerequisite above):

```cpp
void kernels::shared_mem_vec_warp(const GemmArgs &a) {
  constexpr int BM = 128, BK = 8, BN = 128, TM = 8, TN = 8;
  constexpr int WM = 64, WN = 32, WNITER = 1; // pick values satisfying the kernel's static_asserts
  constexpr int NUM_THREADS = (BM * BN) / (TM * TN);

  dim3 block(NUM_THREADS);
  dim3 grid(ceil_div(a.N, BN), ceil_div(a.M, BM));

  if (a.beta == 0.0f) {
    shared_mem_vec_warp_kernel<BM, BK, BN, WM, WN, WNITER, TM, TN, NUM_THREADS,
                               true>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  } else {
    shared_mem_vec_warp_kernel<BM, BK, BN, WM, WN, WNITER, TM, TN, NUM_THREADS,
                               false>
        <<<grid, block>>>(a.M, a.N, a.K, a.alpha, a.A, a.B, a.beta, a.C);
  }
  CUDA_CHECK_LAUNCH();
}
```

(The values `WM=64, WN=32, WNITER=1` satisfy the kernel's static_asserts for the existing `BM=128, BN=128, NUM_THREADS=256, TM=TN=8` tile. Pick whatever you actually want — the static_asserts will tell you off if you choose wrong.)

- [ ] **Step 2: Build, run, verify the `shared mem vectorized warptiled` line.**

```
cmake --build build -j
./build/cuda_learning
```

Look for:

```
shared mem vectorized warptiled   rel RMSE 1.xxe-05   xxx.xxx ms   x.xx TFLOPS
```

Two things to expect:
- **RMSE pass** — the β≠0 specialization is what RMSE checks. If this fails, the `false` branch of `store_result<float4*>` is the suspect.
- **Real TFLOPS** — previously the benchmark hit an empty β=0 epilogue, so the compiler likely DCE'd the entire kernel and the TFLOPS number was meaningless. After this refactor the β=0 epilogue actually stores, so this is now an *honest* benchmark number. You should expect the TFLOPS to **drop** from the previous bogus value — that's the fix surfacing, not a regression.

- [ ] **Step 3: Commit.**

```
git add src/kernels/shared_mem_vec_warp_tile.cu
git commit -m "shared_mem_vec_warp: compile-time dispatch on BetaIsZero; finish β=0 store"
```

---

## Task 8: Validate the β=0 specializations end‑to‑end

Up to now the harness has only RMSE‑checked the `BetaIsZero=false` path (since `kBeta = 0.5f` in the correctness pass). This task temporarily flips `kBeta` to 0 so the correctness pass exercises the `BetaIsZero=true` instantiations end‑to‑end. Then reverts.

**Files:**
- Modify (temporarily): `include/harness.h`

- [ ] **Step 1: Edit `include/harness.h:96`.**

Find:

```cpp
  static constexpr float kBeta = 0.5f;
```

Change to:

```cpp
  static constexpr float kBeta = 0.0f;
```

- [ ] **Step 2: Build and run.**

```
cmake --build build -j
./build/cuda_learning
```

Expected: every kernel line shows a small RMSE (under `kTolerance = 1e-3`). If any kernel prints `FAILED` here, its `BetaIsZero=true` specialization is wrong — typically a mistake in the `store_result` call or in `prod` construction.

- [ ] **Step 3: Revert the harness change.**

Restore `kBeta` to `0.5f`:

```cpp
  static constexpr float kBeta = 0.5f;
```

Don't commit this — it was a one‑shot verification. Confirm with:

```
git diff include/harness.h
```

Expected: empty.

- [ ] **Step 4: Sanity rerun.**

```
cmake --build build -j
./build/cuda_learning
```

Expected: all RMSE lines small, benchmark line populated for every kernel including warptiled.

(If you'd rather make the dual β check permanent — running each kernel twice, once with β=0.5 RMSE'd and once with β=0 RMSE'd — that's a separate harness change worth doing properly later. Out of scope for this plan.)

---

## Done

After Task 8 you should have:
- One new header `include/epilogue.cuh` with two `store_result<bool BetaIsZero>` overloads.
- All six kernels templated on `bool BetaIsZero` with a single‑line epilogue call into `store_result`.
- Six launchers each doing an explicit `if (a.beta == 0.0f) … else …` to pick the specialization.
- A previously‑empty β=0 path in the warp_tile kernel now correctly filled.
- Seven commits — one per task, except Task 8 (no commit).

Where the win actually shows up: the `shared_mem_vec` and `shared_mem_vec_warp` benchmark lines (β=0) should run slightly faster than before, because the `BetaIsZero=true` specialization elides the float4 load of C. The scalar kernels are unchanged in measurable perf — the value there is purely the deduplication and the CUTLASS‑shape consistency.
