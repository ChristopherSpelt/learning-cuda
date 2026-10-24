# SGEMM Optimization — Handover

A self-contained brief so a fresh context can continue the work. Hardware:
**NVIDIA RTX 3090** (GA102, **sm_86 / compute capability 8.6**). 82 SMs, FP32 FMA
peak ~35.6 TFLOP/s, 65536 regs/SM, 48 KB static smem/block (100 KB/SM opt-in),
1536 threads/SM, 4 warp schedulers/SM. GA10x has one shared FP32/INT32 datapath per
partition, so integer/address ops in the inner loop steal FFMA issue slots.

---

## 0. TL;DR

- We walk an SGEMM optimization ladder in `src/kernels/sgemm/`, benchmarking each
  rung against cuBLAS. Row-major `C = alpha*A*B + beta*C`, fp32, no tensor cores.
- **Champion: `double_buffer_prefetch_bank` (kernel 11).** Two-level double buffer
  (global to shared, plus shared to register) with a bank-conflict-free transposed
  `As` (inverted load + padded pitch). **~92-93% of cuBLAS at 4096 batch, exactly
  cuBLAS at 1024 batch, 86% at 8192, +5% over k10 at 256-1024.** SASS-verified (§6).
- **Prior champion: `double_buffer_prefetch` (kernel 10),** same pipeline but the
  `As` transpose store was 4-way bank-conflicted. k11 is k10 + the bank fix.
- **Removed branches (do not revisit):** `cp_async` and `cp_async_swizzle` both LOST
  and have been **deleted from the tree** (the swizzle lineage is closed, see §5).
  NOTE: kernels were renumbered after the deletion. Old "k12" prefetch_bank is now
  **11**; old "k06_1" resolve_bank is now **07**; old "k07" double_buffer is now
  **08**. Old "k08"/"k11" were the two deleted cp.async kernels (no current mapping).
- **NEXT TASK: build k12 = the salykova / YHs "ampere" 128x256x8 kernel** (4-byte
  cp.async writing directly into transposed padded `As`). This is the one lever
  sized to close the large-N gap; it beats cuBLAS on a 3090 in the reference. See §7.

---

## 1. How we work together (READ FIRST)

This is a **hands-on teaching collaboration**, not a do-it-for-them task:

- **The user writes the kernel code.** Provide reviewable snippets and Socratic
  guidance; **do not edit the kernel `.cu` files until the user explicitly says
  "apply/fix."** (Docs, this handover, and memory you may write freely.)
- Match the repo's **inline style**: each kernel is a self-contained `.cu`; don't
  extract shared device helpers/templates prematurely (the user values "design
  seams, but don't build the generic machinery now").
- **Measure before optimizing.** The repo's recurring lesson (`lessons_learned.md`)
  is that the "obvious" optimization is often *not* a win. Verify with the
  benchmark, `cuobjdump`, and reasoning. Predictions overturned so far: BK=8,
  cp.async (twice), the As swizzle.
- No em-dash in prose/comments/docs (use commas, colons, semicolons, parentheses).
- No `Co-Authored-By` trailer; the user prefers one final commit when committing.
- nvcc limitation: `__global__` kernels cannot take class-type non-type template
  params; unpack tile config to integer NTTPs in the host launcher.

---

## 2. Build, run, profile

```bash
# Build (ALWAYS pin the arch: CMake silently defaults to sm_52 in some containers,
# which silently JITs and emits XMAD. Verify with: cuobjdump -sass build/sgemm | grep -c XMAD  (==0))
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --target sgemm

# Run the benchmark (square M=N=K sweep; default 256,512,1024,2048,4096)
./build/sgemm --csv
./build/sgemm 8192 --csv        # large-N steady state

# Registers / smem / spills (no ncu on the runpod, host-level ERR_NVGPUCTRPERM)
cuobjdump -sass -res-usage build/sgemm > sgemm.sass
```

**Harness** (`include/bench.h`): two TFLOP/s numbers plus an auto-validated
`rel_err` (a kernel over tolerance prints as a blank gap, that is the correctness
check):
- **cold**: L2 flushed before each launch, timed in isolation.
- **batch**: back-to-back launches under one timer, cache warm.

Clocks are **not** locked by the harness (and cannot be locked on the runpod), so
`cold > batch` at large N is a boost-clock artifact, and FFMA-dense kernels
self-throttle above N~4000. **Trust batch.** Treat sub-2% as noise; small-N is
microsecond-scale.

> Known oddity: `rel_err` reads `0.00e+00` for ALL kernels at N=4096 (validation
> appears skipped there). Validate correctness at 2048, not 4096.
> Known label bug: a recent run printed bracket sizes at 2x the true size (e.g.
> "[512]" was really 256). Match by cuBLAS rate if in doubt.

---

## 3. The kernel ladder

```
01 naive -> 02 block_tile -> 03 thread_tile_1d -> 04 thread_tile_2d -> 05 vec_loads
-> 06 warp_tile -> 07 resolve_bank -> 08 double_buffer -> 09 inner_loop_prefetch
-> 10 double_buffer_prefetch -> 11 double_buffer_prefetch_bank (CHAMPION)
```
(The two dead cp.async kernels were deleted and the ladder renumbered; 07 resolve_bank
teaches the bank-conflict fix on the simple warp_tile before the pipeline is built,
then 11 re-applies it inside the two-level pipeline.)

A new kernel must be added in three places: (a) `CMakeLists.txt` sgemm sources,
(b) `include/kernels/sgemm.h` decl, (c) the `registry[]` in `src/sgemm.cu`.

**Shared structure of the tiled kernels:** block tile `BM x BN` -> warp tile
`WM x WN` -> thread tile `TM x TN`; `As` **stored transposed** in shared (compute
read is stride-1 in `m`); float4 vectorized loads; compile-time `BetaIsZero` +
`BoundsCheck` dispatch via `cuda_utils::dispatch_bool`; `epilogue::store_result`.

**Tile params for 08/09/10/11:** `BM=128, BK=16, BN=128, WM=64, WN=64, WMITER=1,
WNITER=4, TM=8, TN=4` -> **128 threads (4 warps), 128 accumulator regs/thread.**
ptxas sm_86: **k10 = 216 regs / 32768 B smem / 0 spills, k11 (bank) = 219 regs /
33280 B smem / 0 spills.** Both register-bind at **2 blocks/SM = 8 warps/SM** (16.7%
occupancy). Occupancy is floor-low *by design* (register tiling), so latency hiding
is **structural** (the two-level pipeline), not occupancy-driven. The 1-block case
is only the grid-starved small-N regime (N=1024 -> 64 blocks < 82 SMs), not a
per-SM resource limit. Note 82 = 2*41 (41 prime), so power-of-2 grids are pinned at
64/82 = 78% SM coverage at 1024; only split-K or non-pow2 tiles break that.

---

## 4. Results (RTX 3090, batch TFLOP/s, latest run; true sizes)

| N | k10 prefetch | **k11 prefetch_bank** | cuBLAS | k11 % cuBLAS |
|--:|--:|--:|--:|--:|
| 256  | 1.03  | **1.08**  | 4.10  | 26%  |
| 512  | 4.39  | **4.60**  | 11.28 | 41%  |
| 1024 | 15.45 | **16.29** | 16.29 | **100%** |
| 2048 | 18.53 | **18.55** | 20.83 | 89%  |
| 4096 | 21.27 | **21.48** | 23.18 | 93%  |
| 8192 | 21.31 | **21.44** | 24.87 | 86%  |

- **k11 batch >= k10 at every size.** Big win (+5%) at 256-1024 where the FFMA pipe
  is not saturated and the conflicted store sat naked before `__syncthreads`; a tie
  (+0.1 to +1%) at 2048-4096 where the two-level pipeline already overlaps the
  commit across warps. This is exactly the verifier prediction (+0-3% at large).
- **cleanest proof the conflict was real:** `resolve_bank` (07) is the same bank
  fix on the *non*-double-buffered warp_tile, where nothing hides the stall:
  **+8% at 256, +11% at 2048 batch** over plain warp_tile.
- **k11 is strictly >= k10 on batch at every size 256-8192, no regression: adopted.**
  The 2048 cold -2.4% blip did NOT recur at 8192 (cold tie 21.51 vs 21.50), so it was
  2048-specific noise/quantization, not a systematic scalar-gather penalty.
- resolve_bank over warp_tile shrinks from +11% @2048 to +1.6% @8192: the conflict
  gets increasingly hidden as the FFMA pipe saturates. The remaining ~14% to cuBLAS
  @8192 is bytes/FLOP + L2 locality, which only the 128x256 (k12) build attacks.

---

## 5. The story so far

**The winning line (08 -> 09 -> 10 -> 11):**
- **08 double_buffer:** ping-pong `As[2]/Bs[2]` in shared + register staging, so the
  global->shared load of the next k-tile overlaps the current FMAs. One sync/tile.
- **09 inner_loop_prefetch:** double-buffer at the *shared->register* level
  (`reg_*_a`/`reg_*_b` with an unroll-by-2 role swap), hiding LDS latency.
- **10 double_buffer_prefetch:** the union of 08 + 09 (both memory hops overlap the
  FMAs). Heaviest register user, 0 spills. Long the champion at ~87-92% cuBLAS.
- **The ptxas LDG-sink fix (load-bearing):** with `#pragma unroll` on the outer
  k-tile loop, ptxas **sank** the staged LDGs to the bottom of the unrolled body,
  next to the STS, defeating the double buffer in SASS. **Deleting the one pragma**
  (keeping the loop rolled as a scheduling fence) restored the LDG-early/STS-late
  schedule: **+5-7% everywhere.** ALWAYS check LDG-vs-STS placement in SASS.
- **11 double_buffer_prefetch_bank (CHAMPION):** k10 + a bank-conflict-free `As`.
  k10's transpose store was 16x `STS.32` scattering one global float4 across 4
  k-rows, all landing in banks 0-7 (4-way conflict). k11 **inverts the A load**:
  each thread gathers a 4-row M-strip at fixed k (16x scalar `LDG.32`), then commits
  it as one `make_float4` into `As[k*(BM+4)+m]` (a contiguous `STS.128`). Pitch
  `BM+4 = 132 ≡ 4 (mod 32)` spreads each 8-thread store phase across all 32 banks:
  conflict-free. The read stays `LDS.128`. SASS-verified in §6.

**Removed branches (deleted from the tree; do not revisit):**
- **cp_async (removed):** `__pipeline_memcpy_async` copies contiguous->contiguous, so
  the transposed `As` scatter degraded to 16 tiny 4-byte cp.async/thread (the
  "transpose tax"). Lost -4 to -12% vs double_buffer. NOTE: this was later shown to
  be **tile-coupled, not fundamental** (salykova: cp.async loses at 128x128x8, wins
  at 128x256x8). It also had unpadded `As` (pitch 128, 4-way conflicted stores).
- **cp_async_swizzle (removed):** tried to dodge the transpose with an XOR swizzle of
  a non-transposed `As`. **Measured -27% at 8192.** Root cause (SASS + 3-lens verify):
  the swizzle is conflict-free as derived, but it de-vectorizes the read to 8x
  `LDS.32`, adds per-access XOR/IMAD on the shared FP32/INT port (steals FFMA slots
  1:1), and reused cp_async's inner loop (no register double buffer). The losses
  compose to -27%. **No reference kernel swizzles; they all pad the pitch.** Closed.

**Reference mining (2026-06-07):** a multi-agent workflow read YHs_Sample,
tpoisonooo MMult_cuda_12, salykova sgemm.cu, our double-buffer kernels + SASS, and
prior art, then adversarially verified 8 levers (3 lenses each). Key facts: k10's inner loop
is already ~93% FFMA density (near issue-optimal); the references win on bytes/FLOP
(BN=256 = -25% global traffic) and load-path structure, not the inner loop.
MMult_cuda_12 (faithful 128x128x8) TIES k10 on a 3090, so it has nothing left to
teach. Refuted levers: z-order lane map (our 8x4 grid already gives the good phase
pattern), smem-staged epilogue (our stores are already coalesced), LDS clustering
(ptxas ignores source order), more-CTAs-at-1024 (the 41-prime coverage pin).

---

## 6. SASS audit of k11 (2026-06-09, hot path BetaIsZero / BoundsCheck=0)

Per K-tile loop body, k10 vs k11:

| | k10 | k11 | |
|---|---|---|---|
| As commit (store) | **16x STS.32** (4-way conflict) | **4x STS.128** (conflict-free) | the fix |
| As load | part of 8x LDG.128 | **16x scalar LDG.32** | the cost |
| Fragment read | 18x LDS.128 | **18x LDS.128** | read stays vectorized |
| INT/addr ops (loop) | 38 | **66** (+28) | scalar-gather address math |
| Registers / smem / spills | 216 / 32768 / 0 | 219 / 33280 / 0 | +3 reg, +512 B, still 2 CTAs/SM |

Five confirmations:
1. The bank fix compiled as designed: As commit is now `STS.128`, zero scalar STS.
   (SASS cannot show bank *replays* without ncu; conflict-freedom is analytical +
   the resolve_bank +11% empirical signal.)
2. Read stayed `LDS.128` (18, identical to k10). The padded pitch did not
   de-vectorize the fragment load (the thing that killed the cp_async_swizzle kernel).
3. No spills, occupancy unchanged (2 CTAs/SM = 8 warps/SM).
4. **LDG-sink did NOT recur** (the real risk, since scalar LDGs have destination
   registers): the 16 As-gather loads sit at the **top** of the K-tile iteration,
   before the 256-FFMA compute loop; commit `STS.128` + `BAR.SYNC` at the bottom.
   The no-unroll fence held.
5. The cost is once-per-K-tile (replayed 8x by the rolled compute loop), so dynamic
   FFMA dilution is ~1-2%, not the static 50.6% -> 48.0%. That matches the measured
   cold -2.4% @2048 / batch-tie-at-large signature: the +28 IMAD/IADD steal FFMA
   slots on GA10x's shared port only when the pipe is saturated.

Inherent tension this exposes: at this tile you can vectorize the As store XOR the
As load, not both (LDG.128 reads contiguous-k -> forces scatter store; STS.128
writes contiguous-m -> forces strided gather load). k11 chose vectorized-store +
scalar-load. The only way to get BOTH vectorized is cp.async writing directly into
the transposed layout, which is exactly the §7 next task.

---

## 7. NEXT TASK — k12 = salykova / YHs "ampere" 128x256x8 cp.async

The lever the reference-mining workflow rated **strong on all three lenses**, and
the only one sized to close the large-N gap. salykova's faithful port beats cuBLAS
on a 3090 from N~2500 up (~25-26 vs ~24-25 TFLOP/s, unlocked clocks like ours).

Design (port faithfully from `github.com/salykova/sgemm.cu`,
`src/kernels/128x256x8.cuh`; blog `salykova.github.io/sgemm-gpu`):
- `BM=128, BN=256, BK=8`, **256 threads** (8 warps), `__launch_bounds__(256,1)`,
  fragments `A[2][16] / B[2][8]` = 128 accumulators (same count as k11).
- **A via 4-byte `cp.async.ca` written directly into transposed padded `As[8][132]`**
  (zero STS, zero staging registers, and structurally **immune to the LDG-sink**
  because LDGSTS has no destination register). B via 4-byte cp.async.
- 12 copies spread over k_frags 0-3, `cp.async.wait_all` + one `__syncthreads` at
  k_frag 7. 32 KB smem `__align__(16384)` with a single `^0x2000` ping-pong flip.
  Outer k-loop **rolled** (keep the fence).
- Why faster: bytes/FLOP drops 0.03125 -> 0.0234 (**-25% global traffic**, the
  dominant 8192 lever); FFMA issue fraction rises to ~94%; the whole LDG-sink
  failure class is impossible.

Build notes / risks:
- Build it as the **full 256-wide design**, NOT incrementally from k11. The deleted
  cp_async_swizzle kernel showed a careless cp.async port silently loses 27%.
- Skip the `.L2::128B` hint initially (the measured-winning reference uses plain
  4-byte `cp.async.ca`); add only as a later A/B.
- Port the reference's guard-bitmask + k-remainder bounds handling (keep our
  `BoundsCheck` dispatch). cp.async does no masking on its own.
- Below ~2560 it starves 82 SMs (128x256 -> few CTAs), so **size-dispatch**: keep
  k11 (or k10) for small/mid, k12 for >=2560. First pod action: fresh
  `cuobjdump -sass`, confirm LDGSTS placement and that there are no spills.

**Cheaper contingency levers (only if k12 stalls), per the verified ranking:**
- Threadblock **raster swizzle** (CUTLASS log_tile in {1,2,3}) for L2 reuse at 8192;
  expect +2-4% batch, prologue-only, zero inner-loop change. Diagnostic either way.
- A **64x64-class small-N tier** for N<=512 (the sweep shows +40-57% there), and a
  128x64 tier (512 CTAs = 89% coverage) or split-K=5 at 1024/2048.
- Multistage (4 stages, pow2 rotation, dynamic smem) on top of k12 ONLY if 8192
  still trails >3% after raster. Never modulo-3 rotation (integer ops in the loop).

---

## 8. Open issues / caveats

- **No `ncu` on the runpod** (host-level `ERR_NVGPUCTRPERM`); clock locking also
  blocked. Use `cuobjdump -sass -res-usage` + timing + reasoning. Trust batch.
- **`rel_err` reads 0 at N=4096** for all kernels (validation likely skipped there).
  Validate at 2048.
- All current kernels carry the `BoundsCheck` dispatch (masked loads), so
  non-128-multiple sizes are safe. (The two removed cp_async kernels lacked masking;
  that OOB risk left the tree with them.)
- The local `sgemm.sass` must be regenerated on the pod after any kernel change
  (it is the compiled artifact, not tracked live). The §6 audit used the Jun-9 dump.

---

## 9. Key files

| File | Role |
|---|---|
| `src/kernels/sgemm/11_double_buffer_prefetch_bank.cu` | **CHAMPION** |
| `src/kernels/sgemm/10_double_buffer_prefetch.cu` | prior champion (no bank fix) |
| `src/kernels/sgemm/09_inner_loop_prefetch.cu` | shared->register double buffer |
| `src/kernels/sgemm/08_double_buffer.cu` | global->shared double buffer |
| `src/kernels/sgemm/07_resolve_bank.cu` | bank fix on plain warp_tile (proof) |
| `src/kernels/sgemm/06_warp_tile.cu` | base both pipelines built on |
| `include/loads.cuh` | `masked_load<>`, `masked_load_f4<BoundsCheck>` |
| `include/epilogue.cuh` | `store_result<BetaIsZero>` |
| `include/bench.h` | harness (cold/batch timing, CSV, rel_err) |
| `include/kernels/sgemm.h`, `src/sgemm.cu` | decls + `registry[]` |
| `CMakeLists.txt` | add new kernels here; pin `CMAKE_CUDA_ARCHITECTURES=86` |
| `docs/double_buffering.md` | full double-buffer + cp.async findings |
| `lessons_learned.md` | tuning philosophy (occupancy, measurement discipline) |
| `data/sgemm.csv` | results |
