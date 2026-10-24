# Double Buffering (Software Pipelining) for SGEMM

Findings from adding `src/kernels/sgemm/07_double_buffer.cu` on top of
`06_warp_tile`. Hardware: RTX 3090 (GA102, compute capability 8.6). Style mirrors
`lessons_learned.md`: **formal + intuitive**, with the empirical story.

---

## 1. The problem it targets

`warp_tile`'s K-loop is **serialized**:

```
for each k-tile:
    load tile (global → shared)     # memory pipes busy, FMA idle
    __syncthreads()                 # barrier A (load → compute)
    compute (shared → registers)    # FMA busy, memory idle
    __syncthreads()                 # barrier B (compute → next load)
```

Loading and computing happen **one after the other**, so the global-load latency
(hundreds of cycles to DRAM) sits *fully exposed* on the critical path.

The GPU normally hides that latency by switching to other warps — but register
tiling is deliberately register- and shared-memory-hungry (that's where the
arithmetic intensity comes from), which **starves occupancy**. At our tile shape
the kernel is **register-bound to 2 blocks / 8 warps per SM (~16.7%)**, dropping to
**1 block / 4 warps (~8%)** only when the grid is too small to supply a second block
(small N) — either way far too few warps to cover the stall.

> **Occupancy** hides latency by having *many* warps. **Double buffering** hides it
> by giving *each* warp useful work to do *while its own next load is in flight.*
> When register tiling has spent your occupancy, double buffering is the latency
> hiding you have left.

---

## 2. The technique

Two shared buffers (ping-pong) plus a small per-thread **register staging** area.
The load is pried into two halves so compute can wedge between them:

```
prologue: load tile 0 → buf[0];  __syncthreads()
for each k-tile:
    fetch   next tile  → staging registers    (LDG only — issue, don't store)
    compute current buf[cur]                   (FMAs — latency hides under here)
    commit  staging    → buf[next]             (STS — value has arrived)
    __syncthreads();  cur ^= 1
```

Two facts make it work:

- **The register in the middle.** A global→shared move is always `LDG → register
  → STS`. Writing `smem = global` glues the LDG to its consuming STS, so the warp
  stalls *before* compute. Staging the LDG in a register and doing the STS *after*
  the compute loop is what separates issue from consume — the latency hides under
  the FMAs.
- **One sync, not two.** With two buffers the next load writes the *other* buffer,
  so it can't clobber the tile being read — the write-after-read hazard barrier B
  guarded is gone *by construction*. Per-iteration syncs drop **2 → 1**.

---

## 3. Results (RTX 3090, vs `warp_tile`)

Batch rate (TFLOP/s), the trustworthy column, with grid size in blocks
(`(N/128)²`):

| N | grid blocks | warp_tile | double_buffer | Δ |
|--:|--:|--:|--:|--:|
| 1024 | 64 | 12.39 | **13.93** | **+12.4%** |
| 2048 | 256 | 15.90 | **17.02** | **+7.0%** |
| 4096 | 1024 | 19.25 | 19.46 | +1.1% *(within noise)* |

(256/512 are launch-overhead noise; 4096 ≈ neutral.) At 4096, double_buffer is
**~55% of the 3090's 35.58 TFLOP/s FP32 peak (batch; ~61% on the cold-clock number,
§6)** and **~83% of cuBLAS FP32-strict**.
`rel_err` is identical to the other tiled kernels — numerically correct.

---

## 4. Key insight: occupancy and double buffering are *substitutable*

The win shrinks monotonically as N grows, and grid size explains it exactly:

- **N=1024 → 64 blocks < 82 SMs.** Neither kernel can exceed 1 block/SM (not
  enough blocks to fill them), so `warp_tile` *can't use* its higher occupancy.
  The pipeline is a clean, uncontaminated win (**+12.5%**).
- **N=2048 → ~3 blocks/SM.** `warp_tile` can finally use its occupancy, which
  substitutes for the pipeline; double buffer pays its 2× smem cost. Win narrows
  to **+7%**.
- **N=4096 → ~12 waves.** The machine is saturated with independent blocks;
  `warp_tile`'s occupancy hides latency on its own. The pipeline has nothing left
  to add and is fully paying its occupancy cost → **neutral**.

> Double buffering and occupancy hide the *same* latency. Where you already have
> occupancy, the pipeline adds little. This is the same shape as the `sasum`
> oversubscription finding: the optimization matters where the machine is
> under-fed and washes out once it's full.

---

## 5. Reference reality check — the "30% gains online" mostly don't apply

Classic 2-stage double buffering on a **strong, warp-tiled FP32 baseline** is
known to be marginal-to-harmful:

- **siboehm's own kernel 11** (the canonical reference, A6000 = same GA102/cc 8.6
  arch) **regressed** in his *repo* benchmark: 17,278 GFLOP/s (**74.3%** of cuBLAS)
  vs his kernel 10 warptiling at 21,779 GFLOP/s (**93.7%**). His blog *article* only
  writes up through kernel 10 and lists double buffering as a "WIP / what I'll look
  at next" note — the regression is the measured repo result, not a reason stated in
  the prose. *Our result (a real mid-range win, neutral at scale) is better than the
  reference.*
- **kapilsharma.dev's "30+%"** is **BF16**, against a weaker/unisolated baseline,
  and the author found the *multi-stage* `cuda::pipeline` version **slower** and
  abandoned it for CUTLASS.

> The reason double buffering is famous is **conceptual** — it's the gateway to
> pipelining — not because the 2-stage version is a big win by itself. The real
> gains (CUTLASS, >90% of cuBLAS) come from **`cp.async` multistage** (Ampere
> async copy, 3–4 stages), which hides latency *without* the register/occupancy
> tax that caps the classic method.

---

## 6. Methodology guardrails (per `lessons_learned.md` §11)

- **Lock clocks** (`nvidia-smi -lgc`). The `cold > batch` inversion at 4096
  (21.59 vs 19.46) is a boost-clock artifact, not algorithmic.
- **Noise floor ~2%** — treat 4096's +1.1% as neutral.
- **Trust batch (mid-range) and large-N (steady state)**; small-N is µs-scale noise.
- **Confirm 0 register spills** (`-Xptxas=-v` / `cuobjdump -res-usage`). The
  register-staged variant adds ~32 regs on top of ~128 accumulators.

---

## 7. Status & next step

`double_buffer` is the **fastest custom SGEMM** in the ladder at every size, but
the gain is mid-range only. To beat the large-N plateau:

1. **Sweep tiles for the double-buffered budget** — especially `BK 16 → 8` (halves
   smem — though **registers**, not smem, are the binding limit here) and more-threads/fewer-outputs-per-thread
   (cut register pressure). These params were inherited from `warp_tile`, which
   doesn't pay the 2× smem tax.
2. **Step ②: `cp.async` multistage** — async global→shared (no register staging),
   pipeline >2 stages deep. Built on the `fetch / compute / commit` seam: load and
   compute are functions parameterized by buffer index; generalize {0,1} → {0..S-1}.
   Wrinkle: `As` is stored transposed (a scatter), awkward for `cp.async`'s
   contiguous-only copies — to be resolved.

---

## 8. Confirmed on hardware: ptxas (sm_86) + BK sweep

**ptxas `-v`, sm_86, `BK=16`** (hot path β=0, aligned): **214 registers,
32,768 B smem, 0 spills.** Zero spills rules out local-memory traffic as a
confound. 214 regs/thread is very high — the 128 accumulators dominate — giving
~2 blocks/SM by registers — the binding limit. 32 KB *static* smem is slack: cc 8.6
carveouts are 0/8/16/32/64/100 KB (there is no 48 KB), and the driver auto-picks
≥64 KB so 2 blocks (64 KB) or 3 (100 KB) fit. Net **2 blocks/SM ≈ 16.7% occupancy**
(1 block / 8.3% only when the grid is too small to supply a second) — floor-low, as
predicted.
(Exact figure: `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, a host call — no
`ncu` required.)

**BK sweep (16 → 8).** Halving BK halves smem (→16 KB) and staging registers, but
**doubles the K-loop iteration count** (twice the syncs + boundary prefetch/commit).
Batch TFLOP/s vs `warp_tile`:

| N | warp_tile | double_buffer BK=16 | double_buffer BK=8 | BK=8 vs BK=16 |
|--:|--:|--:|--:|--:|
| 1024 | 12.41 | 13.93 (+12%) | **14.39 (+16%)** | +3% |
| 2048 | 16.15 | **17.02 (+5%)** | 16.48 (+2%) | −3% |
| 4096 | 19.44 | **19.46 (≈0)** | 18.22 (**−6%**) | −6% |

> **Smaller BK shifts the kernel toward the occupancy-favoring regime:** it helps
> (marginally, near noise) where the machine is under-fed, and clearly *hurts* at
> large N where the machine is saturated and compute-bound — there the doubled
> loop/sync overhead is pure cost and the extra occupancy goes unused. Crossover
> ≈ N=2048.

Two lessons:

1. **No single tile is optimal across sizes** — BK=8 wins small, BK=16 wins large.
   This is exactly why cuBLAS/CUTLASS carry `(arch, dtype, M, N, K)` heuristic
   tables and pick the tile per problem (§3 of `lessons_learned.md`). A hardcoded
   tile is always a compromise.
2. **Even size-tuned, double buffering only *ties* `warp_tile` at 4096.** Once the
   grid saturates the machine, occupancy already hides the latency, so the
   pipeline is redundant at scale — regardless of BK. Beating large-N needs
   latency hiding that doesn't *cost* occupancy: `cp.async` (step ②).

**Decision:** keep **`BK=16`** as the single robust value (monotonic, never worse
than `warp_tile`); hold `BK=8` only as a small-N special case if ever
size-dispatched. The large-N lever remains step ②.

**Correction to a prediction:** §1–7 expected `BK=8` to *help* at large N by
raising occupancy. The measurement says the opposite — loop overhead dominates
once the machine is saturated. Textbook "measure, don't assume" (rule #4).
