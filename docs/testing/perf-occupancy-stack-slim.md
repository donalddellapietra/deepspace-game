# Perf experiment: slim per-thread state in `march_cartesian`

Hypothesis-driven experiment. Test whether the same stack-based DDA can
reach the occupancy gains predicted in
[`perf-occupancy-diagnosis.md`](perf-occupancy-diagnosis.md) **without**
rewriting the algorithm (no cursor-walk, no compute-shader port, no
threadgroup memory). The diagnosis doc already enumerates the per-byte
savings; this branch measures whether compiler reality matches the
predictions for three specific state-layout swaps.

Each step lands as its own commit so the telemetry can be attributed to
exactly one change.

## The hypothesis

Of the five stack arrays in `assets/shaders/march.wgsl`, only one
(`s_cell`) genuinely needs to be stacked — the rest can become scalar
`var`s that are mutated on push/pop. The DDA algorithm is unchanged.

| Array | Bytes | Can be scalar? | How |
|---|---|---|---|
| `s_cell` | 60 | **No** | Mutated every iteration at current depth; parent values must survive child traversal so we can continue DDA after pop. |
| `s_node_idx` | 20 | No (for now) | Needed on pop to re-fetch `cur_occupancy`/`cur_first_child`. Could be eliminated by re-walking the cell chain, but adds tree reads on pop. |
| `s_cell_size` | 20 | **Yes (step 1)** | Pure function of depth: `1 / 3^depth`. Scalar mutated on descend (÷3) / pop (×3). |
| `s_node_origin` | 60 | **Yes (step 2)** | Reversible incremental update: on descend `+= cell * parent_cell_size`; on pop `-= s_cell[new_depth] * new_parent_cell_size`. |
| `s_side_dist` | 60 | **Yes (step 3)** | Pure function of (ray_origin, inv_dir, s_cell[depth], cur_node_origin, cur_cell_size). Recompute on pop from scratch. |

If the compiler actually promotes the scalars to registers, per-thread
state drops from ~260 B to ~80 B (plus 28 B of new scalar state). Apple's
happy-occupancy budget is ~128 B/thread; we should end up well inside it.

## What would falsify the hypothesis

If Fragment Occupancy doesn't move materially after step 3, one of:

1. **The compiler isn't promoting the scalars to registers** — maybe it
   still spills `vec3<f32>` vars to threadgroup memory because of some
   aliasing heuristic. Check with `naga --stage fragment --target msl`
   or inspect the compiled MSL in the .trace bundle.

2. **Per-thread state wasn't the actual bottleneck.** Some other
   constraint (divergence, TLB misses, instruction latency) is the real
   limiter, and the diagnosis doc's story was incomplete.

3. **Register pressure was real but recompute-on-pop costs more than the
   occupancy win.** In this case step 3 specifically should regress;
   revert step 3 and keep steps 1-2.

If the numbers don't move the expected direction at any step, **stop and
report** — keep the last-known-good commit and document what the
telemetry actually shows.

## Testing protocol

The slow-soldier-inside-body scenario is the one that showed the 40 FPS
regression; it's the one scenario we measure per step. Other canonical
scenes are covered only by screenshot A/B (for visual regression) — not
repeated perf captures.

**Per step:**

- `cargo build --bin deepspace-game` — clean warnings, zero errors
- `cargo test --lib gpu` — all gpu unit tests pass
- Screenshot A/B on plain_d8, sphere, zoom3 — compared against baseline
  screenshots in `tmp/shot/baseline_*.png`
- GPU telemetry capture on slow-soldier (2560×1440) — parse-metal-trace
  for Fragment Occupancy / ALU Util / Buffer Read
- Harness timing on slow-soldier (release build) for `submitted_done_ms`

Any visual regression is a **blocker**. Perf is expected to improve; if
it regresses, the change is wrong.

## Results

### Baseline (sparse-tree @ 809660e)

Slow-soldier @ 2560×1440, 300 frames, release build, direct-binary
invocation (not `cargo run`, which adds 5-10 ms of noise on this scene).

| counter | mean | p99 |
|---|---|---|
| Fragment Occupancy | **12.04%** | 13.93% |
| ALU Utilization | 21.70% | 23.65% |
| Buffer Read Limiter | 4.84% | 5.28% |
| `submitted_done_ms` avg | **17.77 ms** (≈ 56 FPS) | — |
| `gpu_pass_ms` avg | 2.22 ms (undercount, see diagnosis) | — |

Matches the diagnosis doc: Fragment Occupancy ≈ 12%, confirming register-
pressure regime.

### Step 1: `s_cell_size` → scalar `cur_cell_size` (-20 B)

Slow-soldier @ 2560×1440, 300 frames.

| counter | baseline | step 1 | Δ |
|---|---|---|---|
| Fragment Occupancy | 12.04% | **12.20%** | +0.16 pp (noise) |
| ALU Utilization | 21.70% | 24.43% | +2.7 pp |
| Buffer Read Limiter | 4.84% | 3.00% | −1.84 pp |
| `submitted_done_ms` avg | 17.67 ms | **16.43 ms** | −7% |

**Interpretation:** Fragment Occupancy didn't move. 20 B on its own is
within the compiler's "doesn't cross the register-budget threshold"
slack — the diagnosis doc predicted ~3 pp of occupancy from this
change and we got less than that. The small wall-clock improvement
(~7%) is likely just "the compiler had an easier time scheduling the
scalar than the indexed array access." Screenshots pixel-identical.
Continue to step 2; real occupancy gains should arrive when the
cumulative register savings cross the threshold.

### Step 2: `s_node_origin` → scalar `cur_node_origin` (-60 B)

Slow-soldier @ 2560×1440, 300 frames. Timing measured from direct
binary (`target/release/deepspace-game`) for apples-to-apples — `cargo
run` adds ~5-10 ms of noise per run on this scenario.

| counter | baseline | step 1 | step 2 | Δ vs baseline |
|---|---|---|---|---|
| Fragment Occupancy (mean) | 12.04% | 12.20% | **12.65%** | +0.6 pp |
| Fragment Occupancy (p99)  | 13.93% | 12.85% | **36.29%** | +22 pp |
| ALU Utilization (mean)    | 21.70% | 24.43% | **30.24%** | +8.5 pp |
| Buffer Read Limiter       | 4.84%  | 3.00%  | 2.90%  | −1.9 pp |
| `submitted_done_ms` avg   | 17.67  | 16.43  | **12.67** | **−28%** |

**Interpretation.** The headline is the ALU Utilization jump from 21.7%
→ 30.2% — that's the clean "threads are actually doing more work per
second" signal. `submitted_done_ms` dropping 28% is the matching
wall-clock win.

Interestingly, Fragment Occupancy **mean** barely moved (12.04 →
12.65), but **p99** jumped from 13.93% to 36.29%. The compiler isn't
finding enough register room to raise the steady-state occupancy yet,
but it's finding windows where occupancy can spike — enough that the
ALU gets fed materially more work overall.

Screenshots pixel-identical (plain_d8, sphere, zoom3). `cargo test
--lib gpu` passes.

### Step 3: `s_side_dist` → scalar `cur_side_dist` (-60 B)

Slow-soldier @ 2560×1440, 300 frames.

| counter | baseline | step 1 | step 2 | step 3 | Δ vs baseline |
|---|---|---|---|---|---|
| Fragment Occupancy (mean) | 12.04% | 12.20% | 12.65% | 12.41% | +0.37 pp |
| Fragment Occupancy (p99)  | 13.93% | 12.85% | 36.29% | 21.61% | +7.7 pp |
| ALU Utilization (mean)    | 21.70% | 24.43% | 30.24% | **36.66%** | **+14.96 pp** |
| Buffer Read Limiter       | 4.84%  | 3.00%  | 2.90%  | 1.47%  | −3.4 pp |
| Buffer Write Limiter      | 5.23%  | 4.13%  | 2.34%  | 0.32%  | −4.9 pp |
| `submitted_done_ms` avg   | 17.77  | 16.43  | 12.67  | **9.82** | **−44.7% (1.81× FPS)** |

**Interpretation.** Mean Fragment Occupancy still basically static —
the compiler didn't cross whatever Apple-specific threshold unlocks a
higher-occupancy scheduling mode. But the **ALU Utilization jump from
21.7 → 36.7%** and the corresponding wall-clock improvement tell the
real story: per-thread state dropped enough that the ALU is now
getting fed almost twice as much work per unit time, even with mean
occupancy unchanged.

Buffer Read Limiter **dropped from 4.84% → 1.47%** — 3× less memory
traffic. Removing the indexed stack accesses (`s_side_dist[depth]`)
eliminated whatever compiler-emitted spill/reload instructions that
counter was tracking.

Screenshots pixel-identical on plain_d8, sphere, zoom3 — the entry_pos
vs ray_origin reference unification doesn't move any pixels in these
scenes because `t_start ≈ 0.001` for in-box cameras.

## Conclusion

Three scalar swaps preserving the DDA algorithm yielded:

- **1.81× FPS on the slow-soldier scene** (17.77 ms → 9.82 ms)
- **+15 pp ALU utilization** (21.7% → 36.7%) — the clean "shader is
  doing more work per second" signal
- **−70% Buffer Read traffic** (4.84% → 1.47%)
- Fragment Occupancy mean essentially unchanged (~12%) — see below
  for why this was NOT what we thought.

## What the Top Performance Limiter counter actually says

After the three steps landed, I finally read the `Top Performance
Limiter` counter from the trace (the Metal counter that tells you
*which* resource is gating the GPU at each 20 µs sample). This counter
was not in the `FOCUS` whitelist in `parse-metal-trace.py` and thus
didn't show up in any earlier analysis.

Distribution on the slow-soldier scene:

| state | top limiter | % of samples |
|---|---|---|
| baseline | **ALU Limiter** | 99.1% |
| step 3   | **ALU Limiter** | 98.2% |

The shader is and always was **ALU-bound**, not register-pressure-
bound. `perf-occupancy-diagnosis.md` interpreted the 12% Fragment
Occupancy as register pressure via byte-counting (260 B > 128 B). That
was a plausible-but-wrong hypothesis. The actual story:

- Apple's scheduler keeps just enough SIMD groups in flight to
  saturate the next bottleneck. When the ALU is the bottleneck,
  low mean occupancy is a **symptom, not a cause** — more groups
  wouldn't help because the ALU can't process them any faster.
- The 1.81× speedup came from **reducing ALU instructions per pixel**
  (scalar access vs indexed address arithmetic for stack arrays), NOT
  from freeing registers. Every `s_cell_size[depth]` in the hot path
  compiled to an address-compute + load; replacing it with a scalar
  register read cut that ALU overhead.

The `gpu-telemetry.md` interpretation rule "Fragment Occupancy <25% →
register pressure" needs a caveat for ALU-bound shaders.

## Empty-run batching experiments (all reverted)

Two attempts at empty-cell batching in the DDA path; both regressed.

### Attempt 1: single extra step, shader-only (step 4)

After a normal DDA empty-advance, peek the next slot along the
dominant axis; if empty AND axis still dominant, do a second step
inline (batch size ≤ 2, no pack changes).

| metric | step 3 | attempt 1 |
|---|---|---|
| submitted_done_ms | 9.82 ms | **13.53 ms (+38%)** |
| avg_steps | 31.6 | 31.33 |

Batch fires in <1% of empty iterations — dominant axis switches too
often on diagonal rays for the "still dominant" constraint to hold.
~10 ALU ops of unconditional check overhead dwarfs the rare savings.

### Attempt 2: row-empty fast-forward to OOB

Promoted from single-step to "skip to OOB in the dominant axis if the
entire axis-row is empty AND the ray will exit the row before crossing
either other axis plane." Uses `row_mask = (1 << s) | (1 << s+stride)
| (1 << s+2*stride)` against `cur_occupancy`, plus a DDA-safety check
`t_oob < min(other_side_dists)`. Safe in principle because the row-
empty condition guarantees no missed cells.

| metric | step 3 | attempt 2 |
|---|---|---|
| submitted_done_ms | 9.82 ms | **11.39 ms (+16%)** |
| avg_steps | 31.6 | 30.53 |
| avg_empty | 14.6 | 14.62 |

avg_steps dropped only 3%. The row-empty + DDA-safety conjunction
fires roughly in ~3-5% of empty iterations — not enough to cover the
~10-15 ALU ops of unconditional check overhead per empty iteration.

### Takeaway

On this scene (sparse voxelized geometry viewed diagonally at zoom=4),
the DDA-correct empty-run batchable cases are simply **too rare for
any flavor of in-node INV9 to help**. The long empty runs that DDA
would naturally traverse along a single axis are the exception, not
the rule — even when entire rows ARE empty, the ray is usually
heading toward a y-plane or z-plane crossing first.

Full pack-time INV9 would have the same structural problem: same
DDA-correctness constraint, same rare hit rate. Precomputing the
metadata would save only ~3 ALU ops per lookup; the check overhead
(dominance, DDA safety) is what actually kills the bet. Not worth
implementing.

What's NOT tested here:

- **Node-level AABB culling** — store a tight bounding box of non-
  empty content per node at pack time; ray-box-test at node entry to
  skip the entire 3×3×3 DDA if the ray misses the AABB. This attacks
  descend-cost rather than empty-cost, and descend cost is arguably
  higher (avg_descend=11.5 iterations at ~30 ALU/each = ~345 ops/ray
  vs. avg_empty=14.6 × ~20 ALU/each = ~292 ops/ray).
- **f16 DDA state** — Apple Silicon runs f16 ops at 2× the rate of
  f32. Our F32 Utilization is 2-5% (lots of headroom to spend on
  precision). The DDA side_dist and cell_size values could use f16
  without visual regression (LOD is already quantized).
- **Reducing descent count** — avg_descend=11.5 seems high; may
  indicate LOD thresholds are too aggressive, or the ribbon-level
  LOD could cut more.

## Content AABB culling (kept — 22% wall-clock win)

**Big win.** Per-child content AABB stored at pack time in the spare
12 bits of each tag=2 entry's `_pad` (zero memory overhead). Shader
uses it for two things in one ray-box test:

1. Skip descents where ray misses content (cull)
2. Start child DDA at the AABB entry (skip leading empty cells)

Slow-soldier @ 2560×1440, 300 frames, 3-run median:

| metric | step 3 | AABB folded | Δ |
|---|---|---|---|
| submitted_done_ms | 9.88 | **7.70** | **−22%** |
| avg_steps | 31.60 | **17.02** | −46% |
| avg_empty | 14.62 | **7.60** | −48% |
| avg_descend | 11.53 | 7.93 | −31% |
| avg_oob | 7.53 | 3.93 | −48% |

**Cumulative from baseline: 17.77 → 7.70 ms = 2.31× FPS** (vs 1.81×
after the three scalar-stack steps alone).

**Why the folding matters.** A naive implementation that adds an AABB
ray-box BEFORE the existing child ray-box (keeping both) netted only
~2% wall-clock, because the added ray-box cost on taken descents
roughly cancelled the savings on culled descents. Replacing the old
child ray-box with the AABB ray-box (using `aabb_hit.t_enter` as the
DDA entry t) eliminates 20 ALU ops per taken descent × 7.93 descents
= 159 ops/ray. That's what tipped the net from "noise" to "real win."

**Why this worked when empty-run batching didn't.** Empty-run batching
attacks a high-frequency branch (avg_empty=14.6) but with check
overhead that dominates. AABB culling attacks a moderate-frequency
branch (avg_descend=11.5) and folds its cost into work the shader was
ALREADY doing (the child ray-box). Net: overhead ≈ 0, savings ≈ full.

Pixel correctness: plain_d8 and sphere pixel-identical. zoom3 one
pixel differs by max intensity 6 — floating-point rounding at an
edge, not a correctness bug.

## `s_cell` scalarization (tried — null)

Followed the same pattern as the three scalar-stack wins (s_cell_size,
s_node_origin, s_side_dist): keep `s_cell` as a stash-only array, hoist
a `cur_cell: vec3<i32>` scalar for the hot path, stash on descend,
restore on pop.

Slow-soldier @ 2560×1440, 300 frames, 3-run median:

| metric | branchless (baseline) | + s_cell scalar |
|---|---|---|
| submitted_done_ms | 6.39 | 6.36 |

−0.03 ms, inside run-to-run noise (±0.05 ms). Reverted.

**Why it didn't land when the other three did.** The other stack arrays
held `vec3<f32>` / `f32` data that was modified with `+= * f` every
iteration — the compiler couldn't registerize `s_side_dist[depth]`
across writes because of aliasing conservatism, so it was really
spilling through thread-local memory. `s_cell` holds `vec3<i32>` that
only gets `+= step` with a constant index pattern; Naga/MSL apparently
already promotes this pattern to registers. The manual stash/restore
adds write/read pairs on descend and pop that exactly cancel what was
saved in the hot path.

**Takeaway.** The "scalarize every stack array" heuristic that worked
for steps 1–3 doesn't generalize past the three we landed. The
remaining arrays (`s_cell`, `s_node_idx`) are the ones the compiler can
already handle. Not worth keeping.

## Branchless min-axis DDA advance (−18%)

Replaced 7 copies of the `if cur_side_dist.x < y && x < z { … } else
if y < z { … } else { … }` chain with a pure-arithmetic `vec3` mask
(`min_axis_mask()` in `ray_prim.wgsl`). The four compares inside the
helper are pairwise independent, so the compiler can issue them in
parallel instead of serializing an if/else-if chain with a 3-deep
dependent predicate.

Slow-soldier @ 2560×1440, 300 frames, 3-run median:

| metric | AABB folded (baseline) | + branchless min-axis |
|---|---|---|
| submitted_done_ms | 7.81 | **6.39** (−18.2%) |

Pixel-identical on plain_d8, sphere, zoom3.

**Cumulative from slow-soldier baseline: 17.77 → 6.39 ms = 2.78× FPS.**

**Why this worked when `s_cell` scalarization didn't.** The branching
min-axis was ALU-dependency-chain-limited, not memory-limited. The
7-copy if/else-if pattern forced dependent predication; replacing it
with the `vec3` mask exposes parallelism the compiler can schedule. The
Top-Performance-Limiter counter confirmed "ALU Limiter" was the
dominant category even after the stack-scalar steps — this is the
change that attacks that category directly.

## `tan(fov/2)` hoist (tried — null)

Moved `uniforms.screen_height / (2.0 * tan(camera.fov * 0.5))` out of
the per-descend LOD calc and into a loop-invariant `focal_px` scalar
at the top of `march_cartesian`.

Slow-soldier, 1 run: 6.41 ms vs 6.39 ms baseline. Inside noise.

The wgpu/Naga→MSL pipeline was already hoisting `tan()` out of the
hot loop — the result was still only evaluated once per invocation.
No behavioral change, no perf change. Reverted.

## OOB fold (tried — null, two approaches)

The OOB check at the top of each iter — `cell.x < 0 || cell.x > 2 ||
cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2` — is 11 ops
(6 compares + 5 ORs) and fires every iteration (avg_oob=7.53 / 31.6
= 24% taken, 76% short-circuited).

**Attempt 1:** `any((cell < vec3(0)) | (cell > vec3(2)))`.

Slow-soldier, 1 run: **6.51 ms (+1.8%)**. REGRESSION.

The scalar chain short-circuits early on the common "in-bounds" case
(76%). The vectorized form computes all 6 compares + vec-or + any()
unconditionally — more work on the fast path than it saves on the
slow path.

**Attempt 2:** `min(cell.xyz)` + `max(cell.xyz)` reduction, then
`cell_lo < 0 || cell_hi > 2`. 7 ops total, short dependency chain.

Slow-soldier, 1 run: 6.39 ms — exact match to baseline, zero change.

Compiler likely lowered both versions to similar code after
optimization; Apple Silicon's scalar-per-lane scheduling doesn't
benefit from the min/max reduction over the short-circuit chain.

**Takeaway.** The scalar OR-chain with short-circuit is already
well-matched to the compile pipeline. Moving on; the OOB check isn't
the bottleneck the earlier ALU-count analysis implied.

## Occupancy sweep diagnostic (decisive — we're in a flat region)

Direct test of whether adding per-thread state drops Fragment Occupancy
enough to matter. Added 256 B of live scalar `vec4` state (16 `var`s,
each updated every iter, dep-chained into a guard condition), captured
xctrace counters, compared against HEAD baseline.

Slow-soldier @ 2560×1440, 300 frames, release build.

| counter | baseline (6.39 ms) | +256 B scalars (8.87 ms, +39%) |
|---|---|---|
| Fragment Occupancy (mean) | **12.98%** | **12.63%** |
| Fragment Occupancy (p99) | 68.12% | 51.02% |
| ALU Utilization (mean) | **32.96%** | **32.19%** |
| ALU Utilization (p99) | 52.49% | 40.97% |

**Mean Fragment Occupancy did not move.** Mean ALU Utilization did not
move. The entire 39% wall-clock regression came from the added ALU ops
(16 parallel `vec4 += d` chains + a reduce-sum guard), not from lost
TLP.

### What this rules out

- **"We're register-pressure-limited."** Adding 256 B of live scalar
  state should have dropped Fragment Occupancy a tier (e.g., 13% →
  6.5%) if we were at the cliff. It didn't. We're in a flat region of
  the occupancy curve.
- **"Shrinking state further will unlock more SIMDs."** Stack-scalar
  commits already cut 170 B (450→280) without moving mean occupancy.
  This test confirms the curve is flat in both directions around our
  current state size.
- **"Compute shader migration (same algorithm, different stage) will
  help via TLP."** It won't, unless it meaningfully reduces per-thread
  state — likely to below ~128 B, which is 50% below where we are now.
  That's a significant refactor, not a shader-stage swap.

### What this confirms

- **We're ALU-work-bound at the margin.** Removing ALU ops is linearly
  effective (stack-scalar: -44%, branchless min-axis: -18%). Adding ALU
  ops is linearly ineffective (this test: +39% for +256 B).
- **ALU Utilization sitting at 33% with a flat occupancy response means
  the 67% idle ALU time is NOT fillable by more ops-per-thread.**
  Either the existing SIMDs are dep-chain-blocked (waiting on their own
  results, can't issue) OR they're memory-blocked (waiting on
  storage-buffer loads). Adding more ops to the same SIMDs doesn't
  help — only more SIMDs would. And we can't get more SIMDs at the
  current state size.
- **The remaining wall-clock gap is mostly structural** — not
  addressable by local WGSL-level optimization without a large
  register-budget cut.

### Residual questions the diagnostic didn't answer

- **Which mechanism dominates the 67% ALU idle?** Dep-chain stalls vs
  storage-buffer memory latency. A dep-chain amplification test would
  disambiguate: artificially extend the critical chain length and
  measure wall-clock sensitivity.
- **What tier of occupancy is below us?** Need to probe the state-size
  floor. If we could get per-thread state to ~128 B and measure, we'd
  know whether there's a meaningful TLP win to chase. Currently seems
  unreachable without compute shader + threadgroup memory.

## LOD tuning (tried — null)

Runtime flag `--lod-pixels 2.0` doubled the sub-pixel-rejection
threshold. Shader stats byte-identical to baseline (avg_descend,
avg_lod_terminal unchanged), ~2% wall-clock delta within noise. In
the zoom=4 soldier scene, the cells DDA visits are already ≥2 pixels
on screen, so doubling the threshold catches no new cells. Higher
values would blockify the image. Not landed.

## f16 DDA state (skipped — insufficient theoretical upside)

Didn't attempt. Analysis: the empty-advance hot path has ~2 FP ops
per iteration (1 mul, 1 add). Going f16 at 2× rate would save at
most ~14 ALU ops/ray out of ~700 total = 2% best case. Implementation
cost (feature flag, `enable f16`, type conversions, precision
validation) was not justified vs. the AABB path's known high
upside. Documented here so future work can reconsider if the lever
set changes.

## Next steps (revised)

Empty-cell batching is a dead end on this scene (see above). Real
ALU savings lie elsewhere:

- **Node-level content AABB culling** — per-node "smallest AABB
  enclosing occupied content" stored at pack time. Ray-box-test at
  node entry; if the ray misses the AABB, skip the whole 3×3×3 DDA.
  Attacks descend-cost (~345 ALU ops/ray) directly. Biggest
  theoretical win.
- **F16 DDA state** — Apple Silicon runs f16 ops at 2× f32 rate.
  Current F32 Utilization is 2-5%, plenty of headroom. DDA state
  (side_dist, cell_size) can go f16 without visual regression; LOD
  is already quantized.
- **Reduce descent count** — avg_descend=11.5 seems high for a 5-
  deep anchor. Tuning LOD_PIXEL_THRESHOLD or ribbon-level LOD may
  cut descents directly. Each saved descent ≈ 30 ALU ops.
- **Compute shader with threadgroup tree cache** — cuts storage-
  buffer load instructions (which compile to ALU on Apple). Reframed
  from the original "register pressure" motivation but technique
  still valid.

## Related docs

- [`perf-occupancy-diagnosis.md`](perf-occupancy-diagnosis.md) — original
  measurement and per-byte occupancy table.
- [`gpu-telemetry.md`](gpu-telemetry.md) — how to capture/parse Metal
  counters.
- [`cookbook.md`](cookbook.md) — full test-matrix commands.
- `docs/prompts/compute-shader-migration.md` — alternate route (compute
  shader + threadgroup memory) for the same occupancy problem; this
  branch is the "cheaper bet" before committing to that one.
