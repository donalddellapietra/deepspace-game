# Sparse-fractal traversal cost — what we've measured

Status: open. Last updated 2026-04-19.

This document captures **only what has been empirically verified** for the
Jerusalem-cross deep-zoom rendering problem. Anything speculative
(theoretical speedup estimates, un-measured optimizations, references to
papers without a local benchmark) is called out explicitly as such.

## 1. The problem, as measured

**Scenario**: `--jerusalem-cross-world --plain-layers 20 --spawn-xyz 1.5 1.5 1.5 --spawn-depth 7` at 2560×1440 native resolution. Reproduced via `scripts/repro-jerusalem-lag.sh`.

**Measured cost** (30-frame average, headless render-harness, `--shader-stats`):

| metric | value |
|---|---:|
| submitted_done avg (post-commit 04a2a9b, s_cell packed) | **~52 ms** |
| submitted_done avg (pre-commit 04a2a9b, baseline) | ~77 ms |
| submitted_done worst frame | 75-80 ms (post-pack); 85-95 ms (pre-pack) |
| avg_steps per ray | 136.61 |
| avg_empty | 61.05 |
| avg_oob (ribbon pops) | 28.48 |
| avg_descend | 29.31 |
| avg_lod_terminal | 0.50 |
| hit_fraction | 0.125 |
| avg_steps per hit | **78.49** |
| avg_steps per miss | **144.91** |
| avg_loads_total (storage-buffer u32 reads) | 326.71 |
| &nbsp;&nbsp;tree[] | 208.59 |
| &nbsp;&nbsp;node_offsets[] | 67.80 |
| &nbsp;&nbsp;node_kinds[] | 39.69 |
| &nbsp;&nbsp;ribbon[] | 10.63 |

**88% of rays miss content entirely.** They're not stuck in a loop — they bounce through the fractal's sparse recursive structure looking for a hit that never comes, until Nyquist or OOB terminates them.

**Hit rays also do substantial work.** 78 steps per hit on average — rays that *eventually* find content still bounce through ~78 descents/pops first. The naive model "hits terminate fast, misses do the work" is wrong for fractals of this shape. Both branches are expensive; misses are ~2× hits.

**Session-to-session variance is ~5 ms** on the same binary and scenario. Interleaved A/B measurement within a single session is required to detect anything smaller than ~3 ms.

## 2. The depth cliff

Measured across plain_layers at the same camera pose:

| plain_layers | submitted_done | avg_steps | hit_fraction |
|---:|---:|---:|---:|
| 8 | 3.99 ms | 4.00 | 1.00 |
| 12 | 4.00 ms | 4.00 | 1.00 |
| 16 | 37.80 ms | 115.90 | 0.17 |
| 20 | 52.16 ms | 136.61 | 0.12 |

The cliff is between 12 and 16, not a gradual climb. At `plain_layers ≤ 12`, the carved air tunnel around the camera extends to the tree's leaves; rays exit air and hit Block leaves immediately. At `plain_layers ≥ 16`, there are 4+ levels of fractal recursion **below** the tunnel's bottom; rays bounce through them without finding content.

## 3. The `gpu_pass_ms` metric was unreliable — removed

An earlier iteration had a `gpu_pass_ms` metric from `TIMESTAMP_QUERY` render-pass-boundary timestamps. On Jerusalem it read ~6 ms while `submitted_done_ms` read ~52 ms, which suggested a large "non-shader gap."

That gap was an artifact. On Apple Silicon Metal, per-render-pass timestamp counters are **not guaranteed to be monotonic for fast passes** — end-tick can read *before* start-tick. The code already compensated with `abs(delta)`, but the resulting values were noise for our pass durations. See the removed code at `src/renderer/draw.rs` and commit history for the specifics.

**The timestamp-query scaffolding was deleted.** `submitted_done_ms` (queue.submit → on_submitted_work_done callback) is the only authoritative GPU-side wall-clock signal on this platform.

## 4. The hallway analogy (verified mental model)

Our voxel DDA walks one cell at a time. In a sparse node (Jerusalem: 7 occupied, 20 empty of 27), every empty cell the ray crosses is one DDA step. Nyquist only stops *descent* into sub-pixel cells; it does not stop *lateral* traversal of above-Nyquist empty cells at the current level.

For Jerusalem-nucleus: rays start in a small air pocket, exit it, traverse nested sparse crosses at multiple scales. 61 of 137 steps are empty-cell advances; 29 are descents; 28 are ribbon pops. Descents and pops come in ~matched pairs (ray descends into a kept cell, finds nothing it hits, pops out).

## 5. PySpace comparison — what we verified

**PySpace's native scenes at 2560×1440, z=3.0, headless uncapped:**

| scene | iterations | ms/frame |
|---|---:|---:|
| menger | 8 | 4.15 |
| sierpinski_tetrahedron | 9 | 4.4 |
| mausoleum | 8 | 10.4 |
| mandelbox | 16 | 16.6 |
| butterweed_hills | 30 | 16.9 |
| test_fractal | 20 | 20.7 |
| tree_planet | 30 | **>50 ms** (didn't finish 25-frame benchmark in 18s) |
| snow_stadium | 30 | **>60 ms** (didn't finish) |

**Our Menger at the same resolution: 4.37 ms** — parity with PySpace's menger. Our worst case (Jerusalem nucleus, 52 ms) sits in the same range as PySpace's heavier scenes.

**PySpace can also be slow.** "SDF is always fast" is wrong. Both approaches scale up with complexity.

## 6. What PySpace actually does

Verified by reading `external/PySpace/pyspace/fold.py`, `object.py`, `frag.glsl`:

- Fold iterations (`for _ in range(8)`) are **unrolled at compile time** in the generated GLSL. No runtime loop.
- Every pixel pays the full iteration count in every DE evaluation. No early escape.
- Inside-solid renders degenerately: DE returns ≤ 0, ray terminates on step 0, shader renders a single flat surface color.
- Our `/tmp/pyspace_fork/pyspace/frag.glsl` adds an "inside-escape" patch that steps through starting-solid material before hitting-testing, making the inside view render meaningful content.
- With the patch, PySpace's fold-jerusalem-cross "inside nucleus" view renders in 6.16 ms at 2560×1440 — but the Jerusalem-cross construction in `/tmp/pyspace_uncapped.py` is **structurally wrong** (does not faithfully produce the 7-of-27 Jerusalem attractor per the agent's analysis). So the 6.16 ms benchmark is on a different attractor than ours, not a fair comparison.

## 7. Analyzer results (jerusalem, plain_layers=20)

Run via `cargo run --release --bin perf_opt_analysis -- jerusalem 20`.

### 7a. Per-axis distance field

Every axis is symmetric (by fractal construction):

| axis | mean run | run=1 | run=2 | run=3 |
|---|---:|---:|---:|---:|
| +X / -X / +Y / -Y / +Z / -Z | 1.60 | 60% | 20% | 20% |

An axis-dominant ray can skip an average of 1.60 empty cells per DDA advance — not 1.00 as the older `df_analysis`'s worst-octant metric suggested.

The old metric (worst-octant Chebyshev) = min over all 6 axis directions = 1.00 because any single axis-ray might have run=1. But a real ray isn't constrained to take the minimum — it takes *its own axis's* run. Per-axis DF gives ~60% reduction in empty-cell advances for axis-dominant rays.

### 7b. Path-mask cull rate

For every (node, entry_cell, ray_octant) configuration — enumerating all 20 Jerusalem nodes × 27 entry cells × 8 octants = 4320 triples — checking whether `occupancy & reachable_mask == 0`:

- **25.9% of configurations culled**.
- All Jerusalem nodes have popcount=7; rate is uniform across nodes.
- A culled descent eliminates: the descend itself + ~2 empty-cell advances inside + the eventual pop ≈ 4 DDA steps.

### 7c. Earlier projection — turned out wrong

A prior version of this doc projected "1.6× speedup, 52 → 32 ms" from combining per-axis DF and path-mask cull. **The path-mask portion of that projection was implemented and measured; it delivered zero wall-clock improvement** (see §10). The per-axis DF portion has not been implemented but its mechanism is similar (reduce per-ray step count in empty regions), so the same zero-translation risk applies.

**Lesson**: step-count and load-count reductions do not translate linearly (or at all) to wall-clock on Apple Silicon for this workload. Projections based on step-count math alone should not be trusted without a prototype-and-measure.

## 8. What's verified about our system's internals

- **Tree structure**: base-3 (3×3×3 children per Node), 27-bit occupancy mask per node, `first_child_offset` pointing into `tree[]`, packed child entries (tag, block_type, content_aabb, node_index). Nodes are content-addressed and shared across the fractal's self-similarity (Jerusalem at `plain_layers=20` has only 20 unique nodes in the packed tree).
- **DDA per cell**: check occupancy bit → if set, check tag → if tag=0 or empty-rep, advance DDA one cell; if tag=1, hit Block; if tag=2, check AABB, descend or advance.
- **LOD gate**: single threshold `LOD_PIXEL_THRESHOLD = 1.0` (Nyquist). Stack depth capped at `MAX_STACK_DEPTH = 8` (hardcoded const, fires rarely — see `avg_lod_terminal = 0.50`).
- **Ribbon pops**: when ray exits the current frame, pop to ancestor, recompute `side_dist` from scratch (~6 FMAs per pop).
- **Expand-carve**: ensures the camera's anchor path is tree-walkable down to `anchor.depth() - 1` by inserting fresh empty Nodes where the walk would hit Empty or Block. Critical for `(1.5,1.5,1.5)` spawn to be in air.

## 9. Free space in the existing pack layout (verified)

Per Cartesian node, currently unused bits:

- 5 bits in occupancy u32's high bits
- 4 bits per packed child (in the `_pad` field beyond the 12-bit content_aabb)
- 96 bits in the node_kinds entry (unused for Cartesian; only CubedSphereBody/Face use them)
- 6 bits per child in the tag byte

For a 7-slot Jerusalem node: 171 free bits.
For a 20-slot Menger node: 301 free bits.

A per-axis DF at 2 bits × 6 axes × 27 slots = 324 bits doesn't fit in this free space. A per-slot DF at 2 bits × 27 = 54 bits does. Per-axis requires growing node size OR using a side buffer.

## 9b. What HAS worked (measured, merged, non-trivial)

- **Pack `s_cell` from `vec3<i32>×8` to `u32×8`** (commit 04a2a9b). Cell coords range -1..=3, so 3 bits per axis packs into a single u32 per depth. Reduces per-thread DDA stack storage by 64 B (96 B → 32 B).

  Metal GPU Counters diagnosed the bottleneck beforehand: Fragment Occupancy 9.7% mean (rule: <25% = register pressure), ALU Utilization 29%, Buffer Read Limiter 2.5%. Not compute- or bandwidth-bound; occupancy-bound due to per-thread register budget.

  **Measured wall-clock: 77.6 → 52.2 ms (-25.4 ms, -33%)** on Jerusalem nucleus 2560×1440, interleaved A/B (6 samples each, zero distribution overlap). All shader stats counters identical (avg_steps=136.61, hit_fraction=0.1249 etc.) — same traversal, just runs faster.

  First optimisation in this investigation where step-count and load-count stayed flat but wall-clock moved substantially — validates that on Apple Silicon the dominant performance model is occupancy / register budget, not instruction or load count.

## 10. What we've measured NOT to work (or to be broken)

- **Retiring BASE_DETAIL_DEPTH** (the ribbon-shell LOD gate) cost us ~20% perf on soldier, ~none on plain, but fixed correctness for fractals. Documented in `docs/testing/perf-lod-diagnosis.md` epilogue.
- **Early attempt at path-mask cull** using 7-iter DDA simulation cost more than it saved on an earlier baseline. The per-check cost was too high — the closed-form tensor-product math (implemented later) is cheaper per check.
- **Closed-form path-mask cull (production)** — implemented, measured, reverted. Reduced `avg_steps` 136.6 → 115.0 (−16%), `max_steps` 2723 → 1839 (−32%), `avg_loads_total` 327 → 294 (−10%). Wall-clock submitted_done was **indistinguishable from baseline** (both ~60 ms, within session variance). The GPU's warp scheduler appears to hide the eliminated descent work behind other rays' memory stalls, so removing it doesn't shorten the frame.
- **Inline child occupancy into parent's child entry** (grows child entry 2 u32s → 3 u32s, tree[] size +40%) — implemented, measured. Gave a reproducible ~5 ms improvement (70 → 65 ms, ~7.7%) via interleaved A/B. **Reverted** because the gain is marginal, the storage cost scales with world size, and it could be net-neutral or worse on GPUs with smaller L1 caches.
- **Stack-cache parent headers** (add `s_occupancy[]` and `s_first_child[]` arrays per depth; pop reads from them instead of loading `tree[]`). Implemented, measured. **2× regression**: 60 ms → 120 ms. Adding per-thread arrays pushed the compiler past Apple Silicon's register budget; arrays likely spilled to thread-private memory where accesses are per-thread (not shared across the warp), so they became slower than the cache-resident tree[] loads they replaced. Reverted.
- **MAX_STACK_DEPTH = 5** breaks fractal rendering (rays terminate at LOD before reaching Block leaves, show representative_block = majority color = monochromatic). 8 is the verified floor for our default `plain_layers = 8` fractals.
- **`df_analysis.rs`'s within-node Chebyshev DF ("is a dud")** is correct in its narrow claim but the worst-octant metric that underlies it over-conservatively assumes pure-diagonal rays. Per-axis DF gives 1.60, not 1.00.
- **Multi-layer DF expansion analysis** (`df_analysis --expand N`): ran across 4 fractals at expansion depths 1–3. OUTER (tag=0 anchor-slot) mean DF in parent-cell units ranged 0.20–0.86 and **decreased** with more expansion. A DF-guided DDA would take smaller steps than the current 1-cell stride — strictly worse. DF is ruled out for this tree family.

## 11. What remains unverified

Optimizations discussed but **not implemented and measured** in our system. All are speculative until benched locally:

- **ESVO-style contour planes** (Laine & Karras): paper reports 5–10× on their test scenes, but those scenes have a very different shape (thin surfaces, not volumetric sparse fractals). Our measured `avg_steps_per_hit = 78.49` means a contour-cull targeting miss reduction has a realistic wall-clock ceiling around "misses down to hit-cost" ≈ ~30 ms frame time, **not** the 8–10 ms a naive model would predict (which assumed hits terminate in ~0 steps). The mechanism also overlaps with path-mask cull, which already proved to give zero wall-clock on this GPU — caveat aggressively.
- **Cone marching**: no prototype.
- **Per-axis DF in shader**: no prototype. Its mechanism (reduce empty-cell DDA step count for axis-dominant rays) is the same class as path-mask cull, which delivered 0 ms wall-clock. Speculative.
- **Inline first_child in child entry** (would grow tree[] another ~40%): not tried. Same risk-benefit profile as inline-occupancy (marginal wall-clock, real storage cost) — but no guarantee.
- **Pack `kind` into the 4 free bits of the packed u32** (zero storage cost): not tried. Would save one `node_kinds[]` load per tag=2 encounter. Unclear if on critical path.
- **Reducing plain_layers for rendering-only subtrees**: not tried. Would reduce the 78-step hit cost directly. Design-level tradeoff (fewer content levels below anchor).

## 12. Action items — current reading

The previous version of this section recommended path-mask cull and per-axis DF as the top actions based on a step-count projection (§7c). That projection turned out to be wrong: path-mask cull was implemented and delivered 0 ms wall-clock. Treat the following list as **candidates to measure**, not confident recommendations.

Candidates in decreasing order of expected payoff, all caveated:

1. **Accept the current floor and focus on design choices.** `plain_layers = 20` with camera at anchor 7 puts leaf content 13 levels below the anchor, so ~78 steps/hit and ~145 steps/miss are baked into the workload shape. Reducing the gap between anchor depth and leaf depth (world-gen or adaptive-depth choices) changes the cost more than any of our measured shader optimizations have.

2. **Pack `kind` into packed flags** (zero storage cost). Would remove ~29 `node_kinds[]` loads per ray. Expected wall-clock impact: unknown, probably small. Worth trying because it's free.

3. **Inline first_child in child entry** (+~40% more tree[]). Only experiment IF (2) is a measurable win, because it has the same mechanism (same cache line as already-loaded data) that made inline-occupancy give its small win. Storage cost is real.

4. **A fundamentally different algorithm** (cone marching, ESVO contours, BVH, surface-representation change). These would require architectural work and haven't been prototyped; speculative.

**What we explicitly do NOT recommend** based on measured outcomes:
- Path-mask cull variants (closed-form tested; zero wall-clock benefit)
- Any within-node distance field (DF analyzer proves it can't help this fractal family)
- Per-thread array caching of any kind (stack-header experiment was a 2× regression)
- Chasing further step-count or load-count reductions without a wall-clock hypothesis (we've now seen both counters drop without frame time moving)

## 13. Open questions worth investigation

- At what `plain_layers` does the cliff actually hit (13? 14? 15?)? Current data sweeps `{8, 12, 16, 20}`; the inflection is between 12 and 16, not narrowed further.
- Are there other camera angles at anchor=7 that hit similar costs, or is `(1.5, 1.5, 1.5)` uniquely bad because it's on the body-center axis of the fractal?
- What fraction of the 60 ms wall-clock is "shader work the compiler emitted" vs "memory stalls the warp couldn't hide"? An Xcode Instruments / Metal System Trace capture would answer this — we've been inferring from shader-side counters, which proved unreliable as wall-clock predictors.
- Do the same optimizations (or their failure modes) reproduce on non-Apple-Silicon GPUs (Vulkan / D3D12 via wgpu)? The register-pressure cliff we hit with the stack-cache experiment is Apple-Silicon specific; other GPUs may behave differently. The inline-occupancy win is plausibly GPU-dependent too.
