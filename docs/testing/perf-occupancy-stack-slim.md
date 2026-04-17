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

Slow-soldier @ 2560×1440, 300 frames, release build.

| counter | mean | p99 |
|---|---|---|
| Fragment Occupancy | **12.04%** | 13.93% |
| ALU Utilization | 21.70% | 23.65% |
| Buffer Read Limiter | 4.84% | 5.28% |
| `submitted_done_ms` avg | **17.67 ms** (≈ 57 FPS) | — |
| `gpu_pass_ms` avg | 2.04 ms (undercount, see diagnosis) | — |

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

Results pending.

### Step 3: `s_side_dist` → scalar `cur_side_dist` (-60 B)

Results pending.

## Related docs

- [`perf-occupancy-diagnosis.md`](perf-occupancy-diagnosis.md) — original
  measurement and per-byte occupancy table.
- [`gpu-telemetry.md`](gpu-telemetry.md) — how to capture/parse Metal
  counters.
- [`cookbook.md`](cookbook.md) — full test-matrix commands.
- `docs/prompts/compute-shader-migration.md` — alternate route (compute
  shader + threadgroup memory) for the same occupancy problem; this
  branch is the "cheaper bet" before committing to that one.
