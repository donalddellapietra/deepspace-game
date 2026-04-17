# Perf diagnosis: the ray-march shader is register-pressure-limited

Ground-truth GPU telemetry from Metal on an M2 Max. Run `scripts/capture-gpu-trace.sh`
+ `scripts/parse-metal-trace.py` to reproduce. See
[`gpu-telemetry.md`](gpu-telemetry.md) for how to use the scripts.

## Why this doc exists

For a long time, the working theory on the "40 FPS zoomed-in" problem
was "memory bandwidth bottleneck" or "TBDR tile resolve costs" — both
based on indirect reasoning from `gpu_pass_ms` vs `submitted_done_ms`
gap and resolution sweeps. Apple Silicon's per-pass timestamp undercounts
TBDR shader work (documented in `perf-isolation.md`), so the gap is
misleading — `submitted_done - gpu_pass` doesn't tell you WHAT is taking
the time, only that SOMETHING is.

Real Metal GPU counters tell a different story, and it's actionable.

## The scenario

```
--vox-model assets/vox/soldier_729.vxs
--plain-layers 8 --spawn-xyz 1.15 1.1 1.04 --spawn-depth 5
--harness-width 2560 --harness-height 1440
```

Camera inside the voxelized soldier's body at zoom_level=4 (anchor_depth=5).
Rendering at native retina. **Measures 43 FPS (21.6 ms/frame).**

Shader counters report:
- `avg_steps = 31.6` (iterations per ray)
- `avg_empty = 14.6`, `avg_descend = 11.5`, `avg_oob = 7.5`, `avg_lod_terminal = 4.0`

## What the Metal counters actually show

M2 Max, 300-frame capture:

| Counter | Mean | p99 | Max | Interpretation |
|---|---|---|---|---|
| **Fragment Occupancy** | **12.4%** | 15.2% | 96.3% | **~1/8 of GPU parallel capacity in use** |
| ALU Utilization | 23.4% | 25.5% | 72.9% | Low — compute-starved |
| F32 Utilization | 2.1% | 3.6% | 30.7% | Tiny — most work is integer |
| Buffer Read Limiter | 4.1% | 4.5% | 43.5% | Memory NOT the bottleneck |
| Buffer Load Utilization | 4.2% | 4.5% | 5.0% | Low load traffic |
| Buffer Write Limiter | 5.2% | 6.2% | 57.7% | Tile resolve NOT the bottleneck |
| Threadgroup/Imageblock Store | 0.05% | 1.0% | 14.4% | Nearly zero — confirms not resolve |

All the bandwidth counters are at ~5%. The memory and cache aren't
stressed. But **Fragment Occupancy is only 12.4%.**

## What Fragment Occupancy means

Apple Silicon fragment shaders run in SIMD groups (32 threads each). The
GPU can have many SIMD groups in flight concurrently — that's what
hides memory latency and keeps the ALU units busy. "Fragment Occupancy"
measures the actual number of in-flight SIMD groups relative to the
hardware maximum.

**At 12.4% occupancy, only ~1/8 of the parallel capacity is running.**

At 100% occupancy with our measured 21.6 ms frame, theoretical frame
time would be ~2.5 ms = 400 FPS. The 8× gap between measured 43 FPS and
theoretical 400 FPS is almost entirely occupancy, not memory, not shader
algorithmic complexity.

## Why occupancy is low: register pressure

The `march_cartesian` shader holds a per-depth stack of traversal state:

```wgsl
var s_node_idx:     array<u32, MAX_STACK_DEPTH>;       //  5 × 4  =  20 bytes
var s_cell:         array<vec3<i32>, MAX_STACK_DEPTH>; //  5 × 12 =  60 bytes
var s_side_dist:    array<vec3<f32>, MAX_STACK_DEPTH>; //  5 × 12 =  60 bytes
var s_node_origin:  array<vec3<f32>, MAX_STACK_DEPTH>; //  5 × 12 =  60 bytes
var s_cell_size:    array<f32, MAX_STACK_DEPTH>;       //  5 × 4  =  20 bytes
// plus cur_occupancy, cur_first_child, normal, ...    ≈        40 bytes
// Total ≈                                                     260 bytes per thread
```

Apple Silicon's fragment shader register file supports roughly 128 bytes
per thread at maximum occupancy. At 260+ bytes of state, the compiler
must either spill to threadgroup memory (slow) or reduce occupancy
(reduce in-flight SIMD groups). It chooses reduced occupancy. Hence
Fragment Occupancy = 12%, not 50-100%.

This is also why INV6 reduced `MAX_STACK_DEPTH` from 64 to 5 — the
original arrays totaled ~3.5 KB per thread, which was catastrophic
spilling. 5 is the minimum that doesn't break correctness, but it's
still over the happy-occupancy register budget.

## Why other theories don't hold up

**"Memory bandwidth"**: Buffer Read Limiter mean 4.1%, max 43%. We use
maybe 5-10% of the M2 Max's 400 GB/s. Not memory-bound.

**"TBDR tile resolve"**: Threadgroup/Imageblock Store Limiter mean 0.05%.
Tile resolve is a rounding error.

**"Cache thrashing"**: Working set ~3.5 MB; SLC is 48 MB. Fits 14× over.
Buffer Load Utilization = 4.2% confirms we're not thrashing.

**"Compute-bound"**: ALU Utilization 23%. We're using a quarter of the
ALU. Not compute-bound — we're compute-STARVED (not enough threads
running in parallel to keep the ALU busy).

**"Shader algorithmic complexity"**: avg_steps = 32 per ray. The shader
is doing the right work. The problem isn't that each thread takes too
long — it's that too few threads run simultaneously.

## What to do about it

Reduce per-thread state. Every byte we cut from the shader's per-thread
register footprint buys back occupancy. Rough targets:

| Change | Bytes saved | Expected occupancy |
|---|---|---|
| baseline (current) | 0 | 12% |
| Drop `s_cell_size` (recompute from depth) | -20 | ~15% |
| Compress `s_side_dist` to f16 | -30 | ~18% |
| Combine `s_node_origin` + `s_cell` into one array | -40 | ~22% |
| Drop `s_node_origin` (recompute from `s_cell` chain) | -60 | ~30% |
| MAX_STACK_DEPTH=3 (unsafe for budget=4 LOD) | -88 | ~40% |

Double occupancy → roughly double FPS (on this scenario). Getting to
40% occupancy would take us from 43 FPS → ~140 FPS, a clean 3×. Hitting
the theoretical 100% occupancy gets us to 400 FPS, but is probably
impossible without a wholly different algorithm.

## What to avoid

**Don't add more per-depth state.** Anything "stored per stack level" in
a register array trades 4-12 bytes of occupancy per slot. If you need
to reason about parent cells, do it by recomputation, not by caching.

**Don't chase memory bandwidth optimizations here.** We use 5% of memory
capacity. Any optimization there is invisible in the frame time. Save
them for scenes that actually hit bandwidth (if we ever reach that).

**Don't chase TBDR resolve.** It's 0.05%. Irrelevant.

**Don't add threadgroup memory** as a "cheap spill." Threadgroup memory
ALSO reduces occupancy (it's per-SIMD-group, shared across threads).
Registers-to-threadgroup is not a win on Apple Silicon.

## Related signals

**Resolution scaling** still holds — half the pixels = half the total
work, so rendering at non-retina resolution gets ~2-3× regardless of
occupancy. Stacks multiplicatively with occupancy fixes.

**INV9 (per-node empty-run metadata)** reduces `avg_empty` → fewer
outer-loop iterations per ray. Complementary to occupancy fixes — if
we can do both, the gains compound.

## Reproduce

```bash
cargo build --bin deepspace-game --release
scripts/capture-gpu-trace.sh slow-soldier -- \
    --render-harness --vox-model assets/vox/soldier_729.vxs \
    --plain-layers 8 --spawn-xyz 1.15 1.1 1.04 --spawn-depth 5 \
    --disable-overlay --harness-width 2560 --harness-height 1440 \
    --exit-after-frames 300 --timeout-secs 15 --suppress-startup-logs
scripts/parse-metal-trace.py tmp/trace/slow-soldier.trace
```

Look at `Fragment Occupancy`. If it's still in the 10-15% range, the
register pressure hasn't been fixed. If it's above 30%, the current
optimization worked and the bottleneck has moved.

## Timeline

- 2026-04-17: initial diagnosis. Fragment Occupancy = 12.4% on the slow-
  soldier scenario, matching the ~8× gap between measured and theoretical
  peak perf. Next step: reduce per-thread state in `march_cartesian`.
