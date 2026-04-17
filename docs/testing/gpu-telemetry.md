# GPU Telemetry (Apple Silicon / Metal)

Capture hardware GPU counters while the ray-march harness runs a scenario,
then summarize the distributions. Answers questions that `gpu_pass_ms`
and `submitted_done_ms` can't: is the shader memory-bound, compute-bound,
or occupancy-limited? Is the cache thrashing? Is the ALU starved?

Requires macOS + Xcode command-line tools (for `xctrace`). Zero code
changes to the game — runs against a release build.

## Quickstart

```bash
# 1. Build release.
cargo build --bin deepspace-game --release

# 2. Capture a trace of whatever scenario you want to profile.
scripts/capture-gpu-trace.sh <label> -- <harness-args>

# 3. Summarize the counter distributions.
scripts/parse-metal-trace.py tmp/trace/<label>.trace
```

## Example: the zoomed-in-soldier slow-frame scenario

```bash
scripts/capture-gpu-trace.sh slow-soldier -- \
    --render-harness --vox-model assets/vox/soldier_729.vxs \
    --plain-layers 8 --spawn-xyz 1.15 1.1 1.04 --spawn-depth 5 \
    --disable-overlay --harness-width 2560 --harness-height 1440 \
    --exit-after-frames 300 --timeout-secs 15 --suppress-startup-logs

scripts/parse-metal-trace.py tmp/trace/slow-soldier.trace
```

Sample output:

```
counter                                     samples     mean      p50      p90      p99      max
----------------------------------------------------------------------------------------------------
Fragment Occupancy                           139789    12.44    12.57    12.70    15.17    96.32
Compute Occupancy                              1073     0.14     0.00     0.01     0.82    37.50
ALU Utilization                              139270    23.43    24.16    24.64    25.54    72.94
F32 Utilization                              139296     2.09     2.05     2.21     3.63    30.73
Buffer Read Limiter                          139838     4.11     4.32     4.41     4.48    43.49
Buffer Load Utilization                      137244     4.19     4.32     4.40     4.47     4.97
Buffer Write Limiter                         137942     5.23     5.41     5.88     6.16    57.72
Buffer Store Utilization                     133067     5.32     5.43     5.89     6.16     8.46
Threadgroup/Imageblock Load Limiter            3511     0.82     0.30     2.68     3.53     4.36
Threadgroup/Imageblock Store Limiter         125386     0.05     0.01     0.01     1.02    14.41
```

All values are percentages of peak capacity.

## How to interpret

Each sample is a 20 µs hardware window. Distributions show what the GPU
was doing across the whole capture. Focus on **mean** and **p99** — the
max often catches transient spikes irrelevant to steady-state perf.

| Counter | What it means | Red flag threshold |
|---|---|---|
| **Fragment Occupancy** | % of peak parallel SIMD groups running | **<25% = register pressure** |
| **ALU Utilization** | % of peak compute. Low = stalled; high = compute-bound | >70% + high occupancy → compute-bound |
| **F32 Utilization** | % of peak float math. Usually lower than ALU total | for integer-heavy shaders: low is expected |
| **Buffer Read Limiter** | % of peak buffer load throughput | >50% = memory-bandwidth-bound |
| **Buffer Load Utilization** | Actual buffer load activity | closely tracks Read Limiter |
| **Buffer Write Limiter** | % of peak buffer store throughput | usually low for this workload |
| **GPU Last Level Cache Limiter** | % of peak SLC throughput | >50% → SLC thrashing |
| **MMU TLB Miss Rate** | % of memory accesses missing TLB | >10% → TLB thrashing, scattered access |

**Diagnosis decision tree** based on the top signals:

| Occupancy | ALU | Buffer Read | Likely bottleneck |
|---|---|---|---|
| **<25%** | low | low | **Register pressure.** Shader too fat, can't run enough SIMD groups. |
| >50% | >70% | low | Compute-bound. Optimize shader math. |
| >50% | low | >50% | Memory-bandwidth-bound. Cut loads / use caching. |
| >50% | low | low | Latency-bound on cold misses. Working set too big for cache. |
| high | high | high | Well-balanced; everything is working hard. Still slow? Algorithmic. |

## How to capture different scenarios

The capture script wraps `xcrun xctrace record --instrument "Metal GPU
Counters"`. Any harness args work — just keep the `--render-harness` and
don't use live-loop flags that wait on vsync (would skew the trace).

Good scenarios:

- **At-rest** (what's the normal per-frame cost?)
- **Regression-under-test** (which counter spikes at the slow moment?)
- **Before/after an optimization** (did the counter we targeted move?)

Example comparing two anchor depths:

```bash
scripts/capture-gpu-trace.sh shallow-d3 -- \
    --render-harness --plain-world --spawn-depth 3 --spawn-pitch -1.0 \
    --harness-width 1920 --harness-height 1080 --exit-after-frames 300 \
    --timeout-secs 15 --disable-overlay

scripts/capture-gpu-trace.sh deep-d13 -- \
    --render-harness --plain-world --spawn-depth 3 --spawn-pitch -1.0 \
    --script "wait:60,zoom_in:10,wait:240" \
    --harness-width 1920 --harness-height 1080 --exit-after-frames 300 \
    --timeout-secs 15 --disable-overlay

# Compare
scripts/parse-metal-trace.py tmp/trace/shallow-d3.trace > tmp/shallow.txt
scripts/parse-metal-trace.py tmp/trace/deep-d13.trace > tmp/deep.txt
diff -y tmp/shallow.txt tmp/deep.txt
```

## Caveats

- `xctrace` requires Xcode command-line tools installed: `xcode-select --install`.
- Counter sampling is hardware-dependent. Some counters (e.g. TLB Miss Rate)
  may return no samples in certain workloads even when the hardware supports
  them — the shader wasn't the kind of work that counter measures.
- The XML export from xctrace is massive (1-2 GB for a 20-second trace).
  The parse script streams line-by-line to avoid loading all at once.
- Sample weights vary: counters with more sample windows (e.g. Fragment
  Occupancy with 139K samples) are more reliable than ones with few
  samples (e.g. Compute Occupancy with 1K).
- Release build matters. Debug builds have extra instrumentation that
  confuses the occupancy signal.

## See also

- [`perf-occupancy-diagnosis.md`](perf-occupancy-diagnosis.md) — what we
  found when we first ran this on the ray-march shader: Fragment
  Occupancy at 12%, the actual root cause of the 40 FPS zoomed-in case.
- [`cookbook.md`](cookbook.md) — end-to-end testing commands.
- [`perf-isolation.md`](perf-isolation.md) — the higher-level methodology
  (offscreen harness, resolution sweeps, component isolation) that
  gave us the "something deeper is wrong" signal.
