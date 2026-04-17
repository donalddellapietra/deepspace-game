# Baseline: fragment-shader ray-march (slow-soldier, 2560x1440)

Focused on the motivating scenario only.

## render_harness_timing (5 runs × 300 frames)

| run | gpu_pass_ms | submitted_done_ms | total_ms |
|-----|-------------|-------------------|----------|
| 1   | 2.785       | 21.325            | 21.790   |
| 2   | 2.074       | 21.304            | 21.713   |
| 3   | 2.141       | 21.362            | 21.791   |
| 4   | 2.825       | 21.412            | 21.886   |
| 5   | 2.333       | 21.376            | 21.842   |
| **mean** | **2.432** | **21.356**     | **21.804** |

FPS = 1000 / 21.804 = **45.9**

## GPU counters (Metal, 300-frame capture)

| counter | mean | p50 | p90 | p99 |
|---|---|---|---|---|
| Fragment Occupancy | 12.28% | 12.56% | 12.70% | 19.28% |
| ALU Utilization | 23.17% | 24.15% | 24.63% | 25.23% |
| Buffer Read Limiter | 4.08% | 4.32% | 4.41% | 4.47% |
| Buffer Write Limiter | 5.17% | 5.40% | 5.88% | 6.13% |
| Threadgroup Store Limiter | 0.04% | 0.01% | 0.01% | 1.22% |

## Shader stats

- avg_steps=31.60, max_steps=66, hit_fraction=1.0000
- avg_empty=14.62, avg_descend=11.53, avg_oob=7.53, avg_lod_terminal=4.00
- 91249 packed nodes, ribbon_len=2

Matches `perf-occupancy-diagnosis.md` numbers. Fragment Occupancy at 12% confirms register-pressure regime.

## Step 1 (`--renderer compute`, no threadgroup memory) — 5 runs × 300 frames, 2560×1440

| run | gpu_pass_ms | submitted_done_ms | total_ms |
|-----|-------------|-------------------|----------|
| 1   | 15.715      | 19.613            | 20.076   |
| 2   | 15.855      | 18.327            | 18.807   |
| 3   | 16.761      | 18.144            | 18.615   |
| 4   | 15.880      | 18.146            | 18.617   |
| 5   | 15.336      | 18.213            | 18.707   |
| **mean** | **15.909** | **18.489**     | **18.964** |

FPS = 1000 / 18.964 = **52.7**

### Step 1 vs fragment baseline

- **total_ms**: 21.804 → 18.964  (−13.0%,  +6.8 FPS)
- **submitted_done_ms**: 21.356 → 18.489  (−13.4%)
- **gpu_pass_ms**: 2.432 → 15.909  (the fragment number was undercounting TBDR resolve; the compute timestamp sees real work, so these aren't directly comparable — submitted_done is the honest metric)

Shader ray counts identical (avg_steps 31.60, per-branch splits match) — the compute port reproduces the fragment traversal bit-for-bit, as expected.

**Step 1 verdict**: ~13% faster end-to-end on the motivating scenario, even without threadgroup memory. The compute pipeline alone is a modest win. Step 2 tests whether moving the per-depth stack to threadgroup memory compounds or erases this gain.

## Step 2 (compute + workgroup stack) — 5 runs × 300 frames, 2560×1440

| run | gpu_pass_ms | submitted_done_ms | total_ms |
|-----|-------------|-------------------|----------|
| 1   | 12.124      | 13.644            | 14.094   |
| 2   | 12.084      | 13.642            | 14.076   |
| 3   | 12.107      | 13.642            | 14.082   |
| 4   | 12.127      | 13.680            | 14.118   |
| 5   | 11.952      | 13.667            | 14.114   |
| **mean** | **12.079** | **13.655**     | **14.097** |

FPS = 1000 / 14.097 = **70.9**

### Step 2 vs fragment baseline

- **total_ms**: 21.804 → 14.097  (**−35.3%**,  +25.0 FPS)
- **submitted_done_ms**: 21.356 → 13.655  (−36.0%)
- **gpu_pass_ms**: 2.432 (undercount) → 12.079  (honest signal)

### Step 2 vs Step 1

- **total_ms**: 18.964 → 14.097  (−25.7%)
- **submitted_done_ms**: 18.489 → 13.655  (−26.1%)
- **gpu_pass_ms**: 15.909 → 12.079  (−24.1%)

Shader ray counts unchanged (avg_steps 31.60). Pixel parity bit-identical to Step 1.

### Step 2 GPU counters (300-frame Metal trace)

| counter | fragment | Step 2 | Δ |
|---|---|---|---|
| Fragment Occupancy | 12.28% | 15.21% (blit only, small sample) | n/a |
| Compute Occupancy | 0.09% | **6.00%** | compute path now alive |
| ALU Utilization | 23.17% | 23.33% | flat |
| Buffer Read Limiter | 4.08% | **0.89%** | **−78%** |
| Threadgroup Load Limiter | 0.60% | 2.72% | as expected |
| Threadgroup Store Limiter | 0.04% | 2.91% | as expected |

**Step 2 verdict**: hypothesis **validated** — moving the per-depth stack to workgroup memory gave a clean +54% FPS over the fragment baseline (+35% over Step 1 alone).

**Surprises worth noting**:

1. Compute Occupancy is only 6% — lower than fragment's 12%. The prompt expected 25–50%. Despite lower occupancy, total FPS is up, which means per-thread throughput went up more than occupancy went down (presumably from lower register pressure and register-cached scalars inside the inner loop). ALU utilization stayed flat at ~23%, so we're still compute-starved but less severely.

2. Buffer Read Limiter dropped 4× to 0.89%. The diagnosis doc's "bandwidth is 4%, not the bottleneck" claim was already true for fragment; now it's nearly zero. **This kills Step 3's premise** — Step 3 was designed to reduce Buffer Read Limiter further, but there's almost nothing left to reduce.

### Step 3 decision

Not pursued. Step 3 targets Buffer Read Limiter, but that counter is already at 0.89% (essentially rounded to noise). The remaining headroom is Compute Occupancy (6%), which tile-local caching would worsen rather than improve (it adds more threadgroup memory, which is what's suppressing occupancy).
