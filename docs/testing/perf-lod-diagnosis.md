# Perf diagnosis: the Cartesian shader's LOD is wrong

## Symptoms observed

In the live game on a retina 2560×1440 window, overlooking the plain
grassland at layer 37 (anchor_depth=5), the frame rate collapses
from ~60 FPS to ~5 FPS. Logs show:

```
renderer_slow acquire_ms=1002  present_ms=1355  total_ms=1002
slow_frame render_ms=1000–2000  zoom_level=37  anchor_depth=5  nodes=187
```

The swapchain is starved — the GPU is 1–2 seconds behind. Per-frame
`acquire_ms` and `present_ms` spike because `get_current_texture()`
has no free backbuffer to hand out.

## Instrumenting the shader

To find out *why* the GPU was a full second behind, the harness got
new signal:

- **`gpu_pass_ms`** via `wgpu::Features::TIMESTAMP_QUERY` — GPU time
  of the ray-march pass itself.
- **`submitted_done_ms`** via `queue.on_submitted_work_done` — wall
  time from submit to GPU callback fire (captures TBDR tile-resolve
  that the per-pass timestamps miss).
- **`avg_steps`, `max_steps`** — per-pixel DDA iterations emitted
  via atomics from the fragment shader (opt-in behind `--shader-stats`
  and a WGSL `override` constant so the default path pays nothing).
- **`avg_oob`, `avg_empty`, `avg_descend`, `avg_lod_terminal`** —
  which branch of the inner loop each iteration lands in.

## What the data showed

Live run at 2560×1440, plain world, layer 37:

```
rays=3,686,400  avg_steps=170.2  max_steps=244  hit_fraction=1.0000
avg_oob=40.1  avg_empty=80.7  avg_descend=45.5  avg_lod_terminal=0.0
submitted_done_ms=117–197
```

- **170 DDA iterations per ray.** Every pixel marches through ~170
  inner-loop iterations before finding a hit.
- **47%** of those steps are DDA advances inside an empty-tag child.
- **27%** are descents into a Node child.
- **23%** are out-of-bounds pops back to the parent level.
- **0%** hit the Nyquist LOD terminal (`lod_pixels < 1.0`).

170 × 3.7M rays = **630M DDA iterations per frame**. At ~5 GOps/s
effective (memory-bound atomic-free Cartesian loop on Apple Silicon)
that's ~125 ms of GPU work, which matches the observed
`submitted_done_ms=117–197`.

## Why the Nyquist LOD wasn't helping

The shader had this in the Node-descend branch:

```wgsl
let lod_pixels = cell_world_size / ray_dist * screen_height
                 / (2.0 * tan(fov * 0.5));
let at_lod = lod_pixels < 1.0;
if at_max || at_lod { /* treat as LOD terminal */ }
else { /* descend */ }
```

`at_lod` fires when a child cell would project to less than one
pixel. At 1280×720 with `fov ≈ 1.2` and cell size = 1/27 of an
anchor cell, that threshold sits past `~500` cell-widths of ray
distance — effectively never for close content. So every Node child
got fully descended.

`at_max` used a hardcoded `depth_limit = MAX_STACK_DEPTH = 64`,
which also never tripped.

Meanwhile the CPU was computing `visual_depth()` (typically 3) and
setting `uniforms.max_depth = visual_depth`. **That uniform was
ignored by the Cartesian path.** It only gated the sphere-face
walker's depth.

## The provisional "fix" (INVESTIGATING PERFORMANCE 3)

We capped Cartesian descent at `uniforms.max_depth + 1`:

```wgsl
let cart_depth_limit = min(uniforms.max_depth + 1u, MAX_STACK_DEPTH);
r = march_cartesian(..., cart_depth_limit, ...);
```

Measured impact at 2560×1440, anchor_depth=8:

| | before | after |
|---|---|---|
| `avg_steps`       | 170  | 21.4 |
| `avg_empty`       | 80.7 | 0.0  |
| `avg_descend`     | 45.5 | 7.3  |
| `avg_oob`         | 40.1 | 3.5  |
| `avg_lod_terminal`| 0.0  | 8.9  |
| `submitted_done_ms` | 68  | 34   (harness) |
| `submitted_done_ms` | 117–197 | 67–100 (live) |

Visually, close-up surfaces now render with the Node's LOD-terminal
color instead of per-voxel cell grids. Distant surfaces still show
the grid because they're rendered before hitting the cap.

## Why the fix is wrong

**The cap is a level count, not a distance.** It says "don't
descend more than N levels below the frame root," regardless of how
far those levels are from the camera.

This creates two failure modes:

### 1. Zoom-out is mathematically equivalent to raising the cap

When the user zooms out from layer 37 to layer 36, `anchor_depth`
drops from 5 to 4. The frame root cell is 3× bigger. To keep the
same physical content visible near the surface, the shader needs
**one more level of descent** — the ground that used to be N=3
below the anchor is now N=4 below.

If `visual_depth` adapts by adding a level (e.g.,
`visual_depth = tree_depth - anchor_depth`), the step count climbs
right back up:

- layer 37 with cap=3 → 21 steps/ray (fast)
- layer 36 with cap=4 → ~60 steps/ray (linear in levels)
- layer 33 with cap=8 → back to ~170+ steps/ray (slow)

If `visual_depth` stays fixed at 3, zooming out makes the nearest
content render at coarser LOD than it should — the grass directly
below the camera looks wrong.

Either way, **distance is what matters for LOD, not depth count.**

### 2. Far cells at shallow depth are rendered too finely

At anchor_depth=5 looking at the horizon, a grass cell 50 anchor-
cells away is very tiny on screen — it should render as one sample.
But under the level-count cap, it still gets descended into for 3
levels. That's ~27 extra DDA iterations on a cell that covers 1
pixel.

Conversely, a grass cell directly below the camera is huge on
screen and should get full detail. Under the same cap it gets the
same 3 levels — which is fine for the floor, but only coincidentally.

## What distance-based LOD should look like

The Nyquist criterion (`lod_pixels < 1.0`) is the right *shape* —
project the cell's world size into pixel space via the ray distance
— but the threshold `1.0` is the wrong *value* for close content.

Two levers to pull:

1. **Tighten the Nyquist threshold.** `lod_pixels < 4.0` means "stop
   descending if the next child would be < 4 px." At 4 px/cell we
   lose almost no perceived detail but can cut descent by several
   levels. Gives distance-based LOD out of the box.

2. **Add an absolute-distance floor.** `at_lod = lod_pixels < T ||
   ray_dist > D`. Beyond distance D, don't descend further
   regardless of pixel projection. Useful for the horizon case
   where lod_pixels might still be >1 but the cell is visually
   uniform.

Both are cheap to compute — the shader already has `ray_dist` and
`cell_world_size` in scope at the descend site. The existing
`at_lod` branch is the right place to change.

## Resolution

Three changes landed after the diagnosis. Each solves a specific
flaw that kept revealing itself in the shader-stats breakdown.

### 1. Ribbon-level LOD (INV4, commit `4e7b445`)

Replaced the level-count cap and the 2D Nyquist gate with a
tree-native distance metric: `ribbon_level` (ancestor-pop count).

- Inside the anchor cell (ribbon_level=0): `BASE_DETAIL_DEPTH`
  levels of descent (default 4).
- Each additional ribbon shell: detail budget drops by 1,
  bottoming out at 1.
- Nyquist (`LOD_PIXEL_THRESHOLD`, default 1.0) stays as a
  sub-pixel floor.

Implemented as a WGSL `override` constant `BASE_DETAIL_DEPTH` with
CLI flag `--lod-base-depth N`. Removed the earlier
`LOD_WORLD_RADIUS` stopgap.

Why this is right: ribbon pops correspond to successively-larger
cubic shells (3³ = 27 cells per shell) around the camera in the
tree's own structure. Detail budget per shell = a **cubic LOD
gradient by construction**. Invariant under zoom because
ribbon_level only counts pops from the current anchor, not from
the root.

Measured (2560×1440, plain, 60 frames, `BASE_DETAIL_DEPTH=4`):

| depth | layer | avg_steps | submitted_done_ms |
|---|---|---|---|
| 3  | 38 |  5.9 | 31.9 |
| 5  | 36 | 24.8 | 33.5 |
| 8  | 33 | 24.5 | 33.7 |
| 12 | 29 | 25.2 | 33.9 |
| 17 | 24 | 25.2 | 33.6 |

`avg_steps` flat across the zoom range — the regression under
zoom that killed the level-count cap is gone.

### 2. Interaction radius gate (INV5, commit `4ad74bb`)

Cursor raycast + break/place share a distance cap:

```
max_t_in_frame = interaction_radius_cells × anchor_cell_size_in_frame
anchor_cell_size_in_frame = 3 / 3^K    where K = anchor_depth - frame_depth
```

Since `ray_dir_in_frame` is normalized, `HitInfo.t` is a frame-local
distance and compares directly to `max_t`. Applies to both
Cartesian (render frame) and sphere (body frame) paths.

CLI flag `--interaction-radius N` (default 6). Zoom-aware by the
same mechanism as the LOD shells: anchor cell size scales with
zoom, so the physical reach adjusts automatically. "Floating in
space" semantic: when you're too far from content for your current
zoom level to reach it, the cursor shows no highlight — either
move closer or zoom out.

### 3. Stack-depth shrink (INV6, commit `81a4fad`)

The single biggest GPU win, independent of algorithm: the
`march_cartesian` DDA stacks were declared at `MAX_STACK_DEPTH=64`,
allocating ~3.5 KB of per-fragment scratch across 5 arrays.
Apple Silicon's register allocator can't fit that per invocation,
so **every DDA iteration spilled to threadgroup memory**.

Since ribbon-level LOD caps descent at `BASE_DETAIL_DEPTH=4`, we
only need ~5 stack slots. Dropped `MAX_STACK_DEPTH` to 5:

| MAX_STACK_DEPTH | submitted_done_ms | live FPS |
|---|---|---|
| 64 | 33.9 | ~30 |
|  8 | 24.0 | ~42 |
|  5 | 16.8 | ~60 |

Live game at 2560×1440 retina: `avg_frame_fps = 60.89` across 586
samples, steady through a scripted zoom-out across layers 33→37.

If BASE_DETAIL_DEPTH is raised later, `MAX_STACK_DEPTH` must be
raised to match or descent is silently capped.

## Related instrumentation

- `scripts/perf-breakdown.sh` — matrix runner.
- `--shader-stats` — enable atomics and stats readback.
- `--perf-trace <path>` — per-frame CSV.
- `--lod-base-depth N` — ribbon-level detail budget.
- `--lod-pixels N` — Nyquist floor.
- `--interaction-radius N` — cursor / break reach cap.
- `renderer_slow …` stderr lines in live mode with phase timings
  + branch counters.

See also:
- `docs/testing/perf-isolation.md` — the isolation playbook.
- `docs/testing/harness.md` — flag reference.
