# Sphere dig-down: sub-frame saturates at `m_truncated=5` past layer 10

## Symptom

`e2e_sphere_descent::sphere_dig_down_descent` renders 21 layers (d5..d25).
Past layer ~10 the image stops changing — `d18`, `d19`, `d20` are
byte-identical PNGs (md5 `3ac93b2b52987b…`); `d23`, `d24`, `d25` are
byte-identical too. The user sees "geometry collapses at 8–10 layers
below the surface."

This is NOT the same bug as the shader-rewrite runbook
(`docs/design/sphere-shader-bug-repro.md`, commit `0c1fae9`). The
shader rewrite at commit `12a0692` fixed the hit_fraction=0 issue;
the remaining freeze has a different cause.

## Verified cause: render-frame cap in `with_render_margin`

`target_render_frame` → `render_frame` calls:

```rust
frame::with_render_margin(..., desired_depth = 11, render_margin = 4)
```

`with_render_margin` internally does:

```rust
let target_depth = logical_depth.saturating_sub(render_margin); // 11 − 4 = 7
let render = compute_render_frame(..., target_depth = 7);
```

For a sphere camera, `compute_render_frame` with `desired_depth=7` caps
`m_truncated` at `(7 − body_depth − 1) = 5` regardless of how deep
`sphere.uvr_path` has grown.

**Every deep layer (d10..d25) uploads the same sub-frame:**
`body(1) + face(1) + m_truncated(5) = render_path.depth() = 7`,
`frame_size = 1/3^5 ≈ 4.1e-3`.

The `RENDER_ANCHOR_DEPTH=14 / RENDER_FRAME_K=3 / RENDER_FRAME_CONTEXT=4`
constants come from the Cartesian design where you render a "spatial
bubble" and rely on LOD descent for finer detail. For sphere-sub, the
sub-frame IS the local coordinate system, so capping it at m=5 forces
the shader to resolve deep cells inside a too-coarse local box.

### Evidence

From `tmp/sphere_stderr2.log` (added `SPHERE_UPLOAD` log in
`src/app/edit_actions/upload.rs`):

```
SPHERE_UPLOAD m_truncated=5 logical_uvr_depth=10 walker_limit=5  frame_size=4.115e-3 render_path_depth=7
SPHERE_UPLOAD m_truncated=5 logical_uvr_depth=15 walker_limit=9  frame_size=4.115e-3 render_path_depth=7
SPHERE_UPLOAD m_truncated=5 logical_uvr_depth=24 walker_limit=9  frame_size=4.115e-3 render_path_depth=7
```

`logical_uvr_depth` (the camera's true zoom depth) grows 4 → 24.
`m_truncated` / `render_path_depth` / `frame_size` never change past
layer 10.

### Why pixels look grey (not sky)

Every pixel at d18+ is a REAL palette hit — not a miss. Debug-paint
instrumentation (`SPHERE_DEBUG_PAINT=true` in `sphere.wgsl`) plus a
temporary purple override on the real-hit return path showed d18
turning fully purple. The walker finds a uniform stone cell (the
m=5 sub-frame's content at the camera's location) and shades it —
the result is a grey gradient because the shading is monochromatic
when `pos` drifts smoothly across the single large cell.

The dug pit at layer N is at tree-depth N. The m=5 sub-frame + a
`walker_limit=9` intra-cell descent reach tree-depth ~7+9=16. Layers
past ~16 have their pit geometry below the walker's reach, so the
walker returns the uniform stone that surrounds the pit at depth 16,
and all pixels show that stone.

## Non-causes (ruled out)

- **f32 precision at deep m**: the architecture already uses local
  coords correctly. The shader's precision model in `sphere.wgsl:849`
  is sound; the rewrite at `12a0692` verified this. The saturation is
  a CPU-side frame choice, not numerical decay.
- **`m_truncated` computation in `compute_render_frame`**: it correctly
  clamps to `min(desired - body - 1, logical_m)`. The cap comes from
  the caller passing `desired = 7`.
- **Camera-fit / pixel-density shrink loop in `target_render_frame`**:
  `CAMERA_FITS fits=true pixels=31572` — the loop never shrinks the
  frame. The cap is upstream in `with_render_margin`.

## Fix direction (not yet applied)

For SphereSub cameras, the sub-frame MUST be built at the camera's
full logical depth:

1. Bypass `with_render_margin` for sphere cameras — use
   `compute_render_frame(..., desired_depth = body_depth + 1 +
   sphere.uvr_path.depth())` directly.
2. Or extend `with_render_margin` to detect SphereSub and pass
   `render_margin=0` (Cartesian-only margin).

The walker's intra-cell `walker_limit` stays as-is — once the sub-
frame tracks the camera's deep cell, only ~2–3 levels of intra-cell
descent are ever needed.

## Debugging instrumentation left in place

- `SPHERE_DEBUG_PAINT` (default `false`) in `assets/shaders/sphere.wgsl`
  now covers two previously-silent return paths:
  - **cyan**: `sign_s == 0` (ray exited via `t >= t_exit` without a
    pos-axis crossing)
  - **white**: neighbor-transition interval miss
- `SPHERE_UPLOAD` trace in `src/app/edit_actions/upload.rs` logs
  `m_truncated`, `logical_uvr_depth`, `walker_limit`, `frame_size`,
  `render_path_depth` on every GPU upload. Use this to watch the
  sub-frame cap in real time.

## Reproduction

```sh
cargo test --test e2e_sphere_descent sphere_dig_down_descent
md5 tmp/sphere_descent/d{18,19,20}.png  # all identical
grep "SPHERE_UPLOAD" <captured stderr> | sort -u  # m_truncated stuck at 5
```
