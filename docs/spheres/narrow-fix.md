# Narrow fix — slot-path hit coords + AABB in f64

## Diagnosis

At d=10, the walker's precision is fine (10 ULPs vs 1.7e-5 cell width,
17× headroom). What's breaking is the 8-corner AABB expansion in
`hit_aabb_body_local`:

```
Δbody_xyz[tangent] = du · shell · WORLD_SIZE · dir[tangent]
```

For a face-center pixel, `dir[tangent]` can be ~1e-3. At m=10,
`du·shell ≈ 1.7e-5`, so `Δbody_xyz[tangent] ≈ 1.7e-8` — below the
f32 ULP of `body_xyz ≈ 1.5` (which is 1.2e-7). Adjacent corners
collapse along a tangent axis; the 8-corner AABB degenerates to a
triangle or square.

The walker itself works. Every consumer that derives body-XYZ from
the walker's f32 `u_lo/v_lo/r_lo/size` (and only the AABB does)
inherits this projection collapse.

## Fix

Two changes, both surgical.

### 1. Walker emits integer ratios

Replace on `SphereHitCell`:
```rust
u_lo: f32, v_lo: f32, r_lo: f32, size: f32
```
with:
```rust
ratio_u: i64, ratio_v: i64, ratio_r: i64, depth: u8
```

Semantics: `u_lo = ratio_u / 3^depth`, exact rational. Walker
accumulates `ratio_u = ratio_u_parent · 3 + us` per descent level —
integer, precision-free, independent of depth. i64 covers m ≤ 40.

CPU walker (`walk_face_subtree`) and shader mirror both change to
track u32/i64 ratios instead of f32 accumulators.

### 2. AABB corners in f64

`hit_aabb_body_local` gets a new path:
```rust
let denom = 3.0_f64.powi(cell.depth as i32);
let un_lo = cell.ratio_u as f64 / denom;
let un_hi = (cell.ratio_u + hit_span) as f64 / denom;
// ... vn, rn similarly
// project 8 corners via face_space_to_body_point_f64, bound in f64
```

New f64 variant `face_space_to_body_point_f64` in `cubesphere.rs` —
same math, f64 throughout. `bounding_box` result rounds to f32.

## Deletions (unrelated cleanup, same PR)

The `SphereSubFrame` / `cs_raycast_local` infrastructure is abandoned
and never populated; the `sub_*` fields on `SphereHitCell` and their
shader/renderer plumbing exist only to shim its output. Delete
alongside the fix:

- `src/world/raycast/sphere_sub.rs` (file)
- `SphereSubFrame`, `ActiveFrameKind::SphereSub`, `with_neighbor_stepped`,
  `uvr_corner` in `frame.rs`
- `face_frame_jacobian`, `mat3_inv`, `Mat3`, `mat3_mul_vec` in
  `cubesphere.rs`
- `WorldPos::in_sub_frame`, `Transition::SphereEntry` in `anchor.rs`
- `cpu_raycast_in_sub_frame` in `raycast/mod.rs`
- `SphereHitCell.sub_c_body / sub_j_cols / sub_local_lo / sub_local_size`
- Renderer uniforms: `sub_c_body`, `sub_j_col{0,1,2}`, `sub_uvr_slots`,
  `sub_meta`, `sub_face_corner`, `ROOT_KIND_SPHERE_SUB`
- Shader `sphere.wgsl`: `sphere_in_sub_frame`, `face_frame_jacobian_shader`,
  `mat3_inv_shader`, `mat3_inv_scaled_shader`, `bevel_layered_local`,
  `walk_from_deep_sub_frame_dyn`, `FaceFrameJac`, `Mat3Columns`
- Shader `march.wgsl`: the `ROOT_KIND_SPHERE_SUB` dispatch branch
- Shader `bindings.wgsl`: `sub_uvr_slots`, `sub_meta`, `sub_face_corner`
- `docs/design/sphere-ribbon-pop-{two-step,impl-plan,gpu-port,uvr-state,
  proposal}.md`, `sphere-shader-bug-repro.md`

## What stays

- Body march `cs_raycast` — the only render path, unchanged.
- `face_space_to_body_point` (f32) — shader still uses it per-pixel.
- Walker `walk_face_subtree` — stays, reformulated to emit ratios.
- Cross-face handling — body march handles seams via continuous
  body-XYZ plane normals.

## Precision wall after the fix

- AABB corners (f64): usable to m ≈ 33.
- Integer ratios (i64): overflow at m ≈ 40.
- Body march plane normals: start collapsing at m ≈ 15 — the real
  wall, unchanged by this fix.

So the usable range moves from m ≈ 10 (current AABB break) to m ≈ 15.
Past 15 needs the architectural rewrite (cube-seam primitive + proper
intra-face local DDA).

## Staging

Two commits:

1. **Delete SphereSub plumbing** — pure subtraction, green at head.
2. **Walker ratios + AABB f64** — one diff. New fields replace the
   old; walker CPU + shader updated together.

Net: ~−1200 LoC, ~+200 LoC new.
