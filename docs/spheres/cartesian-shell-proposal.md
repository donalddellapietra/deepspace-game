# Sphere shell → gnomonic plane normals — revised rewrite proposal

> Replaces the previous draft. The previous draft claimed gnomonic gives
> "linear (u, v, r)" and proposed reusing `march_cartesian`'s exact DDA
> arithmetic. That's wrong — gnomonic coords are **rational** in ray `t`,
> not linear. This revision narrows the scope significantly: the real
> change is just deleting `ea_to_cube` / `cube_to_ea` from the hot path.

## Problem recap

Current sphere rendering uses equal-angle cubed-sphere geometry.
`sphere_in_cell` walks the face subtree with ray-plane intersections
against plane normals derived via `ea_to_cube(u_lo · 2 − 1)`
(= `tan(c · π/4)`). At d ≥ 10, adjacent cells' plane normals differ
by ~1e-5 in the n-axis component, and the arg-min of
`t_u_lo / t_v_lo / t_r_lo` flips unstably across adjacent pixels —
mode-4 stripes and "hollow" placed blocks on the ground.

Every fix that stays inside the equal-angle mapping has been
rejected in worktree `sphere-attempt-2-2-3-2`. See
`docs/spheres/d10-hollow-block-debug-state.md`.

## Scope simplification

Shell-only (no digging to core) is granted. The inner shell is the
floor of playable space. This doesn't change the DDA's math; it
only bounds which cells ever get rendered.

## The actual fix

**Delete `ea_to_cube` / `cube_to_ea` from the DDA hot path.**

The equal-angle mapping wraps each cube-plane coordinate `cu`
through `atan(cu) · 4/π` so that `un_abs` is angularly uniform on
the sphere. Inside the hot DDA loop we then UN-wrap via
`ea_to_cube(cell_u_lo · 2 − 1) = tan(...·π/4)` to build each
plane's normal in world space.

If we drop the wrap entirely — use `un_abs = (cu + 1) · 0.5`
directly instead of `un_abs = (cube_to_ea(cu) + 1) · 0.5` — then:

- Each cell boundary at depth `d` is at `cu = k/3^d · 2 − 1` for
  integer `k` ∈ [0, 3^d]. Two exact integers divided once — 0.5 ULP
  error regardless of depth.
- Plane normal is `u_axis − (k/3^d · 2 − 1) · n_axis`. No `tan`, no
  accumulated `child_size /= 3` drift.
- The DDA still does ray-plane for the 4 lateral walls and
  ray-sphere for the 2 radial walls — but each computation is now
  stable at d ≥ 10.

### The DDA is NOT march_cartesian

Important correction from cold review: gnomonic `u(t) = x(t)/y(t)`
is RATIONAL in `t`, not linear. Solving `u(t) = k/3^d` still
requires one divide per candidate boundary — you cannot reuse
`march_cartesian`'s `inv_dir` / `delta_dist` machinery.

What IS reusable:
- The control flow (stack of depths, push/pop on descend/exit,
  integer slot pick per level).
- The walker (`walk_face_subtree`) untouched.
- `bevel_layered` in face-normalized coords.
- `face_lod_depth` pixel threshold.
- Smooth-radial `hit_normal = n` fallback.
- `sphere_debug_*` modes.

The arithmetic in the DDA inner loop stays shaped like the current
sphere DDA, just with different cell-boundary values.

### Angular uniformity trade

Equal-angle: every cell subtends the same solid angle.

Gnomonic: cells at face corners are larger. Face-corner vs face-
center edge-length ratio is `(1 + u² + v²)^(3/2)`. At `(u,v) = (1,1)`
that's `3√3 ≈ 5.2×` area ratio.

Close-range gameplay does not see both center and corner at once,
so the in-view angular scale is roughly uniform. **Empirical check
required before landing**: screenshot a cell at `(un=0.95, vn=0.95)`
and `(0.5, 0.5)` at d=8; measure on-screen projection. Ship only if
the ratio is ≤ 2× at typical gameplay framing.

"Layer UX uniformity" memory exempts sphere-related objects from the
uniformity rule.

## d=30+ support

Preserved via the same frame-local rescale `march_cartesian` uses:
each ribbon pop into a face subtree reapplies the `ray_dir / 3` +
`ray_origin = slot_off + ray_origin / 3` transform so values stay
`O(1)` regardless of depth. The face-local DDA's cell-boundary
fractions `k/3^d` are computed from INTEGER `ratio_u`, so precision
doesn't degrade with depth.

The previous draft missed this: the face-local DDA's ray scale must
compose correctly with the ribbon-pop machinery in `march()`.

## Face-to-face transitions

Previous draft: "~30 lines." Correction: realistic cost is
~80–120 LoC because gnomonic `u` blows up as the ray approaches a
face edge (axis_dot → 0), and three-face corners need a stable
tie-break policy.

**For v1 we defer this entirely**: when the ray exits `u` or `v` of
the current face, terminate as a miss (render sky). This matches
the current shader's behavior — it also terminates rather than
cross (the current bug IS an unstable `pick_face` flip at edges;
"terminate cleanly" is an improvement).

## What gets deleted

- `ea_to_cube`, `cube_to_ea` in `sphere.wgsl` (2 function defs).
- `cube_to_ea` call in `sphere_in_cell` at line 456 (entry UV calc).
- `ea_to_cube` calls on lines 607–610 (plane normal construction).
- Same pair in `src/world/cubesphere.rs` — f32 AND f64 siblings.
- Same pair in `src/world/raycast/sphere.rs` (CPU mirror).
- `ea_cube_round_trip` test.

Walker, bevel, face_lod_depth, sphere_debug modes, highlight AABB
math all stay. `SphereHitCell.ratio_*` stays — already integer.

## What gets changed

### `assets/shaders/sphere.wgsl`

1. Line 456: replace `un_abs = clamp((cube_to_ea(cu) + 1) * 0.5, ...)`
   with `un_abs = clamp((cu + 1) * 0.5, ...)`. Same for `vn_abs`.
2. Lines 590–593: `cell_u_lo_ea = w.u_lo * 2 - 1` becomes
   `cell_u_lo_cube = w.u_lo * 2 - 1`. Value stays the same but is
   now interpreted as a cube-plane coord, not an ea-space coord.
3. Lines 598–601: plane normals — drop `ea_to_cube` wrap.
   `n_u_lo = u_axis - cell_u_lo_cube * n_axis`. Direct cube coord.
4. No other DDA / loop changes.

### `src/world/cubesphere.rs`

1. `body_point_to_face_space`: drop `cube_to_ea` on `cu`/`cv`.
2. `face_uv_to_dir`: drop `ea_to_cube` on `u`/`v`.
3. f64 sibling in same file — same changes.
4. `build_face_subtree`: SDF sampling at cell centers — the cell
   bounds are still `k/3^d` in `(u, v, r)`, just interpreted as
   gnomonic cube coords now. The sampling arithmetic is identical;
   only the interpretation of the resulting point changes.

### `src/world/raycast/sphere.rs`

Mirror `sphere.wgsl` changes. Walker untouched.

### Tests

- Remove `ea_cube_round_trip`.
- Retune `body_face_space_round_trip` tolerances — expected
  bit-exact under gnomonic.
- **Add visual regression test**: render d=10 hollow-block repro,
  assert `tmp/bug_m4.png` ground rows have uniform winning-plane
  hue (one RGB cluster within ΔE < 10). Run via
  `scripts/repro-sphere-d10-bug.sh 0 4`.

## Realistic LoC

- `sphere.wgsl`: ~60 lines changed
- `cubesphere.rs`: ~40 lines changed
- `raycast/sphere.rs`: ~30 lines changed
- Tests: ~50 lines added (visual regression)
- **Total: ~180 LoC single commit**, much smaller than the
  previous draft's estimate.

## Migration ordering (one commit)

Rewrite the three files in a single commit. Walker and
`SphereHitCell.ratio_*` are already integer — no shim needed.
Compiles green throughout if done in one pass; naga validation
passes since we're only changing which values feed into existing
plane-normal expressions.

## Empirical validations BEFORE landing

1. **Face-corner pixel-size ratio.** Screenshot demo_sphere at
   face-center `(0.5, 0.5)` and face-corner `(0.95, 0.95)` at
   d=8 gameplay camera. Must be ≤ 2× pixel-ratio for ship.
2. **Worldgen cache dedup rate.** Gnomonic cell centers differ from
   equal-angle cell centers, so `build_face_subtree`'s hash dedup
   may change. Cache-hit rate should match equal-angle within 5%.
3. **Mode-4 clean at d=10 AND d=15 AND d=20.** The previous draft
   only spec'd d=10. If gnomonic fixes d=10 but d=15 still
   stripes, the integer-ratio depth range (currently `3^15`) needs
   extension.

## Success criterion

`scripts/repro-sphere-d10-bug.sh 0 4` at `--spawn-depth {10, 15, 20}`:

- **Mode 0**: placed block at each depth renders as solid shape —
  no hollow, no stripes. Pixel count of block silhouette within
  5% of analytic projection from the camera position.
- **Mode 4**: ground winning-plane image is monochrome (single
  RGB cluster, ΔE < 10). No whole-screen striping.
- **Perf**: frame time within 10% of d=8 baseline at the same
  camera position.

All three measurable in the existing harness.

## Known fallback (not used in v1)

If the cosmetic distortion is unacceptable, keep equal-angle but
**pack per-cell plane normals into a sibling buffer**: +24 bytes /
node, indexed by `face_node_idx`. Shader reads `plane_normals[idx]`
instead of calling `ea_to_cube`. No trig in hot path, angular
uniformity preserved. Doesn't solve d > 15 depth limits (needs
ratio-based integer-scaled normal encoding), so still not a
complete fix. Kept here as a named alternative; v1 ships the
gnomonic approach.
