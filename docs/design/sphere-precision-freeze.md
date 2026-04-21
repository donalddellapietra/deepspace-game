# Sphere precision-freeze: graceful degradation past the f32 wall

Deep UVR descent inside a `CubedSphereBody` shows visible geometry
collapse at `m ≥ 8` — the shader's smeared-diamond failure is the
f32 Jacobian recompute losing meaning. The fix is **not** to swap
cubed-sphere cells for Cartesian cubes (that produces seam / drift
artefacts). Instead we **freeze** the precision-sensitive pieces of
the sub-frame at a safe depth `M_FREEZE` and reuse that single
linearisation for the whole deeper sub-tree. Cell content, DDA
topology and palette lookups are unchanged — only `J`, `J_inv`,
`c_body`, `un/vn/rn_corner`, `frame_size` stop refining past the
freeze boundary.

## 1. Precision wall per quantity (`r_body = 0.45`, body_size = 3)

Assumptions: f32 rel ε ≈ 6e-8; `rn_corner ≈ 0.5`, so ULP(rn_corner)
≈ 6e-8. Local sub-frame magnitudes use `s = frame_size/3 = 1/3^(m+1)`.

| Quantity | Formula | Wall |
|---|---|---|
| `un_corner` accumulated in `uvr_corner` | Σ `us[k] · 3^-(k+1)` | partial sum is exact to `m ≈ 16` (last addend ≈ 2·3^-17 ≈ 1.5e-8, comparable to ULP(0.5)) |
| `frame_size = 1/3^m` | direct pow | exact representation to `m ≈ 23` (3^-23 ≈ 1.2e-11), subnormal wall at `m ≈ 80` |
| `J` column magnitudes | `O(r_body · s) = O(1.35/3^(m+1))` | representable to `m ≈ 23`, but precision relative to `c_body` drops below 1 ULP at `m ≈ 15` |
| `det(J)` in f32 | `O((r_body · s)^3) ≈ (1.35/3^(m+1))^3` | subnormal at `m ≈ 25` (det ≈ 1.2e-45); *usable* precision gone by `m ≈ 14` because `mat3_inv_shader` carries two 6-multiply products that each have 8-bit mantissa budgets |
| Neighbor step `un_corner += frame_size` | add | silently dropped once `frame_size < ULP(un_corner)` → `m ≈ 16` |
| Tint `rn_abs = rn_corner + r_lo · s` | add | addend below ULP at `m ≈ 16` |

**Observed collapse at `m = 8`** (not predicted by the table above)
comes from shader `mat3_inv_shader`: the 3×3 determinant squares
out a compound error that shows up visibly well before the
subnormal wall. The CPU's `mat3_inv` is safe because it uses f64
internally. So the **shader-specific wall is `m ≈ 8–10`**, the
CPU-specific wall is `m ≈ 16`.

## 2. Curvature bound — what is safe to freeze?

`face_frame_jacobian` produces a linearisation with residual
`O(s² · curvature)` body units, where curvature scales as
`1/r_body ≈ 2.2/body_size`. Concretely:

| m | `s = 1/3^(m+1)` | residual / J-magnitude |
|---|---|---|
| 10 | 1.7e-5 | `s · 2.2 ≈ 3.7e-5` |
| 12 | 1.9e-6 | `4.2e-6` |
| 15 | 7.0e-8 | `1.5e-7` |
| 20 | 2.9e-10 | `6.4e-10` |

At `m ≥ 12` the Jacobian is constant to within 4.2e-6 relative over
the whole sub-frame region — 5+ orders below pixel scale. Freezing
it introduces no visible error.

## 3. What changes vs what stays (with `M_FREEZE = 12`)

**Stays identical:**
- Walker (`walk_from_deep_sub_frame` / `walk_sub_frame`) descends
  via `uvr_path` slots at full `m_truncated` depth. Same tree
  nodes, same palette, same `Child::Empty` handling.
- Intra-cell DDA in local `[0, 3)³`, axis-aligned cell boundaries.
- Ray-box interval math; `ro_local` comes from
  `in_sub_frame(&sub)` which is already symbolic-precise.

**Changes:**
- `SphereSubFrame::with_neighbor_stepped`: past `M_FREEZE` it
  *only* path-steps. No `face_frame_jacobian` / `mat3_inv` call;
  `j`, `j_inv`, `c_body`, `frame_size`, `un/vn/rn_corner` inherit
  from the frozen sub-frame. Corner *sums* are still tracked
  symbolically for the walker's slot prefix — the `_corner` f32
  fields just stop being used.
- `compute_render_frame`: `effective_jacobian_depth =
  min(m_truncated, M_FREEZE)`. Jacobian evaluated at that depth;
  `render_path` / walker `uvr_prefix_slots` keep the full
  `m_truncated` so cell content is correct.
- Shader `sphere_in_sub_frame`: on `out_of_box`, skip the
  `face_frame_jacobian_shader` + `mat3_inv_shader` branch when
  `is_frozen == 1u`. Use uploaded `J`, `J_inv` only.

## 4. Visual artefacts at the freeze boundary — none

- *Within* a frozen sub-tree: identical J shared by all cells →
  same regime as current shallow rendering; no seams.
- *At* the `m = M_FREEZE` → `m = M_FREEZE + 1` boundary: both
  sides reference the `M_FREEZE`-level J. Continuous by
  construction.
- *Across sibling frozen sub-trees* (two neighbours both frozen
  at `M_FREEZE`, each carrying its own J): J-difference is
  `O(s² · curvature)` = `4.2e-6` relative at `M_FREEZE = 12`.
  Below f32 rel-ε. No visible seam.

## 5. Static vs runtime threshold

Recommend **compile-time constant** `M_FREEZE: u8 = 12`. No
runtime cost, predictable, trivial to upload. Promote to a
uniform (`sub_meta.w`) if `outer_r` varies enough in future
content to warrant it. Per-pixel dynamic freeze (detect
`det(J) < ε`) adds branching inside the DDA and defeats SIMD
parallelism — not worth it unless a specific planet radius
demands it.

## 6. Implementation plan, file-by-file

1. **`src/app/frame.rs`** — `SphereSubFrame`: add
   `freeze_depth: u8`. `compute_render_frame` sphere branch:
   ```
   let effective_jac_depth = m_truncated.min(M_FREEZE as u32);
   let (un, vn, rn, size) = uvr_corner(&sphere.uvr_path,
       effective_jac_depth as usize);
   ```
   Render-path push loop still goes `0..m_truncated` so walker
   has the full prefix. Store both.
2. **`SphereSubFrame::with_neighbor_stepped`** — add `if
   self.depth_levels() > M_FREEZE { only path-step, reuse
   frozen J/J_inv/c_body }`. Path-step produces a new
   `render_path`; `un/vn/rn_corner` stop getting bumped
   (they're no longer reference points for anything past freeze).
3. **`src/world/raycast/sphere_sub.rs::cs_raycast_local`** — no
   change to loop body; `with_neighbor_stepped` returns a sub
   with the frozen J on the deep side, which the DDA already
   uses as-is.
4. **`assets/shaders/sphere.wgsl::sphere_in_sub_frame`** — add
   `let is_frozen = uniforms.sub_meta.w;`. In the `out_of_box`
   block, wrap the `face_frame_jacobian_shader` +
   `mat3_inv_shader` calls in `if is_frozen == 0u { … }`.
   Skip the `un/vn/rn_corner` f32 bumps when frozen.
5. **`src/renderer/mod.rs::set_root_kind_sphere_sub`** — add
   `is_frozen: bool` param, store in `sub_meta.w`.
6. **`src/app/edit_actions/upload.rs`** — pass
   `sub.depth_levels() >= M_FREEZE`.
7. **`src/world/cubesphere.rs`** — no change. `mat3_inv`
   stays f64-internal; only CPU path calls it; shader never
   inverts past freeze so `mat3_inv_shader`'s f32 precision
   is a non-issue.

## 7. Cells-per-pixel sanity check

At `m = 25`, freeze at 12: each deep cell is `3/3^13 ≈ 1.9e-6`
local units. Camera at `~1.5` local, FOV 1.2 rad, 480 px: pixel
subtense ≈ `2·tan(0.6)/480 · 1.5 ≈ 4.3e-3` local. Deep cells
are 2270× sub-pixel. Walker must terminate earlier — recommend
`walker_limit = M_FREEZE + 3` (m ≤ 15, cell ≈ 4.3e-4 local,
10× sub-pixel with anti-aliasing headroom). Tunable.

## 8. Walker-limit tuning

`walker_limit` in the shader becomes `min(visual_depth,
M_FREEZE + 3)` where `visual_depth` is derived from
camera-to-cell distance. Past that, cells are sub-pixel and
we short-circuit to the shallower cell's material.

## 9. Editing past freeze

Edits (break / place) use CPU-side symbolic descent via
`WorldPos.sphere.uvr_offset + uvr_path`. No f32 coord
arithmetic at deep m — `last_edit_slots` records exact slot
chain. Freeze only affects *rendering*; the authoritative
tree state keeps full depth. Edits at `m > visual_depth`
are still recorded and propagated to the palette; they just
may not be individually visible until the camera zooms in
past the freeze.

## 10. Testing / validation

- `cs_raycast_local` parity: new unit test at `m = 20` with
  `M_FREEZE = 12` vs baseline unlocked (≤ freeze depth): hit
  cell must match to within `O(s²)` cell offsets; block id
  identical.
- `tests/sphere_ribbon_pop_precision.rs`: new
  `jacobian_freeze_no_position_drift` that casts a bundle of
  rays against a known solid cell at `m = 18`, asserts hit
  cell id constant across `M_FREEZE ∈ {10, 12, 14}`.
- Visual mosaic `tmp/sphere_descent/d{5..25}.png`: healthy
  rendering from `d5` through at least `d18` after the patch
  (18 of the 21 frames previously smeared). Deeper frames
  bounded by walker-limit termination, not precision.
- `cargo test --test e2e_sphere_descent` stays green.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Freeze too shallow → visible flat patches on curved shell | Curvature residual at `M_FREEZE = 12` is 4.2e-6 relative — five orders below pixel; empirical choice, tunable |
| Freeze too deep → keeps the current collapse | Cap `M_FREEZE ≤ 14` (shader `mat3_inv_shader` wall) |
| Path-step bubble-up correctness past freeze | `Path::step_neighbor_cartesian` is pure slot-packed integer math — unaffected by coord precision; unit-tested already |
| Edit / render path divergence | Render only decouples *Jacobian*; `render_path` + `uvr_prefix_slots` keep full depth, so walker dispatches the same cells CPU sees |

## 12. Ambiguity resolved

One judgment call: the spec asks to freeze `un/vn/rn_corner`
"as f32 stable" at freeze depth, but the current
`with_neighbor_stepped` increments them per step to feed the
Jacobian rebuild. Resolution: past `M_FREEZE`, do **not**
update the f32 corner fields — they're unused once the
Jacobian is frozen. Symbolic cell identity comes from
`render_path`'s slot prefix, which is untouched by the
freeze. This keeps the freeze semantics clean (no
silently-wrong f32 arithmetic) without cost.
