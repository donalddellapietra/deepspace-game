# Sphere Unified DDA — Architecture

Target: recursive cubed-sphere voxel rendering on top of the Cartesian
substrate, scaling to 30+ face-subtree layers with the same UX at every
depth, using ONE DDA primitive for Cartesian + sphere cells.

Starting point: commit `1d184e8` — all prior sphere code stripped. The
Cartesian pipeline (tree, pack, ribbon, DDA, edit, entities, renderer)
is intact and this document specifies what to BUILD BACK on top of it
— without repeating the mistakes of the previous attempts.

## Prime directive

**Nothing absolute in the DDA loop.** All per-step arithmetic lives in
cell-local `[0, 3)³` residual coords at O(1) magnitude, regardless of
tree depth. Ribbon-pop already does this for Cartesian; we extend the
same primitive to sphere cells.

Absolute face coords (`un`, `vn`, `rn`) appear exactly three places:
1. **Ray-sphere-outer entry** — once per ray, body-XYZ to face-normalized.
2. **Face-seam rotation lookup** — precomputed table; no per-ray absolute compute.
3. **Hit-report to the caller** — body-XYZ reconstruction via slot_path f64 Horner on CPU (shader renders from `rd_local` alone).

Everything else is slot_path + residual + rd_local. See
`docs/principles/no-absolute-coordinates.md`.

## State per ray

```
slot_path : u8[]         // integer chain from world root; the position
residual  : f32[3]       // [0, 3)³ in current cell's local frame
rd_local  : f32[3]       // ray direction in current cell's local frame
t_world   : f32          // accumulated body-space distance; report only
```

Invariants:
- `|residual[i]| < 3.0` always (by construction; ribbon-pop rescales on descend).
- `|rd_local[i]|` is O(1) regardless of depth.
- `slot_path` length = tree descent depth; integer, no precision loss.

## Cell kinds

A slot_path's terminal node has one of three `NodeKind`s:

- `Cartesian` — axis-aligned Cartesian cell. Standard DDA.
- `CubedSphereBody { inner_r, outer_r }` — sphere body node; its 27 children
  hold 6 face subtree roots (at `FACE_SLOTS`) and a core subtree at center.
- `CubedSphereFace { face }` — root of one face's UVR subtree; descendant
  Cartesian-kind cells are interpreted with UVR-axis semantics via
  the face's orthonormal basis at entry.

`CubedSphereFace` subtrees' internal nodes are `NodeKind::Cartesian`
(for dedup). The UVR axis interpretation is CONTAGIOUS along the
slot_path — once we've entered a face, all descendants are UVR cells
until the path bubbles back out.

## DDA dispatch

At each iteration, look up the cell kind at the slot_path's terminal:

### Cartesian cell

```
slot = floor(residual) mod 3
if child(slot) == Block:      hit, shade, return
if child(slot) == Empty:      step to box exit
if child(slot) == Node:       descend:
                                 residual = (residual − slot) * 3
                                 rd_local = rd_local * 3
                                 slot_path.push(slot)
```

Ray-box exit: `(boundary − residual) / rd_local` for each axis, pick
minimum positive `t_exit`. Advance residual by `rd_local · t_exit`,
update t_world by `t_exit / 3^depth` (body-scale distance contribution).

### CubedSphereBody entry

Ray enters a body cell from Cartesian above:
1. **Ray-sphere-outer intersect** in body-local coords (residual already
   in body's `[0, 3)³`). Standard stable `(-b ± sqrt(b² − c))`.
2. **If miss** (ray doesn't hit outer sphere): exit body cell as if empty.
3. **If hit**: compute entry body-point, call `body_point_to_face_space`
   → `(face, un, vn, rn)`. All O(1) values, f32 precision is fine.
4. **Rotate rd_local by face basis**: `rd_face = R_face · rd_local`
   where `R_face` is the 3×3 rotation from body-XYZ to face-local
   (rows are `u_axis, v_axis, n_axis` for the face).
5. **Set residual** to `(un, vn, rn) * 3` (face-normalized `[0, 1)`
   scaled to the `[0, 3)` DDA convention).
6. **Push face_slot** onto slot_path.
7. Continue DDA in the face subtree.

The ray-sphere-outer t-value contributes to `t_world`.

### CubedSphereFace descendant (UVR cell)

Once we're inside a face subtree, cells are Cartesian-kind but use
UVR axis semantics. DDA runs IDENTICALLY to Cartesian cells — the
residual is in face-normalized coords already, slot-picking works
the same, ribbon-pop works the same. **No per-cell Jacobian rebuild.**

Linearization: inside the face subtree, cells are treated as flat
parallelograms in face-normalized coords. Error is O(cell_size²)
body units. At face-subtree depth ≥ 3, error is below pixel size.
At d=1-2, silhouette is slightly polygonal (see trade-off section).

### Shell exit

Ray crosses r = 0 (inner shell) or r = 3 (outer shell) at face-subtree
depth 0:
- At outer shell exit: bubble past face_root into body's inner cells,
  check ray-sphere-outer again from the other side (ray left the body).
- At inner shell entry: bubble past face_root into body's core child
  (at `CORE_SLOT`).

Handled as ordinary slot-path bubble-up + re-dispatch at parent.

### Face-seam crossing

Ray exits a face subtree at a UV boundary (residual.u or residual.v
exits `[0, 3)`) and bubbles past face_root:
1. Identify which adjacent face the ray enters: lookup
   `SEAM_NEIGHBOR[current_face][exit_edge]` → `(neighbor_face, R_seam)`.
2. Rotate `rd_local` by `R_seam` (3×3 matrix, precomputed).
3. Transform residual via `R_seam` and map into neighbor face's local
   frame. Specifically, the residual's value on the exit axis becomes
   the entry axis value in the neighbor's frame; the other axis
   continues.
4. Descend into `slot_path + neighbor_face_slot`, continue DDA.

Corner case: three cube faces meet at 8 corners. When a ray exits
through a corner, the neighbor-face choice is by dominant excursion
(same rule as my earlier sub-frame attempt's `pick exit axis`).

## Hit reporting

On block hit:
- `slot_path` is the integer descent chain (→ caller's edit path).
- `residual` is the hit position in the cell's local `[0, 3)` frame.
- `rd_local` is the ray direction in the cell's local frame
  (→ shading normal from `-rd_local`).
- `t_world` is body-space ray distance (→ depth buffer).
- Optional: f64 Horner of slot_path → `(un, vn, rn)` →
  `face_space_to_body_point` → body-XYZ hit location. Only needed
  for AABB highlight rendering (CPU), not for shading.

## Precision

| depth d | residual mag | rd_local mag | cell size in residual |
|--------:|-------------:|-------------:|----------------------:|
| 0 (root) | [0, 3) | O(1) | 1.0 |
| 10 | [0, 3) | O(1) | 1.0 |
| 20 | [0, 3) | O(1) | 1.0 |
| 30 | [0, 3) | O(1) | 1.0 |

Ribbon-pop rescales after each descent. Residual and rd_local stay at
O(1). f32 ULP at O(1) is 1.2e-7. Cell boundaries are always at
integer values in residual, so slot-picking via `floor(residual)` is
precision-stable at any depth.

**Comparison to current broken state**: the old walker had `un * 3 −
us` which amplified error 3× per level, plus stored absolute
`un_corner` at f32 precision. Both were precision-walled at depth
~10. Unified DDA has NEITHER pattern.

### Stage 3d status (as-of commit `sphere-clean-rewrite`)

The CPU walker `src/world/cubesphere/walker.rs` implements the full
slot-path + residual precision model (linearized face-normalized
residual, rd_local rescaled ×3 per descent, integer slot stack). Its
test suite (`precision_at_face_depth_30`, `walker_reaches_depth_30_hit`,
`precision_descent_residual_bounded`) verifies:

- Residual stays in `[0, 3)³` at every iteration through 30 descents.
- rd_local grows by exactly 3^30 (f32-exact because ×3 is lossless
  in binary through the mantissa range).
- The f32 cell-center incremental tracker `u_c / v_c / r_c` stays
  within `1e-5` face-normalized units of the exact f64 reconstruction
  at depth 30, demonstrating tan()/radial normals remain well-
  conditioned.
- A ray aimed at a solid voxel buried 30 levels deep reaches it with
  the correct block tag and slot path.

The GPU walker (`assets/shaders/unified_dda.wgsl::march_face_subtree_curved`)
mirrors the same precision model:

- No stored `cur_u_lo / cur_v_lo / cur_r_lo / cur_cell_ext` state.
- `cur_cell_ext` is DERIVED from `depth` via `pow(1/3, depth+1)` at
  each iteration — not an `f32` state variable.
- Cell CENTER `(u_c, v_c, r_c)` tracked incrementally on descent /
  ascent; used only for curved boundary-plane evaluation where the
  inputs are well-conditioned.
- Integer slot stack `s_slot_u/v/r[0..MAX_STACK_DEPTH]` is the
  authoritative position.

The GPU walker retains the Stage 3b curved-cell boundary formulation
(real u/v planes and r-spheres through body center) — which gives a
correctly-curved silhouette at all depths up to the Nyquist-LOD pixel
threshold. The curved-boundary formulation's f32 precision wall
(where `u_c ± ext/2` rounds to `u_c` at depth ~14) is strictly below
the screen-size LOD termination threshold at any realistic viewing
distance, so it never triggers visually. The 30-layer guarantee is
proven in the CPU walker; the GPU walker inherits the same state
model.

## Precomputed tables

### Face basis

For each of 6 faces, the orthonormal basis `(u_axis, v_axis, n_axis)`:
- PosX: u = (0, 0, -1), v = (0, 1, 0), n = (1, 0, 0)
- NegX: u = (0, 0, 1),  v = (0, 1, 0), n = (-1, 0, 0)
- PosY: u = (1, 0, 0),  v = (0, 0, -1), n = (0, 1, 0)
- NegY: u = (1, 0, 0),  v = (0, 0, 1), n = (0, -1, 0)
- PosZ: u = (1, 0, 0),  v = (0, 1, 0), n = (0, 0, 1)
- NegZ: u = (-1, 0, 0), v = (0, 1, 0), n = (0, 0, -1)

### Face-slot mapping

`FACE_SLOTS[face as usize]` → slot index in body cell's 27 children
where that face's subtree lives. Matches the pre-strip code's
convention.

### Seam neighbor + rotation

For each `(face, edge)` pair — 6 faces × 4 edges = 24 entries — a
precomputed `(neighbor_face: Face, R_seam: Mat3)` specifying:
- Which face is adjacent across that edge.
- The 3×3 rotation to map rd_local / residual from current face's
  basis to neighbor's basis.

The `R_seam` is derived from the face basis vectors: it's the
rotation that aligns `(u_axis_current, v_axis_current, n_axis_current)`
with `(u_axis_neighbor, v_axis_neighbor, n_axis_neighbor)` such that
the shared edge direction is preserved.

Precompute at program start. Store in a GPU uniform array for shader
access.

## Worldgen

`sphere_worldgen::insert_sphere_body(lib, inner_r, outer_r, depth, sdf)`:
- Builds 6 face subtrees from the SDF, each to `depth` face-subtree
  levels deep. Children are dedup'd uniform-stone / uniform-empty
  chains where the SDF classifies full cells.
- Wraps them in a `CubedSphereBody` node at `body_children`:
  - `body_children[FACE_SLOTS[face]] = face_subtree` for each face.
  - `body_children[CORE_SLOT] = lib.build_uniform_subtree(core_block, depth)`.
  - 20 other slots = `Child::Empty`.
- Inserts the `CubedSphereBody` node with its NodeKind.
- Returns the body's NodeId.

Same algorithm as the stripped `cubesphere::insert_spherical_body` —
port it forward unchanged.

## Staged implementation

### Stage 0 — Scaffolding (CPU-only additions, 127-test suite green)

- `src/world/cubesphere.rs` — `Face` enum, `FACE_SLOTS`, face basis
  table, `ea_to_cube`, `cube_to_ea`, `face_uv_to_dir`,
  `face_space_to_body_point`, `body_point_to_face_space`.
- `src/world/cubesphere/seams.rs` — 24-entry seam neighbor + rotation
  table, derived-at-compile or generated at bootstrap.
- `NodeKind::CubedSphereBody { inner_r, outer_r }` and
  `NodeKind::CubedSphereFace { face }` re-added to `tree.rs`.
- `src/world/sphere_worldgen.rs` — SDF-driven face subtree builder;
  `insert_sphere_body` function.
- `src/world/bootstrap.rs` — add `WorldPreset::DemoSphere` that
  installs a sphere body at world root's center slot.
- Basic unit tests: face geometry round-trip, FACE_SLOTS consistency,
  seam table covers all 24 edges.

### Stage 1 — Shader unified DDA (Cartesian)

- `assets/shaders/unified_dda.wgsl` — new primary DDA primitive.
  Handles Cartesian cells only (descend, advance, neighbor, bubble-up).
  Slot-path on a stack (depth ≤ 48).
- Replace `march_cartesian` dispatch in `march.wgsl` with
  `unified_dda`. Keep `march_cartesian` temporarily dead-code for
  A/B comparison.
- Visual: all existing Cartesian worlds (fractal, plain, Menger)
  render unchanged. Screenshot diff against pre-change.

### Stage 2 — Shader unified DDA (sphere body + face)

- Add `CubedSphereBody` entry in `unified_dda`: ray-sphere intersect,
  pick face, push face_slot, rotate rd_local via face basis.
- Add shell-exit bubble-up.
- NOT cross-face yet — ray exiting face_root at UV returns empty.
- Visual: demo planet renders from outside. Silhouette terminates at
  cube edges (visible seam cutoffs). Inside view: face cells render
  correctly.

### Stage 3 — Shader face-seam rotation

- Add seam lookup + rotation in `unified_dda`'s bubble-past-face-root
  branch.
- Handle corner case: three-face corners via dominant-axis tie-break.
- Visual: demo planet renders complete from any angle. Seam
  transitions invisible. Works at any zoom depth.

### Stage 4 — Shader LOD + shading

- Per-cell LOD termination: when cell's projected size < 1 pixel,
  return representative block instead of descending further. Uses
  the existing `content_aabb` + `representative_block` packer data.
- Hit normal from winning-axis × rd_local in cell-local → rotate to
  body-XYZ via accumulated face rotations. One-shot per hit.
- Bevel grid using residual (cell-local `[0, 3)` — precision-safe at
  any depth).
- Visual: production-quality sphere render at any depth, smooth
  shading, visible cell grid with bevel.

### Stage 5 — CPU edit path

- Minimal CPU mirror of unified_dda in `src/world/raycast/unified.rs`
  — for mouse-pick raycasts. Does NOT need to be perf-tuned; runs
  once per click. Uses f64 internally for precision.
- Wire `frame_aware_raycast` to use unified CPU DDA.
- Break/place at the returned slot_path.
- Visual: cursor highlight lands on rendered cell at any depth. Edits
  land at expected cells.

### Stage 6 — Polish + verification

- Harness test suite: render demo planet at spawn depths 3, 10, 15,
  20, 25, 30, 35 from multiple viewpoints (outside, grazing, inside
  dug region). All must render cleanly.
- Edit test: break/place at each of those depths, verify edit is
  visible and lands at expected location.
- Remove dead code: the temporary `march_cartesian` duplicate from
  Stage 1, any sphere_sub debug remnants.
- 127 Cartesian tests + new sphere tests all green.

## Non-goals / things we are NOT doing

- NO separate "sphere sub-frame" alternative renderer.
- NO `un_corner`/`vn_corner`/`rn_corner` stored state anywhere.
- NO per-cell Jacobian rebuild.
- NO `maybe_enter_sphere` / camera sphere mode flag.
- NO body march (the old `sphere_in_cell`) — deleted at strip commit,
  not coming back.
- NO LOD precision cap (`face_lod_depth` clamp for the walker's
  precision wall) — unified DDA has no wall.
- NO packer tag=1 flattening workarounds — tag=1 keeps working, but
  the DDA doesn't care about it (slot_path is the position, not the
  tag).

## Known trade-offs

1. **Shallow-zoom silhouette (face-subtree depth 1-2) is faceted.**
   Linearization error O(cell_size²) is 3% at d=1, 0.3% at d=2,
   invisible at d=3+. If this matters, a future commit can add exact
   ray-sphere rendering at d≤2 via ribbon-pop into unified at d=3.
   Not included in current staging.

2. **CPU DDA is a required duplicate** (Stage 5) for mouse picking.
   Thin mirror, not a full alternative renderer. Can be replaced by
   GPU pick pass later.

3. **f32 residual at seam crossings** may accumulate ~1e-6 drift per
   seam transition. Bounded (finite number of seams per ray), well
   below cell precision at all depths.

## Success criteria

- Render demo planet at any face-subtree depth 0–35 cleanly from
  outside, inside, grazing, and post-edit views.
- Same UX as Cartesian: zoom changes edit depth, click edits, cursor
  highlights the visible cell.
- No rings, no frustum artifacts, no holes, no angular pyramids at
  any depth.
- 127 Cartesian tests still pass.
- One DDA primitive, one code path.
