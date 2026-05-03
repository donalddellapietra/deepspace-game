# TangentBlock rotation: making outside/inside views and movement consistent

This is the story of the `--rotated-cube-test` bug and the four-part
fix that made the rotated middle cube render correctly from outside,
from inside, and across the boundary, while keeping movement
precision-stable at deep anchor depths.

## What a TangentBlock is

`NodeKind::TangentBlock { rotation: R }` is a Cartesian Node whose
internal `[0, 3)³` storage frame is rotated by `R` relative to its
parent's slot frame. The cube of stone you see in the world is the
TB's storage cube rotated by `R` about its own centre `(1.5, 1.5, 1.5)`
in storage / `(0.5, 0.5, 0.5)` in the parent slot.

`R` is "local-to-parent": `parent_coords = R · local_coords`, both
expressed centred about the cube centre. The inverse `R^T` takes
parent-frame rays into the TB's storage frame so the renderer's
Cartesian DDA can index the TB's children by their storage slot
positions.

For `--rotated-cube-test` the middle cube uses `R = rot_y(π/4)`, so
its visible silhouette from above is a 45° diamond.

## The four-part bug

The geometry pipeline has four touchpoints where `R` / `R^T` need to
be applied. Each one was either missing, applied at the wrong pivot,
or applied at the wrong scale.

### 1. Direction-only `R^T` at TB entry (GPU + CPU raycast)

**State before fix**: HEAD's `march_cartesian` TB-child dispatch
applied `R^T` to ray *direction* but left ray *origin* untransformed:

```wgsl
let local_origin = local_pre_origin;                       // unrotated
let local_dir = R^T · local_pre_dir;                       // rotated
```

**Symptom**: the rotated cube **moved with the camera**. Rotation
shape always 45°, but the cube *translated* in world space as the
camera moved laterally.

**Why**: with origin not rotated, the inner DDA returns a parameter
`sub.t` for the *rotated-direction* local ray, but the outer code
applies that same `sub.t` to the *unrotated-direction* world ray
(`hit_world = O + sub.t · D`). The two parameters describe different
points; the mismatch scales linearly with the lateral component of
camera position, producing the apparent translation.

**Fix**: rotate origin AND direction by `R^T` about the cube centre
`(1.5, 1.5, 1.5)` in storage:

```wgsl
let centred = local_pre_origin - vec3<f32>(1.5);
let rotated = vec3<f32>(dot(rc0, centred), dot(rc1, centred), dot(rc2, centred));
let local_origin = rotated + vec3<f32>(1.5);
let local_dir = vec3<f32>(dot(rc0, local_pre_dir), dot(rc1, local_pre_dir), dot(rc2, local_pre_dir));
```

Centred rotation is a rigid transform → arc length along the ray is
preserved → `sub.t` IS the world-frame parameter. Same fix in
`world/raycast/cartesian.rs::cpu_raycast_inner` so cursor targeting
matches the visual.

### 2. Centred `R` on the symmetric ribbon pop

The ribbon-pop code is the inverse of TB entry — when a ray exits a
TB and the ribbon walker pops up to the parent frame, it has to undo
the entry's `R^T`. The same direction-only-vs-centred dichotomy
applies. Fixed at `march.wgsl::march()` ribbon pop:

```wgsl
let scaled = ray_origin / 3.0;
let centred = scaled - vec3<f32>(0.5);
ray_origin = slot_off + vec3<f32>(0.5)
           + rc0 * centred.x + rc1 * centred.y + rc2 * centred.z;
ray_dir = rc0 * ray_dir.x + rc1 * ray_dir.y + rc2 * ray_dir.z;
```

Mirror in `world/raycast/mod.rs::cpu_raycast_in_frame` for cursor
targeting.

### 3. NO TB R^T at the frame-root

An earlier attempt also applied centred `R^T` at the start of
`march()` whenever the frame root was a TB, on the theory that
"camera arrives in the TB's unrotated local frame and needs to be
brought into storage." That premise is wrong: `in_frame_rot` already
returns the camera in TB-storage when the anchor crosses a TB
(via the tail walk's `cur_rot` accumulation), and `frame_path_rotation`
already applies `R^T` to the camera basis on the CPU. Applying R^T
at the frame root again was a double-rotation (`R^T · R^T`), wildly
wrong everywhere except the cube centre (the fixed point of any
rotation about itself).

**Removed** in commit `29d86ab` — the camera arrives in the right
frame for the DDA already; no extra transform needed.

### 4. Cell-local rotation-aware path step (`renormalize_world`)

After fixes 1-3, the renderer was geometrically correct. But movement
across the TB boundary teleported the camera off-axis. The bug was
in `step_neighbor_cartesian`'s pop-and-re-descend: when the path
bubbles up across cells, the inherited slot indices for the un-stepped
axes are world-frame integers above a TB but storage-frame integers
below it. Crossing into or out of a TB ancestor, those indices denote
entirely different physical cells.

For 90° rotations the indices reinterpret to a different consistent
permutation (visible glitch but topology preserved). For 45°,
`R · X-axis = (√2/2, 0, -√2/2)` mixes axes — single-axis stepping
cannot represent the boundary correctly. The path step has to
re-derive the slot indices using the rotation.

**Fix**: replace `renormalize_world` with a cell-local pop +
redescend in `world/anchor/world_pos.rs`:

- **Pop**: drop the deepest slot. If the popped cell was a TB, apply
  forward `R · (offset − 0.5) + 0.5` to convert from TB-storage frame
  into the parent's Cartesian slot frame BEFORE the standard
  `(slot + offset) / 3` rescaling.
- **Redescend**: pick the slot via `floor(offset · 3)`. If the picked
  child is a TB, apply `R^T · (offset − 0.5) + 0.5` to the post-floor
  offset to enter the TB-storage frame.
- **WrappedPlane wrap**: preserved on axis 0 — when the deepest cell's
  parent is a slab and offset[0] overflows, the slab edge wraps in
  place instead of bubbling.

All rotation pivots are at the constant `(0.5, 0.5, 0.5)` of a unit
cell. Magnitudes stay ≤ 0.5 regardless of anchor depth, so f32
precision is bounded by the cell-fraction, not by the world position.
Movement remains correct at any anchor depth.

The same R / R^T handling extends to:

- `zoom_in_in_world`: applies `R^T` post-floor when the descended-into
  cell is a TB.
- `zoom_out_in_world` (new): applies `R` pre-rescale when the popped
  cell was a TB. Wired into the runtime scroll-zoom caller in
  `event_loop.rs`.
- `update()` movement: applies `R^T_chain` (transpose of
  `frame_path_rotation` over the anchor) to the world-frame WASD
  step BEFORE adding to offset, so the step has the same axes as
  the offset (deepest cell's children frame) for paths that cross
  TBs.

## Why the absolute-coords approach in BROKEN STATE 2 wasn't viable

Commit `b2b574 BROKEN STATE 2` proved the cause by computing world
XYZ via `in_frame_rot` and re-deriving anchor + offset via a new
`from_world_xyz` (which walks root → anchor applying `R^T` at TB
descents). That fix was correct geometrically but used absolute world
coordinates `[0, 3)³`. At anchor depth 30, cell size is `3/3^30 ≈ 1e-14`,
well below f32 epsilon — movement of a few cells per second produces
zero observable change after world position is rounded to f32.

The cell-local pop-and-redescend in fix 4 keeps the boundary
correctness without the precision penalty.

## Sequence of attempted states

| State | Geometry transforms | Path step | Symptom |
|---|---|---|---|
| `BROKEN STATE 1` (`3691e40`) | Direction-only R^T | Cartesian (TB-blind) | Cube translates with camera; boundary stitched only because the same wrongness applies on both sides |
| `PROGRESS POINT 2` (`872001b`) | Centred R^T (entry + pop + frame-root) | Cartesian (TB-blind) | Geometry correct; couldn't dig past depth 1 because pack flattened TB children to Block |
| `BROKEN STATE 2` (`b2b574`) | Centred R^T entry + pop, no frame-root | Snap via `from_world_xyz` (absolute coords) | Geometry and boundary correct; movement freezes at deep anchors due to f32 precision wall |
| **PROGRESS POINT 3** (this) | Centred R^T entry + pop, no frame-root | Cell-local rotation-aware renormalize | Geometry correct; boundary correct; movement precision-stable at any depth |

## Files touched in PROGRESS POINT 3

- `assets/shaders/march.wgsl`: centred R^T at TB entry; centred R on
  ribbon pop; no frame-root R^T (removed in `29d86ab`).
- `src/world/raycast/cartesian.rs`: centred R^T at TB entry (CPU mirror).
- `src/world/raycast/mod.rs`: centred R on TB ribbon pop (CPU mirror).
- `src/world/anchor/world_pos.rs`: cell-local rotation-aware
  `renormalize_world`; rotation-aware `zoom_in_in_world`; new
  `zoom_out_in_world`.
- `src/world/anchor/path.rs`: `node_kind_at_depth` made `pub(super)`.
- `src/app/mod.rs::update`: `R^T_chain · step_world` before
  `add_local`; reverted from BS2's `from_world_xyz` snap.
- `src/app/event_loop.rs`: scroll-zoom uses `zoom_out_in_world`.
