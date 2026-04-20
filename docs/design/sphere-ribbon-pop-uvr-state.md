# Sphere ribbon-pop — symbolic UVR camera state (stage 5)

Stages 1-4 built the `SphereSub` frame + Jacobian + local-frame DDA
on CPU and GPU. They're architecturally correct. What they lack is
a camera position representation that stays precise past the
body-local f32 wall (~14 face-subtree levels). This stage adds
that representation.

The principle is the same one Cartesian ribbon-pop uses: the camera
knows where it is **symbolically** (as an integer path), and the
only floating-point coord is a small offset within the deepest cell.
Stage 5 applies that principle to the face-subtree side.

## Problem

`WorldPos { anchor: Path, offset: [f32; 3] }` works perfectly as
long as slots are XYZ-semantic. When the camera descends into a
`CubedSphereBody`, further descent is into face subtrees whose slots
are UVR. Trying to reuse `anchor.push(xyz_slot)` for those depths
produces a path the renderer can't interpret.

Even without that — even if you just stop the anchor at the body
cell and let `camera.in_frame(&body_path)` give you a body-local
`[0, 3)³` position — you're bounded by f32 eps of ~5e-8 body-local
units. Block cells at face-subtree depth 30 are ~1e-14 body-local
wide. Nothing you do downstream recovers that.

Stages 1-4 feed `cam_local` to `sphere_in_sub_frame` via body-XYZ
subtraction: they compute `J_inv · (cam_body - c_body)`. Both terms
are body-local; their difference is below f32 eps at deep sub-frame
depth. So the sub-frame's `cam_local` is noise — exactly the bug
making every pixel return a hit in the depth ≥ 5 screenshots.

## Data model

```rust
// anchor.rs
pub struct WorldPos {
    pub anchor: Path,            // Cartesian descent, world-root → body
    pub offset: [f32; 3],        // [0,1)³ in anchor's local frame
                                 // (Cartesian when sphere is None)
    pub sphere: Option<SphereState>,
}

pub struct SphereState {
    /// Cached for convenience; always == anchor when sphere is Some.
    pub body_path: Path,
    pub face: Face,
    /// Symbolic face-subtree descent. Slots use UVR semantics:
    /// `slot_index(us, vs, rs)` where us→u-axis, vs→v-axis,
    /// rs→r-axis on the face. No precision loss — integer data.
    pub uvr_slots: Vec<u8>,
    /// [0,1)³ inside the deepest UVR cell. f32 eps here means
    /// precision relative to the *innermost* cell, not the body.
    pub uvr_offset: [f32; 3],
}
```

Invariants:
- `sphere.is_some()` iff the camera is inside a `CubedSphereBody`
  cell in the tree.
- When `sphere.is_some()`: `anchor` ends at the body cell.
- `uvr_offset ∈ [0, 1)³` always (same contract as `offset`).

## Transitions

### Zoom-in (anchor deepens)

**Case 1: sphere is None**. Cartesian ribbon-pop, unchanged.
Exception: if the child we're entering has kind `CubedSphereBody`,
initialize sphere state at the end of the step:

```
(the normal XYZ slot push + offset rescale runs)
if child.kind == CubedSphereBody { inner_r, outer_r }:
    cam_body = offset * 3                    // body-local [0,3)³
    (face, un, vn, rn) = body_point_to_face_space(cam_body, inner_r, outer_r, 3.0)
    self.sphere = Some(SphereState {
        body_path: self.anchor.clone(),
        face,
        uvr_slots: Vec::new(),
        uvr_offset: [un, vn, rn],
    })
```

This is the **single** lossy XYZ→UVR conversion in the system. It
happens once per body-entry, at the transition boundary where the
conversion is meaningful (camera is on/near the shell).

**Case 2: sphere is Some**. Symbolic UVR refinement only:

```
us = floor(uvr_offset[0] * 3).clamp(0, 2)
vs = floor(uvr_offset[1] * 3).clamp(0, 2)
rs = floor(uvr_offset[2] * 3).clamp(0, 2)
sphere.uvr_slots.push(slot_index(us, vs, rs) as u8)
sphere.uvr_offset[0] = sphere.uvr_offset[0] * 3 - us as f32
sphere.uvr_offset[1] = sphere.uvr_offset[1] * 3 - vs as f32
sphere.uvr_offset[2] = sphere.uvr_offset[2] * 3 - rs as f32
```

`anchor` does NOT get a slot pushed in this case — the anchor stops
at the body and never extends. All deeper symbolic state lives in
`uvr_slots`.

### Zoom-out (anchor shallows)

**Case 2 (sphere Some)**: pop `uvr_slots` first:
```
if !sphere.uvr_slots.is_empty():
    slot = sphere.uvr_slots.pop()
    (us, vs, rs) = slot_coords(slot)        // UVR semantic, same numeric formula
    sphere.uvr_offset = (offset + [us, vs, rs]) / 3
    return
// uvr_slots empty — next pop exits the body:
self.sphere = None
// fall through to Cartesian anchor.pop()
```

When leaving the body: recompute `offset` as the body-local XYZ of
the pre-exit UVR position, so Cartesian continues smoothly. In
practice this is just `face_space_to_body_point(face, un, vn, rn,
...) / 3` to get offset in the body's `[0, 1)³`.

**Case 1**: regular Cartesian pop.

### in_frame

Signature unchanged: `in_frame(&frame) -> [f32; 3]`. Branch on
relationship between `frame` and the camera's state:

- Frame is an ancestor of body (shallow): Cartesian walk as today.
  Sphere state is ignored; the result is the camera's world-body
  position in frame-local.
- Frame == body: if `sphere.is_some()`, reconstruct body-local via
  `face_space_to_body_point(face, un, vn, rn, inner_r, outer_r,
  3.0)` where `(un, vn, rn)` is assembled from `uvr_slots +
  uvr_offset`. This is a lossy reconstruction; precision is f32 eps
  relative to body_size. But for any consumer that needs body-local
  coords at moderate accuracy, this is fine.

### in_sub_frame (new method)

Cheap + exact for the sphere case:
```rust
pub fn in_sub_frame(&self, sub: &SphereSubFrame) -> [f32; 3] {
    let sphere = self.sphere.as_ref().expect("camera not in a sphere");
    // sub-frame depth M = sub.frame_size log — the number of
    // uvr_slots consumed by this sub-frame.
    let m = sphere.uvr_slots.len();
    debug_assert!(m >= MIN_SPHERE_SUB_DEPTH as usize);
    debug_assert!(sphere.face == sub.face);
    // sub-frame is the cell at uvr_slots[..m]; camera is at
    // that cell + uvr_offset. So cam_local = uvr_offset * 3 for
    // each axis (the [0,1)³ → [0,3)³ sub-frame-local scaling).
    [
        sphere.uvr_offset[0] * 3.0,
        sphere.uvr_offset[1] * 3.0,
        sphere.uvr_offset[2] * 3.0,
    ]
}
```

**This is the key mechanical improvement.** No body-XYZ subtraction.
Camera local coord comes directly from symbolic state, representing
position within the deepest cell as a plain `[0, 1)³` scaled to
`[0, 3)³`. Precision is always ~1e-7 of a single cell, independent
of how deep we are.

## compute_render_frame

Reduces to: if `sphere.is_some()` and `uvr_slots.len() >=
MIN_SPHERE_SUB_DEPTH`, build `SphereSub` from `sphere.uvr_slots +
face`. Else keep current logic for Cartesian/Body.

```rust
if let Some(s) = camera_pos.sphere.as_ref() {
    let m = s.uvr_slots.len();
    if m >= MIN_SPHERE_SUB_DEPTH as usize {
        // Accumulate un/vn/rn_corner from uvr_slots (symbolic).
        // Derive (c_body, J, J_inv) from face_frame_jacobian.
        // render_path = body_path + FACE_SLOTS[face] + uvr_slots (all symbolic).
        return ActiveFrame::SphereSub(...);
    } else {
        return ActiveFrame::Body { ... };  // shallow, exact march
    }
}
```

No more descending the Cartesian anchor path looking for face
children, no more reinterpreting XYZ slots as UVR. The data is
already symbolic and correctly tagged.

## gpu_camera_for_frame

SphereSub arm pulls `cam_local` directly:
```rust
ActiveFrameKind::SphereSub(sub) => {
    self.camera.position.in_sub_frame(&sub)  // symbolic, precise
}
```

The `J_inv · basis` transform for `(forward, right, up)` is
unchanged — directions don't carry precision issues.

## Edit actions

`edit_actions::frame_aware_raycast` SphereSub arm:
```rust
ActiveFrameKind::SphereSub(sub) => {
    let cam_local = self.camera.position.in_sub_frame(&sub);
    let ray_dir_body = self.ray_dir_in_frame(&sub.body_path);
    cpu_raycast_in_sub_frame(
        library, world_root, &sub,
        /* render_path */ &sub.uvr_render_path, // body_path + FACE_SLOTS[face] + uvr_slots
        cam_local, ray_dir_body,
        edit_depth, lod,
    )
}
```

`render_path` for the sub-frame is constructed at frame-build time
and stored on `SphereSubFrame`. It's needed for walk-to-node
lookups inside the DDA and for hit-path construction.

## Execution order (one commit)

1. `anchor.rs` — add `SphereState`, extend `WorldPos`, rewrite
   `zoom_in`/`zoom_out` to branch on sphere, add `in_sub_frame`.
2. `cubesphere.rs` — add `body_to_sphere_state(cam_body, inner_r,
   outer_r) -> SphereState` helper if it cleans up the transition.
3. `frame.rs` — rewrite `compute_render_frame` to consume
   `sphere_state` instead of walking XYZ slots past body. Store
   `render_path` and `uvr_render_path` on `SphereSubFrame`.
4. `app/mod.rs::gpu_camera_for_frame` — SphereSub uses
   `in_sub_frame`.
5. `edit_actions/mod.rs` — SphereSub raycast uses `in_sub_frame`.
6. `edit_actions/upload.rs` — pass `sub.render_path` (already done
   via `active_frame.render_path`, verify).
7. Tests:
   - `anchor::sphere_state_initialized_on_body_entry` — build a
     world, simulate a zoom-in that crosses the body boundary,
     assert sphere state is populated.
   - `anchor::uvr_symbolic_precision_at_depth_30` — simulate 30
     sphere zoom-ins, assert `uvr_offset ∈ [0,1)³` and zooming
     out 30 times exactly reverses the state.
   - `anchor::zoom_cycle_preserves_position` — arbitrary sequence
     of zoom-in/zoom-out (mixing boundary crossings) round-trips
     back to the initial WorldPos.
   - `sphere_zoom_seamless_*` — the existing screenshot test at
     depths 3..8 should pass with planet fractions monotonically
     varying, no 100%-hit mode.
8. Single commit; rule in `feedback_no_intermediate_visual_states`.

## Precision argument, restated

The only lossy step in the stage-5 architecture is
`body_point_to_face_space` at body entry. It converts one body-XYZ
point (the camera's pos at the moment it crosses the body shell)
into `(face, un, vn, rn)` with ~1e-7 relative error. That error
propagates into `uvr_offset`, but `uvr_offset` is subsequently
refined symbolically — error doesn't compound across depths. At
face-subtree depth 30, the camera's `uvr_offset` is within its
deepest cell to ~1e-7 of that cell's size. That's exactly the
precision invariant Cartesian ribbon-pop maintains. No wall.

## What stages 1-4 keep

All of it. The Jacobian, `cs_raycast_local`,
`sphere_in_sub_frame`, uniforms, GPU dispatch — they're correct.
What changes is the *input* they receive: `cam_local` now comes
from symbolic UVR state, not from a lossy XYZ subtraction. That's
the one fix. Stage 5 is a camera-state upgrade, not a shader
rewrite.
