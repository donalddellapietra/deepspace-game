# Sphere raycast precision confusion

Notes from a debugging session where I misdiagnosed the f32 precision
issue in the sphere walker. Writing this down so the next person (or
me, next session) doesn't retrace the same wrong path.

## The symptom

At anchor depth ≥ 7 on the sphere world, the CPU raycast produced
cell bounds with `size ≈ 1e-14` (normalized face coords). Cell-
boundary ray-plane / ray-sphere intersections in `cs_raycast`
couldn't distinguish adjacent planes, so the stepping loop failed
to advance and the raycast returned `None` or stopped at the wrong
cell.

## The misdiagnoses I cycled through

1. **"Walker descends too deep through uniform subtrees."** Added
   a `uniform_type == UNIFORM_EMPTY` early-exit to the walker to
   return at the current cell size instead of bottoming out at
   `MAX_FACE_DEPTH`. This worked as a band-aid for shallow zoom
   but broke deep-zoom edits: at anchor 15 the walker would
   terminate at depth 3 (the face root's uniform-empty slot),
   returning a big cell — user sees huge highlight, wrong
   granularity.

2. **"Anchor depth isn't being honored."** Added a
   `max_face_depth` cap to the walker so it terminates at
   `anchor_depth - 2` levels. Cells scaled with zoom. Seemed
   right — until I tested at depth 30 and hit the f32 wall
   anyway (size `≈ 1e-14`).

3. **"Precision is fundamentally bounded by f32. Need f64 or
   cap depth."** Proposed Option A (f64 CPU math, f32 shader).
   Claimed "cells at depth 30 are sub-pixel anyway so the
   shader's coarse LOD is fine." The user called this out:
   **"What are you talking about at depth 30, individual cells
   are sub-pixel on the screen? Do you understand what this
   project is?"**

## The actual confusion

I was thinking of the tree depth as an *absolute* multi-resolution
structure: cells at depth N are `1/3^N` of the *whole world*,
rendered at world-scale, therefore sub-pixel at deep N.

That's the mental model for a fixed-scale octree engine. It's
wrong for this project.

**This is an infinite-zoom voxel engine.** At any anchor depth N,
the render frame is the specific `[0, 3)³` cell the camera lives
inside, expressed in **local coords of that cell**. Cells at the
render-frame level occupy O(1) local units and get rendered at
full screen size. "Depth 30" doesn't mean "tiny cells rendered
small" — it means "zoom into a specific sub-cubic-cell 30 times,
and render what's there at full resolution." A single block at
depth 30 might be 50 pixels wide on screen, same as a block at
depth 5.

Cartesian achieves this via **ribbon-pop**: at every tree level,
ray_origin and ray_dir are rewritten into the child frame's local
coords (`ray_origin = slot_xyz + ray_origin/3; ray_dir /= 3`).
Each frame's math operates in O(1) local units regardless of
absolute depth. That's how Cartesian supports 60+ layers with f32.

## Where my sphere architecture violated this

In `app/frame.rs::compute_render_frame`, I made
`NodeKind::CubedSphereBody` a **terminal render root**. Reasoning
at the time: "the body's 27 children aren't XYZ slots, so the
anchor path can't mechanically descend through it; stop at the
body and let the shader handle sub-body detail via
`sphere_in_cell`."

This is WRONG. It collapses every deep-sphere zoom level into one
monolithic face-subtree descent starting from the face root at
body-local scale. The walker's cell bounds are in face-root-
normalized coords, and at deep levels `size = 1/3^N` hits f32
precision walls.

The correct architecture: `compute_render_frame` should descend
**through** the face subtree just like Cartesian descends through
Cartesian nodes. At anchor depth 30, the render frame IS the
30th-deep face cell. Ray + cell bounds in its local `[0, 3)³`
frame. Cells at that level are size 1 local unit. f32 precision
fine. Matches Cartesian architecturally.

The face UVR geometry IS non-linear at the face-root level (the
cubemap warp), but inside the face subtree each child is just a
slot-aligned sub-cell of its parent — affine nesting, same as
Cartesian. The face-root metadata (inner_r, outer_r, face, base
UVR offset) travels with the frame, letting the ray-march
reconstruct body-XYZ cell planes from local frame coords without
ever needing to represent `1/3^30` directly.

## Lessons

- **"Depth N" in this project means N levels of recursive zoom,
  each with its own local render frame.** It doesn't mean "render
  cells that are 1/3^N of the world at world scale."

- **f32 precision isn't bounded by tree depth.** It's bounded by
  whether each level's math stays O(1) local. If the architecture
  violates that — e.g., by descending monolithically from a shallow
  render root — precision collapses regardless of f64/f32.

- **The fix for a precision issue in this codebase is almost always
  "ribbon-pop at every level" — never "cap depth" or "use wider
  float."** If you're reaching for wider floats, check whether the
  architecture has a level where it's supposed to rescale and isn't.

- **`NodeKind::CubedSphereBody` is not a terminal frame root.** It
  behaves like any other Cartesian-ancestor node. The render frame
  descends through it into the face subtree, inheriting the
  body's radii + face metadata as it goes. The sphere nonlinearity
  lives at the face-root boundary, not at every level.
