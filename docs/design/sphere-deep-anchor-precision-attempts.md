# Sphere mode: deep-anchor camera jitter + CPU raycast offset

Symptom (per user, repeatedly verified):

- At anchor depth ~16+ (layer ~8 and below) in sphere-render mode
  (`--planet-render-sphere 1`):
  - The camera **jerks while moving** — small, ULP-scale jumps
    that grow more visible the deeper the anchor.
  - The CPU raycast for break/place becomes **increasingly off**:
    the broken/placed cell is offset from the cell under the
    crosshair, with the offset growing with anchor depth.

Constraint: rendering itself (the dedicated `march_in_proto_cube`
walker, commit 1d979fd) is accepted and works. The remaining bug
is purely camera-position precision feeding the GPU and CPU
raycast.

## Hard constraints from the user

- `MAX_STACK_DEPTH = 8` globally — do not raise it.
- **Do not** introduce a `TangentBlock` `NodeKind` (the reference
  worktree `sphere-mercator-1-2` does this; explicitly rejected).
- **Do not** introduce a sub-frame architecture for the proto
  cell.
- **Do not** reinvent the wheel — "JUST CARTESIAN BUT ROTATED",
  use the cartesian renderer with per-cell tangent rotation.
- Take the **one** idea from `sphere-mercator-1-2`: a dedicated
  walker with a deeper local stack so deep break/place actually
  shows up. (Done — `march_in_proto_cube`, PROTO_STACK_DEPTH=24,
  no LOD termination.)
- No destructive git (no `reset --hard`); revert with `git revert`.

## Root-cause hypothesis

`WorldPos::in_frame` (in `src/world/anchor.rs`) accumulates the
camera's frame-local position by walking the anchor path in `f32`,
multiplying offsets by `1/3` per step. The Cartesian renderer
sidesteps this: when the camera descends, the **render frame**
deepens with the anchor, so the walk from anchor → render frame
is short (a handful of levels) and `cam_local` stays in
`[0, 3)³` with healthy ULPs.

Sphere-render mode pins the render frame to the WrappedPlane
(`compute_render_frame` short-circuits to the WP because the
slab subtree above ends in `Empty` parents — `slot.y` axis is
sparse). So as the anchor goes from depth 13 (just above the
slab) to depth 18+, the in_frame walk grows by 5+ levels. Each
step is `(slot + offset) / 3`. After 5 levels of `/3`, the
significand-cost of one input ULP becomes the visible jitter
in `cam_local`. The GPU camera (built from `cam_local`) and the
CPU raycast origin both inherit this noise.

## Attempts

### A1. f64-internal `in_frame` + sub-cell `frac` descent in CPU raycast

Commit `b716d75` → reverted by `fe6e28a`.

Idea: do the long anchor walk in `f64` and only narrow to `f32`
at the boundary. Add a sub-cell `frac` traversal inside
`cpu_raycast_sphere_uv`'s cube branch so the f32 cube-local
coords don't need to resolve sub-cell structure.

Result: per user, **neither issue was fixed**. f64 internal walk
must still cross the same f32 boundary into the GPU camera
uniforms; the GPU march is f32 throughout, so any precision win
on the CPU side is invisible.

### A2. Cartesian frame upgrade when camera enters the proto cell

Commit `238b7dc` → reverted by `5b64373`.

Idea: when the camera's anchor descends into the proto cell,
synthesize a Cartesian render frame at that depth so the same
"render frame deepens with anchor" trick used by Cartesian
rendering applies here too.

Result: per user, **same fundamental issue**. The synthesized
frame's basis still has to express the tangent rotation of the
slab cell, and the camera-to-frame conversion still walks the
sphere subtree in f32. (In hindsight: the "frame" abstraction
in this codebase doesn't carry an arbitrary rotation; the upgrade
landed on the wrong cell axes.)

### A3. Shallowed camera anchor for sphere-mode rendering + raycast

Commit `0e86add` → reverted by `999a073` (current HEAD).

Idea: for sphere-render mode, project the camera's WorldPos up
to a shallower anchor (e.g. WP depth) before computing
`cam_local`, so the in_frame walk is bounded by `(camera_anchor
- WP_anchor)`, which is small. Added
`WorldPos::shallowed_to(target_depth)` (still in the codebase as
dead code after the revert).

Result: per user, **doesn't help**. Shallowing is the inverse of
deepening — it accumulates `(slot + offset) / 3` going *up*
instead of going *down*, so the same f32 ULP cost shows up in
the projected position. The walk is just being relocated, not
removed.

## What's still in the tree

- `march_in_proto_cube` walker with `PROTO_STACK_DEPTH = 24` and
  no LOD termination — accepted, fixes deep break visibility.
- `cube_local_hit` branch in `cpu_raycast_sphere_uv` — accepted.
- `WORLD_SIZE` interaction-range cap for sphere mode in
  `interaction_range_in_frame` — accepted.
- `WorldPos::shallowed_to` — dead code from A3; safe to remove.
- `--disable-highlight` flag wired through the harness scripts
  — accepted (highlight overlay was broken in sphere mode long
  before this work).

## What we ruled out by user direction

- Sub-frame architecture rooted at the proto cell.
- `TangentBlock` `NodeKind` (per `sphere-mercator-1-2`).
- Raising `MAX_STACK_DEPTH`.
- Anything not "cartesian but rotated".

## Open question for the next attempt

The architectural mismatch is: Cartesian's precision strategy
is *the render frame moves with the anchor*. Sphere mode
currently keeps the render frame at the WP root because the
slab subtree's slot.y axis is sparse, so `compute_render_frame`
hits Empty above the slab and gives up. Some thought is needed
on how to deepen the render frame *into the slab cell* without
introducing a new NodeKind (TangentBlock) or a sub-frame, and
without breaking the assumption that frames have axis-aligned
basis. The cube basis (u=lon-tan, v=+radial, w=lat-tan) is
**not** axis-aligned in WP-local space — that rotation is what
makes the existing frame abstraction not naturally extend.

A1 / A2 / A3 each attacked the symptom (f32 walk noise) without
removing the walk itself. The fix probably has to remove the
walk, which means either:
- Make the WP slab subtree dense along slot.y so the existing
  Cartesian frame-deepening descends into the slab cell
  naturally (would let `compute_render_frame` find a deep
  Cartesian frame inside the slab without a new NodeKind), or
- Express the slab cell's tangent rotation as a small extension
  to the frame abstraction (but axis-aligned-only is a deep
  assumption — would touch a lot of code).

Neither is in the "JUST CARTESIAN BUT ROTATED" framing yet.
