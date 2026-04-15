# Unified Driver Refactor — Living Architecture Doc

This document describes the *current state* of the renderer and
the path to the unified-driver architecture. It supersedes earlier
scope sketches: where the implementation has diverged from the
sketch, this doc reflects the implementation.

## What "unified driver" means

The renderer is one ray-march loop that walks **the same tree**,
**the same buffer**, **the same dispatch protocol**, regardless of
where in the world the camera sits or how deep the content goes.

- One **NodeKind dispatch** at every cell entry: Cartesian,
  CubedSphereBody, CubedSphereFace.
- One **stack** for the inner DDA, one **ribbon** for popping
  upward through ancestor frames.
- One **buffer layout**: all reachable content packed by BFS from
  the absolute world root, with distance-LOD flattening on
  Cartesian subtrees.
- The render frame chooses **where the GPU starts marching**;
  precision is always frame-local, no matter how deep the camera's
  absolute path.

This is structural unification — one loop, one stack, one
descend/ascend protocol. It is **not** "one identical primitive
for every cell shape": curved face cells need spherical r-axis
tests, Cartesian cells need planar tests. Kind-specific behavior
lives in metadata + small switches *inside* the unified loop, not
in parallel pipelines.

## Current state (as of this commit)

### Done

1. **Anchor coordinates.** `WorldPos = (Path, offset)` keeps f32
   bounded to a single cell at any depth (`world::anchor`).
2. **NodeKind dispatch in pack and shader.** A planet is a
   `CubedSphereBody` node sitting at slot 13 of the world root —
   no parallel pipeline (`world::tree`, `assets/shaders/ray_march.wgsl`).
3. **Kahan-compensated face DDA boundaries.** `walk_face_subtree`
   accumulates cell bounds with Kahan summation rather than
   `cells_d = pow(3, depth)` quantization. Lifts the precision
   wall in the inner sphere DDA.
4. **Frame-root dispatch.** `Renderer.set_root_kind_*` +
   `uniforms.root_kind`. Shader at start of `march()` either
   enters `march_cartesian` (Cartesian or face frame) or
   `sphere_in_cell` (body frame), with explicit face metadata for
   face-root starts and body pops.
5. **Pack from world root with camera in WORLD coords.** Single
   pack call produces a buffer that any frame depth can ray-march
   from. Distance-LOD decisions are at world scale and shared
   across frames.
6. **Build ribbon.** `world::gpu::build_ribbon` walks the GPU
   buffer down the frame path and emits pop-ordered ancestors.
7. **Shader ancestor pop.** `march()` outer loop: dispatch on
   current frame; on miss, pop next ribbon entry, transform the
   ray (`origin/3 + slot_offset`, `dir/3`), continue at the
   ancestor's buffer index. Hop budget 80.
8. **Render-frame walker is face-aware.** `app::frame::compute_render_frame`
   accepts Cartesian, CubedSphereBody, or CubedSphereFace as a
   frame root and can descend into face cells on the camera path.
9. **Highlight transform.** `aabb_world_to_frame` projects the
   cursor's AABB into the frame the shader sees.
10. **Module split.** `world::gpu` is `gpu/{mod, types, pack, ribbon}.rs`;
    frame helpers live in `app/frame.rs`. Shader is one file but
    organized into clearly-marked sections.
11. **CPU / anchor face-frame helpers.** `world::edit` and
    `world::anchor` now have canonical single-body helpers for
    face-frame projection and inverse projection without hardcoded
    absolute body centers.

### Not yet done

1. **Highlight across ancestor pops.** `aabb_world_to_frame` uses
   the *original* frame. If a hit happens after one or more pops,
   the AABB is in the wrong frame and the highlight outline drifts.
   Needs the shader to report the pop level at hit and the CPU
   to pre-compute AABBs for each ribbon level.
2. **Cap removal.** `cs_edit_depth` clamp `[1, 14]`, hardcoded
   `walk_face_subtree d <= 22u`, `MAX_STACK_DEPTH = 16` are still
   in place. Face-cell-as-frame removed the main precision wall;
   these are now just explicit implementation caps to retire.
3. **Face seam transitions inside a single sphere_in_cell.** The
   current code re-projects the ray to a face on every step,
   which works but is wasteful. With face-cell-as-frame in place,
   crossing a u/v boundary becomes a stack pop + push to a
   neighbor face's frame.
4. **Sphere outer-shell exit.** When a ray exits a body upward,
   today we `return` miss and the caller advances Cartesian DDA
   past the body cell. With ribbon in place, this should pop to
   the body's parent and continue from the *exit* point, not from
   the body's bounding-cube boundary.
5. **Performance.** `pack_tree_lod` from world root runs every
   frame and is O(visible nodes). Acceptable today; needs caching
   when content density grows. Not a correctness item.

## File layout

```
src/world/gpu/
  mod.rs       — re-exports + module-level docs
  types.rs     — GpuChild, GpuNodeKind, GpuCamera, GpuPalette
  pack.rs      — pack_tree, pack_tree_lod (BFS from any root)
  ribbon.rs    — GpuRibbonEntry, build_ribbon

src/app/
  mod.rs       — App struct + lifecycle + render_frame method
  frame.rs     — compute_render_frame, frame_origin_size_world,
                 aabb_world_to_frame (pure helpers)
  edit_actions.rs — edit/upload/highlight pipeline
  event_loop.rs  — winit ApplicationHandler
  cursor.rs    — cursor lock/unlock
  input_handlers.rs

src/renderer.rs — wgpu pipeline + GpuUniforms + bind groups
                  (root_kind dispatch, ribbon storage buffer)

assets/shaders/ray_march.wgsl — single-file WGSL with sections:
  1. Uniform / buffer declarations
  2. Tree access helpers (child_packed etc.)
  3. Cubed-sphere geometry helpers (face_normal etc.)
  4. walk_face_subtree (Kahan-compensated boundary accumulator)
  5. sphere_in_cell (face-DDA inside one body cell)
  6. march_cartesian (the inner Cartesian DDA)
  7. march (outer loop: dispatch + ribbon pop)
  8. fs_main (sky color + highlight)
```

## Coordinate systems (precision discipline)

There are exactly three coordinate systems in flight:

1. **Path-anchored** (`WorldPos { anchor: Path, offset: [f32; 3] }`).
   Used by the camera, edits, cursor. Exact at any depth.
2. **World** (`[f32; 3]` in `[0, WORLD_SIZE)`). Used by:
   - Cursor raycast (CPU walks tree in world XYZ).
   - Pack distance-LOD test (consistent across frames).
3. **Frame-local** (`[f32; 3]` in `[0, 3)` of frame's cell). Used by:
   - Camera position handed to the shader (`cam_local`).
   - Highlight AABB after `aabb_world_to_frame`.
   - All shader DDA state.

Conversions live in `app::frame` and `WorldPos::in_frame_in`. Never
do f32 arithmetic on world XYZ at deep zoom — magnitudes grow
unbounded. Always project into frame-local first.

## What lifts the depth limits

Every "depth limit" in the engine is actually one of:

- **f32 precision in the camera's coords.** Solved by frame-local:
  camera in `[0, 3)` regardless of absolute path depth.
- **f32 precision in cell-boundary accumulation.** Solved by Kahan
  summation in the face walker. Sub-ULP error per accumulation,
  total < 1 ULP at any depth.
- **Shader stack budget.** Currently 16 levels of Cartesian
  descent from the frame. Visual depth is bounded by screen
  resolution (~10–15 levels), so 16 is generous; the absolute
  tree depth doesn't matter because the frame can lift.
- **Face walker loop budget.** Hardcoded 22 today (matches demo
  planet's face depth + 2). Face-cell-as-frame means this is no
  longer a correctness wall, but it is still an explicit cap.

So "support 40+ layers" reduces to: the camera must always be in
a frame whose cells are not sub-ULP, and the inner DDA at that
frame must be well-conditioned. Both are infrastructure problems
already half-solved.

## Test inventory

`cargo test --lib` runs all unit tests. Counts are approximate:

- `world::gpu::types`: 4 tests (struct sizes, NodeKind round-trip)
- `world::gpu::pack`: 5 tests (pack_test_world, planet body
  packed, LOD flatten, near subtree kept full)
- `world::gpu::ribbon`: 7 tests (empty, single, multi, non-Node
  stop, OOB safety, real planet world)
- `app::frame`: 14 tests (compute_render_frame across
  configurations, frame_origin_size at various depths, AABB
  identity / scale / round-trip / inflation)
- existing world tests (`tree`, `cubesphere`, `edit`, `palette`,
  `sdf`, `worldgen`, `spherical_worldgen`, `state`)

Total target: 100+ tests post-refactor.

## Next-step recipe (for the session that picks this up)

1. **Highlight cross-pop.**
   - Shader return value gains a `frame_level` field (which pop
     level the hit happened at).
   - CPU pre-computes per-ribbon-level AABBs and the shader picks
     the right one for outline drawing.
2. **Cap removal.**
   - Drop `cs_edit_depth` clamp, `MAX_STACK_DEPTH = 16`,
     `walk_face_subtree d <= 22u`. Replace with `MAX_VISUAL_DEPTH`
     (still 16, but justified by screen resolution).
3. **Sphere outer-shell exit handoff.**
   - `sphere_in_cell` returns the exit point + new ray direction.
   - Caller (whoever popped into the body) continues from that
     point in the parent frame.
4. **Multi-body smoke test.**
   - `spherical_worldgen` install a second body at a different
     slot. Verify ribbon pop renders both bodies from any frame.

When all four land, the engine genuinely supports 40+ layers
without any hardcoded caps, with planet-deep editing working at
arbitrary face depth.
