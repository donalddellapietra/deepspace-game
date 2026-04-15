# Undone / Hacky Pieces — Layer-1 Refactor

Honest accounting of what's still broken or stubbed after the
unified-driver work this session. Each item is a real correctness
gap or shortcut, not a polish nit.

## Hacks that shipped

### 1. Highlight overlay disappears on cross-pop hits

The shader skips the highlight ray-box test entirely when
`result.frame_level > 0`. So when the cursor's hit happens via
ribbon pop (camera in a deep frame, ray exits into an ancestor
and hits something there), the cell outline doesn't draw. The
proper fix: per-ribbon-level AABBs computed on the CPU and
indexed by `frame_level` at fs_main time.

### 2. `t` discontinuity across pops

Each ribbon pop restarts the DDA's `t` parameter from 0 in the
new frame. Visual results are correct (each frame's DDA is
internally consistent), but `result.t` returned from `march()`
is in whatever frame the hit happened in — not world distance.
The highlight's depth-compare uses `result.t`, so cross-frame
depth ordering is broken (works fine in single-frame hits, which
is the common case, but a moon-behind-planet scenario would
order things wrong).

`HitInfo.t` from `cpu_raycast_in_frame` has the same shape: it's
in the popped frame's units, not cumulative world distance.

### 3. `std::process::exit(0)` in test_runner instead of `event_loop.exit()`

`tick_test_runner_after_frame` calls `process::exit` to avoid
threading the `event_loop` reference through the call. anchor-
refactor's version threads it correctly. Works but skips winit
shutdown.

## Architectural pieces deferred

### 4. Face-cell-as-frame

`compute_render_frame` stops at face-cell boundaries. Camera
deep inside a planet's face subtree only gets body-level frame
precision (cells of size 1, body-local). The unified driver
calls for face-cell frames where each face cell can itself be a
frame root with `(face_id, u/v/r bounds)` metadata, dispatching
into a `sphere_in_face_cell` shader entry. Same for cpu_raycast.
Without this, deep face-content edits still hit the precision
floor of body-frame coords once you zoom past where body-frame
cells become sub-pixel.

### 5. Sphere outer-shell exit handoff

When a ray exits a body outward (`r >= cs_outer`), `sphere_in_cell`
returns miss and the caller advances Cartesian DDA past the body
cell. The "right" handoff resumes the parent-frame DDA from the
ray's actual exit point, not from the body cell's bounding-cube
boundary. Today this means: anything embedded right next to a
planet (a moon, debris) gets occluded by an invisible bounding
cube around the planet rather than the planet itself.

### 6. Face-seam transitions still implicit

`sphere_in_cell` re-runs `pick_face` every step rather than
walking through a 24-case face-transition table. Works but
doesn't unify with the ribbon-pop protocol — face-to-face
neighbor traversal isn't expressed in the same vocabulary as
cell-to-cell DDA stepping.

### 7. Hardcoded shader caps

- `MAX_STACK_DEPTH = 16` in `march_cartesian`
- `walk_face_subtree d <= 22u`

These match the demo planet's depth. Unified-driver doc lists
them as "should disappear once face-cell-as-frame lands."

## Test gaps

### 8. cpu_raycast_in_frame ribbon ascent: only structural unit tests

`cpu_raycast_in_frame_pop_finds_hit_in_ancestor` just verifies no
panic. There's no test that asserts the *correct cell* is hit
when the ray exits the deep frame and pops to find content in
an ancestor. End-to-end correctness here only validates via
visual / interactive testing.

### 9. preserve_path doesn't extend through face subtrees

`pack_tree_lod_preserving` only protects slots whose parent
NodeKind makes them LOD-eligible (Cartesian + Cartesian child).
A camera path that descends into a face subtree past the body
won't have face slots preserved — but face nodes don't get
flattened anyway, so it's not a bug *yet*. Becomes a bug if
Cartesian content embedded inside a face subtree starts being
flattened.

### 10. Highlight AABB transform uses *intended* frame, not effective

`update_highlight` pulls `frame` from `render_frame()` (the
intended deep frame). `upload_tree_lod` uses `effective_frame`
(what `build_ribbon` actually reached after pack flattening).
When these differ — typical case for the planet world where the
intended deep frame's empty siblings are flattened — the
highlight AABB is in the wrong frame's coords, so the outline
drifts or vanishes.

This is at least one of the bugs surfacing as
"highlight (frame-local): min=[0,0,0] max=[3,3,3]" — the AABB
ends up matching the entire frame box because the intended
frame's transform inflates a tiny world AABB to the wrong scale.

## What should land next session

1. **Fix bug #10**: pass the *effective* frame (the one the
   shader actually uses) to `aabb_world_to_frame` in
   `update_highlight`. Cheap one-line fix, blocks visible
   placement-target outline at deep zoom.
2. **Fix bug #2/#3**: per-ribbon-level AABB array passed to the
   shader; `fs_main` indexes by `result.frame_level`. Removes
   the "cross-pop hits skip highlight" stub.
3. **Verify cpu_raycast_in_frame end-to-end**: add a planet-world
   test that asserts the cell hit at deep zoom is the cell
   actually under the crosshair.
4. **Face-cell-as-frame** (#4): the big remaining piece.
5. **Sphere-exit handoff** (#5) when multi-body content is added.
6. **Cap removal** (#7) once #4 lands.

Items 1–3 are this-session-grade. Items 4–6 are next architectural
session.
