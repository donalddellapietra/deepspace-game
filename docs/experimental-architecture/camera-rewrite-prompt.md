# Prompt: rewrite the camera so nothing uses absolute coordinates

## Context

You're on branch `attempt1` in
`/Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/attempt1`,
forked from commit `20912b5` ("Revert Commit D: dynamic render frame").
That revert restored the bug: the camera is jittery at deep zoom.

The architectural target is in
`docs/experimental-architecture/anchor-refactor-decisions.md` — read it.
The TL;DR:

- Every world location is `(anchor: Path, offset: [f32; 3])`, offset in
  `[0, 1)³`. The `Position` type in `src/world/position.rs` already
  implements this.
- Rendering happens in a **render frame** that is an ancestor of the
  camera's anchor at `camera.depth - K` (default `K = 3`). The shader
  never sees coordinates outside `[0, 3)³`.
- Nothing in the game should hold or pass a root-frame XYZ. The only
  legitimate XYZ conversion is `Position::pos_in_ancestor_frame(d)`
  with `d` close to the camera's depth, at the render boundary.

## The bug, concretely

`App::render_root_depth()` in `src/app/edit_actions.rs:48` is hard-coded
to return `0`. That means:

- `camera_pos_in_render_frame()` returns the camera's XYZ in the **tree
  root** frame.
- `Camera::gpu_camera(fov, frame_depth)` is called with
  `frame_depth = 0`, so the uniform pushed to the GPU is also
  root-frame XYZ.
- At depth ≥ ~15 a single voxel is narrower than one f32 ulp at root
  scale (`3.0 / 2^23 ≈ 3.6e-7` vs `3^-15 ≈ 6.9e-8`), so the camera
  position snaps per frame → jitter.
- Reverted Commit D had fixed this by returning `camera.depth - K`;
  Commit D also reworked the shader tree walk and broke rendering in
  other ways. **Do not just re-apply Commit D.** Redo the refactor
  cleanly.

## Remaining absolute-XYZ leaks to eliminate

1. `src/app/edit_actions.rs:48` — `render_root_depth()` returns 0.
2. `src/player.rs:27` — gravity uses
   `camera.position.pos_in_ancestor_frame(0) - planet.center`, which
   is the same root-frame coordinate and loses precision at depth.
3. `src/app/event_loop.rs:159` — debug overlay `camera_pos` uses root
   frame (cosmetic but symptomatic).
4. `SphericalPlanet.center: [f32; 3]` (see `world/cubesphere.rs`) is a
   root-frame XYZ. Gravity math orbits this vector. It needs to become
   either a `Position` or be expressed in the render frame each tick.
5. Any raycast / edit call site that passes `camera_pos_in_render_frame`
   implicitly assumes the render frame is the tree root (see usages of
   the accessor in `edit_actions.rs`).

## What to do

Implement the refactor so that **no code path outside
`Position::pos_in_ancestor_frame` ever computes or stores a root-frame
XYZ**. Concretely:

1. **Dynamic render frame.** Make `render_root_depth()` return
   `camera.position.depth.saturating_sub(K)` with a tunable constant
   `K` (default 3). Walk `camera.position.path[..render_root_depth]`
   to resolve the `render_root_id` NodeId (what Commit D tried to do —
   do it correctly this time, dispatching on `NodeKind` so body/face
   subtrees don't corrupt the walk).
2. **GPU packer roots at the render frame.** `gpu::pack_tree_lod` must
   take the `render_root_id`, not `world.root`. Verify the shader still
   produces correct output when the render root is not the tree root.
3. **Gravity in render-frame-or-common-ancestor coordinates.** Convert
   `SphericalPlanet.center` into a `Position` (or equivalent
   path-based handle), and compute `camera → planet_center` by
   expressing both in their **common ancestor's** frame (or the render
   frame, whichever is shallower). Never `pos_in_ancestor_frame(0)`.
4. **Debug overlay.** Either drop `camera_pos` or render it as
   `depth + offset` (e.g. `"d=15, [0.42, 0.18, 0.77]"`).
5. **Spawn.** `Camera::at_spawn(xyz, depth, ...)` stays (one-shot
   worldgen convenience) but add a `Camera::at_position(pos, ...)`
   path-based constructor and migrate `App::new` to use it via
   worldgen returning a `Position`.
6. **Raycast / edit.** `edit::cpu_raycast` takes the camera XYZ in the
   render frame — verify this still works when the render frame is a
   shallow ancestor of the camera, not the tree root. The returned
   `Hit` must be a path from the render root; cursor break/place
   already operate on those paths.
7. **Zoom transitions.** When `zoom_in` / `zoom_out` changes
   `camera.depth`, the render frame shifts. The GPU camera uniform
   must be recomputed and pushed (already in `apply_zoom`) — just make
   sure the packer is also re-rooted and the tree is re-uploaded.

## Tests / verification

- Existing unit tests in `src/world/position.rs` must stay green.
- Launch the game via `./scripts/dev.sh` (per
  `docs/worktree-dev-workflow.md`) and verify:
  - No visible jitter at spawn depth (layer 10).
  - Zoom in to maximum depth: no jitter, camera still moves smoothly.
  - Gravity still pulls you to the planet at every zoom level.
  - Break / place still land on the cell under the crosshair at every
    zoom.
- No call site outside `Position::pos_in_ancestor_frame` and
  `Position::from_world_pos` should pass or return `[f32; 3]` world
  coordinates. Grep for `world_pos(` and `pos_in_ancestor_frame(0`;
  both should have zero hits in runtime paths (tests may keep them).

## Out of scope

- Beyond-render-frame rendering (solar-system view). The skybox handles
  anything outside the render frame.
- Multi-body gravity. One planet for now.
- Retiring `cs_*` uniforms / sphere DDA rewrite (that was Commit C,
  separate concern).

## Commit workflow

You're on a worktree. Per `CLAUDE.md`, commit and push each working
iteration; don't batch. Confirm each intermediate run is not broken
before committing.
