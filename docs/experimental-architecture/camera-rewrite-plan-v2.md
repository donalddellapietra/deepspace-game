# Camera rewrite plan v2 — what I'm actually going to do

Supersedes my earlier attempts in `camera-rewrite-prompt.md` and
`camera-rewrite-first-principles.md`. Those diagnosed the problem;
this is the concrete set of code changes.

## Scope for this pass

Fix the per-frame jitter for the demo-scene player standing on the
planet, without regressing the Cartesian-only path. Defer the full
face-subtree-as-render-root refactor (which requires NodeKind-aware
`Position` primitives and a new face-local shader path).

## What changes

### 1. Rust: dynamic render root (`src/app/edit_actions.rs`)

Replace the hard-coded `render_root_depth = 0` / `render_root_id =
world.root` with a walk that descends the camera's path:

- descends through `Cartesian` node children freely (no precision cost,
  shader's main DDA handles them);
- when it encounters a `CubedSphereBody` child, **descends into it**
  and stops — the body becomes the render root;
- never descends into a `CubedSphereFace` (shader cannot root there);
- also capped at `camera.depth - K` with `K = 3` for the Cartesian
  path.

For the demo scene this means: camera inside body → render root = body
node at depth 1. Camera in open Cartesian space → render root =
`camera.depth - 3`.

`camera_pos_in_render_frame()` continues to call
`Position::pos_in_ancestor_frame(render_root_depth)` — same API, now
exercised with non-zero depths.

### 2. Shader: root-kind dispatch (`assets/shaders/ray_march.wgsl`)

Today the shader's `march` enters the Cartesian DDA at `root_index`
and only checks NodeKind on CHILDREN during descent. When the root
itself is a `CubedSphereBody`, we need to skip the Cartesian walk and
go straight to sphere DDA with:

- `body_origin = vec3<f32>(0.0)`
- `body_extent = 3.0` (render frame spans `[0, 3)³`)
- `inner_r` / `outer_r` from `kinds[root_index]`

One small `if` at the top of `march`, before the Cartesian DDA
initialization. `march_sphere_body` already exists and is correct.

### 3. Keep the CPU raycast in root frame (for now)

`edit::cpu_raycast` does Cartesian-only DDA. When called with body as
root it would walk the body's face subtrees as Cartesian — wrong.

For this pass: leave `do_break`/`do_place`/`update_highlight` calling
`cpu_raycast` with `self.world.root` + camera pos in root frame
(`pos_in_ancestor_frame(0)`). That keeps cursor hits on Cartesian
blocks working. Cursor hits on the planet already go through
`try_cs_break` / `try_cs_place` which do their own cubed-sphere
raycast — unchanged.

Highlight AABB stays in root-frame coords. Today the highlight
uniform goes through the same pipeline as the render, so when the
shader is in body frame the AABB won't match. Two options:

(a) Disable Cartesian highlight while render root ≠ world.root — the
    player is inside the body, so Cartesian-block highlights aren't
    meaningful anyway.
(b) Convert the AABB from root frame to body frame before uploading.

(a) is simpler and correct for the demo. (b) is only needed when the
player is in a mixed region.

I'll start with (a) — a two-line gate in `update_highlight`.

### 4. Gravity stays on root-frame XYZ for now

`player.rs` reads `camera.position.pos_in_ancestor_frame(0) -
planet.center`. At depth 11 in f32, planet.center magnitude is O(1)
and camera XYZ magnitude is O(1) — the subtraction is well-conditioned
even with f32. Visible jitter is about rendering, not gravity. Defer
the Position-based planet.center refactor.

## Verification

1. `./scripts/dev.sh` in the worktree — build must succeed.
2. Spawn: the planet should render correctly (the shader now takes a
   different path — body-as-root — so this is the critical regression
   check).
3. Movement: flying around the planet should no longer show per-frame
   position jitter.
4. Zoom in (scroll up): jitter should stay absent up to a few more
   levels of depth than before.
5. Break / place: cursor hits on the planet continue to work (the cs
   path is unchanged).

## Not in this pass

- Face-subtree-as-render-root. Requires a new face-local shader DDA
  and NodeKind-aware `Position::pos_in_ancestor_frame`. Separate PR.
- `Position`-based `SphericalPlanet.center`. Separate PR.
- Deleting `world_pos()` / retiring the root-frame XYZ API. Several
  call sites still need root frame (gravity, CS raycast).
- Debug-overlay camera pos display. Cosmetic.

## Rollback

If the shader change regresses rendering, reverting
`assets/shaders/ray_march.wgsl` and `src/app/edit_actions.rs` to HEAD
restores the old behavior — both changes are localized.
