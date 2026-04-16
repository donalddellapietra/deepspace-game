# Anchor-based coordinates refactor: design decisions

This document supersedes `refactor-decisions.md`. It describes the
migration to a **path-anchored** coordinate system, which is
structurally close to the earlier "paths" proposal but phrased in
terms of the classical anchor-and-offset pattern.

**Core principle:** every position in the world is `(anchor, offset)`.
- `anchor` is a `Path` through the 27-ary tree — exact, symbolic, no
  f32 precision loss at any depth.
- `offset` is a small `[f32; 3]` in the anchor cell's local `[0, 1)³`
  frame.

Because the offset is bounded to `[0, 1)`, f32 gives plenty of
precision (~7 digits) within any cell at any anchor depth. As the
player moves, the offset overflows and we re-anchor to the next cell.
As they zoom, the anchor's depth changes.

f32 never accumulates across cells. The GPU never sees a huge world
coordinate. Rendering always happens in a frame small enough for f32.

## 1. Core types

### 1a. `Path`

```rust
pub struct Path {
    slots: [u8; MAX_DEPTH],   // MAX_DEPTH = 63, per tree.rs
    depth: u8,                 // how many slot entries are live
}
```

- Root is at `path.depth == 0`; each `push(slot)` descends one level.
- A path equality compare is `memcmp`-fast: compare depths and the
  `slots[..depth]` prefix.
- Two paths are "near" iff they share a long common prefix. The
  length of the shared prefix defines the scale of their separation
  — same semantics as `coordinates.md`.

### 1b. `WorldPos`

```rust
pub struct WorldPos {
    pub anchor: Path,
    pub offset: [f32; 3],  // always in [0, 1)³ by invariant
}
```

**One hard invariant:** `offset[i] ∈ [0, 1)` for each axis, normalized
after every mutation. This is maintained by the coordinate
primitives; callers don't violate it directly.

### 1c. Offset frame is *node-kind-relative*

The axes of the `offset` are interpreted based on the anchor's
deepest-node kind:

- `NodeKind::Cartesian` or `CubedSphereBody`: offset is `(x, y, z)` in
  the cell's Cartesian local frame.
- `NodeKind::CubedSphereFace`: offset is `(u, v, r)` in the face's
  equal-angle + radial local frame.

This means **the offset's semantic axes change when the anchor
descends into or exits a face subtree**. That transition is handled
by the coordinate primitives themselves (the math is deterministic);
the player-visible semantic changes (up-vector rotation, etc.) are
handled by separate game-level transition handlers (section 4).

## 2. Primitives

### 2a. `add_local(delta)` — the only movement entry point

```rust
impl WorldPos {
    pub fn add_local(&mut self, delta: [f32; 3], lib: &NodeLibrary);
}
```

Adds `delta` to `offset`. If any axis overflows `[0, 1)`, steps to
the neighboring cell at the current depth (or bubbles up the path if
the step crosses higher-level boundaries). Restores the invariant
before returning.

Dispatches internally on the deepest node's `NodeKind`:

- `Cartesian` / `CubedSphereBody`: vanilla Cartesian neighbor step.
- `CubedSphereFace`: face-aware step (u/v boundaries may cross cube
  seams; r boundaries may radially exit the sphere).

**After the move**, if the `NodeKind` at the deepest anchor level
changed, the coordinate primitive records the transition so the
caller can run the game-level transition handler (section 4).

### 2b. `zoom_in()` / `zoom_out()`

```rust
impl WorldPos {
    pub fn zoom_in(&mut self);   // push: anchor descends into slot under offset
    pub fn zoom_out(&mut self);  // pop: anchor ascends; offset rescales
}
```

- **Zoom in** pushes a new path slot equal to `floor(offset * 3)`.
  Offset is rescaled so that `offset[i] = fract(offset[i] * 3)`. The
  position is unchanged; we're just expressing it in a finer cell.
- **Zoom out** pops the last path slot. Offset is rescaled so that
  `offset[i] = (offset[i] + popped_slot_coord[i]) / 3`. Clamps at
  root (no virtual ancestors).

Both are O(1) with no tree reads.

**If the zoom descent lands in a `CubedSphereFace` child** (or any
non-Cartesian child), the offset's semantic axes change. Handled the
same way as movement-induced transitions.

### 2c. `Path::step_neighbor` (low-level primitive)

```rust
impl Path {
    fn step_neighbor(&mut self, axis: i8, direction: i8, lib: &NodeLibrary);
}
```

Walks one cell in the indicated axis at the current depth. On
overflow at one level, bubbles up to the parent and retries. On
crossing into a different-axis-kind subtree (e.g., a face seam),
rewrites the slot and subsequent axes per the appropriate transition
table.

Used internally by `add_local` and by transition handlers. Not
usually called by game code directly.

## 3. `NodeKind`

```rust
pub enum NodeKind {
    Cartesian,
    CubedSphereBody { inner_r: f32, outer_r: f32 },
    CubedSphereFace { face: Face },
}
```

- All nodes have exactly 27 children (the load-bearing invariant —
  see `refactor-decisions.md` §1a).
- `NodeKind` is part of the content-addressed hash, so two identical
  subtrees with different `NodeKind`s don't dedup into one.
- `CubedSphereBody`: 6 children at face-center slots are the face
  subtrees; 20 corner/edge slots are `Empty`; center slot is the
  interior filler.
- `CubedSphereFace` children use the same 27-child layout, but the
  axes are `(u_slot, v_slot, r_slot)`.

`inner_r` and `outer_r` are in the body cell's local `[0, 1)` frame,
so `0 < inner_r < outer_r ≤ 0.5`. This keeps a sphere strictly
inside one cell of its Cartesian parent; bigger spheres in world
terms live at shallower anchors.

## 4. Transitions

Transitions are the NAMED events that fire when the anchor crosses a
coordinate-meaning boundary. The coordinate math handling the path
rewrite is **automatic** (in the primitives). The game-visible
effects (camera up-vector rotation, motion feel, UI hints) are
**explicit** game-level handlers.

### 4a. Transition taxonomy

1. **Sphere entry**: anchor enters a `CubedSphereBody` subtree for
   the first time during a zoom or step. Game sets "up" to the
   radial direction from body center, blended smoothly.
2. **Sphere exit**: anchor leaves a `CubedSphereBody` subtree. "Up"
   returns to world frame, blended smoothly.
3. **Face entry** (body → face): the anchor pushes into a face
   subtree. Offset's axes change from Cartesian to `(u, v, r)`.
4. **Face exit** (face → body outward via r=1 crossing): coordinate
   primitive pops the face node; effectively becomes sphere exit if
   that was the outermost containing body.
5. **Cube seam** (face → face across u or v boundary): 24-case axis
   remapping per the cubed-sphere face adjacency table.

### 4b. Where each piece of transition logic lives

| Piece | Layer | Notes |
|---|---|---|
| Path rewrite on overflow | `Path::step_neighbor` | Coordinate primitive |
| Offset axis reinterpretation | `WorldPos::add_local` | Coordinate primitive |
| Cube seam axis remapping | `Path::step_neighbor` + transition table | Coordinate primitive |
| Camera up-vector rotation | Game-level handler | Called on transition event |
| Orientation yaw/pitch re-expression | Game-level handler | Called on transition event |
| UI hints (e.g. "entering orbit") | Game-level handler | Optional |

### 4c. Transition event surface

```rust
#[derive(Debug, Clone, Copy)]
pub enum Transition {
    None,
    SphereEntry { body_path: Path },
    SphereExit  { body_path: Path },
    FaceEntry   { face: Face },
    FaceExit    { face: Face },
    CubeSeam    { from_face: Face, to_face: Face },
}
```

`add_local` and `zoom_in`/`zoom_out` return a `Transition`. Game
code checks it and dispatches:

```rust
let t = camera.position.add_local(vel * dt, &lib);
match t {
    Transition::SphereEntry { body_path } => {
        // Compute body center direction, set target_up, etc.
    }
    // ...
    _ => {}
}
```

This keeps coordinate math self-contained while letting game code
react to the meaningful events.

## 5. Camera

```rust
pub struct Camera {
    pub position: WorldPos,
    pub smoothed_up: [f32; 3],   // direction in the CURRENT anchor's frame
    pub yaw: f32,
    pub pitch: f32,
}
```

- The camera's world location is its `position.anchor + offset`.
- `smoothed_up`, `yaw`, `pitch` are all interpreted in the CURRENT
  anchor's local frame.
- On transitions, `smoothed_up` is updated by the game handler to
  reflect the new frame's meaning.
- The camera has no world-space `[f32; 3]` position. Anywhere.

## 6. Rendering

### 6a. The render frame

Rendering always happens in one f32-safe local frame, called the
**render frame**. The render frame is a `Path` — an ancestor of the
camera's anchor — chosen so the visible frustum fits inside `[0, 1)³`
of the render frame's cell.

Default policy: render frame = `camera.position.anchor` truncated
to `(depth - K)` for some fixed K (default `K = 3`). This gives you
"your cell plus a few layers up" as the rendered volume. Tunable.

### 6b. Camera position in the render frame

Computed each frame by composing offsets up the path:

```rust
fn camera_in_render_frame(camera: &WorldPos, render_depth: u8) -> [f32; 3] {
    let mut result = camera.offset;
    let mut scale = 1.0;
    for slot in camera.anchor.slots[render_depth..camera.anchor.depth].iter().rev() {
        let (sx, sy, sz) = slot_coords(*slot);
        result[0] = (result[0] + sx as f32 * 1.0) / 3.0 * 3.0;
        // (similar for y, z)
        scale /= 3.0;
    }
    // Convert to the render frame's [0, 1)³:
    // walk camera.anchor from render_depth down, composing the offset
    // at each level.
    result
}
```

The result is in `[0, 1)³` of the render frame's cell if the camera
is inside it. Always f32-safe.

### 6c. Beyond the render frame

Deferred. For v1, the shader renders a fixed dark-space skybox
outside the render frame. Orbital / solar-system views are
out-of-scope for the first pass. Later we can pack multiple render
frames (your anchor + its ancestor + its grand-ancestor) and
composite, but not now.

### 6d. Sphere body rendering

Same story as the path plan: the shader encounters a
`CubedSphereBody` child during its walk, dispatches into a
cubed-sphere DDA over its 6 face children, composites by nearest t.

The body's `inner_r`/`outer_r` are read from the `NodeKind` at render
time; the body is self-contained in its own tree node. No parallel
uniforms.

## 7. Editing

### 7a. Cell identity IS the anchor path

To break a block: find the anchor path of the cell you hit via the
cursor raycast, and call `set_cell_at_depth(lib, path, Empty)`.

To place a block: same, with `Block(bt)`.

The cursor raycast returns an **anchor path**, not world coordinates.
It walks the render frame's tree from ray origin along ray direction,
tracking the current cell's slot path. First solid cell hit: that's
the hit path.

### 7b. No separate "edit depth" concept

The depth at which you edit IS the depth the raycast resolved to.
Zoom has no separate effect. If you're zoomed way out, you hit
shallow cells — big chunks. If you're zoomed in, you hit deep cells
— tiny ones. The cell's visual size on your screen tells you the
impact of an edit; that's the UX signal.

No `cs_edit_depth`, no `visual_depth`, no `edit_depth` as separate
engine state.

## 8. Worldgen

### 8a. Planets are tree nodes, not external handles

`spherical_worldgen::build` inserts a `NodeKind::CubedSphereBody`
node at a specific path in the library and returns that path. The
body node owns its face children through normal tree ownership.

`WorldState` has no `cs_planet: Option<SphericalPlanet>` field. If
you want to know "where the demo planet is" you ask worldgen for the
path it returned and look it up.

### 8b. Multi-body is free

Want more planets? `build` a second body at a second path. Want a
universe? Loop.

### 8c. Spawn position

The starting `WorldPos` for the player is constructed from worldgen:
"spawn at path X with offset Y." `App::new` derives this from
wherever worldgen places the demo scene.

## 9. Velocity

Velocity is in **offset-units per second at the current anchor
depth**. One unit = one cell width at the anchor's depth.

- Zoom in: anchor deeper, world-units-per-cell smaller, same velocity
  number means smaller world motion per second. Matches player
  intuition: "when I'm small, moving at the same speed gets me
  through less world."

This is what the current code does; the new model preserves it.

## 10. What goes away

- `Camera.pos: [f32; 3]` — replaced by `Camera.position: WorldPos`.
- `SphericalPlanet` struct — replaced by `CubedSphereBody` node kind
  + the face subtrees being ordinary tree children.
- `WorldState::cs_planet` field — no more singular-planet assumption.
- `App::zoom_level: i32` — replaced by `camera.position.anchor.depth`.
- `edit_depth`, `visual_depth`, `cs_edit_depth` helpers — replaced by
  "whatever depth the raycast resolved to."
- `cs_planet`, `cs_params`, `cs_highlight`, `cs_face_roots_a/b`,
  `cs_blocks` uniforms — all deleted. Sphere content flows through
  `NodeKind` in the tree buffer.
- `Renderer::set_cubed_sphere_planet`, `set_face_roots`,
  `set_cubed_sphere_highlight` — deleted.
- `pack_tree_lod_multi` — collapses back to `pack_tree_lod` with a
  single root (the render frame).
- `cs_edit_depth`, `cs_cursor_hit`, `try_cs_break`, `try_cs_place`
  free functions — collapse into normal editing flow that operates on
  tree paths (no planet-vs-tree distinction).
- `generate_spherical_planet` signature — changed: takes insertion
  path, returns inserted path.

## 11. Execution order

1. Add `Path` + `WorldPos` + `add_local` + `zoom_in`/`zoom_out` +
   `Transition` enum, with tests. Zero behavior change — new types
   exist alongside old XYZ types.
2. Add `NodeKind` to `Node`; default all existing nodes to
   `Cartesian`. Dedup hash includes `NodeKind`. Zero behavior change.
3. Migrate `Camera` to hold `position: WorldPos`. Add a
   `world_pos_f32()` shim returning `[f32; 3]` for old callers; keep
   them working on the shim.
4. Migrate `player::update` to mutate `WorldPos` directly via
   `add_local`. Gravity math uses `world_pos_f32()` shim temporarily.
5. Migrate `pack_tree_lod` to root at the render frame (the
   camera's ancestor at `depth - K`). Shader walks that subtree in
   its local frame.
6. Delete `world_pos_f32()` shim; `edit.rs` CPU raycast returns a
   `Path` instead of XYZ. Break / place / highlight all work on
   paths.
7. Replace `zoom_level` + scroll handling with `zoom_in`/`zoom_out`.
   Camera anchor reanchoring on zoom is handled by the primitive.
8. Wire up sphere transitions: `NodeKind::CubedSphereBody` +
   `CubedSphereFace` are introduced; `add_local` dispatches on
   `NodeKind`; transition handlers for entry/exit/seam are written.
9. Retire `cs_*` uniforms from the shader; sphere bodies render via
   a new `NodeKind` branch in the shader's tree walk.
10. Remove legacy XYZ-position plumbing entirely; rename / retire
    legacy identifiers.

Each step leaves tests green.

## 12. Explicitly deferred

- **Beyond-render-frame rendering**: solar-system / orbital views
  that require packing multiple ancestors. v1 shows a dark skybox
  outside the render frame.
- **Collision at body boundaries with sub-cell precision**: v1
  accepts cell-aligned boundaries (no rounded edges, no "scraping
  the sphere's surface").
- **Networked / multi-camera support**: `WorldPos` serializes as
  `[u8; 64] + [f32; 3] + u8`, but multi-camera coherence is
  out-of-scope.
- **`trait Axes`-style generalization**: the enum is fine for the
  two kinds we have. Extend when a third kind actually appears.

## 13. New decisions surfaced while writing

Three decisions came up that weren't previously settled. I'm
proposing defaults below; they don't require a conversation unless
you disagree.

1. **Render-frame ancestor depth (`K`)**: default 3. Tunable via a
   constant. Means "show 27³ = ~20k cells of my anchor depth around
   me." If K feels wrong in practice, we adjust without architectural
   changes.

2. **Movement-induced anchor-depth changes**: when `add_local`
   bubbles up past a NodeKind boundary (e.g., radially exiting a
   sphere face), the anchor's depth decreases. The player is now
   "zoomed out" by one level. This is correct — they entered a
   different coordinate scale. They can zoom back in to explore at
   the old depth if they want.

3. **Zoom buttons are the ONLY way to voluntarily change anchor
   depth**. Movement-induced depth changes are allowed (as above)
   but they're side effects of crossing geometric boundaries, not
   intentional zoom. This keeps the mental model clean: the player
   intentionally zooms with scroll wheel; the world may adjust their
   depth as they cross boundaries.
