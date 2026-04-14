# Path-based coordinates refactor: design decisions

This document captures decisions made while scoping the migration from
the current absolute-XYZ architecture to the path-based coordinate
model described in `coordinates.md`. It is the authoritative reference
for *what* we're building; `coordinates.md` describes the high-level
model. When the two disagree, this file wins — it's where we've
resolved the under-specified bits.

## Core principle

**No absolute coordinates anywhere in the engine.** Every position is
expressed as a `Position { path, offset }`. The `path` is a sequence
of 27-ary slot indices; the `offset` is the sub-slot fractional
position inside the deepest resolved node, always in `[0, 1)³`. The
only f32s for "where something is" are offsets confined to a single
node's local frame. No f32 ever accumulates across cell boundaries.

This is the only hard constraint. Everything else in this document is
the concrete shape of how we honor it.

## 1. Node representation

### 1a. Every node still has exactly 27 children

The `[Child; 27]` invariant is load-bearing — every tree-walking
primitive in the codebase (`slot_index`, dedup hashing, GPU packing
with `GPU_NODE_SIZE = 27`, the shader's DDA, `pack_tree_lod_multi`)
assumes it. We keep it.

Non-Cartesian node kinds (cubed-sphere body, cubed-sphere face) still
use 27 child slots. The *meaning* of those slots varies by node kind,
but their count doesn't.

### 1b. `NodeKind` enum on every node

```rust
pub enum NodeKind {
    /// Standard Cartesian branching. Children are indexed by
    /// `(x_slot, y_slot, z_slot)` via `slot_index(x, y, z)`.
    Cartesian,

    /// A cubed-sphere body: the cell occupied by a planet/moon/star.
    /// Of the 27 children, 6 specific slots hold the cube-face
    /// subtrees; the remaining 21 slots are either Empty (outside
    /// the sphere's volume at this zoom) or a uniform interior
    /// filler (inside the sphere's inner shell). `inner_r` and
    /// `outer_r` are the radial shell extent, expressed in the
    /// containing cell's local `[0, 1)` frame (so `0 < inner_r <
    /// outer_r ≤ 0.5`, with 0.5 being the cell wall).
    CubedSphereBody { inner_r: f32, outer_r: f32 },

    /// One cube face's subtree. Children are indexed by
    /// `(u_slot, v_slot, r_slot)` using the same `slot_index(u, v, r)`
    /// mapping. `face` identifies which of the 6 cube faces this
    /// node represents — used by `step_neighbor` when a u/v step
    /// crosses the cube seam into a sibling face.
    CubedSphereFace { face: Face },
}
```

Rationale: only two non-standard kinds exist today (sphere body and
sphere face). An enum is honest about the cases we have; a trait would
be premature generalization. We extend the enum if a new axis type
appears (cylindrical rings, icosahedral faces, etc.).

### 1c. Face-to-slot mapping for `CubedSphereBody`

The 6 face subtrees live in the face-center slots of the body's 27
grid:

```
slot_index(0, 1, 1) → -X face
slot_index(2, 1, 1) → +X face
slot_index(1, 0, 1) → -Y face
slot_index(1, 2, 1) → +Y face
slot_index(1, 1, 0) → -Z face
slot_index(1, 1, 2) → +Z face
```

The geometric semantics of the 21 non-face slots:

- **Center slot `(1, 1, 1)`**: the sphere's interior — a uniform
  filler chain of `Block(core_block)` for the solid core, dedup'd.
- **All other 20 slots** (corners, edges of the parent cube): `Empty`.
  Conceptually "outside the sphere at this zoom" — the sphere's faces
  are round and don't fill the corners of its containing cell.

This mapping is purely notational; `step_neighbor` interprets the 6
face slots via the face-transition table, not via their XYZ positions
in the 27-grid. No code should ever compute "which direction is slot
(0,1,1)?" on a CubedSphereBody node — slot identity is opaque, the
NodeKind carries the semantics.

### 1d. Shell radii live on the `CubedSphereBody` node kind

`inner_r` and `outer_r` are fields of `NodeKind::CubedSphereBody`.
They're NOT stored on the face children, NOT on the planet's global
state, NOT on a separate "planet registry."

Consequences:

- Two spheres with identical radii hash to the same `NodeKind`, and
  the content-addressed library deduplicates them if their children
  also match. A universe with many identical moons shares one node.
- Two spheres with different radii are different nodes, correctly —
  the shell extent is part of the node's identity.
- Radii are in the **containing cell's local `[0, 1)` frame**, not
  world units. The absolute size of a sphere is determined by where
  in the tree its body node lives. A sphere at depth 3 is much bigger
  than a sphere at depth 15 with the same `outer_r`.

`Face` and per-face SDF params (noise seed, surface/core block types)
are NOT per-node — they're worldgen-time inputs used to populate the
face subtrees, then baked into child content.

### 1e. `CubedSphereFace` children are `(u_slot, v_slot, r_slot)`

For a face node, `slot_index(u, v, r)` reuses the standard Cartesian
mapping but the axes are face-local equal-angle `u`, face-local
equal-angle `v`, and radial `r`. The shader and `step_neighbor` know
this because the node's `NodeKind::CubedSphereFace` tag tells them.

Rendering per-cell: a child at slot `(u_slot, v_slot, r_slot)` inside
a face node at depth `d` inside a body at depth `d-1` has world
geometry computed via the existing cubed-sphere math
(`face_uv_to_dir`, `coord_to_world`) interpreted in the body's local
frame.

### 1f. Radial-exit rule for `CubedSphereFace`

Stepping past `r_slot = 2` outward on a face node exits the sphere's
outer shell. The path bubbles up: pop the face node, pop the body
node, and continue in the body node's **parent** (which is Cartesian),
stepping laterally from the body's cell into the appropriate neighbor
in the Cartesian parent's grid.

Stepping below `r_slot = 0` inward goes into the sphere's interior
filler (the center slot of the body node). At zoom levels where the
interior is a uniform subtree, descent resolves immediately.

### 1g. Face-seam rule for `CubedSphereFace`

Stepping past `u_slot = 0` or `u_slot = 2` (similarly for `v_slot`)
on a face node crosses a cube-face seam into a sibling face. The
24-case face-transition table (6 faces × 4 edges) is consulted and
the path is rewritten:

1. Pop the current face node from the path.
2. Push the sibling face slot (one of the 6 face-center slots on the
   body node) — this is the new face.
3. The `(u_slot, v_slot)` axes may swap or negate in the new face's
   frame; the face-transition table handles this mapping.
4. `r_slot` is preserved across the seam.

The transition table is the same math already present in
`cubesphere::face_uv_to_dir` and neighbors; it just needs to be
exposed as a pure `step_neighbor`-level function on top.

## 2. Position type

### 2a. `MAX_DEPTH = 63`, `Position.path = [u8; 63]`

Matches the value already in `tree.rs`. 3^63 ≈ 1.7 × 10³⁰ cells per
axis at the deepest layer. Fixed-size array, no allocation, cheap to
copy/compare.

`Position.depth: u8` says how many slots in `path` are populated.
Unused slots beyond `depth` are `0` (irrelevant — never read).

`Position.offset: [f32; 3]` is the sub-slot fractional position in
`[0, 1)³` within the deepest resolved node. Each axis independently.

Note: for `CubedSphereFace` nodes, the "3 axes" of offset are
`(u, v, r)` in the face's equal-angle+radial frame, not `(x, y, z)`.
Same struct field, different interpretation based on the deepest
node's kind.

### 2b. Zoom is a `Position.depth` change

- Zoom in: push a new slot at `depth`, entering one 27-child grid.
  `offset` is multiplied by 3 (same point, now expressed in the
  child's frame) and overflows into the chosen slot index.
- Zoom out: pop the last slot, `offset` is divided by 3 and
  translated by the popped slot's position within `[0, 1)`.
- **Zoom out past root (depth 0)**: clamp. No-op. We never fabricate
  virtual ancestor nodes.

### 2c. `step_neighbor` dispatches on the deepest node's `NodeKind`

```
fn step_neighbor(position: &mut Position, library: &NodeLibrary, axis: i8) {
    let parent_kind = library.lookup(position.parent_path()).kind;
    match parent_kind {
        Cartesian => cartesian_step(position, axis),
        CubedSphereBody { .. } => sphere_body_step(position, axis),
        CubedSphereFace { face } => face_step(position, face, axis),
    }
}
```

Each branch handles its own overflow/bubble-up semantics. Cartesian
is trivial (±1 on the relevant slot, carry to parent on overflow).
Face and body branches consult the rules above.

## 3. Rendering

### 3a. Rendering frame = smallest ancestor containing view frustum

Per frame, the CPU walks up from the camera's deepest resolved node
until the ancestor's cell (in its own local `[0, 3)` frame) fully
contains the camera's view frustum. That ancestor is the "render
root." The shader walks the subtree rooted at the render root.

The camera's ray origin is its offset in the render root's frame
(derived by composing the offsets up the path, with scale-by-3 at
each level). Always small f32 values, always precision-safe.

Recomputed every frame. O(depth) work on the CPU (≤63 steps). Cheap.

### 3b. Sibling subtrees render via the existing multi-root mechanism

A `CubedSphereBody` or any other "foreign axes" subtree is already a
*node* in the render root's subtree — it gets picked up by the BFS
during packing. The shader walks it as a regular 27-child node,
branching on `NodeKind` when interpreting the 6 face children.

No parallel GPU buffers for sphere content. The current
`cs_face_roots` uniform goes away. All tree content is in the single
packed tree buffer.

### 3c. Cubed-sphere DDA stays in the shader

The existing `ray_march.wgsl` cubed-sphere DDA (ray-sphere outer
intersect → local `(u_ea, v_ea, r_n)` → tree walk → exit-plane math)
survives. What changes:

- It's no longer triggered by a separate `cs_planet` uniform.
- It's triggered when the DDA enters a `NodeKind::CubedSphereBody`
  child during the tree walk.
- The body's `inner_r`/`outer_r` are read from the node's kind, not
  from a uniform.

## 4. Velocity and movement

### 4a. Velocity is in "cells per second at the current resolved depth"

Zoom-invariant feel. A step takes you one cell at your current zoom,
regardless of whether your current cell is 1m or 1km across.

Internally: `velocity: [f32; 3]` is the per-second delta to add to
`Position.offset`. When `offset` overflows `[0, 1)`, `step_neighbor`
handles the carry.

### 4b. Movement is a `Position.add_offset(velocity * dt)` operation

Three steps:

1. `offset += velocity * dt` componentwise.
2. For each axis where `offset` overflows `[0, 1)`, call
   `step_neighbor` — which may bubble up multiple levels and re-enter.
3. Reclamp `offset` to `[0, 1)` in the resulting frame.

O(1) in the common case (no overflow). O(depth) when crossing a
high-level boundary. Identical semantics across Cartesian, body, and
face node kinds (the `step_neighbor` dispatch handles the differences).

## 5. Editing

### 5a. Edit target = the camera's deepest resolved node

The cursor ray traces from the render root. The hit cell is at some
path. Break/place operate on that cell at that path — no separate
"edit depth" or "highlight depth" knob.

The depth of the edit is implicit: it's the depth at which the ray
found a terminal (`Block` or `Empty`) during the tree walk. That
depth varies pixel-to-pixel; the first solid hit wins.

### 5b. No absolute `cs_edit_depth` cap

The f32 precision issue goes away because rendering happens in local
frames. The cursor's cell is tracked by its *path* (list of slot
indices), not by an absolute `(iu, iv, ir)` triple. Path comparison
is bit-exact at any depth.

### 5c. Editing at any path depth works

`set_cell_at_depth` already handles arbitrary depths via path
walking. Under the new model the "depth" input is just "how many
slots past the current path" to descend. The existing
`rebuild_with_edit` pattern (descending, expanding uniform
terminals, rebuilding parents) generalizes to any `NodeKind` —
`rebuild_with_edit` doesn't care about axis meaning, it just
re-wraps children.

## 6. Camera

### 6a. Camera state

```rust
pub struct Camera {
    pub position: Position,
    pub smoothed_up: [f32; 3],   // direction, not position; no change
    pub yaw: f32,
    pub pitch: f32,
}
```

`smoothed_up` is a direction in the camera's local frame. When the
camera's path changes (zoom or step), directions are preserved —
directions don't need re-anchoring because they're already in
whatever local frame the current render needs.

### 6b. Orientation tracking across sphere-body entry

Special case: when the camera descends into a `CubedSphereBody`
node, it enters a frame where the "natural up" is radial (outward
from the body's center), not world-Y. `smoothed_up` should rotate
accordingly, blended over a few frames (same mechanism as today's
`update_up`). The trigger is "did my path's deepest `CubedSphereBody`
ancestor change?" rather than "am I within an influence radius?"

## 7. Planet constraints

### 7a. A sphere body fits in one cell of its parent

`outer_r ≤ 0.5` in the containing cell's local `[0, 1)` frame. The
sphere can't span multiple cells of its parent. If a worldgen wants
a bigger sphere, it lives at a shallower ancestor (where cells are
bigger in world terms).

### 7b. Planets are inserted into the tree by worldgen

`spherical_worldgen::build` no longer returns a `SphericalPlanet`
handle held externally. It inserts a `NodeKind::CubedSphereBody`
node at a specific path in the library and returns that path. The
tree-as-single-source-of-truth holds all planet data.

`WorldState` no longer has a `cs_planet: Option<SphericalPlanet>`
field. The space tree contains the planet.

## 8. Backward-compatibility / what goes away

- `App::cs_planet: Option<SphericalPlanet>` — deleted. Planets are
  tree nodes.
- `SphericalPlanet` struct — deleted. Its center/radii become node
  kind data; its face_roots become children in the body node.
- `generate_spherical_planet` — signature changes: takes the path to
  insert at, mutates the library, doesn't return a handle.
- `Renderer::set_cubed_sphere_planet`, `set_face_roots` — deleted.
- Shader uniforms `cs_planet`, `cs_params`, `cs_highlight`,
  `cs_face_roots_a/b` — deleted. Per-node info flows via `NodeKind`.
- `cs_edit_depth`, `cs_cursor_hit`, `try_cs_break`, `try_cs_place` —
  collapse into the single editing path; no sphere-specific flow.
- `zoom_level: i32` on `App` — replaced by `camera.position.depth`
  (plus whatever zoom-anchor state we need for UX).
- `edit_depth()`, `visual_depth()` — derived from `position.depth`.

## 9. Execution order (for reference)

This matches the scoping report's recommended order:

1. Add `Position` + `step_neighbor` + tests (no behavior change yet).
2. Add `NodeKind` enum to `Node`; default all existing nodes to
   `Cartesian`; no behavior change.
3. Migrate `Camera.pos` → `Camera.position`, add `world_pos()`
   compatibility shim temporarily.
4. Migrate `player::update` to path-based offset integration; keep
   gravity math using `world_pos()` shim.
5. Migrate `pack_tree_lod_multi` + shader to use a camera-enclosing
   ancestor as render root.
6. Delete `world_pos()` shim; fully path-based raycast, break, place,
   highlight.
7. Introduce `NavStack` / zoom-anchor state; rewrite
   `handle_scroll_zoom` in path terms.
8. Implement cross-subtree `step_neighbor` for cube-face seams
   (7g) and radial exits (7f).
9. Retire `cs_planet` uniforms from the shader; sphere content flows
   entirely through `NodeKind`.
10. Delete legacy XYZ-position plumbing.

## 10. Explicitly deferred

- Multi-body support (`Vec<SphericalBody>`). The tree naturally
  supports any number of sphere bodies (they're just nodes at any
  paths), so this is "free" once the migration is done — no extra
  engine work, only worldgen choosing where to place them.
- `Trait Axes` abstraction. Extend the `NodeKind` enum when a 3rd
  axis type appears; don't generalize preemptively.
- Networked / serialized positions. Paths serialize as-is (`[u8; 63]`
  + `u8` + `[f32; 3]`). No conversion layer planned.
- Big-int or f64 coordinate fallback. Explicitly not the architecture.
  All precision comes from path locality + per-node f32 offsets.
