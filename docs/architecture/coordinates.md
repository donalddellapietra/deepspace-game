# Coordinates

Every position in the world is `(anchor, offset)`. The anchor is a
symbolic `Path` through the 27-ary tree — exact at any depth. The
offset is a tiny `[f32; 3]` kept in `[0, 1)³` of the anchor's cell.
**f32 never accumulates across cells.** As motion overflows a cell,
the anchor advances; as the player zooms, the anchor's depth changes.

Source of truth: `src/world/anchor.rs`.

## `Path`

```rust
pub struct Path {
    slots: [u8; MAX_DEPTH],   // MAX_DEPTH = 63
    depth: u8,
}
```

Each slot is `0..27`, the child index into a node's 3×3×3 grid.
Root is `depth == 0`. Two paths are "near" iff they share a long
common prefix; the length of the shared prefix defines their scale
of separation.

Equality is `memcmp`-fast on `slots[..depth]`.

## `WorldPos`

```rust
pub struct WorldPos {
    pub anchor: Path,
    pub offset: [f32; 3],     // invariant: each axis ∈ [0, 1)
}
```

**Hard invariant**: `offset[i] ∈ [0, 1)` at all times. The coordinate
primitives preserve this; callers should not poke the field directly.

Because the offset is bounded to one cell, f32 has its full ~7 digits
of precision available locally. No large-world accumulation.

## `WORLD_SIZE`

`WORLD_SIZE = 3.0` is a *frame-local* coordinate constant — one
node's three children span `[0, 3)` on each axis in the local frame
the renderer uses. It is **not** an absolute-world scale measurement.
See [../principles/no-absolute-coordinates.md](../principles/no-absolute-coordinates.md).

## Offset interpretation depends on NodeKind

The semantic axes of `offset` track the deepest node's kind:

- `Cartesian` / `CubedSphereBody`: offset is `(x, y, z)` in Cartesian
  local coords.
- `CubedSphereFace`: offset is `(u, v, r)` in the face's equal-angle
  + radial frame.

When the anchor descends into or exits a face subtree, the offset's
meaning changes. The coordinate primitives handle the rewrite; game-
level effects (camera up-vector, motion feel) are handled by
transition callbacks (see below).

## Primitives

### `add_local(delta)`

```rust
impl WorldPos {
    pub fn add_local(&mut self, delta: [f32; 3], lib: &NodeLibrary) -> Transition;
}
```

The only movement entry point. Adds `delta` to `offset`; if any axis
overflows `[0, 1)`, steps to the neighboring cell at the current
depth (bubbling up the path if needed). Dispatches on the deepest
node's `NodeKind` (Cartesian vs. face seam). Returns a `Transition`
describing any crossing.

### `zoom_in() / zoom_out()`

```rust
impl WorldPos {
    pub fn zoom_in(&mut self);
    pub fn zoom_out(&mut self);
}
```

Both are O(1), no tree reads.

- **Zoom in** pushes a new slot equal to `floor(offset * 3)`, rescales
  `offset[i] = fract(offset[i] * 3)`.
- **Zoom out** pops the last slot, rescales
  `offset[i] = (offset[i] + popped_slot_coord[i]) / 3`. Clamps at root.

Zoom does not move the player — it re-expresses the same position at
a different granularity.

## Transitions

```rust
pub enum Transition {
    None,
    SphereEntry { body_path: Path },
    SphereExit  { body_path: Path },
    FaceEntry   { face: Face },
    FaceExit    { face: Face },
    CubeSeam    { from_face: Face, to_face: Face },
}
```

Returned from `add_local` and `zoom_in`/`zoom_out`. Game code reacts:
sphere entry rotates the camera's "up" to the radial direction; cube
seams re-express yaw/pitch in the new face's axes; and so on. The
coordinate primitives never implement the *game* effects — only the
*math* rewrite.

## Camera

```rust
pub struct Camera {
    pub position: WorldPos,
    pub smoothed_up: [f32; 3],   // in the current anchor's frame
    pub yaw: f32,
    pub pitch: f32,
}
```

The camera has no world-space `[f32; 3]` position — anywhere. Its
location is always `position.anchor + position.offset`. `smoothed_up`,
`yaw`, and `pitch` are interpreted in the current anchor's frame, and
transition handlers re-express them when frames change.

## Velocity

Velocity is in **offset-units per second at the current anchor
depth**. One unit = one cell width at that depth. Zoom in ⇒ same
number means smaller world motion, matching the intuition "when I'm
small, moving at the same speed gets me through less world."

## What this replaces

This model supersedes the old world-XYZ plumbing:

- No `Camera.pos: [f32; 3]` — gone.
- No `SphericalPlanet` / `cs_planet` — the sphere is just a node with
  `NodeKind::CubedSphereBody` in the tree.
- No separate `edit_depth` / `visual_depth` / `cs_edit_depth` — edit
  depth is the depth the raycast resolves to. Zoom in, hit deeper
  cells; zoom out, hit shallower chunks.
- No `to_world_xyz` / `from_world_xyz` helpers. See
  [../principles/no-absolute-coordinates.md](../principles/no-absolute-coordinates.md).
