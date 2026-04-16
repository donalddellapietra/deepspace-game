# Cubed sphere

Planets live *inside* the voxel tree as a `NodeKind::CubedSphereBody`
node. There is no separate `SphericalPlanet` struct, no parallel
raycaster, no dedicated GPU buffers — the same tree walk handles both
Cartesian space and planetary bodies.

Source of truth:
- `src/world/cubesphere.rs` — geometry, `Face`, `insert_spherical_body`.
- `src/world/cubesphere_local.rs` — frame-local sphere math for the
  shader + CPU raycast.
- `src/world/raycast/sphere.rs` — CPU DDA inside a body.
- `assets/shaders/sphere.wgsl`, `face_math.wgsl`, `face_walk.wgsl` —
  GPU walker.

## A body is a node

`NodeKind::CubedSphereBody { inner_r, outer_r }` tags a node whose
27 children are laid out as:

- **6 face-center slots** (`FACE_SLOTS` in `cubesphere.rs`) hold the
  face subtrees. Each face subtree's root is tagged
  `NodeKind::CubedSphereFace { face }` so the shader and CPU raycast
  know its children are indexed on `(u_slot, v_slot, r_slot)` rather
  than `(x, y, z)`.
- **Center slot (1, 1, 1)** holds the uniform interior filler — a
  dedup'd chain of `Block(core_block)`.
- **The other 20 slots** are `Empty` — the containing cube's corners
  and edges that the sphere doesn't fill.

Radii live on the body's `NodeKind` and are expressed in the *body
cell's local `[0, 1)` frame*. The shader scales them by the body
cell's render-frame size at draw time. Consequence: `0 < inner_r <
outer_r ≤ 0.5`; a bigger sphere in world terms lives at a shallower
anchor.

## Face indexing

```rust
pub enum Face { PosX = 0, NegX = 1, PosY = 2, NegY = 3, PosZ = 4, NegZ = 5 }
```

Face subtree children use `(u, v, r)` axes. `u` and `v` are
equal-angle cube-sphere coordinates across the face; `r` is radial
depth from the inner surface to the outer surface. This is why the
offset in a `CubedSphereFace` anchor isn't a Cartesian `(x, y, z)` —
see [coordinates.md](coordinates.md#offset-interpretation-depends-on-nodekind).

## The render frame for a planet

When the camera sits inside a face subtree, `compute_render_frame`
returns an `ActiveFrameKind::Sphere(SphereFrame)`. The frame's linear
render root stays at the **containing body cell** — not at the face
subtree — but carries an explicit `(face, u_min, v_min, r_min, size)`
window restricting the sphere DDA to the player's immediate face
region. This is what keeps the shader in f32 precision while letting
the player stand on a planet surface at deep anchor depths.

The `logical_path` in the frame continues through the face subtree;
editing, highlight, and the crosshair all operate on that deeper path.

## Transitions

Crossing between face subtrees, entering/exiting the body, or walking
radially off the outer surface all fire `Transition` events through
the coordinate primitives (see
[coordinates.md](coordinates.md#transitions)). Game code uses these to
rotate the camera's "up" to the radial direction on sphere entry, to
re-express yaw/pitch when crossing cube seams, and so on.

## Multi-body

Many planets, nested planets, planets orbiting planets — all free.
Insert a second `CubedSphereBody` at a different path and the tree
handles the rest. There is no "the" planet in the engine.
