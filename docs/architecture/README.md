# Architecture

These docs describe the live architecture of the game as it exists in
code today. They are *normative* — if a doc disagrees with `src/`, the
doc is wrong and should be fixed.

For invariants that every piece of the engine must honor, see
`docs/principles/`. For landed refactors and superseded plans, see
`docs/history/`.

## The five-minute tour

The game is a recursive voxel engine. Every node in the world is a
`3×3×3 = 27`-child tree node (`src/world/tree.rs`). The same code
renders, edits, and simulates at every level — there is no "leaf"
layer. A player can zoom from a continent-sized node down through 63
levels of subdivision without changing subsystems.

- **Tree** — `src/world/tree.rs`. Content-addressed immutable nodes
  with `Cartesian`, `CubedSphereBody`, and `CubedSphereFace` kinds.
  See [tree.md](tree.md).

- **Coordinates** — `src/world/anchor.rs`. Every position is a `Path`
  (symbolic, exact) plus an `offset ∈ [0, 1)³` (f32, local). f32 never
  accumulates across cells. See [coordinates.md](coordinates.md).

- **Rendering** — `src/renderer.rs`, `src/app/frame.rs`, WGSL under
  `assets/shaders/`. A GPU ray-march walks the tree in a *frame-local*
  coordinate system selected per-frame from the camera's anchor. See
  [rendering.md](rendering.md).

- **Editing** — `src/world/edit.rs`, `src/world/raycast/`. A CPU ray-
  march mirrors the shader's DDA to select a cell; `propagate_edit`
  rebuilds ancestors clone-on-write. See [editing.md](editing.md).

- **Collision** — Swept AABB against the tree at a collision depth
  one finer than the player's anchor. See [collision.md](collision.md).

- **Zoom** — `E` / `Q` / mouse wheel move the anchor up or down the
  tree. No FOV change. See [zoom.md](zoom.md).

- **Cubed-sphere** — Planetary geometry lives in `CubedSphereBody`
  nodes and six `CubedSphereFace` subtrees. See
  [cubed-sphere.md](cubed-sphere.md).

- **Content** — Worlds are built offline by voxelizing `.vox` meshes
  or procedural rules into tree subtrees. See
  [content-pipeline.md](content-pipeline.md).

- **Streaming** — Content-addressed nodes enable CDN caching and
  merge-friendly edits. See [streaming.md](streaming.md).

- **Scale** — Reference table of what each layer represents in
  approximate real-world units. See [scale.md](scale.md).
