# Architecture

These docs describe the live architecture of the game as it exists in
code today. They are *normative* — if a doc disagrees with `src/`, the
doc is wrong and should be fixed.

For forward-looking designs that the code does not yet implement, see
[`../design/`](../design/). For invariants every subsystem must honor,
see [`../principles/`](../principles/). For landed refactors and
superseded plans, see [`../history/`](../history/).

## The five-minute tour

The game is a recursive voxel engine. Every node in the world is a
`3×3×3 = 27`-child tree node (`src/world/tree.rs`). The same code
renders and edits at every level — there is no "leaf" layer. A player
can zoom from a continent-sized node down through 63 levels of
subdivision without changing subsystems.

- **Tree** — `src/world/tree.rs`. Content-addressed immutable nodes
  with the `Cartesian` kind. See [tree.md](tree.md).

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

- **Zoom** — mouse wheel moves the anchor up or down the tree.
  Anchor-depth change is implemented; side-effect physics (walk
  speed, gravity scaling) is not. See [zoom.md](zoom.md).

- **Scale** — Reference table of what each layer represents in
  approximate real-world units. See [scale.md](scale.md).

## What's *not* here

The following subsystems were designed but are not wired into the
game today:

- **Collision / physics** — `src/player.rs` is a no-op; gravity in
  `sdf.rs` is unused. See [../design/collision.md](../design/collision.md).
- **Content pipeline** — `src/import/` parses `.vox` but nothing
  calls it. See [../design/content-pipeline.md](../design/content-pipeline.md).
- **Streaming / multiplayer** — content-addressed dedup is real,
  network code is not. See [../design/streaming.md](../design/streaming.md).
