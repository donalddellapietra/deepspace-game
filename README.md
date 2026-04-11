# Deep Space

A voxel game where the world is a content-addressed tree and the
camera chooses which layer of the tree to play on.

Built with [Bevy 0.18](https://bevyengine.org/) in Rust.

## Concept

The world is a fixed tree of `MAX_LAYER = 12` layers. Every node —
regardless of which layer it sits at — holds a **25³ voxel grid**
(15,625 cells) and has up to **5³ = 125 children**. Each parent voxel
is the majority-vote downsample of a 5×5×5 region of exactly one
child, so "going up a layer" is well-defined and lossless in the
maximally-dedup'd case. At `25 × 5^12 ≈ 6.1 billion` voxels per axis
the world is effectively unbounded.

"Drilling" isn't a game-state operation — it's pure camera zoom. The
camera carries one integer, `CameraZoom.layer`, and every visible
entity in a frame is emitted at that layer. Pressing **F** zooms in
(shows finer detail), **Q** zooms out (shows larger area at coarser
detail). The player's underlying position never changes; only which
layer of the tree is being rendered, moved through, and edited.

At every view layer the game plays like Minecraft: walk, jump,
break cells, place cells. Editing a cell at a coarse view layer fills
the entire subtree underneath it with that block — content dedup makes
this cheap even when the affected volume is billions of leaf voxels.

## Controls

| Key | Action |
|-----|--------|
| WASD | Move |
| Mouse | Look |
| Space | Jump |
| Shift | Sprint |
| Left click | Break the cell under the crosshair (at current view layer) |
| Right click | Place the active block on the face under the crosshair |
| 1-0 | Select hotbar slot |
| E | Open / close inventory |
| F | Zoom in (view one layer deeper) |
| Q | Zoom out (view one layer shallower) |
| R | Reset player to spawn (recover from falling into the void) |
| Escape | Release cursor |

## Running

```bash
# First time (compiles all dependencies, takes a few minutes)
cargo run

# Fast incremental builds with dynamic linking
cargo run --features dev
```

Requires Rust 1.85+ (edition 2024).

## Testing

```bash
cargo test
```

## Architecture

```
src/
  main.rs              App setup, plugins, lighting
  block/               BlockType enum (10 types), per-type PBR materials
  model/
    mod.rs             BakedSubMesh (mesh + block type)
    mesher.rs          Generic bake_volume(size, sampler) greedy mesher
  world/
    mod.rs             WorldPlugin, re-exports
    tree.rs            Node, NodeLibrary, content hashing, downsample,
                         slot encoding, voxel <-> BlockType
    position.rs        Position (leaf), LayerPos (view-layer cell),
                         neighbor walks
    state.rs           WorldState (root NodeId + library), grassland
                         bootstrap with a ground surface
    generator.rs       generate_grass_leaf, generate_air_leaf (v1)
    edit.rs            edit_leaf, edit_at_layer, edit_at_layer_pos
                         (three-branch dispatch), install_subtree
    render.rs          CameraZoom, RenderState, tree-walk renderer,
                         mesh cache keyed by NodeId
    collision.rs       Bevy <-> Position conversion, view-layer-aware
                         AABB clipping, on_ground
  interaction/         Cell-DDA raycast producing a LayerPos + normal
  editor/
    mod.rs             Hotbar, HotbarItem::Block
    tools.rs           zoom_in/out, place/remove, reset_player
  inventory.rs         Inventory panel UI, click-to-swap-into-hotbar
  camera.rs            First-person camera with view-layer-scaled eye
  player.rs            Path-based spawn, cell-rate movement (speed,
                         jump, gravity all scale with cell_size_at_layer)
  ui/mod.rs            Hotbar, "Layer N" indicator
  diagnostics.rs       Debug logging
```

### Data model

Every node is a `Node { voxels: Box<[u8; 15_625]>, children:
Option<Box<[NodeId; 125]>>, ref_count: u32 }`. A leaf has `children =
None`; a non-leaf's 25³ voxel grid is the cached downsample of its
125 children. The whole tree lives in a `NodeLibrary` — a
content-addressed store with separate hash tables for leaves
(keyed by voxel bytes) and non-leaves (keyed by children array). Every
hash hit is verified by byte comparison so the tree stays sound under
the ~rare 64-bit collision. Inserting a non-leaf refs all 125 of its
children; decrementing a refcount to zero evicts and cascades.

`WorldState` holds one `NodeId` (the root) and owns the library. The
root is rebuilt via an `insert_leaf`+`insert_non_leaf` chain that
bootstraps an infinite grassland with a `GROUND_Y_VOXELS = 125`-deep
solid surface and air above it — thanks to dedup that collapses to 25
library entries total (2 leaves + 2 patterns per layer + 1 root).

### Coordinates

Two bounded position types (see `docs/architecture/coordinates.md`):

- **`Position`** — always leaf-layer. `path: [u8; 12]` slots from root
  to leaf, `voxel: [u8; 3]` each `< 25`, `offset: [f32; 3]` each in
  `[0, 1)`. Used by entities, physics, and `Bevy Vec3` conversion.
- **`LayerPos`** — a cell at an arbitrary view layer.
  `path: Vec<u8>` of length `layer`, `cell: [u8; 3]` each `< 25`. Used
  by the input layer, editor, and raycast. A click at view layer `L`
  is a `LayerPos` with `layer = L`, never a globally-growing integer.

Nothing in the world addressing ever holds a number bigger than 124.
Movement is a neighbor walk on the path — at most `O(MAX_LAYER)`
slot mutations per voxel step. See `docs/architecture/coordinates.md`
for the walk algorithm and worked examples.

### Rendering

`render_world` walks the tree from the root every frame. At view layer
`L`, the walk emits one entity per **layer-`(L + 2)` node** (clamped
to leaves). The `+2` is the "sub-texture trick" ported from the 2D
prototype: one view cell corresponds to exactly one layer-`(L + 2)`
subtree, so rendering at `L + 2` shows the finer voxel grid directly
instead of the `L`-layer majority-vote downsample. Detail never gets
smoothed away by zooming out.

Culling in v1 is not a frustum: the walk skips any node whose AABB is
more than `RADIUS_VIEW_CELLS` cells (at the current view layer) from
the camera. The radius is measured in *view cells*, not Bevy units, so
zooming out doesn't collapse the visible world to a dot. Meshes are
cached per `NodeId` in `RenderState.meshes` — the cache inherits the
library's dedup guarantee, so grassland uploads a handful of unique
meshes regardless of how many entities are on screen.

### Collision

The player's `Transform` lives in Bevy `Vec3` space (`1 unit = 1 leaf
voxel`); `position_from_bevy` and `bevy_from_position` (in
`world/collision.rs`) bridge to `Position`. Collision is standard
per-axis AABB clipping: for each axis in order (Y, X, Z) compute the
maximum safe movement given every overlapping cell, then apply the
clipped delta.

Both the player AABB and the collision lattice scale with view layer:
at view `L`, one "cell" is `cell_size_at_layer(L) = 5^(MAX_LAYER - L)`
Bevy units, the player is `PLAYER_HW × PLAYER_H` of those, and blocks
are sampled at `target_layer = (L + 2).min(MAX_LAYER)` — the exact
layer the renderer reads from, so the visible mesh and the collision
lattice always agree. At `L = 12` (leaves) one cell is 1 Bevy unit and
this is identical to the previous leaf-voxel collision; at lower `L`
everything grows in lockstep.

### Editing

All player edits go through `edit_at_layer_pos(world, &LayerPos,
voxel)`, which dispatches three ways — the same split used by the
prototype's `World::edit_at`:

- **`layer == MAX_LAYER`** — single leaf-voxel edit. Clone the leaf,
  write one byte, intern, and walk up to the root.
- **`layer == MAX_LAYER - 1`** — fill the 5³ region inside exactly one
  child leaf that the clicked cell summarises.
- **`layer <= MAX_LAYER - 2`** — the `(cx, cy, cz)` pair decomposes
  into two more slot steps, so the click names a specific
  layer-`(L + 2)` subtree. Build (or recycle via dedup) a "solid X
  chain" rooted at that layer and splice it in.

All three branches end in `install_subtree`, which walks leaf-to-root
re-downsampling the one affected slot per ancestor and interning each
new node. The edit is `O(MAX_LAYER)` regardless of how much volume it
affected in leaf-voxel terms. See `docs/architecture/editing.md`.

## What was borrowed from the 2D prototype

The 2D proof-of-concept at
`github.com/donalddellapietra/prototype-deepspace-game` is the
structural twin of this codebase with one dimension dropped (25² grid,
25 children per node, `MAX_LAYER = 8`). The prototype was built first
to validate the mechanics without the 3D rendering overhead, and
several pieces have now been ported back into this repo:

- **`LayerPos`** — originally added in the prototype so the renderer,
  camera, and editor never had to touch a leaf `Position`. The 3D
  architecture doc only had `Position`; the prototype proved
  `LayerPos` was load-bearing and now `src/world/position.rs` has both.
- **Three-branch `edit_at_layer_pos`** — the MAX_LAYER / MAX_LAYER-1 /
  lower dispatch, `install_subtree`, and the two-slot `(c/5, c%5)`
  decomposition all come straight from the prototype's `World::edit_at`.
  `src/world/edit.rs` explicitly references the prototype derivation.
- **Sample two layers below** — the prototype's `subtexture_25` shows
  one view cell as the corresponding layer-`(L + 2)` node's 25² voxel
  grid, rendered from raw bytes instead of a downsample. The 3D
  renderer applies the same rule: `target_layer = (L + 2).min(MAX_LAYER)`.
- **Viewport in cells, not pixels** — pan speed, jump height, gravity,
  render radius, and eye height are all expressed in cells per second
  / cells, then multiplied by `cell_size_at_layer(L)` at runtime. The
  player crosses one cell in the same wall-clock time at every view
  layer, and zooming out makes the player "bigger" in lockstep. This
  matches the prototype's constant-visual-angle behaviour.
- **Cell-DDA raycast** — `src/interaction` steps view cells instead of
  leaf voxels, so picking always takes at most `MAX_REACH_CELLS = 16`
  steps regardless of zoom.
- **Per-NodeId caching** — the render mesh cache is keyed on `NodeId`
  (same as the prototype's texture cache), so cache dedup falls out of
  library dedup for free.

Things this codebase has that the prototype doesn't: greedy mesh
baking (`model/mesher.rs`) with per-block-type sub-meshes, Bevy PBR
materials, `Vec3` player physics with AABB clipping, the
`NodeLibrary` refcount-based eviction, and a ground surface in the
grassland bootstrap. The prototype is append-only and everywhere-grass.

## Development log

See [docs/progress.md](docs/progress.md) for the phases that got us
here, including the original 5³-layered system that was thrown out in
favour of the tree architecture.

## License

MIT
