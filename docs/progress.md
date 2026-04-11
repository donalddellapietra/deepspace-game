# Deep Space — Development Progress

## Project Overview

Deep Space is a voxel game built in Bevy 0.18 (Rust). The world is a
fixed tree of `MAX_LAYER = 12` layers where every node holds a 25³
voxel grid and has up to 5³ children. Rendering, editing, and movement
all operate at the layer the camera is currently pointed at — zoom is
pure camera state, never a rebuild of world data.

The current shape took two large reshapes to arrive at: a 5³ layered
prototype that was thrown out, then the content-addressed tree that
replaced it, then a port of the 2D prototype's `LayerPos` /
sub-texture / cell-rate ideas back into the 3D renderer and input
layers.

---

## Phase 1: Engine Selection

### Evaluated
- **Godot 4.6** — Started here with a GDQuest TPS controller template.
  Abandoned because AI-driven development requires text-editable
  files; Godot's GUI-first workflow and binary import cache made CLI
  iteration difficult.
- **Three.js** — Considered for its pure-code workflow. Good for AI
  dev but lacks built-in physics, scene management, and desktop
  export.
- **Bevy 0.18** — Selected. Pure Rust, ECS, everything is code. No
  GUI dependency. Fast incremental builds with dynamic linking.

### Key decision
Bevy was chosen because: (1) ECS is the right pattern for voxel
simulation, (2) Rust provides C++-level performance with memory
safety, (3) everything is text files editable by AI, (4) the
ecosystem has mature voxel examples to learn from.

---

## Phase 2: Initial Prototype (heightmap terrain)

Perlin-noise chunked heightmap, FPS camera, WASD + gravity + jump,
directional + ambient lighting. Abandoned because a heightmap wasn't
voxel-based and there was no layer concept.

### What we learned
- Bevy 0.18 API differences from documentation (`CursorOptions` as a
  component, `BorderColor::all()`, `GlobalAmbientLight`).
- Face winding must be CCW — got this wrong initially and every block
  rendered inside-out.
- Dev profile must optimize dependencies (`opt-level = 3` for deps)
  or Bevy runs at ~5 FPS.

---

## Phase 3: First voxel data model (5³ per-cell grids)

Replaced the heightmap with a proper voxel system:
- 10 block types with per-type PBR materials.
- `VoxelGrid`: 5³ array of `CellSlot::{Empty, Block(BlockType),
  Child(Box<VoxelGrid>)}`. `Child` was the "drill into this" slot.
- `VoxelWorld`: sparse `HashMap<IVec3, VoxelGrid>` at the top layer.
- Face-culled mesh baking, per-block-type sub-meshes so each material
  rendered correctly (metal shiny, glass transparent, etc).

### Key rewrite: vertex colors → per-type materials
Initially used vertex colors with a single `StandardMaterial`. All
blocks shared the same PBR properties — metal looked the same as
dirt. Fixed by splitting the baked mesh into one sub-mesh per block
type present, each with its own `StandardMaterial`.

### Key rewrite: instance vs template
All ground cells shared the same `ModelId` reference. Editing one
cell mutated the template and changed every cell. Fixed by giving
each cell its own copy of the block data.

---

## Phase 4: Drill-in/out layer system

Added an `ActiveLayer { nav_stack }` navigation stack so the player
could drill into any `Child` cell and edit its 5³ interior, then drill
back out. At every layer the coordinate convention was "1 block = 1
unit" and surrounding parent layers were rendered as baked meshes at
the appropriate scale via a `render_ancestors` walk.

### Issues that kept biting

- **Scaling**. Every function that dealt with neighbors had `if
  nav_stack.len() == 1` special cases for the top-layer HashMap vs
  inner grids. They broke at depth 2+. Fixed with generic
  `get_sibling` / `get_sibling_slot` on `VoxelWorld`.
- **Collision ancestor chain**. `block_solid` initially only checked
  one level up. Walking off the edge of a grandparent cell dropped
  you into the void because top-layer ground was invisible to the
  collision system. Fixed by making `block_solid` walk the entire
  `nav_stack`, mapping coords to each ancestor's coordinate system in
  turn.
- **Render ancestor scale bug**. `cumulative_scale *= MODEL_SIZE` was
  at the end of the loop body instead of the beginning. Ancestor cells
  rendered at the wrong scale.
- **Collision iteration order**. Early attempts did swept AABB
  push-out; the best-push selection logic picked the wrong face when
  multiple blocks overlapped (e.g. a parent `Block` filling a 5-unit
  region). Fixed by moving to per-axis AABB clipping — movement is
  clamped before application, no push-out logic, no iteration order
  issues.

---

## Phase 5: Throwing the layered system out for the content-addressed tree

The drill-in/out system worked but was hard to scale. Two recurring
frustrations pushed the rewrite:

1. **The recursive `CellSlot::Child` tree made every invariant
   non-local.** Editing a cell had to walk up, rebuild ancestor grids,
   and every write needed a fresh `Box<VoxelGrid>` to avoid instance
   sharing. Collision, rendering, and the `nav_stack` all had to know
   about these mutations.
2. **There was no dedup.** An infinite grassland needed one
   `VoxelGrid` allocation per cell. Adding more layers or more world
   meant linearly more memory, even though all the grass was
   identical.

The replacement design is spelled out in `docs/architecture/` and is
structurally identical to the 2D prototype at
`github.com/donalddellapietra/prototype-deepspace-game` (which was
built first as a proof of concept):

- Every node is a 25³ voxel grid plus optionally 125 children
  (`5³`). Same shape at every layer.
- Nodes live in a content-addressed `NodeLibrary` — leaves hashed by
  voxel bytes, non-leaves hashed by children array. Refcounts for
  eviction.
- Edits walk leaf-to-root, partially re-downsample one slot per
  ancestor, intern each new node. `O(MAX_LAYER)` per edit.
- Rendering is a per-frame tree walk that emits one entity per node
  at `CameraZoom.layer`. Zoom is a pure view state.

See `docs/architecture/refactor.md` for the phase plan. The refactor
landed as described — `src/world/` was rewritten top-to-bottom into
`tree.rs`, `position.rs`, `state.rs`, `generator.rs`, `edit.rs`,
`render.rs`, and `collision.rs`.

### What the refactor deleted

- The entire drill-in/out `nav_stack` concept. The player never
  changes layer; only the camera does.
- `CellSlot::Child` recursion, `render_ancestors`, ancestor-walking
  collision. The tree walk handles all three uniformly.
- The model registry save/load system. It's deferred — the new
  architecture will eventually let the player "snapshot" any subtree
  by ref'ing its `NodeId`, and that's the cleaner way to do it.
- The two-level hotbar (Block vs SavedModel). Now `HotbarItem::Block`
  only; saved models come back once the snapshot system exists.

### What made the refactor tractable

- The 2D prototype had already shipped the equivalent rewrite and we
  could crib its shape, so nothing was being discovered for the first
  time in 3D.
- `model/mesher.rs::bake_volume` was already generic over grid size,
  so it could be called with `size = 25` from the new render path
  without any mesher changes.
- The `BlockType` / `BlockMaterials` code was reused verbatim. Only
  the voxel representation changed (`Option<BlockType>` → `u8`).

---

## Phase 6: Porting 2D prototype concepts into the 3D code

Once the tree was in, several ideas from the prototype turned out to
be missing from the 3D repo and were ported in:

### `LayerPos`
The original 3D `coordinates.md` only described `Position` — a
leaf-layer position. In the prototype we found that every piece of
code outside physics (the camera, the renderer, the raycast, the
editor) wanted a cell at the *current view layer*, not a leaf.
Projecting from a leaf every time was awkward and meant the renderer
had to know about leaves. `LayerPos` (`path: Vec<u8>` of length
`layer`, `cell: [u8; 3] < 25`) makes the view cell a first-class
value. `src/world/position.rs` now has both types; `LayerPos::from_leaf`
walks up the path applying the downsample's inverse at each step.

### Three-branch `edit_at_layer_pos`
The prototype's `World::edit_at` dispatches on `view_layer` into
three branches:
- **MAX_LAYER** — single-cell leaf edit.
- **MAX_LAYER − 1** — 5³ region inside one child leaf.
- **≤ MAX_LAYER − 2** — `(cx, cy, cz)` decomposes as `(c/5, c%5)`
  into two slot steps, so the click names a specific
  layer-`(L + 2)` subtree. Build or recycle a "solid X chain" and
  splice it in.

All three call a common `install_subtree(path, new_id)`. Ported
verbatim to `src/world/edit.rs`. Tests in that module mirror the
prototype's.

### Sample two layers below view
A naive renderer shows each view cell as the majority-vote downsample
at `view_layer`. That discards most of the tree — at view layer 5,
every cell is a vote over 5^7 leaf cells. The prototype's
`subtexture_25` shows the *actual content two layers deeper* by
looking up the corresponding layer-`(L + 2)` node and rendering its
raw 25² bytes. The `(c/5, c%5)` decomposition is the same as branch 3
of edit: one view cell corresponds to exactly one layer-`(L + 2)`
subtree.

In 3D the same rule gives `target_layer = (view_layer + 2).min(MAX_LAYER)`.
The renderer emits one entity per layer-`target_layer` node and bakes
its mesh from the raw voxel grid. Detail is preserved across zoom —
zooming in doesn't synthesise new information, it just re-lays the
same content at a finer entity grid.

### Viewport counts cells, not pixels
The prototype's viewport is N cells across regardless of cell size,
so zooming out naturally shows a constant solid angle of the world.
The 3D render radius was fixed Bevy units and collapsed at low view
layers — the visible world shrunk to a dot.

Fix: the render radius is `RADIUS_VIEW_CELLS * cell_size_at_layer(L)`,
multiplied at walk time. Same idea threaded through
`world/collision.rs` (player AABB scales with cell size),
`src/player.rs` (walk/sprint/jump/gravity are cell-rates), and
`src/camera.rs` (eye height scales with cell size). At view 12
(leaves) everything is numerically identical to the previous
leaf-voxel behaviour; at lower view layers the player / camera / radius
scale together so one cell subtends a constant visual angle.

### Cell-DDA raycast
The pre-refactor raycast stepped leaf voxels. At low view layers that
could take millions of steps to cross one view cell. Replaced with a
cell DDA: `dda_view_cells` steps at `cell_size_at_layer(view_layer)`
per iteration and caps at `MAX_REACH_CELLS = 16`. Returns a `LayerPos`
and a face normal in cell units — the editor doesn't need to know
about leaves at all.

### Path-based spawn
The spawn point used to be `Vec3::new(0.0, 3.0, 0.0)`. Fine until the
root origin moved and `y = 3` landed inside a solid cell. Replaced
with a path-based spawn point (`spawn_position()` in `src/player.rs`)
that names a slot corner using `BRANCH_FACTOR` and `NODE_VOXELS_PER_AXIS`
constants. Nothing hardcodes a Bevy coord — the translation is
computed via `bevy_from_position`.

---

## Current state

### Architecture

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
    render.rs          CameraZoom, RenderState, tree-walk renderer
    collision.rs       Bevy <-> Position conversion, view-layer-aware
                         AABB clipping, on_ground
  interaction/         Cell-DDA raycast producing a LayerPos + normal
  editor/
    mod.rs             Hotbar, HotbarItem::Block
    tools.rs           zoom_in/out (F/Q), place/remove, reset_player (R)
  inventory.rs         Inventory panel UI, click-to-swap-into-hotbar
  camera.rs            First-person camera with view-layer-scaled eye
  player.rs            Path-based spawn, cell-rate movement
  ui/mod.rs            Hotbar, "Layer N" indicator
  diagnostics.rs       Debug logging
```

### Test coverage (live)

Unit tests co-located with each module:

- `world/tree.rs` — voxel encoding, slot encoding, downsample
  (all-same + one-different), leaf/non-leaf dedup, refcount cascades.
- `world/position.rs` — walks inside/across leaves, multi-layer
  boundary crossings, off-root detection, round-trip walks,
  `LayerPos::from_leaf` projection at every layer.
- `world/state.rs` — grassland root build, library entry count,
  idempotent rebuild, ground / air leaf sampling.
- `world/edit.rs` — single-voxel edit, round-trip content dedup,
  `edit_at_layer` leaf + higher-layer + root, `edit_at_layer_pos`
  all three branches at non-zero cells.
- `world/collision.rs` — Bevy ↔ Position round trips, out-of-root
  rejection, ground surface boundary, grassland solidity above/below
  the surface, `on_ground` at/near the surface.
- `world/render.rs` — scale/extent math, walk emits visits at leaves,
  radius scales sensibly across view layers (guarding a previous
  bug where low view layers culled everything).
- `world/generator.rs` — grass/air leaves are uniform and distinct.

### Known gaps / deferred work

- **No frustum culling.** v1 uses the "within N view cells of the
  camera" radius test. Full frustum culling is planned once this
  shows up in a profile.
- **No snapshot / template system.** The old drill-in/out code had a
  "save current cell as a model" feature. Not ported — the new
  architecture should expose snapshots as "ref-inc this NodeId" but
  there's no UI or persistence yet.
- **No serialization.** World state is built fresh on every launch.
- **Grassland only.** `generator.rs` returns either all-grass or
  all-air. Noise / biomes / structures are explicit non-goals for
  now; grassland is a deliberate low-complexity baseline to reason
  about while the tree is still settling.
- **No early-exit edit propagation.** `editing.md` notes the
  optimization where a re-downsample that hits the same library id
  stops the walk; not implemented because edits are already well
  inside a frame.

---

## Key lessons

1. **Don't use tolerances for physics.** Every epsilon introduced a
   new edge case. Per-axis AABB clipping is correct by construction.
2. **Don't special-case "top layer".** In the drill-in/out system, any
   function that checked `if nav_stack.len() == 1` broke at depth 2+.
   The tree walk erases the distinction — every level of the tree is
   identical.
3. **Content addressing buys you everything.** Dedup means infinite
   grassland is ~25 library entries. Edit propagation gets round-trip
   freeness for free. Mesh caching inherits dedup for free. Trying
   to do any of these by hand on a non-addressed tree is a ton of
   bookkeeping.
4. **Separate the view from the state.** The drill-in/out system
   tangled "which layer is the player editing" with "where in the
   world is the player." Splitting those into `Position` (player) and
   `CameraZoom` (view state) made rendering, physics, and editing all
   obviously correct.
5. **Build the prototype first.** The 2D prototype found the edit
   dispatch, `LayerPos`, the sub-texture trick, and the
   viewport-counts-cells invariant without fighting Bevy. Porting the
   mechanics back into 3D was mechanical.
6. **Per-block-type materials, not vertex colors.** Still true.
   Vertex colors force shared PBR parameters; per-type materials
   allow metal to be shiny, glass transparent, dirt matte.
