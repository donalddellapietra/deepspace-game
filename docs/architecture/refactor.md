# Refactor plan: flat chunks → content-addressed voxel tree

This is the execution plan for replacing `src/world/`'s current
`FlatWorld` + three-level mesh library with the tree architecture
described in `voxels.md`, `coordinates.md`, `editing.md`, and
`rendering.md`.

## Scope

Essentially all of `src/world/` is rewritten. The `src/editor/tools.rs`
edit path changes. The `src/model/mesher.rs` greedy mesher is mostly
reusable because `bake_volume` is already generic over grid size. The
`src/block/` module is untouched. `src/player/` is untouched beyond
whatever minor signature tweaks the new `Position` type forces.

Rough diff size: **+1200 / -1800** lines. Net shrinkage because the
three-level render paths and the flat-world streaming logic collapse
into one generic tree walker.

## Target file layout under `src/world/`

- `mod.rs` — plugin, re-exports
- `tree.rs` — `Node`, `NodeId`, `NodeLibrary`, hashing, downsample
- `position.rs` — `Position`, `NodePath`, walk/neighbor operations,
  slot encoding constant
- `generator.rs` — `generate(path: &NodePath) -> VoxelGrid`
- `edit.rs` — `edit_leaf(world, path, voxel, block)` and
  `edit_at_layer(world, path_prefix, layer, block)`
- `render.rs` — uniform-layer tree walk renderer (replaces all three
  current depth paths)
- `collision.rs` — AABB queries as tree walks
- `state.rs` — `World { root: NodeId, library: NodeLibrary }` and the
  Bevy resource wrapper

Deleted: the current `chunk.rs`, `library.rs`, `render.rs`, `state.rs`,
`terrain.rs`. The new files replace them.

## Constants

Two constants that were conflated in the old code (`MODEL_SIZE = 5`)
now live separately:

```rust
pub const BRANCH_FACTOR: usize = 5;       // 5³ = 125 children per node
pub const NODE_VOXELS_PER_AXIS: usize = 25;
pub const NODE_VOXELS: usize = 15_625;    // 25³
pub const CHILDREN_PER_NODE: usize = 125; // 5³
pub const MAX_LAYER: u8 = 12;
```

## Voxel representation

Voxels become `u8` in the new design, with `0 = empty` and `1..255` a
`BlockType` discriminant. The 25³ grid is then a `Box<[u8; 15_625]>`
instead of the current `[[[Option<BlockType>; 5]; 5]; 5]`. The
`BlockType` enum is unchanged; there's just a `u8 ↔ BlockType`
conversion at the edge.

## Phases

Each phase is a coherent chunk of work. They're ordered so later
phases depend only on earlier ones, and every phase is unit-testable
on its own before the following phase starts.

The whole refactor lands **in one worktree, as one merge back to
main**. Individual phases build and test within the worktree but are
not separately committed to main. This avoids leaving main in a
half-migrated state where the old and new systems both try to own the
world.

### Phase 1 — `tree.rs`: data structures and library

**Goal:** `Node`, `NodeLibrary`, content hashing, and downsampling as
a pure module. No Bevy wiring yet.

**Produces:**
- `Node` struct (voxels, children, baked_mesh, ref_count).
- `NodeId` type (`u64`, `EMPTY_NODE = 0`).
- `NodeLibrary` with separate tables for leaf (hash-by-voxels) and
  non-leaf (hash-by-children-array), each with byte-compare verify
  on hash hits.
- `downsample(children: [&VoxelGrid; 125]) -> VoxelGrid`.
- `ref_inc` / `ref_dec` / eviction on refcount drop.

**Tests:**
- `downsample` of 125 all-grass children → all-grass parent.
- `downsample` with one stone child among 124 grass → majority-vote
  produces the expected mixture.
- Leaf dedup: inserting the same voxel grid twice returns the same id.
- Non-leaf dedup: a children array with all 125 entries equal to
  `solid_grass_leaf_id` hashes to one id regardless of how often we
  insert it.
- Non-leaf distinction: two children arrays that differ in one slot
  produce different ids (catches a hash-only regression).
- Refcount: insert, increment, decrement to zero, confirm eviction.

**Does NOT depend on Bevy** — this module is pure Rust for now. Mesh
handles can be stubbed with `Handle::default()` in tests and wired
through `Assets<Mesh>` in Phase 5.

### Phase 2 — `position.rs`: addressing and walks

**Goal:** The `Position` type and all the walking/neighbor logic.

**Produces:**
- `NodePath`, `Position`, slot encoding constants and helpers
  (`slot_index`, `slot_coords`).
- `walk_axis(position, axis, delta_voxels) -> Option<Position>` —
  integer voxel steps with carry into the path.
- `neighbor_leaf(path, axis, direction) -> Option<NodePath>` —
  the recursive up-walk for crossing node boundaries.
- Drilling helpers (`camera_position_for_layer(position, layer)`) if
  the renderer needs them — might land in `render.rs` instead.

**Tests:**
- Walking +x within a leaf: voxel increments correctly.
- Walking +x that crosses one leaf boundary: path's last slot
  updates, voxel resets, y and z unchanged.
- Walking +x that crosses a two-layer boundary: path updates at two
  levels, voxel resets.
- Walking past the root: returns `None`.
- Neighbor diagonals decomposed per axis.

**Does NOT depend on `tree.rs`.** Pure math over the path/voxel
representation.

### Phase 3 — `generator.rs`: procedural content

**Goal:** A stub generator that produces all-grass content. Locks in
the generator signature so later phases can depend on it.

**Produces:**
- `fn generate(path: &NodePath) -> VoxelGrid` returning a grass-filled
  grid for every input.
- A `World::get_or_generate(path) -> NodeId` helper that walks the
  tree, calls the generator when a node doesn't exist, and inserts
  the result via `NodeLibrary`.

**Tests:**
- Generator is deterministic (same path → same bytes).
- `get_or_generate` on a fresh world returns one `NodeId` at the leaf
  layer; asking for a second leaf at the same depth returns the same
  id (because grassland dedupes).
- `get_or_generate` at layer K < MAX_LAYER (directly asking for a
  non-leaf) builds the non-leaf by pulling its 125 children through
  the same code path and returning the non-leaf's `NodeId`.

Depends on Phase 1.

### Phase 4 — `edit.rs`: the edit walk

**Goal:** Leaf edits and higher-layer edits that walk up the tree
minting new `NodeId`s and updating `World.root`.

**Produces:**
- `edit_leaf(world, position, block) -> NodeId` (new root).
- `edit_at_layer(world, path_prefix, block)` — higher-layer edit,
  builds the solid-X chain then walks up.
- Downsample-one-slot helper (optimized form of `downsample` that
  only rewrites one 5³ region of the parent).

**Tests:**
- Edit a leaf voxel from grass to stone. Confirm:
  - The leaf's `NodeId` changes.
  - Every ancestor up to the root has a new `NodeId`.
  - The root is reachable from `World.root` and points at the new
    tree.
  - Sibling regions of the tree still point at the pre-edit `NodeId`
    chain (sharing / dedup still works).
- Edit the same leaf voxel back to grass. Confirm:
  - The new root's `NodeId` equals the pre-edit root's `NodeId` (full
    round trip via content addressing).
- Edit at layer K = 6 places stone. Confirm:
  - The solid-stone chain has entries at layers 6..=12.
  - The player's descent from root through the edited path hits
    all-stone leaves.
- Refcount accounting: the old subtree's refcounts decrement after
  the walk so orphaned nodes get evicted.

Depends on Phases 1, 2, 3.

### Phase 5 — `render.rs`: tree-walk renderer

**Goal:** Replace `render_super_super_chunks` / `render_super_chunks`
/ `render_chunks` with one generic tree walker that respects
`CameraZoom`.

**Produces:**
- `CameraZoom` resource.
- `render_walk` — recursive descent from root with frustum culling,
  stopping at `zoom.layer` and emitting positions.
- A `RenderState` that maps `NodePath → (Entity, NodeId)` for
  entity reuse across frames.
- The Bevy system that spawns/despawns entities based on the tree
  walk result.
- A hookup for `bake_volume` from `model/mesher.rs`, now called with
  `size = 25` instead of the old `size = 5` / `25` / `125` triplet.

**Tests:**
- Unit: `render_walk` over a fresh grassland world at `zoom.layer = 3`
  returns the expected number of positions for a test frustum.
- Unit: tree walk correctly skips empty subtrees (`EMPTY_NODE`).
- Integration: one smoke test that spawns a Bevy `App`, generates a
  grassland world, runs the render system for one tick, asserts the
  expected number of entities exist.

Depends on Phases 1, 2, 3. Touches `src/model/mesher.rs` only by
calling `bake_volume(25, ..)` from the new code path.

### Phase 6 — `collision.rs`: tree-walk AABB queries

**Goal:** Replace the current leaf-by-leaf collision with a tree walk
that descends only into children overlapping the query AABB.

**Produces:**
- `collide_aabb(world, aabb) -> Vec<Contact>` — recursive walker,
  prunes children whose world-space AABB misses the query, bottoms
  out at leaves, tests per-voxel.
- Keeps the current `SolidQuery` trait as the integration point for
  the player's physics step (so player movement code doesn't change).

**Tests:**
- AABB fully inside solid grass returns a contact on every voxel it
  touches.
- AABB fully in empty space returns no contacts.
- AABB straddling a leaf boundary walks into both neighbors.

Depends on Phases 1, 2.

### Phase 7 — wire-up and old-code deletion

**Goal:** Replace `WorldPlugin`'s systems and resources with the new
ones; delete the old files.

**Produces:**
- `WorldPlugin` now builds around `World { root, library }` instead of
  `WorldState`. New systems added: `render_world` (new), maybe a stub
  `generate_world` for eager startup of the grassland tree.
- `src/editor/tools.rs` — edit hooks call `edit_leaf` /
  `edit_at_layer` instead of the old refcount-aware chunk writes.
- Delete `src/world/chunk.rs`, the old `library.rs`, the old
  `render.rs`, the old `state.rs`, `terrain.rs`, plus any stale
  re-exports in `mod.rs`.

**Tests:**
- `cargo build` — the whole crate compiles with the new module only.
- `cargo test` — all phase-1..6 unit tests pass, plus the integration
  smoke test from Phase 5.
- Manual: run the game, confirm infinite grassland renders, walking
  works, placing a block works, zooming works.

Depends on all prior phases.

## What the refactor does NOT touch

- `src/block/` — `BlockType`, materials, `MODEL_SIZE`. `MODEL_SIZE`
  may be renamed to `NODE_VOXELS_PER_AXIS` or a separate constant
  added, but existing block code keeps working.
- `src/model/mesher.rs` — `bake_volume` is already generic. `bake_model`
  (the 5³-specific wrapper) is unused by the new code and can be
  removed or left alone; I'll remove it in Phase 7.
- `src/player/` — the player's `Transform` stays in Bevy float space.
  Player movement converts `Transform` to `Position` when it needs to
  query the tree; the conversion is a tree walk from the root.
- `src/camera/`, UI, input — untouched beyond adding a `CameraZoom`
  resource.

## Risks and things to watch

**1. Generator call timing.** For grassland v1 the generator is `O(1)`
and always returns the same grid, so `get_or_generate` has no real
cost. For anything richer (noise, biomes) the generator may spike
during tree walks. Not a Phase-1 concern; revisit when the generator
gets interesting.

**2. First-frame library warm-up.** On the very first render, the
tree walk from the fresh root will bake `MAX_LAYER` library entries
(one per layer of all-grass). `MAX_LAYER ≈ 12` bakes × `~200 µs` each
= `~2.4 ms`. Fine.

**3. The render walk cost at high `zoom.layer`.** The tree walk
descends 0..=zoom.layer levels per frame. At `zoom.layer = 8`, a
frustum might visit a few thousand nodes. For grassland these are
dedup'd to one mesh so GPU cost is trivial, but the CPU walk itself
needs to stay under a millisecond. Measure in Phase 5; add delta-walk
caching if it's a problem.

**4. `NodeId` churn during edits.** Every edit mints `MAX_LAYER` new
ids. For paint-heavy workloads the library grows fast. Eviction via
refcount is the safety net, but we should confirm it actually runs
and doesn't leak. Add a Phase-4 test that walks: edit, undo, edit,
undo, ... in a loop and confirms the library size stays bounded.

**5. `Position` ↔ `Transform` precision.** Converting between the
integer position and Bevy's float transform happens per frame for the
camera/player. Get the conversion right in Phase 2 — preferably with
a round-trip unit test.

## Test budget

- Phase 1: ~10 unit tests in `tree.rs`.
- Phase 2: ~8 unit tests in `position.rs`.
- Phase 3: ~5 unit tests in `generator.rs`.
- Phase 4: ~8 unit tests in `edit.rs` (including the round-trip).
- Phase 5: ~5 unit tests + 1 Bevy integration test.
- Phase 6: ~5 unit tests in `collision.rs`.
- Phase 7: cargo build / cargo test / manual smoke.

Target: the refactor lands with roughly **40 new unit tests** covering
the core invariants. The old tests get rewritten against the new API
or deleted if they tested flat-world specifics.

## Rollout

**One worktree, one landing.** I create a worktree, do Phases 1-7 in
order, commit to the worktree branch as I go, and only open a PR
against main once Phase 7 compiles and passes tests end-to-end.
Nothing partial ever touches main.

If the worktree gets too big to hold in one review pass, we split at
Phase 7 (a "prep" PR with Phases 1-6 in new files alongside the old
code, not wired up; then a "cut-over" PR that does Phase 7). This is
a fallback — prefer the single landing.
