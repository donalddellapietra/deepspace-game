# Editing and propagation

## Overview

There are no flags anywhere in the tree. Edits are synchronous tree
walks that leave the tree fully consistent before returning. Voxels
and meshes are always in sync at every layer because the walk updates
both in one pass.

## The three concerns the old dirty flag mixed together

The old `Chunk::mesh_dirty` flag was doing three different jobs at
once. The tree model separates and eliminates them:

1. **"Has this node been edited by the user?"** → irrelevant. In the
   pure content-addressed tree, regeneration never runs over existing
   nodes, so there is nothing to protect against.
2. **"Does the voxel content match the mesh?"** → gone. The edit walk
   keeps them in sync synchronously.
3. **"Does this ancestor's downsampled grid reflect its current
   children?"** → handled by the edit walk itself, which propagates
   the downsample upward as part of the same operation.

Nothing is ever in an "inconsistent and needs a future rebuild" state.
If the walk returned, the tree is clean.

## The edit walk (single leaf voxel)

Player places a block at leaf voxel `(vx, vy, vz)` inside the leaf
node at `path`.

```
1. Leaf node:
   a. Clone the leaf node's voxel grid, update (vx, vy, vz).
   b. Hash the new voxel content (see "content addressing" below).
   c. Look up in the node library. If unseen, bake a mesh and insert
      a new library entry. Otherwise reuse the existing id.
   d. The leaf's path now points at the new NodeId instead of the
      old one — the leaf's parent's children[slot] is updated.

2. Walk up the path.

3. At each ancestor:
   a. Re-downsample only the 5³ slot corresponding to the child
      that just changed. The other 124 slots are untouched.
   b. The ancestor's children[slot] pointer is replaced with the
      child's new NodeId.
   c. Hash the ancestor's now-updated children array (non-leaf
      hashing is by children, not voxels — see below).
   d. Look up in the library. Reuse or bake (the bake operates on
      the downsampled voxel grid, which is also cached on the entry).
   e. Continue up.

4. At the root, update `World.root` to point at the new root NodeId.
```

That's it. `O(MAX_LAYER)` work per edit. No deferred state. The
world's root `NodeId` changes on every edit — this is normal in a
pure content-addressed tree and not a problem.

## Cost

At `MAX_LAYER = 12`, per-layer cost of a single edit:

- Downsample one 5³ slot: 5³ × 5³ = 15,625 voxel reads ≈ **15 µs**
- Hash the node's 25³ grid: ≈ **2 µs**
- Library lookup: ≈ **30 ns**
- Bake greedy mesh if unseen: ≈ **200 µs**

Worst case per layer ≈ **220 µs**. Twelve layers ≈ **2.6 ms** total per
edit. Comfortably inside a 16 ms frame.

## Node structure

```rust
pub struct Node {
    /// 25³ voxel grid. At leaf, this is the authoritative voxel
    /// content. At non-leaf, this is the downsampled representation
    /// of the node's 125 children, cached here for rendering.
    voxels: Box<[u8; 15_625]>,
    /// None at leaf. Otherwise 125 child NodeIds.
    children: Option<Box<[NodeId; 125]>>,
    /// Bevy mesh handle built from `voxels`.
    baked_mesh: Handle<Mesh>,
    ref_count: u32,
}
```

No flags. The only per-node state is the voxel grid, the (optional)
children pointer, and the mesh. Refcount is for library eviction.

## Content addressing

Library lookups hash one thing per node type:

- **Leaf nodes:** hash the 25³ voxel grid.
- **Non-leaf nodes:** hash the 125-element children `NodeId` array.

These live in separate library tables (or a single table with a leaf/
non-leaf tag in the hash key) so a leaf and a non-leaf can't collide.

**Why non-leaf dedup is by children, not by downsampled voxels.** Two
non-leaf nodes with the same children are structurally identical and
must share an id. Two non-leaf nodes that happen to downsample to the
same 25³ grid but have different children are NOT identical — they
represent different sub-trees, and one can be edited without affecting
the other. Hashing by children is the only way to make subtree dedup
actually work.

**Why this gives infinite grassland one entry per layer.** Every leaf
in an unedited grassland world is the "solid grass" leaf NodeId. Every
layer-MAX_LAYER-1 node has 125 children, all equal to that one id —
the children array hashes to one non-leaf NodeId. Recurse up: every
layer-K node's children are all equal to that layer's "all grass"
NodeId, hashing to one entry at layer K. The whole world collapses to
`MAX_LAYER` library entries total, total memory a few hundred KB
regardless of world size.

On hash match, the library verifies by direct byte comparison against
the existing entry (voxels for leaves, children array for non-leaves)
before returning the existing id. xxHash64 collisions are rare but
real; we verify to keep the tree sound.

## Procedural generation

The procedural generator is a pure function `generate(path) ->
VoxelGrid`. It is called **only** when the tree walks into a path that
doesn't yet have a materialised node. The result is stored as a new
node.

The generator never interacts with the edit walk. Edits own the
regions they touch; the generator owns the regions it hasn't touched.
Because the tree is pure content-addressed, once a node is in the
library it is never regenerated — there's no need to flag it as
"protected."

**Starting complexity: infinite grassland.** The initial generator is
a single function: `fn generate(path) -> VoxelGrid { grass }`. Every
unvisited region is grass everywhere. This collapses every unedited
node at every layer to a single library entry — the simplest possible
world to reason about while we build out the tree and the editor.
Noise, biomes, structures, caves etc. come later.

## Player-facing edit API: `edit_at_layer_pos`

All player edits go through `edit_at_layer_pos(world, &LayerPos,
voxel)`, which dispatches on `lp.layer` into three branches — the
same split the 2D prototype's `World::edit_at` uses:

- **`lp.layer == MAX_LAYER`** (leaf view): synthesise a `Position`
  pointing at `(cx, cy, cz)` inside the targeted leaf and call
  `edit_leaf` directly. Single-voxel write.
- **`lp.layer == MAX_LAYER - 1`** (one above leaves): the clicked
  cell summarises a `5³` region of exactly one child leaf at slot
  `(cx/5, cy/5, cz/5)`. Clone that leaf, fill the `5³` region at
  `((cx%5)*5, (cy%5)*5, (cz%5)*5)`, intern, then call `install_subtree`
  with `lp.path + [child_slot]`.
- **`lp.layer <= MAX_LAYER - 2`** (two or more above leaves): the
  `(cx, cy, cz)` triple decomposes into **two more slot steps**:
  `slot_a = slot_index(cx/5, cy/5, cz/5)` and
  `slot_b = slot_index(cx%5, cy%5, cz%5)`. Together these name a
  specific layer-`(lp.layer + 2)` subtree. Build (or recycle via
  dedup) a "solid X chain" rooted at that layer and splice it in via
  `install_subtree(lp.path + [slot_a, slot_b], chain_id)`.

All three branches end in `install_subtree(ancestor_slots, new_id)`,
which does the leaf-to-root walk above: re-downsample one slot per
ancestor, intern each new node, update `World.root`, rotate the root
refcount. The length of `ancestor_slots` is the layer of the replaced
node.

Semantically, editing a layer-`K` cell still means **"fill the
entire leaf region under that cell with block X"** — dispatch just
picks the cheapest shape of edit walk that achieves it. The third
branch is where dedup carries its weight: replacing a trillion leaf
voxels with "solid stone" costs the same as replacing one million,
because the solid-stone subtree at every layer is already a single
library entry.

## Editing at higher layers (conceptual)

When the camera is zoomed out to layer K (K < MAX_LAYER), a click
targets a layer-K voxel, not a leaf voxel. Placing block X at a
layer-K voxel means **"fill the entire leaf region under that
voxel with block X."**

Semantically this is `5^(3 × (MAX_LAYER - K))` leaf voxels being
replaced. At K = 6 with MAX_LAYER = 12, that's 5^18 ≈ 4 × 10^12 leaf
voxels. You obviously don't write that many bytes — content dedup
collapses it to almost nothing.

The edit walk for a layer-K edit:

```
1. Build or reuse the "solid X at leaf" NodeId (one 25³ grid of X).
2. Build or reuse the "layer MAX_LAYER - 1 all-solid-X" NodeId: a
   non-leaf node whose 125 children are all the leaf NodeId from
   step 1.
3. Walk up, building or reusing "all-solid-X" non-leaf NodeIds at
   every layer from MAX_LAYER - 1 up to K + 1. Each is a 125-child
   array where all 125 children are the layer-below "all-X" NodeId.
4. The layer-K node at the edit location has one of its children
   replaced with the layer-(K+1) "all-X" NodeId from step 3.
5. Continue the normal leaf-edit-style walk from layer K up to
   the root: re-downsample the affected slot, mint new NodeIds,
   update children pointers.
```

Steps 1-3 are the "solid-X chain" — at most `MAX_LAYER - K` new
library entries, all of which dedupe across the whole world for that
material. A layer-6 edit places X costs roughly 6 + 6 = 12 node
operations: 6 chain entries and 6 ancestor walks to the root. Fast.

Higher-layer edits are cheaper per unit of affected volume than
low-layer edits, because dedup is maximally effective when the thing
you're placing is uniform.

## Rendering never asks about cleanliness

The renderer walks the tree and emits entities using each visited
node's `baked_mesh` handle. The mesh handle is always current because
the edit walk kept it current. The renderer never checks a flag, never
asks "is this stale," never schedules deferred work. See
`rendering.md` for the walk itself.

## Deferred optimizations (noted, not implemented yet)

These exist in the design but don't need to land with the first
version. Flagged here so we remember them.

### Early exit on propagation

When an edit gets smoothed out by downsampling at some intermediate
ancestor — the downsampled 25³ grid hashes to the same library id as
before the edit — nothing changed at that layer or above. The walk
can stop.

In practice this makes single-voxel edits stop after 2–4 layers
instead of going the full 12. Cost drops from ~2.6 ms to ~500 µs.
Worth adding once we have metrics to confirm it matters; skipped
initially to keep the walk trivial.

### Batch edits per frame

If the player drag-paints 30 voxels in one frame, the naive
implementation runs the full walk 30 times. A batched version
collects all edits during the frame, then does one walk at
frame-end that re-downsamples each affected ancestor slot only once
even if multiple children changed.

Same `O(MAX_LAYER × affected_slots)` work, but paid once per frame
instead of 30 times. Skip until drag-painting exists and shows up
in a profile.

### Lazy mesh bake

A newly-generated procedural node could defer its mesh bake until
the renderer asks for it the first time. Currently we bake eagerly
during the edit walk and during lookup, which is the simpler model.
Revisit if bake cost starts showing up in frame profiles.
