# Voxel tree: per-node grid and downsampling

The world is a tree of voxel nodes. Every node — regardless of which
layer it sits at — holds a **25³ voxel grid** (15,625 voxels) and has
up to **5³ = 125 children** (the next layer down).

The same shape repeats at every layer. "Going up a layer" means taking
the 125 children immediately below a node and compressing them into
that node's own 25³ grid.

## How children tile their parent

- Parent is 25 voxels per axis.
- Parent has 5 children per axis (5³ = 125 children total).
- Each child occupies a **5-voxel-wide region** on each axis inside the
  parent (25 / 5 = 5).
- Each child's slot in the parent is therefore **5³ = 125 voxels**.
- Sanity check: 125 children × 125 voxels per slot = 15,625 = 25³. The
  children tile the parent exactly — no gaps, no overlap.

## How one child is compressed into its slot

- The child's own grid is 25³ = 15,625 voxels.
- The child's slot inside the parent is 5³ = 125 voxels.
- Compression ratio: 15,625 / 125 = **125:1**, which is **5:1 per
  axis**.
- Each voxel in the output slot summarises a **5×5×5 = 125-voxel
  region** of the child.

## Per-parent-voxel algorithm

For each voxel `(px, py, pz)` in the parent's 25³ grid:

1. Figure out which child it belongs to:
   - `child_x = px / 5`, `child_y = py / 5`, `child_z = pz / 5`
     (each in `0..5`).
   - Child slot index = `child_z * 25 + child_y * 5 + child_x`
     (in `0..125`).
2. Figure out which 5³ region of that child this parent-voxel
   summarises:
   - `lo_x = (px % 5) * 5`, spanning `lo_x .. lo_x + 5`.
   - Same shape for `y` and `z`.
   - That's a 5×5×5 = 125-voxel cube inside the child's own 25³ grid.
3. Majority-vote across those 125 child voxels → one parent voxel
   value. (Exact aggregation rule — majority, first-non-empty,
   something smarter — is a separate decision and lives in its own
   section below.)

Each parent voxel reads from **exactly one child**. Parent voxels never
blend across child boundaries, because the children are adjacent blocks
in the parent, not interleaved.

## Total work per downsample

- 25³ parent voxels × 5³ child reads each = 25³ × 5³ = **1,953,125
  voxel reads** per downsample (equivalently, 5⁹).
- At roughly 1 ns per read, ≈ **2 ms per node downsample**.
- Paid **once per unique pattern** thanks to content-addressed dedup in
  the node library. Procedural terrain reduces to a handful of unique
  patterns per layer, so the amortised cost is effectively zero.

## Signature

```rust
fn downsample(children: [&VoxelGrid; 125]) -> VoxelGrid;
```

`VoxelGrid` is a boxed `[u8; 15_625]` (25³). The input is 125 child
grids arranged in the canonical slot order (see "child slot index"
above); the output is the compressed parent grid.

## Content addressing (library dedup)

Nodes are stored in a content-addressed library. The hash key differs
by node type:

- **Leaf nodes** hash their 25³ voxel grid.
- **Non-leaf nodes** hash their 125-element children `NodeId` array.

Non-leaf dedup is deliberately *not* by downsampled voxels, because
two sub-trees can downsample to the same grid without being
structurally identical. Hashing by children is the only invariant
that makes subtree dedup correct. See `editing.md` for the full
reasoning.

The downsample function above is used to compute each non-leaf node's
cached 25³ voxel grid (for rendering and for the mesh bake), but it
is *not* the input to the library's hash.
