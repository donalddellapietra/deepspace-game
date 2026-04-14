# Node

## What a Node Stores

A node stores exactly two things:

```rust
struct Node {
    children: [Child; 27],  // 3x3x3 = 27 children
    ref_count: u32,         // for library eviction
}
```

That's it. No voxel grid. No mesh. No flags. No metadata.

## Child Representation

Each of the 27 children is one of:

```rust
enum Child {
    Empty,              // air / void
    Block(BlockType),   // terminal — a block type (stone, grass, wood, etc.)
    Node(NodeId),       // non-terminal — pointer to another node in the library
}
```

There is no leaf/non-leaf distinction on the Node itself. Every node has 27 children, always. The distinction exists only at the child level: a child is either a block type (terminal) or a reference to another node (recursive).

A node whose 27 children are all `Block` or `Empty` is effectively a "leaf" — but the code doesn't special-case it. It's just a node whose children happen to be terminals.

## What a Node Does NOT Store

**No voxel grid.** The 27x27x27 voxel grid used for mesh baking and collision is computed on demand by resolving 3 layers of children into a flat array. It is a temporary computation artifact, not persisted data. Compute it, use it, discard it.

**No mesh.** The pre-baked triangle mesh is stored in a separate mesh cache, indexed by NodeId. The node doesn't know or care about its own visual representation.

**No hash.** The content-addressed hash for dedup lives in the NodeLibrary's index tables, not on the node. It's rebuilt on load.

**No layer/depth.** A node doesn't know what layer it's at. It's just 27 children. The same node can appear at any depth in the tree — content addressing means a "small rock" node used at layer 6 is the exact same NodeId whether it appears in a forest at layer 12 or a garden at layer 9.

## Content Addressing and Dedup

Two nodes with identical children arrays share one NodeId. The NodeLibrary maintains a hash table from children-hash to NodeId for dedup on insertion.

This means:
- 50 forests that use the same oak tree → one NodeId for that tree
- A flat plane of grass blocks → one node reused everywhere (all 27 children are `Block(Grass)`)
- Editing one block creates a new node (different children array → different hash → different NodeId)

## Ref Counting

`ref_count` tracks how many parent child-slots point to this node. When it reaches zero, the node is evicted from the library.

A parent with all 27 children pointing to the same node contributes 27 refs to that node. This is correct — the node is "used" 27 times.

Ref count is derived (reconstructible by walking all nodes) but maintained incrementally for performance. It is NOT serialized — it's rebuilt on load.

## Size

A node is small:
- 27 children: each `Child` is likely 8 bytes (1 byte tag + padding + u64 NodeId or u8 BlockType). Total: 216 bytes.
- ref_count: 4 bytes.
- **Total: ~220 bytes per node.**

Compare to the old architecture: 15,625 bytes (voxel grid) + 1,000 bytes (children) + 4 bytes (ref_count) = ~16.6 KB per node. The new node is **75x smaller**.

This means the NodeLibrary can hold far more unique nodes in memory, which matters for a 63-layer tree spanning galaxies.

## The 27x27x27 Voxel Grid (Ephemeral)

When a system needs voxel-level data (mesh baking, collision), it resolves 3 layers of children into a 27x27x27 = 19,683 voxel grid:

1. The node's 27 direct children → 3 per axis
2. Each child's 27 children → 9 per axis
3. Each grandchild's 27 children → 27 per axis

Each voxel in the grid is the block type of the terminal at that position. If a child at any level is non-terminal, its content is recursively resolved (or its pre-computed downsample is used).

This grid is used for:
- **Mesh baking** (offline): greedy-mesh the 27³ grid into triangles
- **Collision** (runtime): sample cell solidity for swept-AABB
- **Downsampling** (offline): compress a child's 27³ grid into a 9x9x9 region of the parent's grid via majority vote of 3x3x3 blocks

The grid is never stored on the node. It is always recomputed from the children when needed.
