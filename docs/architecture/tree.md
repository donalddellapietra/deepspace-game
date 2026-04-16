# Tree

The world is a content-addressed recursive tree. Every node has
exactly **27 children** in a 3×3×3 grid. `MAX_DEPTH = 63`. Nodes are
immutable once interned; edits allocate new ancestors clone-on-write.

Source of truth: `src/world/tree.rs`.

## Node

```rust
pub struct Node {
    pub children: [Child; 27],
    pub kind: NodeKind,
    pub ref_count: u32,
    pub representative_block: u8,  // most common non-empty block (LOD)
    pub uniform_type: u8,          // 0–253 = all same, 254 = all empty, 255 = mixed
}
```

Nothing else. No voxel grid. No mesh. No world position. A node does
not know its own depth or location — those come from the *path* that
reaches it.

## Child

```rust
pub enum Child {
    Empty,             // air
    Block(u8),         // palette index; terminal
    Node(NodeId),      // subtree; non-terminal
}
```

A `Child::Block` is a terminal. A `Child::Node` points into the
shared library and can itself hold any mix of `Empty`/`Block`/`Node`
slots, recursively.

## NodeKind

```rust
pub enum NodeKind {
    Cartesian,
    CubedSphereBody { inner_r: f32, outer_r: f32 },
    CubedSphereFace { face: Face },
}
```

- **Cartesian** — default. Children fill a 3×3×3 grid; slot indices
  are row-major `z*9 + y*3 + x`.
- **CubedSphereBody** — a node whose six face-center slots are
  `CubedSphereFace` subtrees. `inner_r` and `outer_r` are in the
  body cell's local `[0, 1)` frame.
- **CubedSphereFace** — children are indexed on `(u, v, r)` axes; the
  ray-march dispatches to a sphere-aware walker when it descends into
  these. See [cubed-sphere.md](cubed-sphere.md).

`NodeKind` is part of the content-addressed hash. Two nodes with
identical children but different kinds do *not* dedup.

## Slot encoding

`slot_index(x, y, z) = z*9 + y*3 + x`, with `x, y, z ∈ {0, 1, 2}`.
The center child is `slot 13 = slot_index(1, 1, 1)` — exported as
`CENTER_SLOT`.

## Content-addressed dedup

`NodeLibrary::insert_with_kind` hashes `(children, kind)`. If the
hash matches an existing node and the contents compare equal, the
existing `NodeId` is returned. Consequences:

- A uniform tree of air is one `NodeId` regardless of depth.
- A procedurally-generated planet shares subtrees wherever patterns
  repeat.
- Memory grows with *unique* subtree count, not with volume.

## LOD helpers

Two fields on `Node` are computed at insert time and make cheap LOD
possible without re-walking the subtree:

- `representative_block` — the most common non-empty block type in
  the subtree, or `255` if all empty. *Presence-preserving*: a single
  wood voxel in a sea of air still gets `representative = Wood`, so
  thin features survive cascaded LOD.
- `uniform_type` — if the whole subtree is one type, its value;
  `254` for all-empty, `255` for mixed. Uniform subtrees flatten to a
  single `Block` during GPU packing.

See `gpu::pack::pack_tree_lod` in `src/world/gpu/pack.rs` for how the
renderer uses these.

## Refcounting

`NodeLibrary` tracks `ref_count` per node for eventual eviction of
orphan subtrees after edits. Edits swap the root via
`WorldState::swap_root`, which increments the new root's subtree and
decrements the old.
