# LOD Architecture (Layer 3) -- Full Specification

## 1. Tree Structure

### 1.1 Branching Factor

The world is a recursive tree with **branching factor 3**. Every non-terminal node has exactly 3x3x3 = 27 children. Children are indexed by `(x, y, z)` where each coordinate is in `{0, 1, 2}`. The center child is `(1, 1, 1)`.

### 1.2 Why Base 3

Base 3 is the smallest odd branching factor greater than 1. Odd branching factors have a center cell; even ones do not.

**Why the center matters:**

Natural structures have center-line features: a tree trunk, a character's spine, a doorway, a pillar. In a base-2 or base-4 tree, the geometric center of a node falls at the junction where ALL children meet. Accessing or editing the center requires:

- Base 2: touching 8 children (all of them)
- Base 4: touching 8 inner children

In base 3, the center is child `(1, 1, 1)` -- one child, one tree walk, one edit path. The most common spatial location (the center of things) is the cheapest to access, not the most expensive.

**Why not base 2 specifically:**

A 2x2x2 = 8-child node is too coarse to represent any meaningful structure. It's a spatial subdivision, not a content unit. You cannot build a recognizable room, face, pillar, or window in 8 voxels. With base 3, a single 3x3x3 node (27 children) is the smallest grid where the node can be a "thing in the world" -- a small room, a character's head, a section of wall with a window.

The entire SVO/octree literature treats nodes as spatial indices (acceleration structures for rendering). In this architecture, nodes ARE content -- meaningful things at their own zoom level. This changes the branching factor calculus entirely. Base 3 has never been used in the literature; this is novel.

### 1.3 Node Types

There are exactly two kinds of nodes:

1. **Terminal nodes**: Block types. Stone, grass, wood, leaf, sand, water, air. These are the atoms of the world. They have no children and no mesh. They are represented as an enum value in a parent's child slot.

2. **Non-terminal nodes**: Compositions of 27 children. Each child is either a terminal (block type) or another non-terminal (NodeId pointing into the library). Every non-terminal has a pre-baked mesh (see Section 3).

There is no special "leaf layer." The distinction between terminal and non-terminal is the ONLY structural distinction in the entire tree. Every non-terminal layer is identical in code and behavior. "Every layer is Minecraft."

### 1.4 Content-Addressed Library

Nodes are stored in a content-addressed `NodeLibrary`. Two nodes with identical children arrays share one `NodeId`. This is the deduplication mechanism:

- 50 forests that all use the same oak tree point to one `NodeId` for that tree.
- The tree's mesh, voxel data, and children are stored exactly once.
- Refcounting handles eviction when no parent references a node.

Dedup is by children array, not by mesh content. Two structurally identical nodes always share an ID regardless of where they appear in the world.

## 2. Content Pipeline (Canned Structures)

### 2.1 Bottom-Up Authoring

Content is authored bottom-up. Each "structure layer" is a curated library of pre-designed arrangements:

| Structure Layer | Example | Built From | Library Size |
|----------------|---------|-----------|-------------|
| Block types | stone, grass, wood, leaf | Terminal (atomic) | ~20 types |
| Micro-structures | bark pattern, grass tuft | 27x27x27 block types | ~100 |
| Objects | trees, rocks, cacti, houses | 27x27x27 micro-structures | ~200 |
| Patches | forest clearings, dune fields | 27x27x27 objects | ~50 |
| Biomes | temperate forest, desert | 27x27x27 patches | ~20 |
| Regions | continent sections | 27x27x27 biomes | ~10 |
| Planets | Earth-like, Mars-like | 27x27x27 regions | ~5 |
| Systems | solar systems | 27x27x27 planets | ~3 |

Each layer's vocabulary is finite and small. No runtime procedural generation. The world is a specific arrangement of specific things, all the way down. This is the "canned structures" approach.

### 2.2 Offline Generation

A `gen_world` binary runs offline and:

1. Generates terminal block type libraries
2. Generates micro-structure libraries (procedural or hand-authored voxel models)
3. Composes them into objects, patches, biomes, etc.
4. For each non-terminal node, bakes a mesh (see Section 3)
5. Serializes the entire tree + mesh cache to disk
6. Runtime loads this pre-built world

The generation process can use expensive algorithms (noise functions, L-system trees, erosion simulation) because it runs once, offline. Runtime never generates content.

### 2.3 Why Not Procedural

Procedural generation at runtime requires evaluating every layer bottom-up before you can render. You can't compute layer 5's mesh without first having layers 4, 3, 2, 1, 0. This means either:

- Massive startup cost (generate everything)
- Lazy generation with visible pop-in
- Both (the problems we hit with the sphere generator)

Canned structures avoid this entirely. Everything is pre-built. Runtime is pure lookup.

## 3. Mesh System ("Meshes of Meshes")

### 3.1 The 27x27x27 Voxel Grid

Each non-terminal node has a pre-baked **27x27x27 voxel grid**. This grid is computed by resolving 3 branching levels of children into a flat voxel array:

- Level 1: 3 per axis (the node's 27 direct children)
- Level 2: 9 per axis (each child's 27 children)
- Level 3: 27 per axis (each grandchild's 27 children)

Each of the 19,683 voxels in the grid holds the block type of the terminal node at that position (or empty if the terminal is air). If a child at any level is non-terminal, its own voxel grid is downsampled to fill the appropriate region.

### 3.2 Downsampling

When composing a parent's voxel grid from children that are themselves non-terminal:

Each child occupies a 9x9x9 region of the parent's 27x27x27 grid (27/3 = 9 per axis). The child's own 27x27x27 grid must be compressed to 9x9x9. This is done by majority-voting each 3x3x3 block:

- For each 3x3x3 = 27 voxels in the child's grid, pick the most common non-empty value
- If all 27 are empty, the parent voxel is empty
- This is **presence-preserving**: a single non-empty voxel in a 3x3x3 block surfaces a non-empty parent voxel

The presence-preserving property is critical. It ensures thin features (a single tree trunk voxel, a wire, a fence post) survive cascaded downsampling across many layers instead of being washed out by majority-empty neighbors.

### 3.3 Mesh Baking

The 27x27x27 voxel grid is greedy-meshed into a triangle mesh. This mesh is the node's visual representation at its own zoom level. The mesh includes:

- Per-face block type (for material/texture selection)
- Ambient occlusion per vertex (computed from neighbor occupancy)
- Optimized face merging (greedy meshing reduces triangle count)

The mesh is baked offline by `gen_world` and cached. At runtime, the renderer places the cached mesh without recomputing.

### 3.4 The Key Insight: 3 Layers of Sub-Cell Detail

Each cell in the player's 27x27x27 view grid is NOT a flat colored cube. It is a **mesh** -- a triangle mesh baked from 27x27x27 sub-voxels. This mesh preserves 3 layers of visual detail below the cell level.

This is the fundamental LOD mechanism. When the player zooms out, the grid cells get coarser (each cell covers more world), but the mesh within each cell compensates by showing internal structure. The visual detail doesn't drop until the structure is so far away that the mesh itself is sub-pixel.

### 3.5 Concrete Example: A Tree Through All Zoom Levels

A tree is a 27x27x27 block structure. The player starts at layer 3 (blocks are layer 0).

**Layer 3 (starting view):**
- Grid cells = layer-0 blocks
- Tree = 27x27x27 cells (fills the view)
- Each cell's mesh shows sub-block detail (layers -1, -2, -3)
- The player walks under the tree. They see bark texture, individual leaves. They can break blocks.

**Layer 4 (zoom out once):**
- Grid cells = layer-1 nodes (3x3x3 blocks each)
- Tree = 9x9x9 cells
- Each cell's mesh shows its 3x3x3 blocks with sub-block detail
- The tree's block structure is fully visible. No block-level detail loss. Trunk, branches, canopy all clear.

**Layer 5 (zoom out twice):**
- Grid cells = layer-2 nodes (9x9x9 blocks each)
- Tree = 3x3x3 cells
- Each cell's mesh shows 9x9x9 blocks
- Full block-level resolution still preserved in the mesh. The tree looks like a tree.

**Layer 6 (zoom out three times):**
- Grid cells = layer-3 nodes (27x27x27 blocks each)
- Tree = 1 cell
- That cell's mesh = the entire tree at 27x27x27 block resolution
- The tree is one cell but LOOKS like a tree. Its block structure is fully visible in the mesh.
- Sub-block detail (bark texture, leaf veins) is no longer visible -- the mesh resolves to block level only.

**Layer 7:**
- Grid cells = layer-4 nodes
- Tree doesn't fill a cell. It occupies ~9x9x9 voxels within a cell's mesh.
- Still recognizable as a tree shape, but losing block-level detail.

**Layer 8:**
- Tree = ~3x3x3 voxels within a cell's mesh. A green-brown cluster.

**Layer 9:**
- Tree = 1 voxel within a cell's mesh. A single pixel. Six zoom levels above the starting view.

**Summary:** the tree retains full block-level fidelity through layers 3-6 (the mesh absorbs the zoom). It loses resolution at layers 7-8 and becomes a single pixel at layer 9. The "3 layers of mesh preservation" buys 3 extra zoom levels of visual fidelity beyond what the cell grid alone provides.

## 4. Player and Gameplay

### 4.1 The 27x27x27 Gameplay Grid

The player always exists in a 27x27x27 grid of cells. This grid is the player's world -- they walk on it, break blocks in it, build in it. The grid resolves 3 branching levels from the current view node.

At layer N, the grid cells are at layer N-3. The player interacts with cells at layer N-3.

### 4.2 Zoom as Navigation

Zooming is tree navigation, not a camera operation:

- **Q (zoom out):** View layer decreases by 1. The grid cells get coarser. Each cell now covers 3x more world per axis. The player effectively shrinks (or the world grows). Movement speed, gravity, jump height, and collision all scale with cell size so the player crosses one cell in the same wall-clock time.

- **E (zoom in):** View layer increases by 1. The grid cells get finer. The player enters a child node. The NavStack records where they came from so they can zoom back out.

### 4.3 Scale-Invariant Physics

Movement constants are in cells-per-second, not world-units-per-second:

- Walk speed: 8 cells/s
- Sprint speed: 16 cells/s
- Jump impulse: 8 cells/s
- Gravity: 20 cells/s^2

At runtime, these are multiplied by `cell_size_at_layer(view_layer)` to convert to world units. The player feels the same at every zoom level.

### 4.4 Collision

Collision operates on the 27x27x27 grid. Swept-AABB per axis against solid cells. The collision system reads the voxel grid of the current view node (or resolves it from children) to determine which cells are solid.

The collision layer may be one level finer than the view layer for better precision (checking 3x3x3 sub-cells per view cell). This is a tuning parameter, not a structural decision.

## 5. Editing

### 5.1 Block Edits

Breaking or placing a block modifies a terminal node. In the content-addressed tree:

1. Clone the parent's children array
2. Replace the target child with the new block type (or empty)
3. Insert the new children array into the library (dedup may return an existing NodeId)
4. The old parent is left for refcount-based eviction

This is O(1) per edit, plus O(depth) for propagating the change upward.

### 5.2 Upward Propagation

After an edit, every ancestor's voxel grid and mesh are stale. The edit walk goes from the edited node up to the root:

1. At each ancestor, recompute the affected 9x9x9 region of the voxel grid (incremental downsample -- only the changed child's region, not all 27)
2. Re-bake the mesh for that region (or mark it dirty for lazy re-bake)
3. Insert the new node into the library

With base 3, the incremental downsample patches 9 voxels per axis (the changed child's region), not the full 27. This is 729 voxels recomputed per layer, times the tree depth.

### 5.3 Edit Scope

The player edits at their current view layer's cell resolution. At layer 3 (blocks), they break individual blocks. At layer 6 (trees), they could break entire trees. At layer 9 (forests), they could clear entire forests. The same edit mechanism works at every layer -- it's always "replace one child slot."

## 6. Serialization

### 6.1 Canned World Format

The pre-built world is serialized as:

- **Node library:** All nodes (children arrays + voxel grids), content-addressed. Bincode + LZ4 compression.
- **Mesh cache:** Pre-baked meshes indexed by NodeId. Compact vertex format (u8 positions, u8 normals, u8 AO, u32 indices). Per-entry zstd compression. Indexed file (meshes.idx + meshes.bin) for on-demand loading.

### 6.2 Save Files

Player edits create new nodes. Save files store only the delta:

- Nodes with `id >= canned_node_count` are player-created
- The save file contains these nodes plus the new root NodeId
- Loading = load canned world + apply save overrides

### 6.3 Mesh Streaming

Not all meshes fit in memory. The mesh streamer loads meshes on demand:

- Visible nodes: synchronous load (blocking, must be fast)
- Adjacent zoom levels: asynchronous prefetch (background I/O thread with priority queue)
- Priority = inverse distance to camera

## 7. Rendering Pipeline

### 7.1 Per-Frame Walk

Each frame, the renderer walks the tree from the root to the emit layer (view_layer - 1), collecting visible nodes via AABB culling against a view radius:

1. Start at root, push onto DFS stack
2. At each node, compute AABB in camera-relative coordinates
3. If AABB is outside the view radius, skip
4. If at emit layer, emit a Visit (node_id, origin, scale)
5. Otherwise, push 27 children onto stack

### 7.2 Reconciliation

The Visit list is reconciled against the previous frame's entity list:

- Same path + same node_id: keep entity, update transform if origin changed
- Same path + different node_id: despawn old, spawn new
- New path: spawn
- Missing path: despawn

### 7.3 Entity Structure

Each emitted node becomes a Bevy entity tree:

- Parent: `Transform` + `Visibility` + `WorldRenderedNode(NodeId)`
- Children: one entity per sub-mesh (per block type), each with `Mesh3d` + `MeshMaterial3d`

### 7.4 Floating Anchor

The `WorldAnchor` resource tracks the player's integer leaf coordinate. All Bevy-space positions are computed as `(target_leaf_coord - anchor_leaf_coord)` -- an i64 subtraction cast to f32. The anchor moves with the player every frame, so f32 precision is never lost regardless of world position.

## 8. Relationship to Current Codebase

The current codebase on `sphere-planet-clean` implements this architecture with BRANCH_FACTOR=5 and a 25x25x25 voxel grid. The structural changes to reach the base-3 design are:

1. **tree.rs**: Change `BRANCH_FACTOR` from 5 to 3. Children array becomes `[NodeId; 27]`. Voxel grid becomes `[Voxel; 19683]` (27^3). Update downsample to use 3x3x3 blocks instead of 5x5x5.

2. **position.rs**: Update path math for base-3 coordinates.

3. **view.rs**: Update `scale_for_layer` to use powers of 3 instead of 5. Update `slot_index`/`slot_coords` for 3x3x3.

4. **All other modules**: Constant changes only (BRANCH_FACTOR, grid size). The algorithms are identical.

The architecture is already recursive. The change is the branching factor and the mesh-within-cell visual preservation model, not the tree structure itself.
