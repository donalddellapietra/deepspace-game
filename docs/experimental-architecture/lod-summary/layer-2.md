# LOD Architecture (Layer 2)

## Tree Structure

The world is a recursive tree with branching factor 3. Every node has 3x3x3 = 27 children. The tree has no special layers -- every layer is structurally identical. Terminal nodes (the deepest layer) are block types: stone, grass, wood, leaf, air. Non-terminal nodes are compositions of 27 children, where each child is either a block type or another node.

Base 3 was chosen deliberately as the smallest odd branching factor. Odd bases have a center cell (1,1,1); even bases (2, 4) do not. This matters because natural structures have center-line features -- a tree trunk, a character's spine, a doorway. In an even-base tree, the geometric center falls at the junction of all children, so accessing or editing the center requires touching multiple children. In base 3, the center is one child, one tree walk, one edit path.

## Content Pipeline (Canned Structures)

Content is authored bottom-up. Each "structure layer" is a library of pre-designed arrangements in a 27x27x27 space:

- **Block types** (terminal): stone, wood, leaf, sand, water. Atomic. No children.
- **Objects** (e.g. trees): 27x27x27 arrangements of blocks. Library of ~200 tree variants, each a voxel model. Pre-generated or hand-authored.
- **Biome patches** (e.g. forests): 27x27x27 arrangements of objects. Library of ~50 forest variants, each placing ~10 trees from the object library.
- **Regions**: 27x27x27 arrangements of biome patches. Forests, deserts, oceans tiled together.
- **Planets, solar systems, galaxies**: same pattern, continuing upward.

Each layer's vocabulary is finite and small (hundreds, not millions). No runtime procedural generation -- everything is pre-computed offline. This is the "canned structures" approach: the world is a specific arrangement of specific things, all the way down.

Content-addressed dedup handles sharing. 50 forests that all use the same oak tree point to one NodeId. The tree's mesh and data are stored once.

## Rendering and LOD

The player always sees a 27x27x27 grid. They are always 3 branching levels above their finest resolution. At layer N, the grid cells are at layer N-3.

The critical LOD mechanism: **each cell contains a pre-baked mesh that preserves 3 layers of visual detail below the cell level.** The mesh is a triangle mesh baked from 27x27x27 sub-voxels (3 branching levels of children resolved into a voxel grid, then greedy-meshed). This means each cell is not a flat colored cube -- it has internal geometric detail.

### Example: A Tree Through Zoom Levels

A tree is a 27x27x27 block structure authored at layer 3 (assuming blocks are layer 0).

| View Layer | Cell Level | Tree in Grid | Tree in Mesh | Visual |
|------------|-----------|-------------|-------------|--------|
| 3 (start) | 0 (blocks) | 27x27x27 cells | Sub-block detail in each cell | Full detail, break individual blocks |
| 4 | 1 | 9x9x9 cells | Each cell's mesh shows its 3x3x3 blocks with sub-detail | No block-level loss |
| 5 | 2 | 3x3x3 cells | Each cell's mesh shows 9x9x9 blocks | Still full block resolution |
| 6 | 3 | 1 cell | Cell mesh = the entire tree at 27x27x27 block resolution | Tree looks like a tree! |
| 7 | 4 | Inside 1 cell | Tree = 9x9x9 within the cell's mesh | Some detail loss begins |
| 8 | 5 | Inside 1 cell | Tree = 3x3x3 in the mesh | A small cluster |
| 9 | 6 | Inside 1 cell | Tree = 1 voxel in the mesh | Single pixel |

The tree retains full block-level fidelity through layers 3-6 (the mesh absorbs the zoom). It only starts losing resolution at layer 7+, and becomes a single pixel at layer 9 -- six zoom levels above the starting view.

## Why Base 3, Not Base 2 or 4

The standard voxel literature uses base-2 octrees or base-4 (64-trees) for hardware alignment. Nobody has used base 3. The reason is that octrees are spatial indices -- acceleration structures for rendering where the node has no inherent meaning.

In this architecture, nodes ARE content. Each node is a meaningful thing at its own zoom level. This changes the branching factor calculus:

- **Base 2 (octree)**: 2x2x2 = 8 children. No center cell. Too coarse to represent any meaningful structure. To reach the center, you cut through all 8 children. The center of anything is the most expensive location to access.
- **Base 4**: 4x4x4 = 64 children. No center cell (even). Same center-access problem as base 2.
- **Base 3**: 3x3x3 = 27 children. Center cell at (1,1,1). The smallest grid where a node can represent a meaningful thing (a small room, a face, a pillar). Natural structures fit within odd-axis grids.
- **Base 5**: 5x5x5 = 125 children. Has a center, but larger nodes, more memory. The current codebase uses base 5 with a 25x25x25 voxel grid and it works, but base 3 is more minimal.

## Zoom as Navigation

Zooming is not a camera operation -- it is tree navigation. Pressing Q zooms out (the view layer decreases by 1), pressing E zooms in. The player's 27x27x27 grid re-centers on a different layer of the tree. Movement speed, gravity, and collision all scale with the cell size at the current view layer, so the player crosses one cell in the same wall-clock time regardless of zoom.

The NavStack tracks the player's path through zoom levels so they can zoom back in to where they were.
