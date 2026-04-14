# Pathing and Coordinates

## Layer Numbering

Layer 0 is the bottom — individual block types (atoms). Higher layer numbers are higher in the tree, covering more world.

- Layer 0: atoms (stone, grass, air). 1 block.
- Layer 1: 3x3x3 = 27 blocks.
- Layer 2: 9x9x9 = 729 blocks.
- Layer 3: 27x27x27 = 19,683 blocks. The player's starting gameplay grid.
- Layer N: 3^N blocks per axis.

The root is at layer 63 (the maximum). The total world resolution is 3^63 ~ 1.7 x 10^30 voxels per axis — enough to span billions of light-years at millimeter resolution.

## Zoom Direction

- **Q (zoom out)**: layer increases. The player ascends the tree. Each cell covers 3x more world per axis. The world appears to shrink around the player.
- **E (zoom in)**: layer decreases. The player descends into a child node. Each cell covers 3x less world. The world appears to expand.

Higher layer number = higher up = more world visible = coarser cells.

## Path Representation

A position in the tree is a path from the bottom (layer 0) upward. Each entry is a **slot index**: a `u8` encoding which of the 27 children this position occupies at that branching level.

```rust
const MAX_DEPTH: usize = 63; // note that this can be changed later and is arbitrary -- just under a cache line

struct Path {
    slots: [u8; MAX_DEPTH],  // slots[0] = finest level, slots[26] = coarsest
    depth: u8,               // number of valid entries (= the layer this path reaches)
}
```

- `slots[0]` is the slot at the finest resolved level (closest to atoms).
- `slots[depth - 1]` is the slot at the coarsest level.
- Entries beyond `depth` are zero (pre-allocated, unused).
- The array is stack-allocated, `Copy`, fixed size. No heap allocation.

### Slot Index Encoding

Each slot index encodes an `(x, y, z)` position within a 3x3x3 grid:

```
slot = z * 9 + y * 3 + x       // range 0..26
x = slot % 3                    // range 0..2
y = (slot / 3) % 3              // range 0..2
z = slot / 9                    // range 0..2
```

The center child is slot `1*9 + 1*3 + 1 = 13`, i.e. `(1, 1, 1)`.

### Why Flat Slot Indices

The alternative — storing each level as `[u8; 3]` — uses 3x the memory (81 bytes vs 27 bytes per path) and doesn't simplify any code. Every function that indexes into a node's 27-element children array needs a flat index anyway. The `slot_coords()` conversion is a single inline function for the rare cases where you need `(x, y, z)`.

## Position (Full Player Location)

A complete position in the world adds a voxel coordinate and sub-voxel offset to the path:

```rust
struct Position {
    path: [u8; MAX_DEPTH],   // slot indices, bottom-up
    depth: u8,               // how many levels are resolved
    voxel: [u8; 3],          // position within the node's 27x27x27 grid (0..26 each)
    offset: [f32; 3],        // sub-voxel fractional position, each in [0.0, 1.0)
}
```

The `voxel` field indexes into the node's 27x27x27 voxel grid (the mesh-resolution grid, not the 3x3x3 children grid). Each component is 0..26.

The `offset` field provides sub-voxel precision for smooth movement and collision. It is always in `[0.0, 1.0)` per axis. Crossing a voxel boundary carries into `voxel`; crossing a node boundary carries into `path`.

## Leaf Coordinates

For large-world precision, positions can be converted to integer **leaf coordinates** — an `[i64; 3]` giving the absolute block-level position in the root's frame. This is computed by walking the path top-down, accumulating each slot's contribution:

```
coord = sum over i in 0..depth of: slot_coords(slots[i]) * 3^i
        + voxel (as the layer-0 contribution)
```

Leaf coordinates are used for:
- The floating anchor (Bevy-space = position - anchor, always small)
- Exact i64 subtraction between two positions (no f32 precision loss)
- Serialization and save files

## The 27x27x27 Voxel Grid

Each non-terminal node has a 27x27x27 = 19,683 voxel grid. This grid resolves 3 branching levels of children into a flat voxel array (3 children per axis × 3 per axis × 3 per axis = 27 per axis). It is the basis for:

1. **Mesh baking**: greedy-mesh the grid into triangles.
2. **Collision**: sample the grid to determine solid cells.
3. **Downsampling**: compress a child's 27x27x27 grid into a 9x9x9 region of the parent's grid (majority vote of each 3x3x3 block).

The grid index function:

```
voxel_idx(x, y, z) = z * 27 * 27 + y * 27 + x    // range 0..19682
```

## Tree Depth and World Scale

Layer 0 atoms are sub-block voxels — fine enough to capture pixel-level detail from imported 3D models. A GLB mesh can be voxelized at layer 0 resolution, preserving curves, angles, and surface detail that would be lost if the atomic layer were block-sized. This means a "block" the player interacts with (at their gameplay layer) is not a flat-colored cube — it contains a pre-baked mesh with thousands of sub-voxels of geometric detail.

With 63 layers, branching factor 3, and a layer-0 voxel of approximately 1mm:

| Layer | Size per axis | Scale |
|-------|--------------|-------|
| 0 | ~1mm | A sub-block voxel |
| 3 | ~27mm | A small detail — a nail, a tooth |
| 6 | ~73cm | A chair, a rock, a barrel |
| 9 | ~20m | A tree, a house |
| 12 | ~530m | A park, a few city blocks |
| 15 | ~14km | A city |
| 18 | ~390km | A small country |
| 21 | ~10,500km | A planet (Earth ~ 12,700km) |
| 24 | ~282,000km | Lunar orbit scale |
| 27 | ~7.6M km | Inner solar system |
| 36 | ~15B km | Outer solar system (Pluto ~ 6B km) |
| 42 | ~4 light-years | Nearest star distance |
| 48 | ~1,100 light-years | A star cluster, a nebula |
| 54 | ~300,000 light-years | A galaxy (Milky Way ~ 100,000 ly) |
| 57 | ~8M light-years | A galaxy cluster |
| 60 | ~220M light-years | A supercluster |
| 63 | ~6B light-years | Observable universe scale |

63 = 3 x 21 (base-3 friendly). The path array is 63 bytes — fits just under a cache line. The total world resolution is 3^63 ~ 1.7 x 10^30 voxels per axis.

The absolute scale (1mm per layer-0 voxel) is a design parameter. What matters is the relative scale: a tree at layer 9 has 19,683 voxels per axis — enough to capture bark texture, individual leaves, and branch geometry from an imported GLB model. The mesh baking pipeline voxelizes imported 3D models at layer 0 resolution and builds the tree structure bottom-up, so every zoom level has correct LOD automatically.

The player's default gameplay layer is around layer 9 (walking among trees and buildings). At layer 21 they walk among planets. At layer 54 they walk among galaxies. Same code, same UX, same physics (scaled by cell size).

## Mesh Import Pipeline

3D models (GLB/glTF) are imported by voxelizing them at layer 0 resolution:

1. The model's bounding box is mapped to a target layer (e.g., a tree maps to layer 9 = 19,683 voxels per axis).
2. Each layer-0 voxel is tested against the model's triangles to determine occupancy and material.
3. The voxelized model is inserted into the tree bottom-up, with dedup at every level.
4. Meshes are baked at each layer automatically via the standard 27x27x27 grid pipeline.

This means any 3D model — a character, a vehicle, a building — can be imported at full geometric fidelity and will automatically have correct LOD at every zoom level. A statue imported from a high-poly GLB looks detailed up close (layer 3-6 shows individual chisel marks) and becomes a recognizable silhouette from far away (layer 12+ shows the overall shape), with no manual LOD authoring required.
