# Content Pipeline

## Overview

3D models (GLB/glTF) are imported into the tree through offline voxelization. The output is a single NodeId that can be placed anywhere in the world as a child of any node. LOD, streaming, and rendering come for free from the tree structure.

## Step 1: Choose Resolution

The target resolution must be a power of 3 to map cleanly to tree layers:

| Resolution | Layers | Voxels | Good for |
|-----------|--------|--------|----------|
| 27³ | 3 | 19K | A simple block, an icon |
| 81³ | 4 | 531K | Furniture, a rock |
| 243³ | 5 | 14M | A detailed statue, a small vehicle |
| 729³ | 6 | 387M | A tree with individual leaves |

The choice depends on the object's visual complexity. A smooth boulder needs less resolution than a fern with individual fronds.

## Step 2: Voxelize

For each voxel in the 3D grid, determine if it intersects the mesh and what material it has.

### Algorithm

Rasterize each triangle into the 3D grid, like 2D rasterization but extended to 3D:

1. For each triangle in the mesh, compute its axis-aligned bounding box in voxel coordinates.
2. For each voxel in that bounding box, test if the voxel overlaps the triangle (point-in-triangle test on the closest face, or triangle-box intersection).
3. On hit, sample the triangle's material/texture at the intersection point.
4. Map the sampled color to the nearest BlockType.

### Material Mapping

The voxelizer maps mesh materials to the game's BlockType enum. This can be:

- **Automatic:** Map the material's base color to the nearest BlockType by color distance. Brown → Wood, green → Leaf, gray → Stone.
- **Tagged:** The 3D model's material names map to block types. A material named "bark" maps to Wood, "leaves" maps to Leaf. This gives artists control.
- **Palette-based:** The artist paints the model using a palette of exact BlockType colors. The voxelizer matches colors exactly.

### Handling Interiors

A naive surface voxelization leaves the inside of the mesh hollow. For solid objects (rocks, logs), a flood fill from outside marks all unreached interior voxels as solid. The fill uses the dominant material of the surrounding surface.

For intentionally hollow objects (a house, a barrel), the artist leaves an opening or tags the mesh as hollow. No interior fill.

## Step 3: Build the Tree

Convert the flat voxel grid into a base-3 tree, bottom-up:

```
Layer 0: each voxel → Child::Block(type) or Child::Empty
Layer 1: group 3×3×3 = 27 adjacent layer-0 children → insert as one Node
Layer 2: group 3×3×3 = 27 adjacent layer-1 nodes → insert as one Node
...
Layer N: single root Node
```

Each `insert` goes through the NodeLibrary's content-addressed dedup. Identical subtrees (all-empty air, uniform solid stone) collapse to a single NodeId. A tree mesh that is 90% air stores only the trunk and canopy nodes — the air is represented once.

## Step 4: Output

The output is a single NodeId — the root of the imported structure. This NodeId is placed into the world by assigning it as a child in a parent node:

```rust
parent.children[slot] = Child::Node(imported_root_id);
```

A forest is built by placing different tree NodeIds into a parent's 27 child slots. A city is built by placing different building NodeIds. The content pipeline produces the vocabulary; world authoring arranges it.

## Batch Import

The `gen_world` binary runs the content pipeline for all assets:

1. Voxelize each GLB model at its target resolution.
2. Insert all resulting nodes into a shared NodeLibrary (dedup across assets — two trees with the same trunk pattern share trunk nodes).
3. Compose structures: place imported NodeIds into parent nodes to build forests, cities, planets.
4. Serialize the NodeLibrary + root NodeId to disk.

## Performance

Voxelization is offline and needs no real-time performance:

| Resolution | Typical time | Output size (deduped) |
|-----------|-------------|----------------------|
| 27³ | <1 second | ~100 nodes, ~22KB |
| 81³ | ~1 second | ~1,000 nodes, ~220KB |
| 243³ | ~5 seconds | ~10,000 nodes, ~2.2MB |
| 729³ | ~30 seconds | ~50,000 nodes, ~11MB |

Output sizes are after content-addressed dedup. Sparse objects (mostly air) are much smaller. Dense objects (solid rock) are also small (uniform regions dedup).
