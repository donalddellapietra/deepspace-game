# Collision

## How It Works

Collision operates on the immediate children of the player's current gameplay blocks — one layer below the interaction layer. This gives Minecraft-style swept-AABB collision at sub-block resolution.

The player is at view layer N. Their gameplay blocks are at layer N-3 (the cells of the 27×27×27 grid). Collision samples one layer below that: layer N-4.

At layer N-4, each gameplay block is subdivided into 3×3×3 = 27 sub-cells. The collision system checks solidity of these sub-cells for the swept-AABB algorithm.

## Per-Frame Collision

Each frame:

1. Compute the player's AABB (half-width, height) in cell units at the gameplay layer.
2. Expand the AABB by the movement delta to find the sweep region.
3. For each gameplay block in the sweep region (~20-50 blocks), read its 27 children to determine which sub-cells are solid.
4. Run per-axis swept clipping (Y first, then X, then Z) against the solid sub-cells.
5. Update the player's `Position` via offset carry / path stepping.

Reading a block's 27 children is one node lookup in the library. For ~40 blocks in the sweep region, that's ~40 node lookups per frame. Trivial.

## Solidity Check

A sub-cell (child of a gameplay block) is solid if:
- It is `Block(type)` where `type` is a solid block type (not air, not water)
- It is `Node(id)` (a non-terminal child has structure inside it, treat as solid)

Empty children are passable. This gives correct collision at the sub-block level without resolving the full tree depth.

## No Voxel Grid Needed

The old architecture resolved a 25×25×25 voxel grid for collision sampling. The new architecture reads children directly from nodes. No intermediate grid. Each solidity check is one child lookup in one node — an array index into 27 elements.

## On-Ground Check

To determine if the player is standing on something:

1. Probe slightly below the player's feet (small negative Y delta).
2. Check if any sub-cell in that probe region is solid.
3. If the probe clips against a solid sub-cell, the player is on the ground.

Same algorithm as the current `on_ground`, just reading children instead of a voxel grid.

## Snap-to-Ground on Zoom Change

When the player zooms out (Q), the gameplay blocks get coarser. A thin floor that was solid at layer N-4 might not exist at layer N-5 (the sub-cells are larger, the floor might be "inside" a sub-cell rather than on top of one). The player could fall through.

On zoom change:
1. Check if the player's feet are inside a solid sub-cell at the new layer.
2. If yes, push the player upward until they're in an empty sub-cell.
3. If the player is floating (no solid below), drop them down until they hit ground.

This is the same snap-to-ground behavior as the current code, just operating on children instead of a voxel grid.

## Scale-Invariant Physics

Movement constants are in cells-per-second at the gameplay layer:

- Walk: 8 cells/s
- Sprint: 16 cells/s
- Jump: 8 cells/s impulse
- Gravity: 20 cells/s²

The cell size at the gameplay layer scales with zoom. At layer 9, a cell is ~20m / 27 ≈ 0.74m. At layer 12, a cell is ~530m / 27 ≈ 20m. The player crosses one cell in the same wall-clock time regardless of zoom.
