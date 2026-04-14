# Coordinates

## The Path IS the Coordinate

There are no absolute coordinates. No i64 leaf coordinates. No global (x, y, z). The path through the tree IS the position:

```rust
struct Position {
    path: [u8; 63],    // slot indices, bottom-up
    depth: u8,         // how many levels are resolved
    offset: [f32; 3],  // sub-voxel fractional position, each in [0.0, 1.0)
}
```

This is exact at every scale. There is no floating-point accumulation, no precision loss at large distances, no need for a floating anchor to keep numbers small. A position at the center of a galaxy and a position at the center of a grain of sand are both 63 bytes + 12 bytes of offset. Neither is "bigger" or less precise than the other.

## Why No Absolute Coordinates

The old architecture converted paths to i64 leaf coordinates for Bevy-space math. This had two problems:

1. **Overflow.** 3^63 ~ 1.7 × 10^30 voxels per axis. i64 max is 9.2 × 10^18. It doesn't fit. i128 would work but is awkward and slow.

2. **Unnecessary.** Absolute coordinates exist to answer "how far apart are two things." But in a recursive tree, distance is relative to the layer you're viewing from. Two trees 10 meters apart are "close" at layer 9 but "the same point" at layer 18. The path already encodes this — two positions that share a long common prefix are close; positions that diverge early in their paths are far.

## Camera Position

The camera's position is a `Position` — a path in the tree. To render, the GPU needs the camera in tree-local coordinates. This is computed per-node:

For each node the ray enters, the camera's position within that node's 3×3×3 grid is:

```
local_pos = (camera path diverges from this node at which child?)
```

The camera's offset within the deepest shared node gives the local floating-point position for the ray origin. This is always small (0.0 to 3.0 per axis within a 3×3×3 grid), so f32 precision is perfect.

## Distance and Proximity

Two positions are "near" if their paths share a long common prefix. The length of the shared prefix tells you the scale of their separation:

- Same path down to depth 60: separated by at most 3^3 = 27 layer-0 voxels (~27mm)
- Same path down to depth 50: separated by at most 3^13 ~ 1.6M voxels (~1.6km)
- Diverge at depth 10: separated by at most 3^53 voxels (~galactic scale)

For view-radius culling, the renderer compares the camera's path against each node's path. If they diverge above a certain depth, the node is too far to render at this LOD level.

## Movement

The player moves by modifying their `Position`:

1. Add the movement delta to `offset`.
2. If `offset` overflows [0, 1) on any axis, carry into the path: step to the neighboring child at the appropriate depth.
3. The `step_neighbor` operation walks up the path until it finds an ancestor where the step doesn't overflow, then walks back down resetting lower slots.

This is O(depth) in the worst case (crossing a high-level boundary) but O(1) in the common case (moving within the same deepest node). No coordinate conversion, no floating-point accumulation.

## Bevy Transform

The player needs a Bevy `Transform` for the camera. This is derived from the `Position`:

```
transform.translation = position.offset  // always in [0, 1)³
```

The world moves around the player, not the player through the world. The renderer (ray marching) takes the camera's `Position` and computes rays directly from the path. The Bevy `Transform` is only used for the camera's local orientation (yaw/pitch), not for world-space positioning.

## Coordinate Comparison with Old Architecture

| | Old (sphere-planet-clean) | New |
|---|---|---|
| Position | path + voxel + offset + i64 leaf coord | path + offset (no leaf coord) |
| Precision | i64 (overflows at 63 layers) | exact at any depth |
| Anchor | WorldAnchor (i64 leaf coord, updated per frame) | none needed |
| Bevy translation | (leaf_coord - anchor) as f32 | offset only, always tiny |
| Distance | i64 subtraction | path prefix comparison |
| Movement | f32 delta → i64 leaf → Position | f32 delta → offset carry → path step |
