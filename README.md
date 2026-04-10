# Deep Space

A hierarchical voxel game where every block contains a world within it.

Built with [Bevy 0.18](https://bevyengine.org/) in Rust.

## Concept

Voxels exist at multiple layers of abstraction. At Layer 0, you place individual blocks (stone, dirt, grass, etc.). Zoom out to Layer 1 and each "block" is actually a 5x5x5 model you built from Layer 0 blocks. Zoom out further and those models become blocks at Layer 2 — a 5x5x5 grid of models, each containing 5x5x5 blocks.

You can drill into any block to edit its interior, then drill back out. At every layer, the game plays like Minecraft: walk, jump, break blocks, place blocks. But the blocks themselves are worlds.

## Controls

| Key | Action |
|-----|--------|
| WASD | Move |
| Mouse | Look |
| Space | Jump |
| Shift | Sprint |
| Left click | Break block / cell |
| Right click | Place block / cell |
| 1-0 | Select hotbar slot |
| E | Open / close inventory |
| F | Drill into targeted block (zoom in one layer) |
| Q | Drill out (zoom back to parent layer) |
| P | Save current cell as a reusable model template |
| Escape | Release cursor |

## Running

```bash
# First time (compiles all dependencies, takes a few minutes)
cargo run

# Fast incremental builds with dynamic linking
cargo run --features dev
```

Requires Rust 1.85+ (edition 2024).

## Testing

```bash
cargo test
```

22 automated tests covering collision physics, grid navigation, raycasting, rendering math, and model save/load.

## Architecture

```
src/
  block/           Block types (10) and per-type PBR materials
  model/           Model templates, face-culled mesh baking
  layer/           Navigation stack for N-layer drill-in/out
  world/
    mod.rs         VoxelGrid, VoxelWorld, recursive rendering
    collision.rs   AABB clipping collision (Minecraft/Quake algorithm)
  editor/          Hotbar, block placement/removal, drill transitions
  interaction/     DDA voxel raycasting, block highlighting
  inventory.rs     Inventory panel UI
  camera.rs        First-person camera, cursor management
  player.rs        Movement, gravity, collision integration
```

### Data Model

- `CellSlot::Block(BlockType)` — a solid block at the lowest level
- `CellSlot::Child(Box<VoxelGrid>)` — a nested 5x5x5 grid (drill into this)
- `CellSlot::Empty` — air

The world is a `HashMap<IVec3, VoxelGrid>` at the top layer. Each `VoxelGrid` contains a `[[[CellSlot; 5]; 5]; 5]` array. The structure is recursive — a `Child` slot contains another `VoxelGrid` whose slots can themselves be `Child` grids.

### Collision

Standard AABB clipping algorithm. For each axis (Y, X, Z):
1. Collect all solid blocks near the player
2. Clip the movement delta against each block face
3. Apply the clipped (safe) movement

`block_solid()` walks the entire navigation stack to detect solids at any ancestor layer — ground from the top-layer HashMap is visible even when drilled 3+ layers deep.

### Rendering

When inside a cell, the current grid's blocks render as individual cubes. `render_ancestors` walks up the navigation stack and renders every ancestor layer's cells as baked meshes at the correct offset and scale. The full world is always visible at every depth.

## Development Log

See [docs/progress.md](docs/progress.md) for the full development history, architectural decisions, rewrites, and lessons learned.

## License

MIT
