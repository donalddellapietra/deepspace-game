# Zoom

## What Zoom Is

Zoom is not a camera operation. It is tree navigation. The ray marcher always renders full visual detail (automatic per-pixel LOD). Zoom changes what the player can **interact with** — the scale of blocks they break, the grid they collide with, and how fast they move.

## Controls

- **Q (zoom out):** View layer increases by 1. The player ascends the tree. Gameplay blocks get coarser (each covers 3× more world per axis). Movement speed scales up proportionally. The player effectively becomes a giant.

- **E (zoom in):** View layer decreases by 1. The player descends into a child node. Gameplay blocks get finer. Movement speed scales down. The player effectively becomes tiny.

## What Changes on Zoom

| System | Effect |
|--------|--------|
| Interaction layer | Blocks the player breaks/places are at layer N-3 (one cell of the 27³ grid) |
| Collision layer | Solid/empty checks are at layer N-4 (one layer below gameplay blocks) |
| Movement speed | walk_speed × cell_size_at_layer(N). Same cells/second at every zoom. |
| Gravity | gravity × cell_size_at_layer(N). Same jump height in cells at every zoom. |
| Player AABB | Scales with cell size. Always 0.3 × 1.7 cells. |

## What Does NOT Change on Zoom

| System | Why |
|--------|-----|
| Visual detail | The ray marcher always renders to per-pixel LOD. Zooming out doesn't make things look worse — things just get smaller on screen, and the ray stops descending earlier because cells are sub-pixel sooner. |
| Render performance | Same number of pixels, similar number of ray steps. The LOD cutoff adjusts automatically. |
| Tree data | No entities spawned or despawned. No mesh invalidation. The GPU buffer may need different nodes loaded, but the streaming system handles this. |

## Zoom Transition

On zoom change:

1. Snap the player to ground at the new collision layer (the grid changed, the player might be inside a block or floating).
2. Zero vertical velocity (prevent residual velocity from the old scale launching the player).
3. Optionally animate the camera height from the old cell size to the new one (cosmetic smoothing).

## NavStack

The NavStack records which child the player entered when zooming in, so they can zoom back out to the same position:

```rust
struct NavEntry {
    child_index: u8,  // which of 27 children we entered
}

struct NavStack {
    entries: Vec<NavEntry>,
}
```

- **E (zoom in):** Push the current child index onto the NavStack. The player enters that child.
- **Q (zoom out):** Pop the NavStack. The player returns to the parent node at the recorded child position.

If the NavStack is empty and the player presses Q, they're at the highest zoom level. No-op (or the maximum layer is capped).

## Why Zoom Exists If Rendering Is Automatic

The ray marcher renders everything at correct detail regardless of zoom. So why have zoom at all?

Because the game is about **interacting** at different scales. At layer 9, you place blocks to build a house. At layer 12, you place trees to build a forest. At layer 18, you place continents to shape a planet. At layer 54, you arrange galaxies.

The visual is always correct. Zoom controls what "one click" means — what scale of thing you're manipulating. It's the difference between a brush size in Photoshop and the zoom level of the canvas. Both matter independently.
