# Coordinates and positions

## Model

The world is a tree of voxel nodes (see `voxels.md`). Every position in
the game is **always at the leaf layer**. Zooming the camera in or out
shows different layers of the tree, but the player, entities, physics,
and collision all live at the leaf. This is "Model A" — a single
coordinate system.

The tree's depth is fixed at world creation. No re-rooting, no dynamic
growth. Pick a `MAX_LAYER` large enough for the world you want and be
done with it.

## Layer numbering

- **Layer 0 = root** (coarsest — one big node covering the whole world).
- **Layer MAX_LAYER = leaves** (finest — individual voxel nodes).
- A node's layer number equals the length of its path from the root:
  `layer = path.len()`. Zero arithmetic, no offset.
- Adding more layers to a world extends downward into larger layer
  numbers. Layer 0 stays layer 0 forever.

Design choice: `MAX_LAYER = 12`. That gives `25 × 5^12 ≈ 6 billion`
voxels per axis — effectively unbounded for a single-origin space game,
and still well inside `f32` / `u64` / `SmallVec` budgets.

## Position type

```rust
pub const MAX_LAYER: u8 = 12;

/// A position in the world. Always leaf-layer.
pub struct Position {
    /// Slot indices from the root to the leaf node. Length is always
    /// exactly `MAX_LAYER`. Each slot is 0..125.
    path: SmallVec<[u8; MAX_LAYER as usize]>,
    /// Which voxel inside the leaf node's 25³ grid. Each 0..25.
    voxel: (u8, u8, u8),
    /// Sub-voxel offset inside that voxel. Each 0.0..1.0.
    offset: (f32, f32, f32),
}
```

Invariants:
- `path.len() == MAX_LAYER` always.
- `voxel.x, voxel.y, voxel.z < 25`.
- `0.0 <= offset.x, offset.y, offset.z < 1.0`.

Nothing is ever a "big number." Every component is bounded.

## Slot encoding

A child's slot index in its parent's 5³ child array is:

```rust
pub const fn slot_index(x: u8, y: u8, z: u8) -> u8 {
    z * 25 + y * 5 + x
}

pub const fn slot_coords(slot: u8) -> (u8, u8, u8) {
    (slot % 5, (slot / 5) % 5, slot / 25)
}
```

This is the one encoding used everywhere. It's arbitrary but must be
consistent.

## Camera is separate

The camera carries its own "which layer am I showing":

```rust
pub struct CameraZoom {
    /// Which tree layer the camera renders. 0..=MAX_LAYER.
    /// Clamped to a UX-friendly range (e.g. 2..=MAX_LAYER-2) so the
    /// player never zooms out to "one voxel" or into sub-voxel noise.
    layer: u8,
}
```

The camera's `layer` is a view state. Changing it does not change any
`Position`. "Drilling" / "zooming" is purely a camera operation.

## Walking

Movement is expressed as a delta in leaf-voxel-space, applied per axis.
A step is split into three carries:

1. **Offset carry.** `offset.x += dx`. If `offset.x` leaves `0.0..1.0`,
   carry into `voxel.x`.
2. **Voxel carry.** If `voxel.x` leaves `0..25`, carry into `path` — a
   **neighbor walk** across leaf node boundaries.
3. **Path carry.** If the neighbor walk needs to cross the root, the
   walk returns `None` — the edge of the world. No re-rooting happens.

### Neighbor walk

To find the leaf node to the `+x` of the current one:

1. If `voxel.x` is still inside `0..25`, no walk needed.
2. Pop the last slot off `path`. Compute `slot_coords(popped)` → 3D.
3. If the popped slot's x is `< 4`, increment it, push the new slot,
   set `voxel.x = 0`.
4. If the popped slot's x is already `4`, recurse: the walk needs to
   cross the parent's x boundary too. Pop again, repeat.
5. When re-descending, all popped layers pick the opposite-axis slot
   (`x = 0` when crossing `+x`) with the same `(y, z)` slot indices
   as on the way up.

Worst case walks all the way to the root — `O(MAX_LAYER)`. Normal case
is one step.

Diagonal movement is decomposed into independent per-axis steps. Each
axis does its own carry/walk.

## Worked example

Setup: `MAX_LAYER = 3` (small, for readability).

### Start at a leaf

```
path   = [72, 37, 16]
voxel  = (0, 5, 20)
offset = (0.0, 0.0, 0.0)
```

`path.len() == 3 == MAX_LAYER` ✓. We're at the leaf.

### Walk +x by 0.5 leaf voxels

`offset.x` goes `0.0 → 0.5`. No carry.

```
path   = [72, 37, 16]
voxel  = (0, 5, 20)
offset = (0.5, 0.0, 0.0)
```

### Walk another +0.7

`offset.x = 1.2` — carries. `voxel.x = 1`, `offset.x = 0.2`.

```
path   = [72, 37, 16]
voxel  = (1, 5, 20)
offset = (0.2, 0.0, 0.0)
```

### Walk far enough to cross the leaf node (+24 voxels)

`voxel.x` walks up to `24`. One more step → `voxel.x = 25`. Carry into
path.

- Pop `16`. `slot_coords(16) = (1, 3, 0)`.
- x is `1 < 4`, increment to `2`. New slot = `slot_index(2, 3, 0) = 17`.
- Push `17`. Reset `voxel.x = 0`.

```
path   = [72, 37, 17]
voxel  = (0, 5, 20)
offset = (0.2, 0.0, 0.0)
```

### Keep walking until we cross slot 37 too

After four more leaf-node crossings, we walked through slots
`17 → 18 → 19 → 20` — wait, `slot_index(4, 3, 0) = 0*25 + 3*5 + 4 = 19`,
and `(5, 3, 0)` doesn't exist. So the sequence is `17 → 18 → 19`, then
we try to cross the parent too.

- Current last slot `19 = (4, 3, 0)`. x is already `4`. Recurse.
- Pop `19`. Look at the next slot up: `37 = (2, 2, 1)`. x is `2 < 4`,
  increment to `3`. New slot = `slot_index(3, 2, 1) = 38`.
- Push `38`, then re-descend. The new leaf we enter picks `x = 0`
  (opposite face) and keeps `(y, z) = (3, 0)` from the popped slot.
  New last slot = `slot_index(0, 3, 0) = 15`.
- Push `15`. Reset `voxel.x = 0`.

```
path   = [72, 38, 15]
voxel  = (0, 5, 20)
offset = (0.2, 0.0, 0.0)
```

That was a two-level neighbor walk: we crossed a layer-3 node boundary
*and* a layer-2 node boundary in the same step. Still `O(MAX_LAYER)`.

### Camera zoom

The player's `Position` above never changes when the camera zooms. The
camera carries its own `zoom.layer: u8`; incrementing it shows a deeper
node centred on the player; decrementing shows a shallower one. All
rendering does is walk the tree to `zoom.layer` and emit entities for
the nodes it finds there.

## Why this never overflows

- `voxel.x, y, z < 25` → `u8`.
- `offset.x, y, z ∈ [0, 1)` → `f32` with no precision drift, ever.
- `path` slots are each `< 125` → `u8`.
- `path.len() == MAX_LAYER == 12` → `SmallVec<[u8; 12]>`, always
  stack-allocated.
- No coordinate spans more than one tree layer. No integer ever grows
  past `125`. No float ever leaves `0..1`.

## Not covered here

- **Editing and propagation.** How edits flow up the tree and keep
  voxels + meshes consistent at every layer. See `editing.md`.
- **Rendering LOD.** How the renderer picks which layer to emit
  entities at. Separate document.
