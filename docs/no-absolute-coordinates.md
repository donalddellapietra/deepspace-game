# No Absolute Coordinates

Companion to `locality-prime-directive.md`. That document says all
runtime decisions must be local. This document explains WHY — and
bans the specific patterns that break at depth.

## The math

The world tree is 63 levels deep. Each level subdivides by 3. The
ratio between the root cell and a depth-63 cell is `3^63 ≈ 1.7e30`.

An `f32` has ~7 decimal digits of precision. That means:

- At depth **15**, a cell is `3^(-15) ≈ 7e-8` of the root. An f32
  centered at 1.5 (root midpoint) can't distinguish two points inside
  the same depth-15 cell. The offset is below the ULP.
- At depth **20**, `3^20 ≈ 3.5e9` overflows f32's integer-exact
  range. Scale factors become approximate.
- At depth **40**, `3^40 ≈ 1.2e19`. No f32 operation involving this
  number is meaningful.

**No single float can address the world.** The path IS the position.

## What's banned

Any computation that accumulates from the root to a deep node:

```rust
// BROKEN: origin drifts past f32 precision at depth > ~15
let mut origin = [0.0f32; 3];
let mut size = WORLD_SIZE;
for slot in path.slots() {
    let child = size / 3.0;
    origin += slot_coords(slot) * child;
    size = child;
}
```

```rust
// BROKEN: 3^depth overflows useful f32 range past depth ~20
let scale = 3.0f32.powi(path.depth() as i32);
```

```rust
// BROKEN: same accumulation, packaged as a method
let world_xyz = pos.to_world_xyz();
let cell_size = pos.cell_size();  // = WORLD_SIZE / 3^depth
let (origin, size) = frame_origin_size_world(&path);
```

These all share the same flaw: they express a deep position or scale
as a number relative to the root. Past depth ~15, the result is
noise.

## What's correct

Operations that start from the **common ancestor** of the two things
being compared, not from the root:

```rust
// CORRECT: in_frame walks from the common prefix
// If the camera and the frame share 37 levels of path,
// the walk is only 3 levels — f32 values stay in [0, 3).
let local = camera.position.in_frame(&frame_path);
```

```rust
// CORRECT: offset_from walks from the common prefix
// Precision scales with proximity, not absolute depth.
let delta = a.offset_from(&b);
```

```rust
// CORRECT: zoom_in/zoom_out are pure slot arithmetic
// Each step: slot = floor(offset * 3), offset = fract(offset * 3)
// No accumulation from root. Precision is always [0, 1).
pos.zoom_in();
```

```rust
// CORRECT: from_frame_local builds from a known frame
// Walks from the frame (not root) to the target depth.
let pos = WorldPos::from_frame_local(&frame, local_xyz, depth);
```

## The rule

Every coordinate operation must work between two **nearby** reference
points: a camera and its frame, two positions in the same
neighborhood, a position and its parent cell. The number of tree
levels traversed in any single computation must be bounded by the
**visual depth** (the levels the screen can actually show, ~7-15),
never by the absolute tree depth.

If a function takes a `Path` and walks it from slot 0 to
`slot[depth-1]`, accumulating floats, it is broken. If a function
computes `3^depth` or `WORLD_SIZE / 3^depth`, it is broken.

The only valid root-to-leaf walk is the **symbolic** one: iterating
path slots to look up tree nodes by index. That's integer work —
no float accumulation, no precision loss.

## Constructing positions at arbitrary depth

To place something at depth 40, do NOT pass absolute coordinates
through `from_world_xyz(xyz, 40)` — the float decomposition produces
garbage slots past depth ~15.

Instead:

1. Construct the position at a shallow depth where floats are precise
   (depth 1-8).
2. Use `deepened_to(40)` to push the anchor deeper. Each zoom step
   is pure slot arithmetic (`floor(offset * 3)`) — no root-relative
   accumulation.

Or construct the path symbolically (slot by slot) if you know the
exact cell you want.

## WORLD_SIZE is not an absolute unit

`WORLD_SIZE = 3.0` is the local frame convention: every node's
children span `[0, 3)` on each axis because there are 3 children per
axis. This is a local coordinate convention, not a world-scale
measurement. Code that uses `WORLD_SIZE` as a frame-local constant
(e.g., `in_frame` mapping to `[0, WORLD_SIZE)`) is correct. Code
that uses it to compute absolute sizes (`WORLD_SIZE / 3^depth`) is
not.
