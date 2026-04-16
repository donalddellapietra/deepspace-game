# No Absolute Coordinates

This is the narrow companion to `docs/locality-prime-directive.md`.
The prime directive says runtime logic must stay local. This document
states the coordinate rule directly:

**Do not express deep positions or scales as floats relative to the
root.**

## Why

The tree subdivides by `3` each level. A depth-`d` cell has size:

`WORLD_SIZE / 3^d`

That is fine as a symbolic fact. It is not fine as a runtime `f32`
computation at deep depth.

- Around depth `15`, root-relative offsets become smaller than an
  `f32` can reliably distinguish near values like `1.5`.
- Around depth `20`, `3^depth` is already far outside the range where
  these scale computations remain meaningful for addressability.
- At deeper layers, converting a path into a single absolute float
  coordinate just destroys the very detail the path is supposed to
  preserve.

The path is the position. Floats are only allowed inside a local frame.

## Banned runtime patterns

These are all forbidden in gameplay/runtime code:

- Converting a deep `WorldPos` to absolute XYZ by walking from the root.
- Constructing a deep `WorldPos` from absolute XYZ by repeatedly
  decomposing from the root.
- Computing cell size as `WORLD_SIZE / 3^depth` for runtime decisions.
- Computing frame origin/size by accumulating from the root.
- Scaling a ray direction by an absolute frame-size ratio.

Examples of banned APIs:

- `to_world_xyz`
- `from_world_xyz`
- `cell_size`
- `frame_origin_size_world`

## Required runtime replacements

Use only local/path-anchored operations:

- `WorldPos::from_frame_local(frame, local, depth)`
- `WorldPos::deepened_to(depth)`
- `WorldPos::in_frame(frame)`
- `WorldPos::offset_from(other)`
- `zoom_in` / `zoom_out`

These are valid because they operate from a nearby frame or common
prefix. Their float work stays bounded by local visual depth, not total
tree depth.

## Construction rule

If you need a position at depth `N`:

1. Construct it in a shallow precise frame, usually the root or another
   nearby known frame.
2. Then deepen it symbolically with `deepened_to(N)`.

Do not feed absolute XYZ into a deep constructor.

## Projection rule

If you need to compare two things:

- project one into the other's frame with `in_frame`, or
- subtract them with `offset_from`.

Do not route the comparison through root-relative world coordinates.

## WORLD_SIZE rule

`WORLD_SIZE = 3.0` is a local convention: each node's children occupy
`[0, 3)` on each axis. It is not a license to compute absolute world
meters at arbitrary depth.
