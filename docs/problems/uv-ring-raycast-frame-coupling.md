# UV Ring Raycast / Frame Coupling

## Problem

Exterior UV-ring raycast edits can land far from the mouse, sometimes looking like the edited subframe content has been rotated about 90 degrees. The camera transition itself should not depend on raycast, but raycast returns edit paths, and those paths are later interpreted by render/upload/subframe systems. If raycast builds a valid path in the wrong coordinate basis, the edit is valid but spatially wrong.

The current code is vulnerable because UV chart coordinates and tangent-local cell coordinates are both represented as plain `x/y/z` triples and tree slots.

## Coordinate Spaces

There are at least two separate spaces:

- Macro UV-ring chart coordinates:
  - `x`: angular/longitude cell
  - `y`: radial shell
  - `z`: height/latitude row

- Tangent-local subframe coordinates:
  - `x`: tangent direction around the cylinder/sphere
  - `y`: radial direction
  - `z`: up/height direction

At the macro cell level these look related, but they are not interchangeable below the selected UV-ring cell.

## Why Recursive UV Descent Broke It

The bad raycast change continued to recursively subdivide in UV/polar coordinates after selecting a macro UV-ring cell:

```text
theta -> child x
rho   -> child y
height -> child z
```

That keeps recomputing child slots from global curved coordinates.

The subframe path, however, enters a selected UV-ring cell and treats deeper child slots as tangent-local Cartesian coordinates:

```text
local x -> tangent
local y -> radial
local z -> up
```

Those are not the same basis. In particular, radial and tangent directions are 90 degrees apart:

```text
radial  = [ cos(theta), 0, sin(theta)]
tangent = [-sin(theta), 0, cos(theta)]
```

So recursive UV descent can produce a structurally valid edit path whose deeper slots are selected in the wrong basis. Rendering may still show the edit consistently in both views, but the block that changed is not the block under the exterior mouse ray.

## Immediate Fix Direction

Separate exterior ray traversal from UV-ring frame transformation.

Exterior raycast should:

1. Walk the curved UV-ring shell only to select the macro cell:
   - `cell_x`
   - `cell_y`
   - `cell_z`

2. Use the shared UV-ring frame transform for that macro cell to convert the exterior ray into tangent-local coordinates.

3. Run normal Cartesian raycast inside the selected cell subtree.

4. Prefix the macro-cell path to the Cartesian child hit path.

The raycast module should not own an independent copy of UV-ring tangent/radial/up math. It should call the same frame conversion code used by subframe transition/rendering.

## Cleanliness / Guardrail Fix

Move pure UV-ring geometry and frame math out of `src/app/uv_ring.rs` into a shared module, for example:

```text
src/world/uv_ring.rs
```

That module should own:

- `UvRingCellFrame`
- `uv_ring_cell_frame`
- `uv_ring_cell_frame_at_local_x`
- `uv_ring_cell_path`
- `uv_ring_cell_coords_from_path`
- root/ring/world to tangent-local conversion helpers

Then:

- app/subframe code uses the shared module
- CPU raycast uses the shared module
- renderer-facing code mirrors the same definitions

Later, add stronger type names for coordinate spaces instead of raw triples:

```text
UvCellCoord
TangentLocalPoint
TangentLocalDir
TreeSlotCoord
```

This avoids accidentally feeding UV/polar coordinates into tangent-local Cartesian code.

## Test Strategy

Uniform subtrees hide this bug. Tests need asymmetric content inside a selected macro cell.

Useful regressions:

- Put a block only in a known tangent-local child slot, cast from exterior at that visible subcell, and assert the returned path ends in that slot.
- Repeat for tangent, radial, and up offsets.
- Add a test that exterior raycast uses the same shared `UvRingCellFrame` conversion as subframe entry.

The tests should not assert that exterior and interior raycasts are globally identical. They start from different frames. The invariant is narrower: after exterior raycast selects a macro UV-ring cell, child descent must use the same tangent-local coordinate interpretation as the subframe.
