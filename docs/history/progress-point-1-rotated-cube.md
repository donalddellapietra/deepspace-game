# Progress Point 1 — Rotated Cartesian Cube Primitive

**Branch**: `sphere-mercator-1-2-2`
**Commit at this milestone**: `7bad504`
**Test world**: `--rotated-cube-test`

## Summary

A single rotated cartesian cube renders correctly inside an
otherwise pure-Cartesian world tree at any depth and any zoom
level, with no spherical math anywhere in the render path. Rotation
is data on the node, applied uniformly through the existing
cartesian DDA — there is no "outside vs inside" distinction.

The visible test scene is three sibling depth-29 uniform-stone
subtrees in a row at the deepest layer of a 30-deep tree:

- Slot 12: plain Cartesian (axis-aligned A)
- Slot 13: `TangentBlock { rotation: rot_y(π/4) }` (rotated centre)
- Slot 14: plain Cartesian (axis-aligned B)

From above, looking down with pitch −π/2, the row reads as
**square / diamond / square**. Zooming in (the renderer's frame
descends past the `TangentBlock` and the active frame becomes a
rotated subtree) keeps the diamond rotated relative to camera
flight — no abrupt 45° flip across the boundary.

## Architecture

### Rotation is data on the node

`NodeKind::TangentBlock { rotation: [[f32; 3]; 3] }` carries a
column-major 3×3 matrix that maps the block's LOCAL `[0, 3)³`
frame to its PARENT slot's frame. `NodeKind::Cartesian` is the
implicit identity-rotation case. Two TBs with bit-distinct
rotations do not dedup; bit-identical rotations dedup as usual via
custom `PartialEq` + `Hash` using `f32::to_bits`.

`IDENTITY_ROTATION` and `rotation_y(radians)` helpers live in
`src/world/tree.rs`.

### No world-absolute coordinates

Every operation that touches positions stays in a `[0, 3)³`
frame-local coordinate system at every level — at root, in any
descendant, inside the rotated cube interior. No code multiplies
slot offsets by `3^N` to express positions in a global "world"
frame. The frame-local invariant is what makes precision survive
arbitrary depth.

### Dispatch at child descent

Inside `march_cartesian` (shader and CPU mirror), when the DDA
descends into a child whose `NodeKind` is `TangentBlock`:

```text
scale = 3 / cur_cell_size                    # parent slot extent → child [0, 3)³
local_pre  = (ray - child_origin) * scale    # ray in unrotated child frame
centred    = local_pre - vec3(1.5)           # relative to cube centre
rotated    = R^T · centred                   # apply inverse rotation
local_origin = rotated + vec3(1.5)           # back to [0, 3)³ frame
```

`R^T` is constructed inline from the three `vec4` columns the
`GpuNodeKind` carries (xyz column + w-pad for std140 alignment).
`R · sub.normal` rotates the local-frame surface normal back to the
parent's frame on hit.

The dispatch fires UNCONDITIONALLY before the LOD/at_max check —
the rotation is part of the descent geometry, not a special-cased
LOD splat.

### Camera position is rotation-aware

`WorldPos::in_frame_rot(library, world_root, frame)` walks the
anchor path with rotation awareness: at each step it tracks both
the cell's CENTRE in common-ancestor coords and the cumulative
rotation, applying the rotation around each cube's centre when
adding the next slot's centred offset and the final fractional
offset. For all-Cartesian paths the result is bit-identical to
plain `in_frame`.

Wired into `App::gpu_camera_for_frame` and the entity-bbox helper
in `edit_actions/upload.rs`. Without this, the camera's reported
world position diverged the moment the anchor crossed a TB.

### Camera direction follows the frame chain

When the active render frame is INSIDE a rotated subtree (the
frame's path crosses a TangentBlock), the frame's local axes are
rotated relative to world. `frame_path_rotation` walks the frame
path from world root, accumulates the cumulative `R`, and the
camera basis (`forward / right / up`) is converted via `R^T` before
upload. This is what makes the cube stay rotated across the
zoom-in transition where `render_path` deepens from `[13]` (frame
= wrapper, TB is a child the DDA dispatches on) to `[13, 13]`
(frame IS the TB; no descent event would have fired).

For all-Cartesian frame paths the rotation is identity and the
basis is unchanged.

## What was deleted to get here

- The entire spherical render path: `march_wrapped_planet`,
  `sphere_uv_in_cell`, `sphere_descend_anchor`, `make_sphere_hit`,
  `cpu_raycast_wrapped_planet`, `ray_meridian_t`, `ray_parallel_t`,
  `ray_sphere_after`. Net ~1500 LoC of compounding-precision
  spherical primitives gone.
- The `--planet-render-sphere` mode flag. WrappedPlane just renders
  through cartesian DDA now; if the world has rotation, the data
  drives it via `TangentBlock`.

## Debug tooling

The held-`]` debug overlay (`src/app/event_loop.rs` +
`ui/src/components/DebugOverlay.tsx`) shows everything needed to
read off "where is the camera and is it in a rotated chain":

- **zoom**: tree/edit/visual/anchor depths, anchor cell width
- **camera**: rotation-aware root XYZ, frame-local XYZ, fov
- **frame**: active kind, render path slots, anchor path slots
- **rotation**: `TB on anchor path` yes/no, cumulative yaw degrees

`[` (with overlay open) copies the current overlay state to the
clipboard as plain text — Rust handler bumps a counter, UI watches
it and runs `navigator.clipboard.writeText(...)`. Plumbed this way
because the webview forwards keystrokes to Rust before any JS
listener fires, so a vanilla `keydown` on `window` was unreliable.

## Test world

`--rotated-cube-test` (`WorldPreset::RotatedCubeTest`) builds the
3-sibling row described above. `CUBE_SUBTREE_DEPTH = 29` is the
single knob; total tree depth is `CUBE_SUBTREE_DEPTH + 1 = 30`.
Library size ≤ 32 — uniform stone at every level dedups across all
three subtrees plus the centre's TB-wrapped interior.

Camera spawns at anchor depth 1, slot 16 = `(1, 2, 1)` of root
(the cell directly above the row), offset `(0.5, 0.5, 0.5)`,
default pitch −π/2.

## Commit history relevant to this point

```text
7bad504  fix(rotated-cube-test): each cube IS a depth-29 recursive subtree
fa1b7d2  feat(rotated-cube-test): scale to tree depth 30
494da7e  fix(tangent): apply frame-path rotation to camera direction
2bba498  debug-overlay: [ copies state to clipboard while overlay open
3add32e  debug-overlay(tangent): rich positional + rotation-chain stats
0858ee1  fix(tangent): rotation-aware in_frame for camera-inside-rotated-cube
e2b20d5  feat(tangent): rotated-cartesian primitive — Step 1.0 minimal test
cc16e14  Revert "wip(tangent): ..." (discarded broken depth-30 attempt)
e5ffba9  wip(tangent): NodeKind::TangentBlock — broken (preserved on remote)
2ccac5f  refactor(planet): rotated-cube planet — strip all spherical traversal
```

## What's known to still be wrong (open work)

Per the user, abrupt transitions still appear in some scenarios.
Need a concrete repro (overlay copy-paste before/after) before
chasing further. Possibilities:

- Rotation chain inside `march_in_tangent_cube` — when descending
  into a nested rotated child of a TB, the inner walker uses
  absolute coords within the cube's `[0, 3)³`. At deep zoom inside
  the TB, this could drift.
- LOD termination inside `march_in_tangent_cube`: the inner walker
  ignores `LOD_PIXEL_THRESHOLD`. If the visible cell at LOD splat
  is a Block at the cube's outermost level, the splat is the
  representative without rotation context.
- Edit/break dispatch via `cpu_raycast` may not yet fully match the
  shader's hit when the camera is inside a rotated frame.
