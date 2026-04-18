# Layer-descent e2e (sphere)

Sphere counterpart to [e2e-layer-descent.md](e2e-layer-descent.md). Same
protocol — scripted descent through the render harness, breaking a cell
at each layer, verifying three independent signals agree — but rooted
in a cubed-sphere planet instead of plain ground.

The real claim under test is the same **self-similarity** claim: the
scene at layer 26 (anchor_depth 5) and the scene at layer 10 (anchor_depth
21) are visually the same structural layout — cells of terrain below,
nested coaxial holes above, sky at the top of the shaft. Only the numbers
on the UI change.

## What a human sees

Setup: `--sphere-world` (tree_depth = 30, planet at the body cell of the
root). Camera at the PosY (north pole) face, hovering one anchor cell
above the sphere surface. `--spawn-on-surface` places the camera using a
path-based spawn that tracks the SDF surface at any depth — no f32
quantization.

One iteration at anchor_depth `N` (UI layer `31 − N`):

1. **Look down — baseline.** Grid of layer-`N` cells on the planet
   surface fills the frame; cursor at center.
2. **Break.** The cursored cell vanishes.
3. **Probe.** Straight-down CPU raycast; records the new anchor path.
4. **Hole shot.** Screenshot of the dark square + surrounding grid.
5. **Zoom in.** `anchor_depth` increments (UI layer decrements).
   `teleport_above_last_edit` positions camera inside the bottom child
   (`slot_index(1, 0, 1)`) of the cell we just broke, repeated to fill
   the depth delta.
6. **Zoomed-down shot.** Visually similar grid at the new depth.
7. **Look up.** Pitch flips. You should see **sky** through the coaxial
   chain of nested holes.
8. **Sky shot.** Upper half must pass a sky-dominance pixel check.
9. **Look back down.** Ready for the next iteration.

After N iterations you've dug an exponentially-nested tower of holes.
Line-of-sight radial (not axial) — the face subtree's `r` axis points
outward from the sphere center, so "up" on the PosY face is +Y in world
space, matching camera pitch `+π/2`.

## Three-way verification per break

1. **CPU probe.** `HARNESS_PROBE hit=true anchor=[...]` on stdout with
   the expected anchor path and depth.
2. **Edit record.** `HARNESS_EDIT action=broke changed=true anchor=[...]`
   on stdout, with `anchor == probe anchor`.
3. **Screenshot capture.** A down-view PNG is written at every depth.
   The sky-dominance pixel check from the Cartesian test is deferred
   here — the sphere GPU render currently produces a uniform tan
   fill when the camera sits inside the outer shell (a pre-existing
   issue; `sphere_zoom_invariance.sh` works because it spawns at
   `y=2.0`, above the shell). Restore once that rendering path is
   fixed.

## World + spawn configuration

| Parameter | Value |
|---|---|
| `--sphere-world` | set |
| `--spawn-on-surface` | set (dispatches to `demo_sphere_surface_spawn`) |
| `--spawn-depth` | starting `anchor_depth` (e.g., `5` for UI layer 26) |
| `--spawn-pitch` | `-π/2` (straight down; ray_dir = probe_down) |
| `--spawn-yaw` | `0` |
| `--interaction-radius` | `36` (boosted from default 12 — see below) |
| Starting UI layer | `26` (= `31 − anchor_depth`) |

## Why `--interaction-radius = 36`

The default (`12 × shell × 3⁻⁴ ≈ 0.147` body-frame units) is already
floored at the SDF min cell size, so at `anchor_depth ≥ 5` the reach
stays near `0.147` regardless of zoom. Each descent-break opens a
cumulative tunnel through the pole; by `anchor_depth ≥ 22` the next
solid lives just past that reach, even though the camera is still
looking straight down. Bumping to 36 anchor-cells gives `~0.44` reach
— enough for the full 20-layer descent.

See [sphere-harness-navigation.md](sphere-harness-navigation.md) for
the layer ↔ anchor_depth mapping and the normalize-ray-dir background.

## Why `--spawn-pitch = −π/2` exactly

`probe_down` fires a ray with direction `(0, −1, 0)`. `do_break` uses the
camera's forward vector. If pitch is `−1.5` (instead of `−π/2 ≈ −1.5708`)
the camera forward is `≈ (0, −0.997, 0.071)` — tilted ~4° off. At
shallow zoom this still lands on the same anchor cell, but at
`anchor_depth ≥ 12` the ~4° tilt can land on a different face of the
cubed sphere than `probe_down`, producing anchor paths that diverge at
the face-slot step. `−1.5707963` keeps probe and break on the same ray.

## Descent range

`tree_depth = 30` limits how deep we can go. Practical range:

- **Start:** `anchor_depth = 5` (UI layer 26). The sphere is recognizably
  a sphere at this level; breaks carve visible craters.
- **End:** `anchor_depth = 25` (UI layer 6). 20 iterations. At
  `anchor_depth > 25` the physical cell is below the SDF detail limit
  (`shell · 3^{−SDF_DETAIL_LEVELS}`) and the interaction gate floors it
  instead of letting the reach shrink further.

## Pixel assertions

Deferred in v1 — see note above. Screenshots are captured but only
verified for *existence*, not pixel content. Once the sphere GPU
render path renders correctly when the camera is inside the outer
shell, re-introduce:

- **Sky dominance** on look-up frames.
- **Center darkened** on break pre/post diffs — validates that
  `sphere_depth_tint` (see `sphere.wgsl`) produces a visibly different
  pixel color for exposed deeper cells.

## Test layout

```
tests/e2e_layer_descent/
    harness.rs       # ScriptBuilder, run, Trace, sky_dominance_top_half
tests/e2e_layer_descent.rs           # Cartesian suite
tests/e2e_layer_descent_sphere.rs    # this suite
```

The harness module is shared between the two suites (`ScriptBuilder`,
`run`, `Trace`) — same protocol, only the world preset + a few CLI
args differ.

## `respawn_on_surface` vs `teleport_above_last_edit`

The Cartesian test's descent step is `zoom_in:1, teleport_above_last_edit`,
which pushes `slot_index(1, 0, 1) = 10` onto the anchor path to
"enter the bottom child of the last-broken cell." That works because
a Cartesian tree subdivides cells by `(x, y, z)` at every level.

Sphere face subtrees index children by `(u, v, r)`, while `WorldPos`
arithmetic is unconditionally Cartesian. So appending slot 10 to a
face-subtree path translates to `(x=1, y=0, z=1)` = bottom in
Cartesian y, which for the PosY face means "middle u, low v, middle r"
= drift horizontally along `-Z`, not radially into the sphere. By
~10 iterations the camera wanders off the face and every probe misses.

`respawn_on_surface` is the sphere-specific replacement: it re-invokes
`demo_sphere_surface_spawn(current_anchor_depth)` so the camera gets
placed path-accurately above the surface at the new depth. Each
iteration tests "break at anchor_depth `N`" on fresh surface geometry
— not "descend into the previously-broken cell" — which is the
meaningful thing to verify on sphere without adding a sphere-aware
teleport.

## Explicitly out-of-scope

- Testing descent across all six faces (one face is enough for v1).
- Color-consistency across layers.
- Nested-hole line-of-sight (the Cartesian test's sky-dominance
  assertion). The sphere analog is radial line-of-sight, blocked
  currently by the rendering issue noted above.
- Re-emergence test (zoom all the way back out, assert all nested
  holes still visible). Separate follow-up.
