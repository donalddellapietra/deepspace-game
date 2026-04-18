# Navigating to the sphere surface in the render harness

Notes from debugging "sphere renders as a tiny warped speck on this branch
but fills the frame on `ray-march-engine-2`." The visible bug was a missing
`normalize()` on `ray_dir` in `assets/shaders/main.wgsl` — sphere-intersection
math in `sphere.wgsl` assumes a unit ray direction, so off-axis rays missed
and only a small central cone ever hit the planet. Fix: wrap the
`camera.forward + right*ndc.x + up*ndc.y` expression in `normalize(...)`.

The rest of this doc captures what navigation levers the harness actually
exposes, because the wrong mental model (e.g. "just zoom out") burned a lot
of time during the debug.

## Layer ↔ anchor_depth for the demo sphere

`tree_depth = 30` for `bootstrap_demo_sphere_world` (planet face subtree
depth 28 + 1 body + 1 root).

```
ui_layer = tree_depth − anchor_depth + 1
anchor_depth = tree_depth − ui_layer + 1 = 31 − ui_layer
```

| UI label         | `--spawn-depth` |
| ---------------- | --------------: |
| Layer 30         |               1 |
| Layer 26         |               5 |
| Layer 15         |              16 |
| Layer  1         |              30 |

## `zoom_*` does NOT move the camera

`--script zoom_in:N` / `zoom_out:N` (and the in-game zoom keys) change
**anchor_depth only** — they don't translate the camera in world space. A
`zoom_out` screenshot shows the same world-space view with a different
render-frame size; it cannot be used to "pull back" to see more of a planet.

To frame the planet, use `--spawn-xyz` to place the camera and (optionally)
`--spawn-pitch` to aim it. `--spawn-depth` picks the anchor cell containing
that point.

## Minimal sphere-visible invocation

```bash
./target/debug/deepspace-game --render-harness \
    --sphere-world \
    --spawn-xyz 1.5 2.0 1.5 \
    --spawn-depth 8 \
    --spawn-pitch -1.5 \
    --disable-overlay --disable-highlight \
    --harness-width 1280 --harness-height 720 \
    --screenshot tmp/shot/sphere.png \
    --exit-after-frames 60 --timeout-secs 12 \
    --shader-stats
```

- World center is `(1.5, 1.5, 1.5)`; sphere outer radius `0.45` world-units;
  so `(1.5, 2.0, 1.5)` is just above the north pole.
- `--spawn-pitch -1.5` points nearly straight down (mouse clamp is `[-1.5,
  1.5]`). `-π/2 = -1.5707` works too but the clamp limit is `-1.5`, and any
  in-game input will clip back to that. `-1.5` is what sticks.
- `--spawn-depth 8` yields `render_path=[13,16]` (top face subtree of the
  body cell) and `visual_depth ≈ 6`, which resolves terrain tiles.

## Diagnosing "planet is missing" in under 30 s

Per `docs/testing/spawn-placement.md` the first question is always
`render_harness_shader ... hit_fraction`:

| `hit_fraction`  | meaning                                               |
| --------------- | ----------------------------------------------------- |
| `0.0000`        | no ray hits anything — the camera cone misses content |
| `< 0.01`        | only a tiny cone hits — suspect unit/scale bug        |
| `~ 0.5 – 1.0`   | camera is pointed at content                          |

The "only a tiny cone hits" symptom is the signature of the `normalize()`
bug that this doc was written to explain: sphere intersection `disc = b² −
c` is only valid for `|ray_dir| = 1`. Non-unit `ray_dir` produces apparent
hits at positions outside `cs_outer`, which the very next check in
`sphere.wgsl` (`if r >= cs_outer || r < cs_inner { break; }`) rejects. Only
rays close to `camera.forward` (where `|ray_dir| ≈ 1`) survive, so the
planet appears as a small central disc — hence "warped and weird."

## Coordinates quick reference

- `WORLD_SIZE = 3.0`. Root cell is `[0,3]³`; each sub-cell is a 1/3 nested
  `[0,3]` region in its parent's frame.
- Planet body is at `slot 13` of the root = world cell `[1,2]³`.
- Body frame `[0,3]³` maps to body cell `[1,2]³`, so world `y=2.0` →
  body-frame `y=3.0` (and that is what `cam_local` reports in gpu_camera
  logs when the active frame kind is `Sphere`).
- Sphere radii in `root_radii.xy` are in body-cell-local `[0,1)` units; the
  shader multiplies by `3.0` to get body-frame units (`cs_outer = 1.35`,
  `cs_inner = 0.36`).

## Harness quirks worth remembering

- `cursor_locked` is auto-enabled when `--spawn-depth` OR `--screenshot` is
  set, so mouse motion on a dev workstation won't fight the headless view.
- `--spawn-pitch` is respected at spawn, but in-game mouse motion clamps to
  `[-1.5, 1.5]`. A headless run never moves the mouse so `-1.5707` works,
  but `-1.5` is safer.
- The yellow crosshair is drawn unconditionally in `main.wgsl`'s fragment
  path. It is NOT suppressed by `--disable-overlay` or `--disable-highlight`.
- `--suppress-startup-logs` hides the `gpu_camera basis` and `spawn:`
  diagnostics you almost always want when debugging a blank screen. Leave it
  off during investigation.
