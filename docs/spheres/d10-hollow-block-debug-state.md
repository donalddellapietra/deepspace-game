# Sphere d=10 hollow-block bug — debug state

Rolling doc tracking the diagnosis loop for: "placing a block at
anchor_depth 10 on the demo planet's surface renders a hollow cube and
stripes the nearby ground". Failure is sharp at d=9–10; d≤8 renders
cleanly.

## Reproducer (headless)

```bash
scripts/repro-sphere-d10-bug.sh              # cycles all 7 debug modes
scripts/repro-sphere-d10-bug.sh 0 4          # just modes 0, 4
SKIP_BUILD=1 scripts/repro-sphere-d10-bug.sh # skip cargo build
```

Outputs: `tmp/bug_mN.png` (post-place) and `tmp/bug_init_mN.png`
(initial / end-of-run).

Camera config baked into the script (matched to a user-captured
screenshot of the bug):
- `--sphere-world`
- `--spawn-xyz 1.5 1.7993 1.4988`
- `--spawn-depth 10`  (layer 21 = tree_depth 30 − anchor_depth 10 + 1)
- `--spawn-pitch -0.5`
- `--interaction-radius 10000` (override lets cursor reach surface)

Place lands the block at full d=10: `HARNESS_EDIT action=placed
anchor=[13,16,13,13,22,13,25,19,22,22] changed=true anchor_depth=10`.
That's 1 body + 1 face (PosY) + 8 UVR slots = total depth 10 ✓.

## What's been built

1. **Integer slot-pick walker** (`assets/shaders/sphere.wgsl`). Scales
   `un_abs → u32 ∈ [0, 3^15)` once at sphere_in_cell entry, then the
   walker picks slots by integer divide. Zero f32 rounding in the
   slot-pick path. Mirror still pending on CPU (`world/raycast/sphere.rs`).
2. **Sphere debug paint modes** (`assets/shaders/sphere_debug.wgsl`,
   cycled by F6 or `--sphere-debug-mode N`). Six modes: step count,
   terminal depth, walker result, winning plane axis, log cell size,
   ratio checkerboard.
3. **Rich debug overlay** (`ui/src/components/DebugOverlay.tsx`). All
   positional stats: root XYZ, anchor cell size in root, anchor / render
   path CSVs, sphere state, body-local distances to center / outer /
   inner shells, `cells above outer`, debug mode.
4. **Surface-nav harness tooling**: `fly_to_surface` refactored to land
   exactly 1 anchor-cell above the hit and call `apply_zoom` so the
   frame state refreshes. `--sphere-debug-mode N` threads through both
   live event-loop and render_harness init paths.
5. **Reproducer script** (`scripts/repro-sphere-d10-bug.sh`) matching
   user's bug-captured screenshot coords.

## Diagnosis so far

Running the reproducer across modes (see `tmp/bug_m0.png` ... `bug_m6.png`):

- **Mode 0 (normal)**: hollow-looking cube where placed block is, with
  visible "interior walls" of the cube; horizontal stripes banding the
  surrounding ground; overall symptom matches user screenshot.
- **Mode 1 (steps)**: block center uses ~128+ steps (yellow heat),
  ground ~few steps (blue). Block is where the DDA spends most work —
  not a runaway loop, just descent depth.
- **Mode 2 (terminal depth rainbow)**: block interior is cyan = d≈8–9.
  Ground around the block is yellow-green = d≈5–7. Walker IS reaching
  deep, but the visible SURFACE terminates before d=10.
- **Mode 3 (walker result)**: block AND ground both paint green = walker
  returned content. No pixels that should be on content show red
  (empty-advance). So the walker is finding the block's content — it's
  not an invisibility bug.
- **Mode 4 (winning plane)**: ground shows alternating stripes of
  **light blue (r_lo)**, **green (v_lo)**, **pink (u_lo)** at pixel-
  row granularity. Block walls show two-color vertical bands.
- **Mode 5 (log cell size)**: cyan block, green-yellow ground; mirrors
  mode 2.
- **Mode 6 (ratio checkerboard)**: ground is nearly uniform blue = all
  those pixels land in the same modulo-8 ratio bucket. Block has
  different hues per wall.

## Ruled out

- **Sub-pixel aliasing**: rejected. Camera scales with anchor depth;
  d=10 cells at anchor 10 project to normal-voxel size (many pixels).
  Reasoning-by-absolute-world-units is the wrong mental model here; the
  app is an infinite-zoom recursive voxel world. See memory
  `feedback_cells_not_subpixel.md`.
- **Walker slot pick jitter**: integer-ratio walker eliminates the
  `floor((un − u_lo)/child_size)` ULP flip. Mode 6 confirms walker
  lands consistently in the same ratio buckets on the ground.
- **Walker bails to empty on content**: mode 3 is all green on the
  block's visible surface. Walker finds content at the placed cell.
- **Block not placed**: `HARNESS_EDIT changed=true` with full 10-slot
  path confirms the tree was edited at the intended d=10 cell.

## Open hypothesis (current focus)

The stripes in mode 4 map pixel-row-by-pixel-row onto different
winning-plane IDs (last_side=0/2/4 alternating), which means for
adjacent ground rows the DDA's `argmin(t_u_lo, t_u_hi, t_v_lo, t_v_hi,
t_r_lo, t_r_hi)` picks a different axis per row. Two possibilities:

1. **Plane-normal f32 collapse**: at d=10, `ea_to_cube(u_lo·2−1)` for
   adjacent u_lo and u_hi of the same cell differ by ~1e-5 on the
   `n_axis` component. Combined with a ray `oc` at magnitude ≈ 1 in
   body-local, `dot(oc, n_u_lo)` and `dot(oc, n_u_hi)` differ by the
   same ~1e-5. f32 ULP at `|oc|=1` is ≈ 1e-7, so the absolute difference
   is well above ULP — but the RATIO `-dot(oc, n) / dot(dir, n)` that
   yields `t` can still amplify cancellation when `dot(dir, n)` is
   small (grazing ray on a lateral plane). Adjacent pixels with slightly
   different dir could swap t-order on any two of the six planes.

2. **LOD-terminal ancestor representative**: mode 2 shows the ground
   terminating at d≈6–7, not d=10. At d=7, each node has representative_
   block = dominant child. If the representative logic mishandles
   "mostly-empty + one placed block" subtrees, the visible surface at
   d=7 could switch from "dirt representative" (before place) to
   "stone + empty stripe pattern" (after place). Doesn't explain the
   HOLLOW look of the block directly, but does explain why the GROUND
   appearance CHANGED when a block was placed.

Both hypotheses are plausible. The striping pattern (hyp 1) seems to
account for mode 4 directly; the surface-appearance change (hyp 2)
accounts for the user's "ground breaks when I place a block" complaint.

## Next action in the loop

Pick the smaller change and run the reproducer:

- **Hyp 1 test**: rewrite the 4 ray-plane intersections in
  `sphere_in_cell` to compute the plane-normal DIFFERENCE (`n_u_hi −
  n_u_lo`) directly from `ea_to_cube'(midpoint) · 2 · size`, so the
  `dot(oc, n_plane)` computation has better-conditioned cancellation
  resistance. Then re-run the reproducer and compare mode 4 stripes.

- **Hyp 2 test**: walk the tree for the placed block's ancestor path
  and print each ancestor's `uniform_type` / `representative_block`.
  If a d=7–8 ancestor is UNIFORM_MIXED with representative=EMPTY while
  its subtree contains our d=10 stone cell, representative pick is
  lossy under partial occupancy. Compare before/after place.

Starting with hyp 1 (smaller scope, directly testable visual).

## Running notes

### 2026-04-21 late — experiments run (all REJECTED)

All of the following tested in the reproducer; none killed the stripes:

- Exp 1: `face_lod_depth` forced to `10u`. Stripes identical.
- Exp A: `w.u_lo/v_lo/r_lo/size` re-quantized to fixed d=9 before plane math. Stripes identical.
- Exp B: uniform-flatten disabled in `pack.rs:217-220`. Stripes identical.
- Exp B + Exp 1 combined. Stripes identical.
- Exp hardcode body_origin/body_size/inner_r/outer_r to demo_planet literals in `sphere.wgsl:400-403`. Stripes identical (minor view shift from f32 literal vs computed).
- Plane-normal midpoint-plus-delta rewrite (from first opus consult). Pixel-identical rendering, no change.

All reverted. Shader back to baseline.

### Confounding evidence

- gpu_camera logs bit-identical pre vs post place (`cam_local=[1.4999998, 2.3978996, 1.4964]`, `render_path=[13]`, `frame_kind=Body { inner_r: 0.12, outer_r: 0.45 }`).
- Mode 2 walker depth: yellow-green ground (d≈7) in both states.
- Mode 3 walker result: green everywhere on ground (walker returns content) in both states.
- Mode 6 ratio bucket: uniform blue on ground (walker lands in same ratio mod 8) in both states.
- Mode 4 winning plane: **UNIFORM r_lo (light-blue) BEFORE place, STRIPED r_lo/v_lo/u_lo AFTER place**. This is the ONLY mode that changes.

Placing a single d=10 block globally alters the DDA's per-pixel winning-plane arg-min outcome for ground rays spatially distant from the placed cell. None of the investigated mechanisms (LOD cap, cell-bound variance, pack uniform-flatten, body-dim drift) account for this.

### Next steps

Per-pixel walker state dump is the path forward: write `winning`, `steps`, `w.depth`, `w.block`, `w.u_lo`, `w.v_lo`, `w.size` to an SSBO and diff pre vs post place for a single pixel row. That pins down which variable actually mutates between the two runs.

