# Layer-descent e2e

The flagship end-to-end test. Drives the render harness through a
scripted descent from UI layer 37 down to layer 1, breaking a cell at
each layer and verifying three independent signals agree on what
happened.

The real claim under test is **self-similarity**: the scene at
layer 37 and the scene at layer 5 are visually the same structural
layout. Only the numbers on the UI change.

## What a human sees

Setup: plain world, 40 layers. Zoom-out clamped at layer 37. Camera
starts hovering one layer-37 cell above the grass surface, looking
straight down, cursor on the cell directly below.

One iteration at layer `N`:

1. **Look down — baseline.** Grid of layer-`N` cells fills the
   frame; outlines between them; yellow cursor highlight at center.
2. **Break.** The cursored cell vanishes, leaving a dark square.
3. **Probe.** Silent — a straight-down CPU raycast writes the new
   anchor path to stdout for the Rust test to check.
4. **Hole shot.** Screenshot of the dark square + surrounding grid.
5. **Zoom in.** UI drops to `N−1`. Camera teleports horizontally to
   the broken cell's center, vertically to one layer-`(N−1)` cell
   above the new ground.
6. **Zoomed-down shot.** A **3×3 grid** of layer-`(N−1)` cells —
   visually identical to step 1, one iteration later.
7. **Look up.** Pitch flips. You see **sky — blue**. All the nested
   holes you've dug are coaxial; line-of-sight straight up is open.
8. **Sky shot.** Upper half must be >50% blue pixels (pixel check).
9. **Look back down.** Ready for the next iteration.

After 37 iterations you've dug an exponentially-nested tower of
holes. Total world-space descent: a geometric series summing to less
than one layer-37 cell. **If any layer breaks the pattern — the grid
disappears, the hole doesn't appear, the sky isn't blue, the frame
freezes — that layer is a concrete failure of self-similarity at a
named depth.**

## Three-way verification per break

Each break must produce all of:

1. **Screenshot diff.** `layer_{N}_down_post.png` differs from
   `layer_{N}_down_pre.png` at the frame center (center-darkened).
2. **CPU probe.** `HARNESS_PROBE hit=true anchor=[...]` on stdout
   with the expected anchor path and depth.
3. **Edit record.** `HARNESS_EDIT action=broke changed=true
   anchor=[...]` on stdout.

Disagreement among these three = rendering is out of sync with the
tree. See [harness.md](harness.md) for the protocol details.

## World + spawn configuration

| Parameter | Value |
|---|---|
| `--plain-world` | set |
| `--plain-layers` | `40` |
| `--spawn-depth` | `4` (= `tree_depth - ui_layer + 1` for layer 37) |
| `--spawn-pitch` | `-π/2` (straight down) |
| Spawn XYZ | `(1.5, 1.5 + cell_size(37), 1.5)` |
| Zoom-out clamp | `ui_layer ≤ 37` |

See [../workflow/gotchas/layer-vs-depth.md](../workflow/gotchas/layer-vs-depth.md)
for the layer ↔ anchor_depth conversion.

## Perf gates

All runs enforce:

```
--max-frame-gap-ms 400
--frame-gap-warmup-frames 2
--timeout-secs <per-test>
```

A stall at any layer fails the test at that layer, with the layer
number in the failure message.

## Pixel assertions

1. **Sky dominance.** On every `layer_{N}_up.png`, count top-half
   pixels where `b > r && b > g && b_normalized > 0.5`. Require
   `count / total > 0.5`. Failure here means the nested-aperture
   line-of-sight is broken.
2. **Center darkened.** On every `layer_{N}_down_post.png` vs.
   `layer_{N}_down_pre.png`, the frame-center square must be darker
   (per-channel mean) after the break. Failure means the edit
   landed CPU-side but didn't render.

## Test layout

```
tests/e2e_layer_descent/
    main.rs          # #[test] fns
    harness.rs       # spawn binary, script builder, stdout parser
tests/e2e_layer_descent.rs    # [[test]] entry point
```

## Explicitly out-of-scope

- Grid-line pixel detection (color dominance only for v1).
- Color-consistency across layers (materials may change).
- Sphere-body descent (Cartesian-only; sphere layers have their own
  asymmetries deferred — see `../history/anchor-refactor-decisions.md` §12).
- Re-emergence test (zoom all the way back out, assert all nested
  holes still visible). Separate follow-up test.
