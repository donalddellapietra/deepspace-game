# Layer-descent e2e: UI flow (what a human sees)

This document describes the end-to-end test as a story — what a person
watching the screen would perceive, frame by frame, with no
implementation detail. The technical spec is in
[`layer-descent-technical.md`](layer-descent-technical.md).

The test's real claim is **self-similarity**: the scene at layer 37
and the scene at layer 5 are visually the same structural layout.
Only the numbers on the UI change.

## Setup

- World: plain world, 40 layers.
- Zoom-out is clamped at layer 37 for the duration of the run
  (scrolling up past layer 37 does nothing; you cannot see layer 38 or
  beyond).
- The camera starts **hovering one layer-37-sized cell above the grass
  surface**, looking straight down, cursor on the cell directly below.
- The UI reads `Layer 37`.

## The opening frame (layer 37)

The screen is filled with the grass field. Because cell borders render
as outlines, you don't see pure green — you see a **grid pattern** of
layer-37-sized cells, with outlines between them. The cell directly
under the crosshair is highlighted (yellow).

## One iteration (repeats 37 times, once at each layer 37 → 1)

The iteration at any layer `N` consists of the following actions and
screenshots. At every layer it looks structurally identical — that is
the claim under test.

1. **Look down — baseline shot.**
   You see a grid of layer-`N` cells filling the frame, with outlines
   between them and a yellow highlight at the center. The material
   color is whatever cell exists immediately below the camera at that
   layer (grass on the first iteration; after that, whatever got
   exposed by the previous break — most likely dirt, then stone
   eventually; we'll see).

2. **Break.**
   Click. The cell directly under the crosshair vanishes. The frame
   now shows a dark square at its center (the empty cell) surrounded
   by unbroken grid-patterned material.

3. **Probe — invisible to a viewer.**
   The engine confirms via a straight-down CPU raycast that the cell
   below is empty and prints the anchor path of the cell that was
   broken. (See technical doc for the trace format.)

4. **Hole shot.**
   Screenshot of the dark square surrounded by grid.

5. **Zoom in.**
   The UI indicator drops from `N` to `N−1`. The camera teleports:
   - horizontally, to the center of the cell we just broke,
   - vertically, to one layer-`(N−1)` cell above the **new ground**
     (the top of the cell that was exposed when the previous cell was
     broken).

   The camera is still pointed straight down.

6. **Zoomed-down shot.**
   You now see a **3×3 grid** of layer-`(N−1)` cells filling the
   frame, with outlines. This is the same scene as step 1 one
   iteration later — same structural layout, 3× finer grid. The
   material color may or may not have changed depending on whether
   the break exposed a different material.

   *(It should look indistinguishable from the opening frame except
   for the grid density and possibly the color.)*

7. **Look up.**
   Pitch flips to straight up, yaw unchanged. You see **sky — blue**.
   You are inside the hole you dug at layer `N`, and that hole opens
   directly above you. From deeper iterations you are inside a
   telescope of nested holes, but they are all coaxial, so the
   line-of-sight straight up is always clear to the open sky.

8. **Sky shot.**
   Screenshot of the up-facing view. The upper half of the frame is
   expected to be >50% sky-blue pixels. (That's the pixel-level
   assertion.)

9. **Look back down.**
   Pitch returns to straight down. The scene now matches step 1 for
   the next iteration (at layer `N−1`): a grid of cells filling the
   frame, cursor on the cell below. Ready for the next break.

## After 37 iterations

The last break happens at layer 1 — the finest resolution the tree
supports in this world. You've dug an exponentially-nested tower of
holes. Total world-space descent: vanishingly small (geometric series
summing to less than 1 cell at layer 37). Total number of iterations:
37. Total expected look: **the same frame, 37 times, with the layer
counter ticking down.**

If any layer breaks that pattern — the grid disappears, the hole
doesn't appear, the sky isn't blue, the frame freezes, a curve
distorts the image — that layer is a concrete failure of the engine's
self-similarity claim, at a named depth.
