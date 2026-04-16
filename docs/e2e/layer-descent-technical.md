# Layer-descent e2e: technical spec

Implementation-facing companion to
[`layer-descent-ui.md`](layer-descent-ui.md).

## Goal

Assert the engine's self-similarity claim in code: the scripted
sequence `{break below, zoom in, look up at sky, look back down}`
produces a structurally identical frame layout at every layer from
37 down to 1, with no frame-time stalls, no rendering degradation,
and no CPU/GPU disagreement about which cell the cursor is on.

## World / spawn configuration

| Parameter | Value | Notes |
|---|---|---|
| `--plain-world` | — | flat grass-over-dirt-over-stone plain |
| `--plain-layers` | `40` | tree depth |
| `--spawn-depth` | `4` | = `tree_depth - ui_layer + 1` for layer 37; see `docs/gotchas/layer-vs-depth.md` |
| `--spawn-yaw` | `0` | irrelevant (looking straight down) |
| `--spawn-pitch` | `-π/2` | straight down |
| Spawn XYZ | centered horizontally at `(1.5, y, 1.5)`, `y` = `1.5 + cell_size(layer=37)` | one layer-37 cell above grass surface |
| Zoom-out clamp | `ui_layer ≤ 37` (equivalently `anchor_depth ≥ 4`) | prevents scrolling out past the intended starting layer |

Zoom-out clamp is a new harness option. Behavior: ignore scroll-up
inputs when they would reduce `anchor_depth` below 4. Zoom-in is
unconstrained (the test will take it all the way to layer 1 =
`anchor_depth = 40`).

## Per-layer action sequence (executed once per layer)

At layer `N`:

1. `screenshot:layer_{N}_down_pre.png` — baseline shot, expect grid pattern.
2. `break` — break the cell the cursor points at (straight down).
3. `probe_down` — CPU raycast straight down, emit `HARNESS_PROBE`.
4. `screenshot:layer_{N}_down_post.png` — expect dark square at center.
5. `zoom_in:1` — decrements `ui_layer`, increments `anchor_depth`.
6. Teleport camera to `(center_x_of_broken_cell, top_of_exposed_material + cell_size(N-1), center_z_of_broken_cell)` with pitch still straight down.
7. `screenshot:layer_{N-1}_down_zoomed.png` — expect 3×3 grid.
8. `pitch:π/2` — look straight up.
9. `screenshot:layer_{N-1}_up.png` — expect sky-blue dominance in upper half.
10. `pitch:-π/2` — look back down, ready for next iteration's step 1.

Note: the step that starts iteration `N−1`'s "look down baseline shot"
is the same frame as iteration `N`'s "zoomed-down shot." Skip the
duplicate capture in practice; it's listed separately above for
clarity.

## Three-way verification per break

Each `break` must produce:

1. **Screenshot** — `layer_{N}_down_post.png` exists on disk and
   differs from `layer_{N}_down_pre.png` at the frame center.
2. **CPU raycast (probe)** — `HARNESS_PROBE` line on stdout with
   `kind=Empty` at the expected anchor path and depth.
3. **Edit anchor** — `HARNESS_EDIT` line on stdout with
   `action=broke kind_before=<non-Empty> kind_after=Empty anchor=<path>`
   where the path's depth equals `anchor_depth + 1` (one level below
   the camera's anchor) OR equals `edit_depth()` per the current
   harness contract.

## New harness script commands (additions to `src/app/test_runner.rs`)

| Command | Effect |
|---|---|
| `screenshot:PATH` | Capture current frame to PNG. Today's `--screenshot PATH` is end-of-run only; this is mid-script. |
| `pitch:RAD` | Set `camera.pitch = RAD` (absolute). |
| `yaw:RAD` | Set `camera.yaw = RAD` (absolute). |
| `probe_down` | Run `cpu_raycast_in_frame` with direction `(0, -1, 0)` from the camera; emit `HARNESS_PROBE anchor=<path> kind=<Empty|Block(ty)|Node(id)> ui_layer=<L> anchor_depth=<D>`. |
| `emit:LABEL` | Emit `HARNESS_MARK label=<LABEL> ui_layer=<L> anchor_depth=<D> frame=<F>` — a free-form timeline marker for correlating screenshots to actions. |
| `teleport_to_center_of_last_broken` | Composite: look up the path of the most recently broken cell, set `camera.position` to its horizontal center with `y = top_of_exposed_material + cell_size(current_anchor_depth)`. Used right after `zoom_in`. |

The existing commands (`break`, `place`, `wait:N`, `zoom_in:N`,
`zoom_out:N`, `debug_overlay`) stay. `break` and `place` are modified
to emit `HARNESS_EDIT action=<broke|placed> anchor=<path>
kind_before=<k> kind_after=<k> ui_layer=<L> anchor_depth=<D>` on
stdout — the third leg of verification.

## New harness flags

- `--max-ui-layer N` — clamp zoom-out so `ui_layer ≤ N`. Set to `37`
  for this test.

Existing flags we rely on (all already in `test_runner.rs`): `--render-harness`,
`--plain-world`, `--plain-layers`, `--spawn-depth`, `--spawn-yaw`,
`--spawn-pitch`, `--disable-highlight` *(NOT used for this test —
we want the cursor highlight visible)*, `--screenshot`,
`--max-any-frame-ms`, `--max-frame-gap-ms`,
`--frame-gap-warmup-frames`, `--timeout-secs`, `--harness-width`,
`--harness-height`, `--script`.

## Hard perf gates (from `docs/render-perf-isolation-playbook.md`)

All runs must pass with:

```
--max-any-frame-ms 250
--max-frame-gap-ms 400
--frame-gap-warmup-frames 2
```

A stall at any layer fails the test at that layer, with the layer
number in the failure message.

## Pixel assertions

Two assertions run in the Rust test harness after the binary exits,
over the generated PNGs:

1. **Top-half-is-sky** on every `layer_{N}_up.png`:
   - Sample every pixel in the top half.
   - Count pixels where `b > r && b > g && b > 0.5` (normalized).
   - Assert count / total > 0.5.
   - If this fails at layer `N`, the nested-aperture line-of-sight is
     broken at that layer — a correctness failure per
     `locality-prime-directive.md`.

2. **Center-darkened** on every `layer_{N}_down_post.png` vs.
   `layer_{N}_down_pre.png`:
   - Sample a small square at the frame center in both images.
   - Assert the post-break center is darker (per-channel mean) than
     the pre-break center.
   - If this fails, the break didn't render — even though the edit
     may have succeeded CPU-side.

Skipped-for-now pixel checks (deferred): grid-line detection,
color-consistency across layers.

## Harness stdout trace schema

Every HARNESS_* line is a single whitespace-separated record on
stdout. Parser in `tests/e2e_layer_descent/harness.rs` tokenizes
these into typed structs.

```
HARNESS_MARK  label=<str> ui_layer=<u32> anchor_depth=<u32> frame=<u64>
HARNESS_EDIT  action=<broke|placed> anchor=<path> kind_before=<k> kind_after=<k> ui_layer=<u32> anchor_depth=<u32>
HARNESS_PROBE anchor=<path> kind=<k> ui_layer=<u32> anchor_depth=<u32>
```

`<path>` format: the slot sequence in dotted form, e.g.
`root.13.13.13.13`. `<k>` format: `Empty`, `Block(<ty>)`, or
`Node(<id>)`.

## Test file layout

```
tests/e2e_layer_descent/
  main.rs       # #[test] fns. Single-layer test first, then full descent.
  harness.rs    # spawn binary, script builder, stdout parser
```

Plus `[[test]] name = "e2e_layer_descent" path = "tests/e2e_layer_descent/main.rs"`
in `Cargo.toml`.

## Phased implementation

1. **Harness primitives first.** Land the new script commands, the
   `--max-ui-layer` flag, and the `HARNESS_*` stdout lines. Verify by
   eye on a handful of manual invocations. No test code yet.
2. **Single-layer proof.** Write one test that runs the full
   iteration at layer 37 only. Assert everything: trace, screenshots
   on disk, pixel checks. Green or fail — fix until green.
3. **Full descent.** Extend to generate the full 37-iteration script
   programmatically in the Rust test. One `cargo run` invocation,
   ~148 screenshots on disk, one structured trace.
4. **Watchlist gates.** Layer in the perf gates, the highlight
   visibility assertion, and (later) zoom-out-and-reverify to confirm
   nested holes are visible when re-emerging.

## Explicitly out-of-scope for v1

- Grid-line pixel detection (just color dominance for now).
- Color-consistency assertions across layers (materials may change;
  we don't predict).
- Sphere-body descent (Cartesian-only — sphere layers have their own
  asymmetries deferred per `anchor-refactor-decisions.md §12`).
- Re-emergence test (zoom all the way back out, assert all nested
  holes still visible). Deferred to a follow-up test once descent is
  green.
