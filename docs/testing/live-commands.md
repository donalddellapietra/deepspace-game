# Useful Live Commands

Commands for interactive exploration (GUI window, mouse/keyboard input). Paste each as a single line — shell line-wrapping can split `--flag value` pairs and break things.

For automated harness testing (headless, perf gates, screenshots) see [cookbook.md](cookbook.md) instead.

## Prerequisite

`scripts/dev.sh` forwards its arguments to the game binary (after starting the Vite overlay in the background). This means any `--flag` that the binary accepts also works on the dev script:

```bash
scripts/dev.sh <flags>
```

## Menger sponge (fractal stress test)

The Menger sponge is a ternary fractal — 20 of 27 children non-empty per level — useful for stressing the renderer with content that can't be uniform-collapsed by the packer.

```bash
scripts/dev.sh --menger-world --plain-layers 15 --lod-base-depth 20
```

- `--menger-world` — selects the preset.
- `--plain-layers 15` — sponge depth (15 gives fully interactive, visible detail down to small scales).
- `--lod-base-depth 20` — effectively disables the ribbon-level budget decay (the shader clamps to `MAX_STACK_DEPTH=5` regardless, and pixel-size Nyquist still stops sub-pixel descent). Use this when you want full per-shell detail for visual inspection, not the default 4-3-2-1 fall-off.

Default spawn is at `(2.5, 2.0, 2.5)` looking into the sponge from a corner. Scroll the mouse wheel to zoom in/out; WASD to move.

**With shader stats printed to stderr every 60 frames:**

```bash
scripts/dev.sh --menger-world --plain-layers 15 --lod-base-depth 20 --shader-stats --live-sample-every 60
```

`render_live_sample` lines show CPU-side phase timings; `renderer_slow` lines fire for any frame >30ms with the full GPU breakdown.

## Plain world (standard baseline)

```bash
scripts/dev.sh
```

No flags → plain world, default spawn at grassland surface.

## Sphere world

```bash
scripts/dev.sh --sphere-world
```

## Common flag reference for live use

| flag | effect |
|---|---|
| `--menger-world` / `--sphere-world` / `--plain-world` | World preset. |
| `--plain-layers N` | Tree depth for the selected preset (default 40 for plain, 15 is a good Menger starting point). |
| `--lod-base-depth N` | Ribbon-level LOD budget. 4 (default) uses 4/3/2/1 shell decay. 20 effectively disables shell decay; pixel Nyquist still applies. |
| `--lod-pixels N` | Pixel-size Nyquist floor. 1.0 (default) = strict sub-pixel rejection. Raise to 2-4 if you want cheaper distant content at the cost of some detail. |
| `--shader-stats` | Enable per-pixel atomic counters for DDA step-count output. Adds ~0.5-1ms per frame at 1280×720 from atomic contention. |
| `--live-sample-every N` | Print a `render_live_sample` line every N frames to stderr with acquire/encode/submit/present/total timings. 0 disables. |
| `--disable-overlay` | Skip the WKWebView overlay (no hotbar, no debug panel). Useful when you want a pixel-stable window. |

## Avoid in live mode

These flags trigger **test-runner mode** which auto-exits the game after 120 frames (or `--run-for-secs N` seconds). Use them in `cookbook.md` harness commands, not for interactive sessions:

- `--render-harness`
- `--screenshot PATH`
- `--exit-after-frames N`
- `--script "..."`
- `--spawn-xyz X Y Z` — positional spawn, also triggers auto-exit
- `--spawn-depth N`, `--spawn-yaw RAD`, `--spawn-pitch RAD`
- `--min-fps`, `--run-for-secs`, `--max-frame-gap-ms`

If you *do* pass one of these and still want a long-lived interactive window, add `--run-for-secs 3600 --timeout-secs 3600` to keep it alive for an hour.

## Switching branches for A/B comparison

The Menger preset + dev.sh forwarding live on the `sparse-tree` branch (in the `.claude/worktrees/sparse-tree/` worktree). The dense-layout baseline with the same preset lives on the `menger-test` branch (in `.claude/worktrees/testing-infra/`). Run the same `scripts/dev.sh ...` command from each worktree to compare FPS side-by-side.
