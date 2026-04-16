# Render harness

The game binary has a built-in deterministic test driver: pass CLI
flags and it runs a headless render loop, optional script, optional
screenshot, and auto-exits. No external input synthesis, no overlay,
no sleeps — the whole thing is reproducible from a command line.

Source of truth:
- `src/app/test_runner.rs` — flag parsing + harness loop.
- `src/app/harness_emit.rs` — script dispatcher + `HARNESS_*` stdout.
- `tests/e2e_layer_descent/harness.rs` — Rust e2e test parser.

Any flag left off ⇒ interactive mode.

## Core flags

| Flag | Effect |
|---|---|
| `--render-harness` | Headless deterministic mode: no overlay/webview, forced redraws, auto-exit. |
| `--show-window` | Keep the native window visible (debugging). |
| `--disable-overlay` | Skip WKWebView + overlay flushing; keep the native surface. |
| `--disable-highlight` | Don't draw the cursor AABB (for pixel-stable shots). |
| `--harness-width W` / `--harness-height H` | Override render target dimensions. |

## World + spawn

| Flag | Effect |
|---|---|
| `--plain-world` / `--sphere-world` | Select world preset. |
| `--plain-layers N` | Tree depth for the plain preset (default 40). |
| `--spawn-depth N` | Starting camera anchor depth. |
| `--spawn-xyz X Y Z` | Explicit camera position (world-XYZ, bootstrap only). |
| `--spawn-yaw RAD`, `--spawn-pitch RAD` | Starting orientation. |
| `--force-visual-depth N` / `--force-edit-depth N` | Override LOD budgets. |

Note: `--spawn-xyz` is a bootstrap shortcut, not a general world-XYZ
API. See [../principles/no-absolute-coordinates.md](../principles/no-absolute-coordinates.md).

## Screenshots + exit

| Flag | Effect |
|---|---|
| `--screenshot PATH` | Capture to PNG after the script settles, then exit. |
| `--exit-after-frames N` | End after N rendered frames (default ~120). |
| `--run-for-secs S` | End after S wall-clock seconds. |
| `--timeout-secs S` | Hard kill switch (default **5.0**). Catches hung shaders. |

## Perf gates

Fail the run if any threshold is tripped after warmup:

| Flag | Effect |
|---|---|
| `--min-fps FPS` | Mean rendered-frame FPS floor. |
| `--fps-warmup-frames N` | Ignore the first N frames (default 10). |
| `--min-cadence-fps FPS` | Present cadence floor (tracks `dt`, not pure work time). |
| `--cadence-warmup-frames N` | Default 10. |
| `--max-frame-gap-ms MS` | Fail if the gap between any two rendered frames exceeds MS. |
| `--frame-gap-warmup-frames N` | Default 30. |
| `--require-webview` | Fail if the WKWebView overlay never comes up. |

## Perf trace (per-frame CSV)

| Flag | Effect |
|---|---|
| `--perf-trace PATH` | Write one CSV row per rendered frame to `PATH`. Header and column meanings documented in `src/app/test_runner/runner.rs::PerfTraceWriter`. |
| `--perf-trace-warmup N` | Skip the first N rendered frames before recording (default 0). Use to exclude startup stalls. |

Captures every measurable phase: CPU (`update`, `pack`, `ribbon_build`,
`tree_write`, `highlight_*`, `encode`, `submit`, `wait`), GPU (`gpu_pass_ms`
via Metal `TIMESTAMP_QUERY`, `submitted_done_ms` via `on_submitted_work_done`),
and workload context (`packed_node_count`, `ribbon_len`, `effective_visual_depth`,
`reused_gpu_tree`). See `docs/testing/perf-isolation.md` for interpretation.

The harness also emits three structured summary lines to stderr at end-of-run:

```
render_harness_timing avg_ms update=... camera_write=... pack=... ... gpu_pass=... submitted_done=... total=...
render_harness_worst total_ms=...@frameN gpu_ms=...@frameN upload_ms=...@frameN
render_harness_workload frames=... avg_packed_nodes=... max_packed_nodes=... avg_ribbon_len=... max_ribbon_len=...
```

## Script

`--script "cmd1,cmd2,..."` runs commands in order.

| Command | Effect |
|---|---|
| `wait:N` | Skip N frames. |
| `break` / `place` | Left- / right-click on the world. |
| `zoom_in:N` / `zoom_out:N` | Zoom N steps (anchor depth changes). |
| `pitch:RAD` / `yaw:RAD` | Absolute camera orientation in radians. |
| `screenshot:PATH` | Capture mid-script to PNG. |
| `probe_down` | CPU raycast straight down in world-space, emit `HARNESS_PROBE`. |
| `emit:LABEL` | Timeline marker; emit `HARNESS_MARK`. |
| `teleport_above_last_edit` | Position camera inside the bottom child of the last broken cell at current anchor depth. |
| `debug_overlay` | Toggle the debug overlay panel. |

## `HARNESS_*` stdout protocol

All records are single-line, whitespace-separated, `key=value`. Parsed
by `tests/e2e_layer_descent/harness.rs`.

```
HARNESS_MARK  label=<str>         ui_layer=<u32> anchor_depth=<u32> frame=<u64>
HARNESS_EDIT  action=broke|placed anchor=[slot,slot,...] changed=<bool> ui_layer=<u32> anchor_depth=<u32>
HARNESS_PROBE direction=down      hit=<bool> anchor=[slot,slot,...] ui_layer=<u32> anchor_depth=<u32>
```

- `MARK` fires from `emit:LABEL`. Use it to correlate screenshots
  with script state.
- `EDIT` fires from `do_break` / `do_place`. `changed=false` means
  the edit was issued but the raycast didn't hit anything (or the
  world didn't mutate).
- `PROBE` fires from `probe_down`. The camera's pitch/yaw are
  temporarily overridden for the raycast and restored, so the next
  render is unaffected.

`anchor` is the slot sequence from the hit's root-to-leaf `HitInfo`
path — the final slot is the cell that was hit.

## Example

```bash
cargo run -- --render-harness --plain-world --plain-layers 40 \
    --spawn-depth 4 --spawn-xyz 1.5 1.01 1.5 --spawn-pitch -1.5707 \
    --script "wait:10,emit:before,break,wait:5,probe_down,screenshot:/tmp/post.png" \
    --exit-after-frames 100 --timeout-secs 30
```

Always wrap in `timeout 6` (or similar) when iterating from Bash —
the default `--timeout-secs 5` gets you there, but the enclosing
wrapper is insurance against a hang that happens before the harness
loop starts.
