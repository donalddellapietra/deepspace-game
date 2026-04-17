# Useful Commands

Paste each as one line. Shell line-wrapping can split `--flag value` pairs and break things.

## Live GUI

`scripts/dev.sh` forwards `"$@"` to the binary. Any game flag works:

```bash
scripts/dev.sh                                                      # plain world, default spawn
scripts/dev.sh --sphere-world                                       # planet demo
scripts/dev.sh --menger-world --plain-layers 15 --lod-base-depth 20 # ternary fractal
```

`--lod-base-depth 20` disables the ribbon-level budget decay (pixel Nyquist still applies — the shader won't render sub-pixel detail).

### With stats in the terminal

Append `--shader-stats --live-sample-every 60` to any live command. `renderer_slow` fires on any >30 ms frame; `render_live_sample` prints CPU phase timings every N frames.

### Flags that cause auto-exit in live mode

These trigger test-runner mode (exits after 120 frames): `--render-harness`, `--screenshot`, `--script`, `--spawn-xyz`, `--spawn-depth`, `--spawn-yaw`, `--spawn-pitch`, `--exit-after-frames`, `--run-for-secs`, `--min-fps`. Use them only in harness commands below, or add `--run-for-secs 3600 --timeout-secs 3600` if you want them in an interactive session.

## Headless benchmarks (harness)

The harness runs offscreen, no vsync, reports pure GPU time. Use this — not the live loop — for perf measurement.

### Single run

```bash
timeout 12 cargo run --bin deepspace-game --quiet -- --render-harness --plain-world --spawn-depth 8 --spawn-pitch -1.0 --harness-width 1920 --harness-height 1080 --exit-after-frames 120 --timeout-secs 12 --suppress-startup-logs --shader-stats 2>&1 | grep -E "render_harness_timing|render_harness_shader"
```

Read `gpu_pass=X.XXX` (pure shader time) and `submitted_done=X.XXX` (includes TBDR resolve). `avg_steps`, `avg_empty`, etc. come from `render_harness_shader`.

### 5-run mean (drop first as warmup)

```bash
for i in 1 2 3 4 5; do timeout 12 cargo run --bin deepspace-game --quiet -- --render-harness --plain-world --spawn-depth 8 --spawn-pitch -1.0 --harness-width 1920 --harness-height 1080 --exit-after-frames 120 --timeout-secs 12 --suppress-startup-logs --shader-stats 2>&1 | grep "render_harness_timing" | grep -oE "gpu_pass=[0-9.]+ |submitted_done=[0-9.]+"; echo "---"; done
```

First run is shader compile + cache warmup — discard. Average runs 2-5.

### Fractal stress scenario

Camera inside a Menger sponge, looking down the central corridor:

```bash
for i in 1 2 3 4 5; do timeout 12 cargo run --bin deepspace-game --quiet -- --render-harness --menger-world --plain-layers 15 --spawn-xyz 1.5 2.8 1.5 --spawn-depth 8 --spawn-pitch -0.8 --harness-width 1920 --harness-height 1080 --exit-after-frames 120 --timeout-secs 12 --suppress-startup-logs --shader-stats 2>&1 | grep "render_harness_timing" | grep -oE "gpu_pass=[0-9.]+ |submitted_done=[0-9.]+"; echo "---"; done
```

### Zoom-in perf regression (INV8 scenario)

Zooms from shallow layer to deep via script, captures sustained-post-zoom perf:

```bash
timeout 20 cargo run --bin deepspace-game -- --plain-world --plain-layers 40 --spawn-depth 3 --spawn-pitch -1.0 --disable-overlay --shader-stats --live-sample-every 120 --script "wait:60,zoom_in:10,wait:600" --run-for-secs 14 --timeout-secs 20 2>&1 | grep -E "renderer_slow|heartbeat" | tail -8
```

## Visual correctness screenshots

```bash
timeout 8 cargo run --bin deepspace-game -- --plain-world --spawn-depth 8 --spawn-pitch -1.0 --disable-overlay --screenshot tmp/shot.png --harness-width 1280 --harness-height 720 --timeout-secs 8
```

Change `--plain-world` to `--sphere-world` or `--menger-world --plain-layers 15` for other scenes. Read the PNG to verify — timing numbers don't catch visual regressions.

## A/B comparison across branches

Same command, different worktree:

```bash
cd /Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/sparse-tree && <command>
cd /Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/testing-infra && <command>
```

The `menger-test` branch in the testing-infra worktree has the Menger preset cherry-picked for A/B against the dense layout baseline.

## Prime directives

- One command per shell invocation. Never chain with `&&`.
- Always wrap `cargo run` with `timeout N` (matches `--timeout-secs N`).
- Never `cargo clean` (20-min rebuild). If incremental cache breaks: `rm -rf target/debug/incremental/deepspace_game-*`.
- Always read the output. A command launching is not a success.

See [cookbook.md](cookbook.md) for deeper perf-debugging methodology and [harness.md](harness.md) for the full flag reference.
