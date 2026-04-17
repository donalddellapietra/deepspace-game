# Testing Cookbook (Agent-Facing)

Problem-oriented quick reference. Pick the scenario, copy the command, interpret the output. For deep flag reference see `harness.md`; for perf-debugging methodology see `perf-isolation.md`.

## Prime directives

1. **One shell command per Bash call.** Never chain `&&`. If one step fails, you lose signal on the others.
2. **Always wrap with `timeout 6`** (or `timeout N` matching the scenario's `--timeout-secs`). The harness defaults to `--timeout-secs 5`; a perf regression or hung shader must not stall the loop forever.
3. **Never `pkill`**, `cargo clean`, `git reset --hard`, or similar. Only stop processes this session started. `cargo clean` costs ~20 min of rebuild; use `rm -rf target/debug/incremental/deepspace_game-*` instead if you need a fresh incremental.
4. **Always read the output of each command.** Launching is not success. "The command ran" is not a result. Read the last 5-20 lines of output and cite specific numbers.
5. **Hard cap on max_body_steps**: 1024. Anything higher has caused macOS lockups. Pack node count cap: 1M (same reason).

## "I changed something. What do I test?"

Decision tree:

```
Did I change shader code (assets/shaders/*.wgsl)?
  → screenshot A/B at three scenarios (see §Visual correctness)
  → shader-stats perf run (see §Perf regression)

Did I change pack logic (src/world/gpu/pack.rs, ribbon.rs)?
  → cargo test --lib gpu
  → screenshot A/B
  → shader-stats perf run

Did I change renderer plumbing (src/renderer/*.rs, buffers)?
  → cargo build --bin deepspace-game
  → live game smoke test (see §Live smoke)
  → screenshot A/B

Did I change edit/raycast (src/world/raycast/*, edit_actions/)?
  → cargo test --test e2e_layer_descent
  → live game smoke test
```

## Visual correctness (screenshot A/B)

Three canonical scenes that must stay pixel-identical across any non-visual change:

```bash
# Scene 1: plain world, depth 8, looking down at grassland
timeout 8 cargo run --bin deepspace-game -- \
    --plain-world --plain-layers 40 --spawn-depth 8 --spawn-pitch -1.0 \
    --disable-overlay --screenshot tmp/shot/plain_d8.png \
    --harness-width 1280 --harness-height 720 --timeout-secs 8
```

```bash
# Scene 2: sphere world at default spawn
timeout 8 cargo run --bin deepspace-game -- \
    --sphere-world --spawn-depth 8 \
    --disable-overlay --screenshot tmp/shot/sphere.png \
    --harness-width 1280 --harness-height 720 --timeout-secs 8
```

```bash
# Scene 3: zoomed-in empty-sky regression scenario (the INV8 fix target)
timeout 8 cargo run --bin deepspace-game -- \
    --plain-world --plain-layers 40 --spawn-depth 3 --spawn-pitch -1.0 \
    --disable-overlay --screenshot tmp/shot/zoom3.png \
    --harness-width 1280 --harness-height 720 --timeout-secs 8
```

**Verify**: use the `Read` tool on each PNG and visually inspect. Plain-d8 shows yellow ground + grey walls. Sphere shows the dark planet ball against blue sky. Zoom3 shows coarse ground tiling.

**Interpretation**: if the image differs from the expected baseline, you have a correctness regression. Screenshots don't lie — timing numbers can.

## Perf regression (shader-stats)

The INV8 regression scenario is the canonical perf check:

```bash
timeout 20 cargo run --bin deepspace-game -- \
    --plain-world --plain-layers 40 --spawn-depth 3 --spawn-pitch -1.0 \
    --disable-overlay --shader-stats --live-sample-every 120 \
    --script "wait:60,zoom_in:10,wait:600" \
    --run-for-secs 14 --timeout-secs 20 2>&1 | \
    grep -E "renderer_slow|heartbeat" | tail -8
```

Look for `renderer_slow` lines with stats fields. Read these values:

| field | meaning | target (post-INV8) |
|---|---|---|
| `avg_steps` | DDA iterations per ray | ≤ 35 |
| `avg_empty` | empty-cell advances | ≤ 20 |
| `avg_descend` | tag=2 descents | ≤ 5 |
| `avg_lod_terminal` | LOD budget hits | ≤ 3 |
| `hits` / `miss` | ray outcomes | hits = 3.69M, miss = 0 |
| `gpu_pass_ms` | actual shader time | ≤ 40 ms |

If `miss` starts growing above 0, your change has a **correctness bug** — the shader is returning sky where geometry should be. Investigate before looking at perf.

If `avg_steps` grows significantly, your change has a **perf regression**. Compare against the last known-good numbers in `docs/testing/perf-lod-diagnosis.md` or the INV commit messages.

## Live smoke (normal gameplay FPS)

```bash
timeout 12 cargo run --bin deepspace-game -- \
    --plain-world --plain-layers 40 --spawn-depth 8 --spawn-pitch -1.0 \
    --disable-overlay --live-sample-every 60 \
    --script "wait:600" --run-for-secs 8 --timeout-secs 12 2>&1 | \
    grep -E "render_live_sample|heartbeat" | tail -6
```

Target: `avg_frame_fps ≥ 60` on the heartbeat lines. `render_live_sample` lines should show `total_ms ≈ 16-17` (vsync-paced).

**Caveat**: when `--run-for-secs` is passed, the test runner switches to `AutoNoVsync` presentation mode. `cadence_fps` will overshoot 60 wildly (200+, 400+) while `avg_frame_fps` remains accurate. Read the correct column.

## Unit tests

```bash
# GPU pack + ribbon tests (most likely to break from a layout change)
cargo test --lib gpu 2>&1 | tail -3
```

```bash
# All lib tests
cargo test --lib 2>&1 | tail -10
```

```bash
# E2E tests (spawn the harness, parse HARNESS_* stdout)
cargo test --test e2e_layer_descent --test render_perf --test render_visibility 2>&1 | tail -10
```

Read the final `test result: ok. N passed; M failed` line. If M > 0, re-run the specific failing test with `--nocapture` to see stderr:

```bash
cargo test --test e2e_layer_descent the_failing_test_name -- --nocapture 2>&1 | tail -40
```

## Common pitfalls

### Shader-stats poll cascade

`--shader-stats` triggers `device.poll(PollType::Wait)` on slow frames to read back GPU timestamps. This stalls the CPU thread for the full GPU drain time. On a slow frame that stall can be 100-900 ms, which then poisons the next few frames' `acquire_ms` (the swapchain couldn't get a free backbuffer). In the `renderer_slow` log you'll see `acquire_ms=900` followed by `gpu_pass_ms=0.00` for the next 5-10 frames.

**What to do**: ignore any `renderer_slow` line with `gpu_pass_ms=0.00` or anomalous acquire times. The subsequent lines after the cascade clears are the real signal. Or re-run without `--shader-stats` to measure pure frame time.

### Shader-stats counts differently from what you expect

`avg_steps` counts *outer loop iterations*. Each branch counter (`avg_empty`, `avg_descend`, etc.) counts iterations that hit that specific branch. They should sum to roughly `avg_steps` but the addition has statistical noise (±10%) because of counter-division rounding.

If `avg_empty` goes up after a perf fix, that's not necessarily bad — it can mean more cells went through the cheap empty-fast-path instead of the expensive descend path. Read the OTHER counters before panicking.

### Test runner auto-switches present mode

When `--run-for-secs N` or `--min-fps` or similar perf-related flags are set, the runner enables `AutoNoVsync`. When you pass only `--render-harness`, vsync is respected. This matters: a "60 FPS" test with `--run-for-secs` isn't actually vsync-limited; a "60 FPS" test with just `--render-harness` is.

### Screenshot path must include filename

`--screenshot tmp/foo.png` works. `--screenshot tmp/foo/` silently fails — the trailing slash makes it a directory target and the harness doesn't create it.

### Chaining `&&` in a Bash call

If you do `cargo build && cargo test && ...` in one Bash call and the build fails, the remaining commands don't run, you don't see their output, and you've lost progress-tracking signal. Use one Bash call per step. If you truly need sequential execution with early-abort, that's what TaskCreate is for — not shell chaining.

### `cargo clean` vs incremental cache

Never `cargo clean` — 20-minute rebuild. If the incremental cache genuinely seems corrupted (symptoms: phantom type errors, macro misbehavior), clear just the incremental directory:

```bash
rm -rf target/debug/incremental/deepspace_game-*
```

This loses ~2 minutes of cache but keeps the ~20 minutes of crate compilation.

## Change-verification checklist

Before reporting a change as done, run these in order:

1. ☐ `cargo build --bin deepspace-game 2>&1 | tail -5` — clean warnings, zero errors
2. ☐ `cargo test --lib gpu 2>&1 | tail -3` — all gpu unit tests pass
3. ☐ Screenshot Scene 1 (plain-d8) — visually identical to baseline
4. ☐ Screenshot Scene 2 (sphere) — visually identical
5. ☐ Screenshot Scene 3 (zoom3) — visually identical
6. ☐ Perf run at the INV8 scenario — `hits=3.69M miss=0`, `avg_steps ≤ baseline`
7. ☐ Live smoke at depth=8 — `avg_frame_fps ≥ 60`

Each item is a separate Bash call. Each output is read and verified before proceeding to the next. If any fails, fix before moving on — don't accumulate regressions.

## When tests are hanging or behaving oddly

**"Command hangs past timeout"**: the binary is ignoring the timeout. Check: are you using `timeout 6 cargo run -- --timeout-secs 5 ...`? The shell `timeout` kills the process; `--timeout-secs` is the binary's own kill switch. The `timeout 6` wrapper is your safety net.

**"FPS numbers look impossible"**: check present mode. `AutoNoVsync` triggers on perf-measurement flags and uncaps the cadence. Read `avg_frame_fps` (the rendered-frame rate), not `avg_cadence_fps`.

**"Screenshot is all-black"**: the harness exited before the render completed. Bump `--exit-after-frames` to 60+ or add `wait:60` at the start of your `--script`. The first few frames are warmup.

**"avg_steps is 0 in shader stats"**: you forgot `--shader-stats`. Without it, the atomic counters are compiled out and all branch counters stay zero.

**"miss count keeps growing after a fix"**: you have a correctness bug. Rays that should hit ground are missing. Look at your shader change — likely you're advancing the ray or setting state in a way that skips over geometry. Revert and add one line at a time.

## See also

- `harness.md` — complete flag reference
- `perf-isolation.md` — methodology for isolating perf regressions
- `perf-lod-diagnosis.md` — the full INV arc; numbers to compare against
- `screenshot.md` — macOS window-capture tricks if you need to debug the live game visually
- `e2e-layer-descent.md` — how to write a new e2e test
