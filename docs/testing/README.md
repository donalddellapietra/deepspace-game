# Testing

Three tiers of tests keep the engine honest:

1. **Render harness** — a built-in deterministic mode of the game
   binary. Drives the renderer from a script, captures screenshots,
   enforces perf gates. See [harness.md](harness.md).

2. **Rust e2e** — `cargo test` suites in `tests/` that spawn the
   harness, parse its `HARNESS_*` stdout, and assert on traces +
   screenshots. Flagship: [e2e-layer-descent.md](e2e-layer-descent.md).

3. **Playwright UI** — browser tests under `ui/tests/` that exercise
   the React overlay in isolation over a Vite dev server.

When chasing a rendering perf regression, start with
[perf-isolation.md](perf-isolation.md). For capturing macOS windows
in headless mode, see [screenshot.md](screenshot.md).

## Quick commands

```bash
# Rust e2e
cargo test --test e2e_layer_descent --test render_perf --test render_visibility

# One e2e test with logs
cargo test --test e2e_layer_descent layer_37_break_below_is_registered_three_ways -- --nocapture

# Dev server (Vite + game, interactive)
./scripts/dev.sh

# Native overlay smoke test
./scripts/test-native.sh

# Overlay boot sequence check
./scripts/test-overlay.sh

# Playwright UI tests (requires trunk serve on TRUNK_PORT, default 8084)
npx --prefix ui playwright test
```

## Timeouts

All render-harness invocations should run under `timeout 6` (or the
test's own `--timeout-secs`) so a hung shader can't stall the test
loop. The harness defaults to 5 s, deliberately low.
