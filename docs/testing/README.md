# Testing

Two tiers of live tests:

1. **Render harness** — a built-in deterministic mode of the game
   binary. Drives the renderer from a script, captures screenshots,
   enforces perf gates. See [harness.md](harness.md).

2. **Rust e2e** — `cargo test` suites in `tests/` that spawn the
   harness, parse its `HARNESS_*` stdout, and assert on traces +
   screenshots. Flagship: [e2e-layer-descent.md](e2e-layer-descent.md).

When chasing a rendering perf regression, start with
[perf-isolation.md](perf-isolation.md). For capturing macOS windows
in headless mode, see [screenshot.md](screenshot.md).

## Quick commands

```bash
# All Rust e2e
cargo test --test e2e_layer_descent --test render_perf --test render_visibility

# One e2e test with logs
cargo test --test e2e_layer_descent layer_37_break_below_is_registered_three_ways -- --nocapture

# Interactive dev loop (Vite + native game via wry overlay)
./scripts/dev.sh
```

## Timeouts

All render-harness invocations should run under `timeout 6` (or the
test's own `--timeout-secs`) so a hung shader can't stall the test
loop. The harness defaults to 5 s, deliberately low.

## What's *not* live

The `ui/tests/*.spec.ts` Playwright specs and `scripts/test-*.sh`
are **legacy** — they reference a Bevy/WASM build path and a
WebSocket state server that the current native wry + winit runtime
does not have. See [../workflow/gotchas/legacy-code.md](../workflow/gotchas/legacy-code.md).
