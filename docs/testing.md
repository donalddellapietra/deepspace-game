# Testing

## Native render perf harness

For native frame-loop perf checks, do not rely only on warm-up-gated FPS.
Always include hard stall gates that apply from startup:

```bash
cargo run --bin deepspace-game -- \
  --disable-overlay \
  --spawn-depth 17 \
  --run-for-secs 2 \
  --timeout-secs 6 \
  --max-any-frame-ms 250 \
  --max-frame-gap-ms 400 \
  --frame-gap-warmup-frames 2 \
  --min-fps 50 \
  --min-cadence-fps 20
```

Key rule:
- `--max-any-frame-ms` and a small frame-gap warm-up (`2`) are required for
  regression tests, so startup acquire/present stalls cannot be hidden by
  FPS warm-up windows.

## Playwright (UI overlay tests)

Tests live in `ui/tests/`. They run against a live `trunk serve` instance.

### Prerequisites

```bash
cd ui && npm install
npx playwright install chromium
```

### Running tests

```bash
# Start the game server first (separate terminal)
# Use --no-autoreload to prevent live-reload from interfering with tests
trunk serve --no-autoreload

# All tests
cd ui && npx playwright test

# Single test by name
cd ui && npx playwright test -g "Hotbar renders with 10 slots"

# Single test file
cd ui && npx playwright test tests/ui-overlay.spec.ts

# Headed (see the browser)
cd ui && npx playwright test --headed -g "Hotbar renders"

# Debug mode (step through)
cd ui && npx playwright test --debug -g "Hotbar renders"
```

### Writing tests

Tests use `page.evaluate()` to push game state via `window.__onGameState()` and verify React renders correctly. For testing commands sent back to Rust, wrap `window.__pollUiCommands` to capture drained commands (the WASM game loop drains the queue every frame).
