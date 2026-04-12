# Testing

## Important: disable auto-reload when running tests

Trunk's live-reload injects a WebSocket that navigates the page on file changes. This causes Playwright test failures (DOM detach, execution context destroyed). Always use `--no-autoreload`:

```bash
# Terminal 1
trunk serve --no-autoreload

# Terminal 2
cd ui && npx playwright test
```

## Running tests

```bash
# All tests
cd ui && npx playwright test

# Single test by name
cd ui && npx playwright test -g "Hotbar renders with 10 slots"

# Headed (visible browser)
cd ui && npx playwright test --headed -g "test name"

# Debug mode (step through)
cd ui && npx playwright test --debug -g "test name"
```

## Writing tests

Tests live in `ui/tests/`. They run against a live trunk server at localhost:8080.

For testing React renders: use `page.evaluate()` to push game state via `window.__onGameState()`, then assert on DOM elements.

For testing commands sent to Rust: the WASM game loop drains the command queue every frame via `__pollUiCommands()`. Wrap that function inside `page.evaluate()` to capture drained commands, and do the click/interaction in the same evaluate block to avoid DOM detach from re-renders.
