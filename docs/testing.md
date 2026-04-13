# Testing

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

### Custom port

Both trunk and Playwright support custom ports (useful when another instance is running):

```bash
# Terminal 1: serve on a different port
trunk serve --no-autoreload --port 8081

# Terminal 2: point Playwright at that port
TRUNK_PORT=8081 npx playwright test -g "horizon"
```

The `TRUNK_PORT` env var is read by `playwright.config.ts` (defaults to 8080).

### Writing tests

Tests use `page.evaluate()` to push game state via `window.__onGameState()` and verify React renders correctly. For testing commands sent back to Rust, wrap `window.__pollUiCommands` to capture drained commands (the WASM game loop drains the queue every frame).

## Visual / GPU testing (WebGPU)

### The problem

Headless Chromium cannot capture WebGPU canvas content in screenshots — the canvas reads as `[0,0,0,0]` (transparent black). This is because headless mode requires `--disable-vulkan-surface` for WebGPU, which prevents canvas surface rendering. This is a known open issue (as of 2026) with no upstream fix.

### The workaround

Use **real Chrome** (`channel: "chrome"`) instead of bundled Chromium, with GPU flags:

```typescript
// playwright.config.ts
use: {
  channel: "chrome",
  headless: true,
  launchOptions: {
    args: [
      "--enable-gpu",
      "--enable-unsafe-webgpu",
      "--enable-features=Vulkan",
      "--use-angle=metal",  // macOS; use "vulkan" on Linux
    ],
  },
}
```

This gives real GPU rendering in headless mode. `page.screenshot()` captures the composited output correctly. However, `canvas.toDataURL()` / `drawImage` to a 2d canvas still returns blank — use screenshots for visual comparison, not pixel sampling.

### Limitations in headless

- **Pointer Lock** does not work (`"The root document of this element is not valid for pointer lock"`). Camera cannot be rotated via mouse. Keyboard/scroll input may work.
- **Canvas pixel readback** (`toDataURL`, `getImageData` via `drawImage`) returns `[0,0,0,0]`. Only `page.screenshot()` captures the rendered frame.

### Enabling atmosphere on WASM

The atmosphere requires storage buffers (compute shaders), which need WebGPU — WebGL2 does not support them. To enable atmosphere on WASM:

1. Add `"webgpu"` to Bevy features in `Cargo.toml` (overrides the default `webgl2`)
2. Remove `#[cfg(not(target_arch = "wasm32"))]` guards on `Atmosphere`, `AtmosphereSettings`, `ScatteringMedium` in `src/camera.rs`
3. Unify `ClearColor` in `src/main.rs` (remove the WASM/native split)

All major browsers now support WebGPU (Chrome 113+, Firefox 141+, Safari 18.2+).

### Worktree setup for visual testing

When creating a worktree for visual testing:

```bash
git worktree add -b my-branch ../deepspace-game-my-branch new-features

# Symlink shared dirs not tracked by git
ln -s /path/to/main/external ../deepspace-game-my-branch/external
ln -s /path/to/main/target ../deepspace-game-my-branch/target

# Create dirs that Trunk.toml ignore list expects to exist
mkdir -p ../deepspace-game-my-branch/{.claude,ui/dist,tests,docs,dist}

# Install npm deps and build UI
cd ../deepspace-game-my-branch/ui && npm install && npm run build
```
