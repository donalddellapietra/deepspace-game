# Playwright UI tests

Browser-level tests that exercise the React overlay against a live
Vite/Trunk dev server.

Source of truth:
- `ui/tests/` — spec files.
- `ui/playwright.config.ts` — browser + GPU setup.

Playwright is configured to use Chrome (not Chromium) with Vulkan/Metal
angle flags so WebGPU is actually available. The tests default to
`http://localhost:8084`; override with `TRUNK_PORT` or by running
Trunk on a different port.

## Running

```bash
# Must be running separately:
./scripts/dev.sh

# Then, in ui/:
npx --prefix ui playwright test
npx --prefix ui playwright test ui-overlay.spec.ts       # single file
npx --prefix ui playwright test -g "hotbar renders"      # by name
```

## What's covered

| Spec | Covers |
|---|---|
| `ui-overlay.spec.ts` | React mount, hotbar slot rendering, key hints, mode indicator, hidden-by-default panels, CSS vars, bridge wiring. |
| `wasm-health.spec.ts` | WASM module boots. |
| `wasm-render.spec.ts` | Canvas renders a frame. |
| `wasm-debug.spec.ts` | Debug overlay surfaces. |
| `mesh-loading.spec.ts` | `.vox` imports through the UI. |
| `npc-spawn.spec.ts`, `npc-perf.spec.ts` | NPC content pipeline. |

## Wiring to the game

The overlay pushes state via `window.__onGameState(json)` and drains
UI commands via `window.__pollUiCommands()`. Tests poke these
directly rather than synthesizing mouse/keyboard events, so a spec
that asserts "hotbar slot 3 is selected" simply calls
`__onGameState({hotbar: {selected: 3}})` and checks the DOM.

## Not covered here

Anything that requires the native window, the wry WebView bridge, or
the render harness. Those live in
[e2e-layer-descent.md](e2e-layer-descent.md) and the `scripts/test-*.sh`
smoke tests.
