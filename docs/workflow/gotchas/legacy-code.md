# Legacy code still in the tree

These paths are compiled or copied along with the repo but are **not
on the active runtime path**. Don't model the live architecture from
them — they reference a prior Bevy-WASM build that was replaced by
the current native-winit + wry-overlay stack.

**Why they're still here**: ripping them out is mechanical but
out-of-scope for any given session. Flagging them is cheaper than
deleting them the wrong way.

## Legacy — do not use as a reference

| Path | What it was | Why it's dead |
|---|---|---|
| `ui/tests/*.spec.ts` | Playwright specs | Wait on `__perfData` from a Bevy-WASM build; assert `window.canvas` layout that doesn't exist. |
| `scripts/test-native.sh` | Native overlay smoke test | Drives Playwright against Vite; assumes a WASM game running in Chrome. |
| `scripts/test-overlay.sh` | Overlay init test | Expects `ws://localhost:9000` — the current overlay uses wry IPC, not WebSocket. |
| `scripts/deploy.sh` | Vercel deploy | Calls `trunk build --release` but `src/main.rs` has no wasm entry point. |
| `Trunk.toml` | WASM bundler config | Same reason: no active wasm target. |
| `assets/shaders/bsl_voxel.wgsl` | Bevy StandardMaterial shader extension | Not referenced in `shader_compose::SHADER_SOURCES`. The live shader is `main.wgsl`. |
| `src/bin/winit_probe.rs` | Scratch winit probe binary | No `[[bin]]` in `Cargo.toml`, no script invokes it. |
| `src/import/` | `.vox` importer | Pub module in `lib.rs` but no caller: `src/app/`, `src/world/`, `src/renderer.rs`, `src/bridge.rs` never touch `import::`. `SavedMeshes` stashes NodeIds from the normal edit path. |
| `assets/characters/*.glb` | Model files | Not loaded by any code. |
| `assets/npcs/*.blueprint.json` | NPC blueprints | Not loaded by any code. |
| `#[cfg(target_arch = "wasm32")]` branches | WASM-specific code | Cargo.toml has wasm-bindgen deps but no active wasm entry point binds them in. |

## What *is* live

- `./scripts/dev.sh` — the canonical dev loop (Vite + native game).
- `cargo test --test e2e_layer_descent --test render_perf --test render_visibility`.
- `src/main.rs` → `winit` event loop → `src/app::App`.
- React overlay in `ui/src/` served over Vite into a wry WebView
  embedded in the native window. IPC via `window.__onGameState` /
  `window.ipc.postMessage` (not WebSocket).

## Before deleting

If you're tempted to delete any of the above, first grep the project:

```bash
rg 'bsl_voxel|winit_probe|import::|Fox\.glb|Soldier\.glb|blueprint\.json'
```

The current state is "nobody references these," but a partial wiring
somewhere would make the delete a break. Verify before cutting.
