# NPC Instancing: Findings & Bugs

**Date:** 2026-04-13
**Branch:** gpu-instancing
**Goal:** Scale from 3 FPS at 1K NPCs to millions

## Final Results

| NPCs | FPS | Architecture |
|-----:|----:|:-------------|
| 1K | 3.0 | Original (ECS per-entity, per-part mesh entities) |
| 10K | 122 | GPU instanced rendering + flat buffer |
| 100K | 125 | WebGPU, no CPU physics |
| 300K | 48 | CPU overlay iteration becomes bottleneck |
| 1.8M | 11.9 | Maximum tested |

## Architecture Evolution

### Phase 1: CPU Optimizations (incremental, ~45% improvement)
- **Frustum culling** in `collect_overlays`: skip NPCs behind camera or outside view radius
- **Allocation recycling**: reuse `Vec<OverlayPart>` across frames via `spare_parts` pool
- **Staggered updates**: AI/animation run on 1/4 of NPCs per frame

These helped but couldn't break 10 FPS at 10K. The bottleneck was entity count.

### Phase 2: GPU Instanced Rendering
Replaced per-entity overlay reconciliation with custom Bevy render pipeline:
- One entity per unique `(NodeId, voxel_color)` mesh group (~12 entities total)
- Instance buffer with per-NPC transform matrices + colors
- Custom WGSL vertex shader reads instance data at locations 10-14
- Custom fragment shader matches BSL lighting (ambient + diffuse + tone mapping)

**Key decision:** Render to `Transparent3d` phase (sorted) instead of `Opaque3d` (binned). Opaque3d uses `BinnedPhaseItem` which has a completely different API. Transparent3d works fine for opaque NPCs — depth writes are inherited from the base `MeshPipeline`.

### Phase 3: Flat NPC Buffer
Replaced per-NPC ECS entities with a single `Vec<NpcState>` resource:
- Zero ECS query overhead for NPC iteration
- Fixed-size `[(Vec3, Quat); 8]` array for part transforms (no heap allocation per NPC)
- Raw `f32` countdown timer instead of Bevy `Timer` (no allocation, no tick overhead)

### Phase 4: WebGPU + Compute Shader (scaffolded)
- Enabled `bevy = { features = ["webgpu"] }` for compute shader support
- Compute pipeline, render graph node, and data types implemented
- Not yet driving simulation (CPU systems disabled for scale testing)

## Bugs & Gotchas

### 1. WASM `std::time::Instant` panics
**Symptom:** Game crashes on load with "time not implemented on this platform"
**Root cause:** `std::time::Instant::now()` and `std::time::SystemTime::now()` are not available on WASM
**Fix:** Replace with `bevy::platform::time::Instant::now()` (uses `web_time` internally) and `js_sys::Date::now()` for RNG

### 2. BSL shader fails to load on WASM
**Symptom:** Green screen (terrain invisible), "Failed to deserialize meta for asset shaders/bsl_voxel.wgsl"
**Root cause:** Trunk's dev server returns 200 for `.meta` file requests on non-existent files. Bevy tries to parse the shader source as meta JSON and fails. Then the shader never loads.
**Fix:** Embed shader at compile time via `OnceLock<Handle<Shader>>` + `include_str!()` + `Shader::from_wgsl()`. Load in `BlockPlugin::build()`. `fragment_shader()` returns the handle from the OnceLock.

### 3. Atmosphere crashes on WebGL2
**Symptom:** wgpu validation error: "Too many bindings of type StorageBuffers in Stage ShaderStages(FRAGMENT)"
**Root cause:** Bevy's `Atmosphere` component creates bind group layouts requiring storage buffers in fragment shaders, which WebGL2 doesn't support
**Fix:** `#[cfg(not(target_arch = "wasm32"))]` on `Atmosphere` spawn, `sync_atmosphere_scale` system, and related imports

### 4. Shader vertex attribute location conflicts
**Symptom:** "Two or more vertex attributes were assigned to the same location in the shader: 5"
**Root cause:** Bevy's `MeshPipeline` reserves locations 0-9 for Position, Normal, UV, Tangent, Color, Joints, Weights, etc. Custom instance attributes at locations 2-6 conflict.
**Fix:** Use locations 10-14 for instance data (transform matrix columns + color)

### 5. Stale wasm-bindgen cache produces `import "env"` errors
**Symptom:** JS file starts with `import * as import1 from "env"` — module doesn't load
**Root cause:** After `git checkout` or feature flag changes, the compiled WASM binary in `target/wasm32-unknown-unknown/debug/` is stale but cargo thinks it's up-to-date (fingerprint matches). wasm-bindgen processes the stale binary and produces invalid JS.
**Fix:** Delete the binary explicitly: `find target/wasm32-unknown-unknown -name "deepspace*" -delete` before `trunk build`. Do NOT rely on `touch src/main.rs` — cargo's fingerprinting may still skip recompilation.

### 6. `js_sys::global()` doesn't return `window` in Bevy WASM
**Symptom:** `js_sys::Reflect::get(&js_sys::global(), &"__spawnNpcs".into())` always returns undefined
**Root cause:** In Bevy's WebGPU WASM context, `js_sys::global()` returns a different global scope than the page's `window` object
**Fix:** Use `js_sys::eval("window.__someVar")` to read page globals. Use `wasm_bindgen(inline_js)` for calling page functions from WASM. For writing, `js_sys::eval(&format!("window.x = {}", val))` works.

### 7. wasm_bindgen `inline_js` doesn't persist state across calls
**Symptom:** `inline_js` function reads `window.__spawnNpcs` correctly on first call, returns 0 on all subsequent calls even though the value was set to 9000
**Root cause:** Unknown — possibly the ES module created by `inline_js` has its own module scope that doesn't see mutations to `window` made by Playwright's `page.evaluate()`
**Fix:** Use `js_sys::eval("window.__getAndResetSpawnCount()")` to call a function defined in `<head>` instead. The `<head>` function shares the same window context as Playwright.

### 8. Compute pipeline `prepare_bind_group` panics before shader compiles
**Symptom:** "unreachable" WASM panic after first NPC spawn
**Root cause:** `pipeline_cache.get_bind_group_layout()` panics if the compute pipeline hasn't finished compiling yet. The system runs every frame but the shader may take multiple frames to compile.
**Fix:** Guard with `pipeline_cache.get_compute_pipeline_state(pipeline.pipeline_id)` — only proceed if `CachedPipelineState::Ok`.

### 9. Pointer lock errors misidentified as panics
**Symptom:** Test reports "unreachable" panic but game is actually running fine
**Root cause:** Bevy tries to lock the cursor on canvas click. In headless Chrome, this throws "The root document of this element is not valid for pointer lock." which is a `pageerror` event. The test treated all `pageerror` events as WASM panics.
**Fix:** Filter `pageerror` events: `if (err.message.includes("pointer lock")) return;`

### 10. Trunk build needs `<head>` scripts before WASM init
**Symptom:** `wasm_bindgen` extern functions fail because the JS functions don't exist when WASM initializes
**Root cause:** Trunk injects the WASM `<script type="module">` in `<body>`. If JS bridge functions (`__setPerfData`, `__getAndResetSpawnCount`) are defined in a `<script>` after the WASM link, they don't exist when WASM calls them.
**Fix:** Move all JS bridge definitions to a `<script>` block in `<head>`, before any Trunk directives.

### 11. Trunk `cd ui` changes shell working directory permanently
**Symptom:** `trunk build` fails with "Unable to find any Trunk configuration" after building UI
**Root cause:** `cd ui && npx vite build && trunk build` — the `cd ui` persists, so `trunk build` runs from `ui/` where there's no `Trunk.toml`
**Fix:** Always use subshells: `(cd ui && npx vite build)` — parentheses create a subshell that doesn't affect the parent's cwd.

## Build Workflow

Correct sequence for building and testing:
```bash
cd /path/to/worktree
pkill -f "trunk" 2>/dev/null
(cd ui && npx tsc -b && npx vite build 2>&1 | tail -1)
find target/wasm32-unknown-unknown -name "deepspace*" -delete 2>/dev/null
rm -rf target/wasm-bindgen dist
trunk build 2>&1 | tail -3
nohup trunk serve --port 8080 --no-autoreload > /tmp/trunk.log 2>&1 &
# Wait for ready:
tail -f /tmp/trunk.log  # look for "server listening"
# Test:
cd ui && npx playwright test <test-name> --reporter=line
```

## What's Next

The remaining bottleneck at 1M+ NPCs is CPU-side `collect_overlays_from_buffer` iterating every NPC to build instance data. The compute shader needs to:
1. Replace CPU overlay collection with GPU-side instance buffer writes
2. Use a heightmap texture for ground collision instead of tree walks
3. Move animation keyframe interpolation to the vertex shader
4. Share the NPC state storage buffer between compute and vertex shaders
