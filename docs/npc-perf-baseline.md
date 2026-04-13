# NPC Performance Baseline (pre-instancing)

**Date:** 2026-04-12
**Branch:** worktree-npc-instancing
**Platform:** WASM (headless Chromium, SwiftShader software renderer)
**Bevy:** 0.18

## Test Setup

- `trunk serve` on port 8080, Playwright headless Chromium
- Each NPC = 1 root entity + 6 part entities + ~11 mesh children = **~18 entities per NPC**
- Mass spawn via M key (1000 NPCs per press, grid pattern around player)
- FPS measured via Bevy's `FrameTimeDiagnosticsPlugin` (smoothed)

## Baseline Results (no instancing)

| NPCs  | Entities | FPS  | Frame Time (ms) |
|------:|--------:|-----:|----------------:|
|     0 |      25 | 12.7 |             78.5 |
| 1,000 | 18,025  |  3.3 |            300.0 |
| 2,000 | 36,025  |  2.1 |            481.9 |
| 3,000 | 54,025  |  1.5 |            649.3 |
| 4,000 | 72,025  |  1.2 |            830.3 |
| 5,000 | 90,025  |  1.0 |          1,023.2 |
| 6,000 | 108,025 |  0.9 |          1,159.8 |
| 7,000 | 126,025 |  0.8 |          1,331.7 |
| 8,000 | 144,025 |  0.7 |          1,488.7 |

## Analysis

**Entity count is the bottleneck.** Each NPC spawns ~18 Bevy entities
(root + 6 parts + ~11 mesh children). At 1,000 NPCs that's 18,000 entities.

Frame time scales roughly linearly with entity count:
- 0 NPCs: ~80ms/frame
- 1,000 NPCs: ~300ms (+220ms for 18K entities = ~12us per entity)
- 8,000 NPCs: ~1,489ms (+1,409ms for 144K entities = ~10us per entity)

The ~10-12us per entity cost comes from:
1. **Transform propagation** through the entity hierarchy (root -> part -> mesh)
2. **CPU-side AI** (`npc_ai` system iterates all NPCs every frame)
3. **CPU-side animation** (`npc_animate` system iterates children per NPC)
4. **Physics** (`npc_physics` system does collision per NPC)
5. **Draw call overhead** (each mesh entity = separate draw submission)

**Note:** These numbers are from SwiftShader (software GPU in headless
Chromium). Real GPU performance would be better for rendering but the
CPU-side entity/system overhead would remain the same.

## Target

10,000 NPCs at 30+ FPS on native, which means reducing per-NPC entity
count and CPU system overhead by roughly 100x.

## Planned Optimizations (ordered by expected impact)

1. **GPU instancing** — collapse all same-mesh parts into single instanced
   draw calls. ~6 draw calls instead of 18K * 6 = 108K. Eliminates mesh
   child entities entirely.

2. **Flatten entity hierarchy** — store part transforms inline instead of
   as child entities. Eliminates transform propagation overhead.

3. **Frustum culling** — skip off-screen NPCs in AI/animation/rendering.

4. **Batch AI/physics** — replace per-entity Timer with raw f32 countdown.

## How to Run

```bash
# From project root
cd ui && npm install
cd ..
trunk build
trunk serve --port 8080
cd ui && npx playwright test npc-perf --reporter=line
```

## WASM Build Fixes (applied in this branch)

The WASM build was broken since commit 3ae6f54. Fixes:
- `std::time::Instant` -> `bevy::platform::time::Instant` (WASM compatible)
- `std::time::SystemTime` -> `js_sys::Date::now()` for random seeding
- Atmosphere component cfg-gated to native only (needs storage buffers)
- BSL shader embedded at compile time (WASM asset server meta issue)
- Assets dir copied to dist/ via trunk config
