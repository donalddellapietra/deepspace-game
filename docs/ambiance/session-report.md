# Ambiance Session Report (2026-04-12)

## What We Built (working, committed)

### 1. BSL-Style Ambiance
- **AcesFitted tonemapping** — filmic S-curve, previously caused shadow acne (fixed by HDR rebalancing)
- **Bloom** — threshold 0.8, energy-conserving, max_mip_dimension 256 for perf
- **Color grading** — BSL-style: shadow saturation 1.1, shadow lift 0.01, subtle contrast
- **SSAO** — High quality, constant_object_thickness 4.0 to reduce graininess
- **Warm sun / cool ambient** — directional light `(1.0, 0.98, 0.93)` at 10k illuminance, ambient `(0.8, 0.85, 1.0)` at 600 brightness
- **Enhanced BSL shader** — smoothstep shadow blending, base-color-aware ambient tint, improved SSS with power-6 falloff and warm tint

### 2. Coordinate Normalization (architectural change)
Added `norm` field to `WorldAnchor` that divides all leaf-to-Bevy conversions by `scale_for_layer(target_layer)`.

- Keeps Bevy-space bounded at ~800 units regardless of zoom layer
- Fixed atmosphere aerial-view LUT banding (concentric rings at layer 6+)
- 9 files changed, 75 tests pass
- Render entities always at scale 1.0
- Collision/player physics unchanged (operate in leaf space)
- `anchor.cell_bevy(layer)` helper for Bevy-space cell sizes

### 3. Zoom Transition Fix
`ZoomTransition.start()` computes each layer's cell size using its own target norm via `scale_for_layer(layer) / scale_for_layer(target_layer_for(layer))`, avoiding stale-norm animation bugs.

### 4. Shader Fog Cleanup
Removed leftover custom distance fog (BslParams fog_start/fog_end, sync_fog_distances system) from failed horizon approach.

## What Remains Unsolved

### The Double Horizon
Terrain edge vs atmosphere horizon. 11 approaches tried, none fully work. See `docs/ambiance/horizon-problems.md` for full analysis.

**Root cause:** No geometry between terrain edge and atmosphere. Bevy's physically-based atmosphere produces dynamic horizon colors that can't be matched with a static fog color.

**Most promising untried approaches:**
1. **Fix atmosphere compositing** — `render_sky.wgsl` transmittance bleed lets ClearColor leak through. Making sky pixels opaque would fix it.
2. **Flat LOD imposters** — only approach that actually merged horizons (ground plane proved it). Needs per-node surface meshes.
3. **Replace atmosphere with simple sky** — nuclear option, guaranteed to work.

## Files Changed

| File | Change |
|------|--------|
| `assets/shaders/bsl_voxel.wgsl` | BSL shader: SSAO curve, ambient blending, SSS |
| `src/block/bsl_material.rs` | Tuned ambient color, AO strength 0.6 |
| `src/block/mod.rs` | (unchanged) |
| `src/camera.rs` | AcesFitted, bloom, color grading, SSAO, atmosphere sync, zoom transition fix |
| `src/main.rs` | HDR lighting rebalance, ClearColor, shadow cascades with normalized coords |
| `src/world/view.rs` | WorldAnchor.norm, delta_as_vec3 normalization, cell_bevy() helper |
| `src/world/render.rs` | Normalized walk coords, scale=1.0 entities |
| `src/world/collision.rs` | local_anchor norm field |
| `src/player.rs` | sync_anchor_to_player sets norm, spawn_anchor with norm |
| `src/editor/tools.rs` | zoom_in/out transition.start() updated |
| `src/interaction/mod.rs` | cell_bevy for Bevy-space distances |
| `src/npc.rs` | cell_bevy for spawn/scale, leaf-space for physics |
| `docs/ambiance/` | Horizon problem analysis, session report |

## Branch State

- Branch: `ambiance-shaders` at commit `4651476`
- Worktree: `/Users/donalddellapietra/GitHub/deepspace-game-ambiance`
- All changes pushed to origin
- ClearColor: `(0.7, 0.78, 0.65)`
- No fog active
- Coordinate normalization active
