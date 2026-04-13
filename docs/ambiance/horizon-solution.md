# Horizon Solution: Shader Clip

## The Working Approach

The double horizon problem is solved by a single shader-level change: **discard terrain fragments beyond the render radius in XZ distance from the camera.**

### How It Works

1. **BslParams** gains a `clip_radius` field (f32, uniform)
2. The BSL voxel fragment shader checks XZ distance from camera to fragment
3. Fragments beyond `clip_radius` are discarded
4. `clip_radius` is set to `radius_bevy` each frame from `render_world`

```wgsl
// In bsl_voxel.wgsl, at the top of fragment():
if (bsl.clip_radius > 0.0) {
    let dx = in.world_position.x - view.world_position.x;
    let dz = in.world_position.z - view.world_position.z;
    if (dx * dx + dz * dz > bsl.clip_radius * bsl.clip_radius) {
        discard;
    }
}
```

### Why It Works

The original double horizon had two problems:
1. **Patchy terrain edge** — sphere culling of cube blocks creates an irregular boundary
2. **Gap between terrain and atmosphere** — no geometry between terrain edge and sky

The shader clip fixes #1 by creating a **perfectly smooth circular boundary**. And #2 was less severe than we thought — the atmosphere's aerial-view inscattering already fogs distant terrain, creating a natural transition to sky. We couldn't see this before because the patchy edge was so visually distracting.

### Why Previous Approaches Failed

| Approach | Why It Failed |
|----------|---------------|
| Flat annulus | Color mismatch — flat PBR surface gets different hue than 3D blocks (warm vs cool from BSL shader). Visible through dug holes. |
| Cylinder wall | Creates visible vertical wall line at render boundary |
| DOF blur | Only blurs mesh fragments, can't fill the gap |
| DistanceFog | Only tints mesh fragments, can't fill the gap |
| Raising annulus Y | Creates a second visible horizon (floating band) |
| Vertex AO on annulus | Can darken but can't shift hue — annulus stays warm/yellow |

### What's Disabled (TODOs)

- **SSAO** — Creates a dark halo ring at the terrain clip boundary due to depth discontinuity. Confirmed via A/B test. Needs edge-aware depth masking to re-enable. Vertex AO in BSL shader still provides per-block occlusion.

### Files Changed

- `assets/shaders/bsl_voxel.wgsl` — Added `clip_radius` to BslParams, fragment discard, `mesh_view_bindings::view` import
- `src/block/bsl_material.rs` — Added `clip_radius: f32` field to BslParams, default 0.0
- `src/world/render.rs` — Sets `clip_radius = radius_bevy` on all materials each frame via `mat_assets.iter_mut()`
- `src/camera.rs` — SSAO disabled (TODO: re-enable with edge masking)

### Properties

- Works at all zoom layers (clip_radius scales with cell size)
- No color matching needed (no separate imposter geometry)
- No digging artifacts (no flat surface below terrain)
- Block interaction works (discard only affects rendered fragments, not raycasting)
- Single entity overhead: zero (no imposter mesh)
- Works with any terrain type (grass, mountains, sphere planet)
