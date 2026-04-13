# Horizon Problems: Analysis and Approaches

## Status

**SOLVED: LUT banding (Problem 2)** — Coordinate normalization (commit e6965fd) fixed concentric rings by keeping Bevy-space bounded at ~800 units.

**UNSOLVED: Double horizon (Problem 1)** — Terrain edge vs atmosphere horizon. 11 approaches tried, none fully work.

## The Double Horizon

Two visible horizons at any zoom layer:
1. **Terrain edge** — where rendered meshes stop (~800 normalized Bevy units)
2. **Atmosphere horizon** — the real sky rendered by Bevy's `Atmosphere`

The gap between them shows sky where terrain should be. The terrain is dark green; the atmosphere at ground level is lighter desaturated green. The mismatch creates a visible seam.

## Why This Is Hard

**There is no geometry between the terrain edge and the atmosphere horizon.** The atmosphere's `render_sky.wgsl` composites as `inscattering + transmittance * background`. Where there's no mesh, it draws sky. No shader trick can create pixels where no mesh exists.

**Bevy's atmosphere is not like Minecraft's sky.** Minecraft uses a simple colored plane + fog color matching. Bevy uses physically-based Rayleigh/Mie scattering with LUTs. The atmosphere's horizon color is dynamic (changes with sun angle, altitude, view direction) and cannot be easily approximated with a static color. This is why the Minecraft-style "fog color = sky color" approach fails — the static fog color never matches the dynamic atmosphere.

## All 11 Approaches Tried

| # | Approach | Why It Failed |
|---|----------|---------------|
| 1 | Custom shader fog (static color) | Fog color doesn't match atmosphere → creates additional horizon band |
| 2 | Atmosphere LUT concentration (0.6x) | Worsened banding at zoomed-out layers |
| 3 | Shader desaturation at distance | Can't fill void beyond terrain edge |
| 4 | Bevy DistanceFog component | Only affects mesh fragments, can't fill void |
| 5 | ClearColor matching (grey-green) | Static color can't match dynamic atmosphere |
| 6 | ClearColor black | Eliminated bleed-through band but terrain/sky gap remains |
| 7 | Ground plane mesh | Horizons merge! But quad is visible, wrong color for varied terrain |
| 8 | Two-tier LOD shell (coarser nodes) | Coarser nodes render as full 3D blocks sticking up |
| 9 | DistanceFog + black ClearColor | Fog darkens terrain edge but color mismatch with atmosphere |
| 10 | Far plane clipping to render radius | Sphere cull means terrain doesn't fill full frustum — gap remains |
| 11 | Minecraft-style fog=sky=clear color matching | Bevy's dynamic atmosphere color can't be matched with a static fog color — the fog darkens terrain but doesn't match the atmosphere's actual horizon rendering |

## Key Insights

1. **The ground plane (approach 7) is the only approach that actually merged the horizons.** When geometry extends to the horizon, the atmosphere naturally fogs it. Every other approach tries to hide the gap; only geometry fills it.

2. **Bevy's atmosphere is the obstacle.** If we used a simple colored skybox (like Minecraft), fog color matching would work trivially. The physically-based atmosphere produces a dynamically varying horizon color that can't be matched with a static fog color.

3. **The core constraint: we can't increase render distance** (RADIUS_VIEW_CELLS) because of CPU cost in the tree walk/bake.

## Approaches NOT Yet Tried

### A: Disable Bevy Atmosphere, Use Minecraft-Style Sky
Replace Bevy's `Atmosphere` with a simple colored sky (skybox or procedural gradient). Then fog color = sky color at horizon, matching Minecraft's proven technique. Lose the fancy scattering but get a seamless horizon.

**Pros:** Proven to work (every voxel game does this). Simple. Fast.
**Cons:** Lose the beautiful atmospheric scattering. Might look "gamey."

### B: Read Atmosphere Color from Sky-View LUT at Runtime
Sample the atmosphere's sky-view LUT at the horizon direction each frame to get the actual dynamic horizon color. Use that as the fog color and ClearColor. This is the "match the atmosphere dynamically" approach.

**Pros:** Works with the existing atmosphere. Fog always matches sky.
**Cons:** Requires reading GPU texture back to CPU, or passing it through a compute shader. Non-trivial Bevy render pipeline work.

### C: Custom Post-Process that Reads Depth + Sky-View LUT
A full-screen post-process that, for pixels with no geometry (depth == far plane), samples the atmosphere's sky-view LUT directly and composites it. This would replace ClearColor entirely with the actual atmosphere color, even in the gap.

**Pros:** Perfect visual result. Works with any terrain.
**Cons:** Requires implementing a custom ViewNode in Bevy's render graph. The sky pass already does this (`render_sky.wgsl`), but the compositing math (transmittance blending) leaks ClearColor through. Fixing the compositing is the real fix.

### D: Fix the Atmosphere Compositing
The root issue: `render_sky.wgsl` outputs `vec4(inscattering, mean_transmittance)` which blends with the framebuffer. At the horizon, transmittance is non-zero, so ClearColor bleeds through. If the sky pass wrote opaque pixels (transmittance → 0) for the sky case (`depth == 0`), there would be no bleed. This might be a one-line fix in the sky shader if Bevy exposes it, or it could require forking the atmosphere plugin.

**Pros:** Fixes the actual bug. Everything else stays the same.
**Cons:** Modifying Bevy's built-in atmosphere shader. May need a Bevy fork or a shader override.

### E: Flat LOD Imposters (revisit approach 8, done right)
The LOD shell is the right concept but needs flat meshes, not full voxel bakes. For each coarser node in the outer ring, emit a single flat quad at the node's surface Y-level, colored by the node's dominant surface voxel. With future mountains/valleys, extract a heightmap from the coarser node's voxel grid instead of a flat quad.

**Pros:** Real geometry that the atmosphere fogs naturally. Adapts to terrain. Proven concept (ground plane worked).
**Cons:** Needs new mesh generation path for the outer tier. Heightmap extraction adds complexity with non-flat terrain.
