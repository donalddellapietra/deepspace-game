# Horizon Problems: Analysis and Failed Approaches

## Current State (commit 7ba5af2)

Two distinct problems visible when zooming out:

### Problem 1: Triple Horizon

At any zoom layer, three distinct visual bands appear between terrain and sky:

1. **Terrain edge** — the hard cutoff where rendered meshes stop (at `RADIUS_VIEW_CELLS * cell_size_at_layer(zoom.layer)` Bevy units)
2. **ClearColor band** — the gap between terrain edge and atmosphere, filled by `ClearColor(0.7, 0.78, 0.65)`
3. **Atmosphere horizon** — the actual sky rendered by Bevy's `Atmosphere` component

The atmosphere renders as a full-screen pass behind all geometry. Where there are no meshes, the ClearColor shows. The atmosphere then composites on top. But the atmosphere's aerial perspective (haze/fog) only applies to mesh fragments — it cannot fill the void between the last mesh and the sky.

### Problem 2: Black Spots and Waves at Layer 9 and Below

When zooming out past layer 10, black artifacts and wavy distortions appear at the terrain edge. These get worse at lower layers (9, 8, etc.), suggesting a **coordinate scaling issue**.

At layer 9: `cell = 5^3 = 125`, `radius = 32 * 125 = 4000 Bevy units`. The atmosphere's `scene_units_to_m = 1/125 = 0.008`. The `aerial_view_lut_max_distance` is `4000 * 0.6 / 125 = 19.2 meters`. This tiny distance means the aerial-view LUT is trying to distribute fog samples over just 19 meters, which may cause precision issues or sampling artifacts in the LUT texture.

The waves likely come from the atmosphere LUT running out of resolution — 32 depth slices (default `aerial_view_lut_size.z`) spread over such a small meter-distance causes banding/aliasing.

## Approaches Tried

### Approach 1: Custom Shader Fog (commit 7ba5af2)
Added `fog_start`/`fog_end` to BslParams, computed distance from camera in the fragment shader, blended toward a static fog color `(0.7, 0.78, 0.72)`.

**Result:** Created a FOURTH horizon — the shader fog color didn't match the atmosphere, producing another visible band. The shader fog only applies to existing mesh fragments; it cannot fill the void beyond the render distance.

### Approach 2: Atmosphere LUT Concentration
Set `aerial_view_lut_max_distance = view_radius * 0.6 / cell` to concentrate atmosphere fog samples closer, making haze thicker near the terrain edge.

**Result:** Made the black spots/waves worse at lower layers because 0.6x of an already-small meter distance further compressed the LUT.

### Approach 3: Shader Desaturation at Distance
Instead of painting a fog color, desaturated and lightened terrain at the render edge.

**Result:** No visible effect — the terrain edge is still a hard cutoff, and desaturation doesn't help when the problem is the void beyond terrain.

### Approach 4: Bevy DistanceFog Component
Added `DistanceFog` with `FogFalloff::Linear` to the camera, with `sync_distance_fog` updating start/end per zoom.

**Result:** No visible effect. The fog IS applied (Bevy calls `apply_fog` inside `main_pass_post_lighting_processing`), but it only affects mesh fragments. The void beyond the render distance is still filled by ClearColor, not fog. DistanceFog fades meshes toward its color but can't create new pixels where there are no meshes.

### Approach 5: ClearColor Matching
Changed ClearColor from grass green `(0.3, 0.6, 0.2)` to atmosphere haze `(0.7, 0.78, 0.65)`.

**Result:** Partially helped — the terrain-to-void transition is less jarring — but it's a static color that can't match the atmosphere's dynamic haze which changes with camera angle, sun position, and altitude.

## Root Cause Analysis

The fundamental issue is architectural: **there is a void between the last rendered mesh and the atmosphere sky, and nothing fills it.**

- The render system stops emitting meshes at `RADIUS_VIEW_CELLS * cell`.
- Beyond that, only `ClearColor` exists.
- The atmosphere renders the sky, but the aerial perspective (fog/haze applied to surfaces) requires mesh fragments to exist.
- Any fog approach that only modifies existing fragments (shader fog, DistanceFog, desaturation) cannot solve this — the void has no fragments to modify.

## Possible Solutions (Untried)

### A: Ground Plane Mesh
Spawn a large flat mesh at ground height extending to the atmosphere horizon. This gives the atmosphere aerial perspective and distance fog something to render on. The ground plane would be a single giant quad at `y = ground_level`, colored to match grass, extending well past the render radius. Distance fog would fade it into the atmosphere naturally.

**Pros:** Simple, works with all existing fog/atmosphere systems.
**Cons:** Only works for flat terrain; at ground level the camera might see the edge; needs per-layer scaling.

### B: Skybox/Skydome with Terrain Color
Render a skydome or cylinder at the render distance that's colored to match the terrain at the bottom and transitions to sky color at the top. This fills the void with a gradient that bridges terrain and sky.

**Pros:** Works at any altitude, fills the void properly.
**Cons:** Needs to match terrain color dynamically; doesn't interact with atmosphere fog.

### C: Extend Render Distance with LOD Imposters  
Instead of hard-stopping at `RADIUS_VIEW_CELLS`, render simplified geometry (flat quads per chunk, or a heightmap mesh) beyond the main render distance. These imposters would be cheap and give the atmosphere something to fog.

**Pros:** Most realistic result; matches Distant Horizons approach from the Minecraft pack.
**Cons:** Complex to implement; needs a second render pass or simplified mesh generation.

### D: Fix the Atmosphere to Not Need Meshes
Use the atmosphere's raymarched mode instead of LUT mode, which might handle the ground plane implicitly. Or post-process: render the scene, then composite the atmosphere on top with proper depth-aware blending that treats the ClearColor regions as "ground level" rather than "infinitely far sky."

**Pros:** Cleanest solution.
**Cons:** Raymarched mode is more expensive; modifying the atmosphere compositor is deep engine work.

### E: Fix Aerial-View LUT Banding (CONFIRMED ROOT CAUSE of artifacts)

**Layer 6 screenshot proves this.** Concentric circles centered on camera = depth-slice banding in the aerial-view LUT.

The math:
- `aerial_view_lut_max_distance = RADIUS_VIEW_CELLS * 0.6 = 19.2 meters` (constant after simplification)
- Default LUT has 32 depth slices → each slice = 0.6 meters
- `scene_units_to_m = 1/cell`. At layer 6, cell = 15625
- 0.6 meters = 0.6 * 15625 = 9375 Bevy units per depth slice
- View radius = 500,000 Bevy units → ~32 visible bands (matches screenshot exactly)

At layer 12: 0.6m = 0.6 Bevy units per slice → invisible. Only appears when zoomed out.

**Attempted fixes (all failed):**

1. **Increase `aerial_view_lut_size.z` dynamically** — The needed slices scale with `cell`, which is 15625 at layer 6. Even capped at 256 slices, each still covers thousands of Bevy units. Impossible to fix within the LUT architecture when the Bevy-space view distance is 500k units.

2. **Floor `scene_units_to_m` at 0.01** — Doesn't help: the ratio view_radius/slices in Bevy-space is the fundamental problem, not the meter conversion.

3. **Set `scene_units_to_m = 1.0` always** — Same problem: view_radius at layer 6 is 500k Bevy units, 32 slices → 15625 BU/slice regardless of meter scaling.

4. **Cap `aerial_view_lut_max_distance` to 100m** — Restricts aerial perspective to near-camera, but same banding visible within that range since cell multiplier still applies.

**Root cause is architectural:** The renderer scales Bevy-space coordinates by `5^(MAX_LAYER - layer)` per zoom level. At layer 6, meshes span 500k Bevy units. Bevy's atmosphere (and all post-processing that uses screen depth) sees these as real coordinates, and any depth-sliced LUT will band because the ratio of scene extent to LUT resolution is unbounded.

**The real fix (Option F):** Normalize all rendered geometry to a fixed Bevy-space range regardless of zoom layer. Instead of scaling `Transform` by `5^(MAX_LAYER-layer)`, keep all node meshes at scale 1.0 and adjust the camera/viewport. From Bevy's perspective, every layer should look geometrically identical — same coordinate range, same distances, same atmosphere parameters. The zoom effect comes from what voxel data is shown, not from coordinate scaling.
