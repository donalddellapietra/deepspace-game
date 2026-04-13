# Horizon Problems: Analysis and Approaches

## Status

**Problem 2 (LUT banding) is SOLVED** — coordinate normalization (commit e6965fd) fixed the concentric rings by keeping Bevy-space bounded at ~800 units across all zoom layers.

**Problem 1 (double horizon) remains** — terrain edge vs atmosphere horizon. Down from triple to double after ClearColor fix (black background).

## The Double Horizon

At any zoom layer, two visible horizons:

1. **Terrain edge** — where rendered meshes stop (sphere of `RADIUS_VIEW_CELLS * cell_bevy` ≈ 800 Bevy units)
2. **Atmosphere horizon** — the real sky/ground horizon rendered by Bevy's `Atmosphere`

Between them: the atmosphere draws sky (since `depth == 0` there), but the sky at ground level doesn't match the terrain color. The terrain is dark green; the atmosphere's ground-level sky is a lighter desaturated green-blue. The mismatch creates a visible seam.

## Why This Is Hard

The fundamental constraint: **there is no geometry between the terrain edge and the atmosphere horizon.** The atmosphere's `render_sky.wgsl` composites as `inscattering + transmittance * background`. Where there's no mesh, it draws sky. No fog, color matching, or shader trick can create geometry where none exists.

## All Approaches Tried (10 total)

| # | Approach | Result |
|---|----------|--------|
| 1 | Custom shader fog (static color) | Created 4th horizon — fog color didn't match atmosphere |
| 2 | Atmosphere LUT concentration (0.6x) | Worsened banding at zoomed-out layers |
| 3 | Shader desaturation at distance | No visible effect — can't fill void |
| 4 | Bevy DistanceFog component | Only affects mesh fragments, can't fill void |
| 5 | ClearColor matching (grey-green) | Partially helped but static color can't match dynamic atmosphere |
| 6 | ClearColor black | Eliminated one horizon band (no more bleed-through), but terrain/sky gap remains |
| 7 | Ground plane mesh | Horizons merge when jumping! But quad is visible, not future-proof for varied terrain |
| 8 | Two-tier LOD shell (coarser nodes) | Coarser nodes render as full 3D blocks sticking up — wrong mesh type for distance |
| 9 | DistanceFog + black ClearColor combo | Fog darkens terrain edge but can't fill void and color doesn't match atmosphere |
| 10 | Far plane clipping to render radius | Clips terrain but doesn't help — sphere cull means terrain doesn't fill the full frustum |

## Key Insights

1. **The ground plane (approach 7) proved the concept works.** When geometry extends to the horizon, the atmosphere fogs it naturally and the horizons merge. The problem was only that a single flat quad is too crude.

2. **The LOD shell (approach 8) is the right architecture** but the wrong mesh. Full voxel bakes at coarser layers produce tall 3D blocks. What's needed is a **flat, top-face-only mesh** — essentially a color swatch at ground level that gives the atmosphere something to fog without sticking up.

3. **ClearColor = black (approach 6) should be kept** regardless of final solution — it eliminates the atmosphere bleed-through band.

4. **Coordinate normalization (Option F, now implemented)** was essential — without it, any atmospheric effect bands at zoomed-out layers.

## Recommended Path: LOD Shell with Flat Imposters

The LOD shell approach is correct but needs a different mesh strategy:

### What went wrong with approach 8
The walk emitted coarser tree nodes and they were baked with the full `BakedNode::new_cold` pipeline, producing 3D voxel meshes with all six face directions. At distance, these appear as blocks sticking up above the terrain.

### What's needed
For the outer shell, don't bake full voxel meshes. Instead, for each coarser node in the shell:
1. Read the **top-face color** from the node's downsample data (the dominant surface voxel)
2. Emit a **single flat quad** at the node's Y position (ground level), colored by that voxel
3. The quad is just the top face of the node — no sides, no bottom
4. The atmosphere's aerial perspective fogs these quads at distance, blending them into the horizon

This is essentially the ground plane approach but with **per-node color** and **proper Y positioning** from the tree data. It adapts to any terrain, any biome.

### Implementation sketch
- In the walk, nodes beyond `inner_radius` but within `outer_radius` get flagged as "shell" visits
- The reconciler for shell visits spawns a simple flat quad mesh (shared across all shell nodes) with the node's dominant color as the material
- The quad sits at the node's origin Y + ground-surface offset, spans the node's XZ footprint
- 1-2 layers coarser than the main emit layer, extending 3-5x the render distance

### Performance
At 3x render distance with nodes one layer coarser (5x bigger): the shell ring covers `(3r)² - r²` area = 8r². Each coarser node covers 25x the area. So the shell adds `8r² / 25 ≈ 0.32 * r²` nodes. For r = 32 cells, that's ~330 flat quads. Negligible.
