# Horizon Problems: Analysis and Approaches

## Status

**SOLVED: LUT banding (Problem 2)** — Coordinate normalization (commit e6965fd) fixed concentric rings by keeping Bevy-space bounded at ~800 units.

**UNSOLVED: Double horizon (Problem 1)** — Terrain edge vs atmosphere horizon. 16 approaches tried, none fully work. Next step: LOD imposters (approach E).

## The Double Horizon

Two visible horizons at any zoom layer:
1. **Terrain edge** — where rendered meshes stop (~800 normalized Bevy units)
2. **Atmosphere horizon** — the real sky rendered by Bevy's `Atmosphere`

The gap between them shows sky where terrain should be. The terrain is dark green; the atmosphere at ground level is lighter desaturated green. The mismatch creates a visible seam.

## The Fundamental Problem

**There is no geometry between the terrain edge and the atmosphere horizon.** Every approach that operates on mesh fragments (fog, blur, shader effects) can only affect the terrain itself — it cannot fill the void beyond the terrain edge. The only solution is to put geometry there.

This was confirmed by testing 16 approaches across two sessions:
- Approaches 1-11: Fog, color matching, shader tricks — all fail because they can't fill the void
- Approaches 12-16: Transmittance fix, DOF blur, DistanceFog + DOF combo — confirmed the fundamental issue: mesh-only effects produce a visible band ON the terrain but leave the gap untouched

## Why This Is Hard

**Bevy's atmosphere is not like Minecraft's sky.** Minecraft uses a simple colored plane + fog color matching. Bevy uses physically-based Rayleigh/Mie scattering with LUTs. The atmosphere's horizon color is dynamic (changes with sun angle, altitude, view direction) and cannot be easily approximated with a static color.

**The core constraint: we can't increase render distance** (RADIUS_VIEW_CELLS) because of CPU cost in the tree walk/bake.

## All 16 Approaches Tried

### Session 1 (ambiance-shaders branch)

| # | Approach | Why It Failed |
|---|----------|---------------|
| 1 | Custom shader fog (static color) | Fog color doesn't match atmosphere → creates additional horizon band |
| 2 | Atmosphere LUT concentration (0.6x) | Worsened banding at zoomed-out layers |
| 3 | Shader desaturation at distance | Can't fill void beyond terrain edge |
| 4 | Bevy DistanceFog component | Only affects mesh fragments, can't fill void |
| 5 | ClearColor matching (grey-green) | Static color can't match dynamic atmosphere |
| 6 | ClearColor black | Eliminated bleed-through band but terrain/sky gap remains |
| 7 | Ground plane mesh | Horizons conceptually merge, but quad is visible and wrong color for varied terrain |
| 8 | Two-tier LOD shell (coarser nodes) | Coarser nodes render as full 3D blocks sticking up |
| 9 | DistanceFog + black ClearColor | Fog darkens terrain edge but color mismatch with atmosphere |
| 10 | Far plane clipping to render radius | Sphere cull means terrain doesn't fill full frustum — gap remains |
| 11 | Minecraft-style fog=sky=clear color matching | Bevy's dynamic atmosphere color can't be matched with a static fog color |

### Session 2 (horizon-fix branch)

| # | Approach | Why It Failed |
|---|----------|---------------|
| 12 | Atmosphere transmittance fix (render_sky.wgsl: `transmittance = vec3(0.0)` for sky pixels) | A/B comparison showed zero visual difference — ClearColor bleed is not the visible problem |
| 13 | Increased atmosphere density (`scene_units_to_m * 3`) | Zero visual difference in automated comparison |
| 14 | DistanceFog (gray tint) | Creates a visible 2D gray band on the terrain. Doesn't fill the gap. Flat, not 3D. |
| 15 | BSL shader distance fog (blend toward horizon color) | Same problem as #14 — only affects mesh fragments. Also broke block interaction. |
| 16 | Bevy DOF (Depth of Field) + DistanceFog | DOF blur works (after fixing sensor_height for our coordinate scale). But fog+blur only affect mesh fragments — the gap between terrain and atmosphere remains. Creates a blurred gray band ON the terrain, gap untouched. |

### Key Debugging Notes from Session 2

- **DOF requires correct sensor_height**: Default 18.66mm produces 0.19px CoC at 800 units (invisible). Need `sensor_height: 10.0` for ~21px CoC at terrain edge. DOF math is designed for real-world meter distances, not ~800-unit game coordinates.
- **`DEPTH_TEXTURE_SAMPLING_SUPPORTED`**: On WASM, DOF requires the `webgpu` Bevy feature (WebGL2 doesn't support depth texture sampling). The constant is in `bevy_core_pipeline/src/core_3d/mod.rs`.
- **Atmosphere compositing**: `render_sky.wgsl` runs between `main_opaque_pass_3d` and `main_transparent_pass_3d`. Post-process effects (DOF, bloom) run after. DistanceFog runs during PBR lighting (before atmosphere).
- **Transmittance fix**: Setting `transmittance = vec3(0.0)` for `depth == 0` in `render_sky.wgsl` prevents ClearColor bleed at the horizon. Technically correct but produces no visible improvement because the ClearColor bleed wasn't the primary visual issue.

## WASM/WebGPU Testing Infrastructure

Established during session 2 for iterative visual testing:

- **WebGPU on WASM**: Add `"webgpu"` to Bevy features in Cargo.toml. Overrides default `webgl2`. Enables atmosphere, DOF, SSR on WASM.
- **Playwright with real Chrome**: Bundled headless Chromium can't capture WebGPU canvas. Use `channel: "chrome"` + `--enable-gpu` + `--enable-unsafe-webgpu` + `--use-angle=metal` for real GPU rendering in headless mode.
- **Canvas pixel readback**: `toDataURL`/`drawImage` returns `[0,0,0,0]` on WebGPU. Only `page.screenshot()` works.
- **Pointer Lock**: Doesn't work in headless — camera can't be rotated via mouse in tests.
- **Custom port**: `trunk serve --port 8082` + `TRUNK_PORT=8082 npx playwright test`.
- **Worktree setup**: Symlink `external/` and `target/`, create `.claude/`, `ui/dist/`, `tests/`, `docs/`, `dist/` dirs, `npm install && npm run build` in ui/.

See `docs/testing.md` for full details.

## The Solution: LOD Imposters (Approach E)

The only approach that can solve the double horizon is **putting geometry in the gap**. Every shipped open-world game does this — Minecraft, Skyrim, GTA, No Man's Sky all render low-detail terrain to the horizon.

### Design

Beyond the normal render radius (`RADIUS_VIEW_CELLS`), emit cheap geometry for an additional ring of coarser tree nodes:

1. **Inner ring** (0 to RADIUS_VIEW_CELLS): Full voxel meshes at emit layer (current behavior)
2. **Outer ring** (RADIUS_VIEW_CELLS to 2× or 3×): Flat quads or low-res heightmaps per coarser node, colored by dominant surface voxel

The atmosphere's aerial-view LUT naturally fogs the outer ring, blending it into the sky. DistanceFog can further soften the transition.

### Requirements

- Extract surface Y-level and dominant color from coarser tree nodes
- Generate simple quad or heightmap mesh per outer-ring node
- Must work with both flat grassland AND sphere/terrain (the sphere-planet worktree uses the same render walk and has the same horizon problem)
- Must be cheap enough to not blow the CPU budget (the whole point is these are simpler than full voxel bakes)

### Open Questions

- How many extra rings? 1 ring at 2× radius, or multiple tiers?
- Flat quads or heightmap extraction? Flat is simpler, heightmap is more correct for terrain.
- How to handle the transition between full-detail inner ring and imposter outer ring?
