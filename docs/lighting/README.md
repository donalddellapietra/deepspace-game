# Lighting roadmap

This directory is the plan for evolving the renderer from its current
state — direct sun + constant ambient, no GI, no PBR — to a full
deferred indirect-lighting pipeline, **without breaking the recursive
content-addressed architecture**.

The reference we're benchmarking against is the voxel-raymarching
project in `external/voxel-raymarching/`. Its renderer at 1920×1080 on
Sponza (4.5M voxels) runs at ~17 ms/frame on the same macOS hardware.
Ours currently renders a simpler frame but is slower per-ray because we
lack the scene-scale streaming machinery that makes secondary passes
affordable.

## Thesis

**Perf infrastructure before lighting.** Every lighting feature (GI,
specular, shadows, denoisers) dispatches over the set of surfaces the
primary raymarch hit. Without a visibility buffer and indirect
dispatch, secondary passes run per-pixel (~2M threads/frame) instead
of per-hit-voxel (~50k). That's a 40× multiplier on every lighting
pass, and it's the reason "add GI" sounds trivial on the reference
codebase and terrifying on ours.

The chunk pool, visibility buffer, indirect dispatch, and `.vxs`
on-disk format are **one coupled system** — skipping any piece breaks
the chain. Phase 1 is non-negotiable.

## Recursive-architecture invariants

Every lighting feature must honor these:

1. **Every layer identical.** No special-case leaf layer. Probes,
   chunks, visibility, and indirect args live at every depth.
   Exception (per memory): sphere-related objects (planets, moons,
   stars) may have scale-specific code.
2. **Content-addressed chunks.** Chunk pool slots are keyed by
   `NodeId`, not by scene coordinates. Two repeating subtrees share
   one slot automatically.
3. **Canned, not procedural.** GI probes, IBL maps, and material atlas
   are baked offline by `tools/scene_voxelize` (and descendants),
   never computed at runtime.
4. **No hardcoded caps.** Pool sizes, probe counts, and visibility
   buffer sizes derive from tree depth and layer count, not from
   magic numbers.
5. **WASM-compatible.** Every buffer fits WebGPU storage-buffer
   limits (128 MB per binding, 512 MB total).

## Phase graph

```
             Phase 1: Perf infra
      (chunk pool, visibility, indirect, .vxs)
                     │
                     ▼
        Phase 2: Deferred + PBR + postprocess
        (G-buffer, PBR palette, tonemap, FXAA)
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    Phase 3:     Phase 4:     Phase 6:
    Shadows      IBL          Specular
    (sun rays)   (skybox)     (reflections)
                     │
                     ▼
              Phase 5: Diffuse GI
       (probes, ambient trace, denoiser)
```

- **Phase 2 depends on 1**: G-buffer needs the visibility buffer to
  pack material IDs efficiently; PBR palette is loaded from `.vxs`.
- **Phase 3 depends on 2**: shadow rays need G-buffer depth + normal.
- **Phase 4 depends on 2**: IBL is sampled during the deferred
  resolve pass using G-buffer normal.
- **Phase 5 depends on 4**: probes use IBL as the sky-visibility
  fallback when a ray escapes the scene.
- **Phase 6 depends on 2**: specular trace reads G-buffer roughness
  and uses the same march primitive.

Phases 3, 4, 6 can land in parallel after Phase 2. Phase 5 is the
most complex and should land last.

## Phases

1. [phase-1-perf-infra.md](phase-1-perf-infra.md) — chunk pool, visibility buffer, indirect dispatch, `.vxs` on-disk format.
2. [phase-2-deferred-pbr.md](phase-2-deferred-pbr.md) — G-buffer pass, PBR material palette, deferred lighting, tonemap + FXAA.
3. [phase-3-shadows.md](phase-3-shadows.md) — sun shadow rays dispatched over visible voxels.
4. [phase-4-ibl.md](phase-4-ibl.md) — skybox lightmap, prefiltered irradiance + radiance cubemaps.
5. [phase-5-diffuse-gi.md](phase-5-diffuse-gi.md) — per-layer irradiance probe volumes, ambient trace, screen-space denoiser.
6. [phase-6-specular.md](phase-6-specular.md) — glossy specular trace + spatial denoise + resolve.

## Acceptance for each phase

Every phase ships with:

- A deterministic test in the harness that renders a canonical scene
  and compares against a reference screenshot.
- A perf number: target GPU time per pass on macOS M-series at 1920×1080.
- A commit (or small stack) on a dedicated worktree. Never merge to
  `fractal-presets` until the phase is stable.
- Updated CLAUDE memory if a non-obvious decision was made.

## Non-goals

- **Path tracing.** Too expensive for interactive. We ship probes +
  screen-space filters like the reference does.
- **Dynamic content-addressed rebuild on edit.** Edits already create
  new NodeIds via clone-on-write. The chunk pool naturally handles
  this — old NodeIds stay cached, new ones load on first ray hit.
- **Mesh rendering paths.** Voxel-only. The reference has
  `rasterized.wgsl` for mesh fallback; we don't need it.
- **SSAO / SSGI.** Screen-space GI is strictly worse than our
  probe-based GI and doesn't compose with layer transitions.

## Estimated total scope

~4k–6k LoC net across Rust + WGSL, landing over ~6 worktrees. Each
phase is 500–1500 LoC. The biggest phases are 1 (infra rewrite) and
5 (probes + denoiser).
