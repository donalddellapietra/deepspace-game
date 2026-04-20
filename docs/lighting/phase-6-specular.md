# Phase 6 — Specular reflections

**Goal.** Glossy reflections beyond what IBL provides. A puddle
shows a smeared reflection of nearby geometry. A polished floor
reflects the columns above it. Metallic surfaces show crisp world
reflections, not just sky.

**Dependencies.** Phase 2 (needs G-buffer roughness + metallic).
Phase 4 (IBL prefilter is the fallback for rays that escape).

**Deliverables.**
- Specular ray trace pass (indirect dispatch over visible voxels)
- Spatial specular denoiser (reuses variance-clamping patterns)
- Specular resolve combines traced + prefiltered IBL based on roughness

## What you see after this phase

Screen-space reflections without the SSR artifacts (no missed
geometry outside the frustum, no screen edges cutting reflections
off). Rough surfaces get near-field reflections instead of
always-blurry IBL. Correct behavior on fractal-scale geometry.

## Architecture

### Specular trace

One ray per visible voxel, GGX-importance-sampled around the
reflection direction:

```wgsl
// specular.wgsl (new)
@compute @workgroup_size(64)
fn specular_trace(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= visibility.hit_count) { return; }

    let entry = hit_list[i];
    let pos = unpack_voxel_position(entry);
    let normal = unpack_voxel_normal(entry);
    let roughness = unpack_roughness(entry);
    let view_dir = unpack_view_dir(entry);

    // Importance-sampled half-vector for GGX
    let jitter = blue_noise_rotation(entry.voxel_slot, frame.id);
    let H = ggx_sample_half(normal, roughness, jitter);
    let L = reflect(-view_dir, H);

    let result = march(pos + normal * 0.001, L, frame.specular_max_distance);
    var radiance: vec3f;
    if (result.missed) {
        // IBL prefilter fallback
        radiance = textureSampleLevel(
            prefilter_tex, sampler_lin, L, roughness * 7.0
        ).rgb;
    } else {
        // Direct + ambient at the secondary hit
        radiance = result.direct + result.ambient * result.albedo;
    }
    hit_list[i].specular = pack_rgba16f(radiance);
}
```

The `result.direct + result.ambient * result.albedo` evaluation at
the secondary hit requires the secondary hit's G-buffer. Two options:

**Option A — one-bounce only with constant ambient.** At the
secondary hit, evaluate sun + sky only. Skip ambient at the bounce.
Cheap but looks slightly wrong on low-roughness near-mirror surfaces.

**Option B — sample the secondary hit's probe.** Requires probe
grid access at the bounce. More expensive but correct.

Go with A for Phase 6 initial; upgrade to B if it matters.

### Spatial denoise

Specular denoise is tricker than diffuse because roughness drives
the filter radius. Use a roughness-weighted spatial bilateral:

```wgsl
// specular_spatial.wgsl (new)
// Similar to atrous but filter radius scales with roughness
// and edge-stopping includes BRDF lobe alignment
```

Reference has `specular_spatial.wgsl` + `specular_resolve.wgsl`.
Port both. ~300 LoC WGSL combined.

### Resolve

Blend traced specular with prefiltered IBL based on confidence:

```wgsl
// deferred.wgsl (modified)
let traced_spec = textureLoad(specular_denoised, p, 0);
let ibl_spec = sample_ibl_specular(...);

// traced_spec.a is confidence (how many samples survived variance clamp)
let spec = mix(ibl_spec, traced_spec.rgb, traced_spec.a);

let color = (diffuse_ibl * (1 - F) + spec) * occlusion + direct + emission;
```

## Shaders touched

- **New:** `specular.wgsl`, `specular_spatial.wgsl`,
  `specular_resolve.wgsl`
- **Modified:** `deferred.wgsl` — specular is now a traced+IBL blend
- **Modified:** `bindings.wgsl` — specular_tex bind group

## Rust code touched

- **New:** `src/renderer/specular.rs` — specular pipelines
- **Modified:** `src/renderer/draw.rs` — sequence specular between
  ambient and resolve (can run in parallel with ambient trace on
  different hit_list slices)
- **Modified:** `src/renderer/gbuffer.rs` — adds `specular_tex`,
  `specular_denoised_tex`

## Recursive architecture integration

Identical to ambient (Phase 5). Specular rays march the same
primitive; secondary hits in different layers look up that layer's
G-buffer equivalents (only primary G-buffer exists — at bounce, we
estimate direct lighting analytically from sun + IBL, no G-buffer
needed).

**Fractal geometry caveat.** Self-similar geometry across layer
boundaries produces specular reflections with visible seams if the
filter doesn't respect layer identity. The atrous-style edge-stop
uses depth + normal, which already varies across layer boundaries
(different ribbon depths → different depth values). Should just
work, but verify visually.

## Acceptance criteria

- A polished column at roughness 0.1 shows a crisp vertical smear
  of the scene behind the camera's reflection direction.
- Gradient roughness test: column varying roughness from 0 to 1
  top-to-bottom shows smooth transition from mirror to IBL blur.
- No fireflies after denoise on a metallic, high-contrast scene.
- Layer-boundary test: specular ray crossing a layer doesn't
  produce seams.

## Perf target

| Pass | Target |
|---|---|
| Specular trace (indirect) | ≤3.0 ms (50k rays × 60 ns) |
| Spatial denoise | ≤0.8 ms |
| Resolve (blended into deferred) | ≤0.2 ms |
| **Added to Phase 5** | **≤4.0 ms** |
| **Phase 6 frame total** | **≤24 ms** |

Note: 24 ms exceeds our 17 ms reference benchmark, because we're
targeting a richer pipeline than their sponza demo (our Phase 5 GI
ran 2 ambient chains instead of 1). On macOS M-series this is
acceptable at 1080p. Aggressive optimizations (half-res specular,
checkerboard trace) get us back under 20 ms if needed.

## Risks & open questions

- **Variance explosion at low roughness.** At roughness → 0, GGX
  importance sampling collapses to the mirror direction → 1-sample
  estimate is high-variance. Mitigation: clamp min roughness to
  0.04 (standard Cook-Torrance practice) and rely on temporal
  accumulation for sub-0.1 roughness.
- **Checkerboard dispatch for perf.** If 3 ms is too expensive,
  dispatch specular over every other voxel per frame (odd/even
  frame parity). Resolve reprojects + averages. ~40% perf at
  minimal quality cost.
- **One-bounce-only approximation.** Metal-inside-metal scenes
  (chrome ball inside chrome box) lose second-bounce light.
  Acceptable for the scenes we target.

## Scope estimate

~1000 LoC (400 Rust, 500 WGSL, 100 tests). 1 week.

## After Phase 6

With all six phases landed, our renderer has:

- Chunk-pool streaming (scales to multi-layer scenes larger than VRAM)
- Visibility buffer + indirect dispatch (secondary passes affordable)
- Full PBR deferred pipeline
- Ray-traced shadows, diffuse GI, specular
- IBL fallback for all rays
- Tonemap + FXAA + TAA
- `.vxs` canned content format

...and we're a superset of the reference's lighting, plus our
recursive/layer/path-coordinate architecture on top. The renderer
on a single large `.vxs` layer should match the reference's perf
within 10–20%; on multi-layer fractal scenes it surpasses it
architecturally (their renderer can't do our scenes at all).
