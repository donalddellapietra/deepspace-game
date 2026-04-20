# Phase 4 — IBL (image-based lighting)

**Goal.** The sky contributes indirect light. A pre-filtered
skybox gives us a constant-cost ambient term that's far better than
the placeholder flat ambient from Phase 2. Also the fallback for
any ray that escapes the scene (e.g. primary rays into sky, GI rays
that miss).

**Dependencies.** Phase 2 (samples G-buffer normal during deferred
resolve).

**Deliverables.**
- Skybox cubemap (raw radiance for primary-ray sky hits)
- Irradiance cubemap (cosine-weighted, for diffuse ambient)
- Specular prefilter cubemap (GGX-weighted, for specular ambient)
- BRDF integration LUT (F0 × scale + bias)
- Deferred resolve samples all of these

## What you see after this phase

The scene has a believable ambient tint matching the sky. A
cloud-scattered sky fills in shadowed areas with soft blue fill.
Metal materials show a blurry sky reflection proportional to
roughness. Primary rays that miss the scene see the skybox directly.

## Architecture

### Cubemap set

```rust
pub struct IblCubemaps {
    pub skybox:        wgpu::Texture, // 1024² × 6, RGBA16F; raw sky radiance
    pub irradiance:    wgpu::Texture, // 32² × 6, RGBA16F; ∫cos(θ)L dω
    pub prefilter:     wgpu::Texture, // 256² × 6, RGBA16F, 8 mips;
                                       //   mip i → roughness i/7
    pub brdf_lut:      wgpu::Texture, // 256² 2D, RG16F; GGX split-sum
}
```

Total ~30 MB across all four. Sits in one bind group.

### Baking the cubemaps (offline)

`tools/scene_voxelize/generate/src/lightmap.rs` (new) — takes a sky
input (HDRI equirect or procedural), emits:

1. **Skybox**: equirect → cubemap projection, downsample to 1024²
   faces, pre-exposed.
2. **Irradiance**: 32² faces, each pixel = cos-weighted integral
   over hemisphere. Use Monte Carlo with 4k samples (compile-time
   or offline); samples cache to the cubemap output.
3. **Prefilter**: 256² faces, mip 0 = mirror reflection (same as
   skybox downsampled), mips 1–7 = GGX-weighted sums. Use the
   split-sum approximation (Karis '13).
4. **BRDF LUT**: 256² 2D table of (NdotV, roughness) → (scale, bias).
   Trivial to bake once per target.

Output: `<name>.lightmap` file. Same zlib format as reference.
Reuse `partly_cloudy.lightmap` from reference as a starting point.

### Deferred resolve integration

```wgsl
// deferred.wgsl (modified)
// Replace ambient placeholder:
//   let ambient = albedo * 0.03;
// With:
let R = reflect(-view_dir, normal);
let irradiance = textureSample(irradiance_tex, sampler_lin, normal).rgb;
let diffuse_ibl = irradiance * albedo * (1.0 - metallic);

let prefiltered = textureSampleLevel(prefilter_tex, sampler_lin, R,
                                     roughness * 7.0).rgb;
let brdf = textureSample(brdf_lut, sampler_lin, vec2(NdotV, roughness)).rg;
let specular_ibl = prefiltered * (F * brdf.x + brdf.y);

let kd = (vec3(1.0) - F) * (1.0 - metallic);
let ambient = (kd * diffuse_ibl + specular_ibl);
```

Cost: 4 texture samples + a handful of arithmetic per pixel.
Negligible compared to the deferred pass's existing work.

### Primary ray sky hits

The primary raymarch, when a ray exits the ribbon without hitting
geometry, needs to sample the skybox. Currently `main.wgsl` renders
a gradient. Replace:

```wgsl
// main.wgsl (modified)
if (result.missed) {
    let sky = textureSample(skybox_tex, sampler_lin, ray.direction).rgb;
    return vec4(sky, 1.0);
}
```

## Shaders touched

- **Modified:** `main.wgsl` — sky fallback samples cubemap
- **Modified:** `deferred.wgsl` — IBL ambient replaces placeholder
- **New (offline):** `tools/scene_voxelize/generate/src/shaders/`
  — prefilter and irradiance bakers (run once at content build time)

## Rust code touched

- **New:** `src/renderer/ibl.rs` — cubemap loading, bind group
- **New:** `src/world/lightmap.rs` — `.lightmap` file format reader
- **Modified:** `src/renderer/deferred.rs` — new bind group for IBL
- **Modified:** `tools/scene_voxelize/generate/src/` — baker entry
  points

## Recursive architecture integration

- **One IBL set per layer.** Each `.vxs` can reference a
  `.lightmap` file; descent swaps the bound IBL set the same way
  chunks stream in.
- **Top layer**: sky cubemap is the actual visible sky (blue,
  clouds, sun disk).
- **Deeper layers**: IBL represents "what you'd see from within this
  layer looking at the sky". For a layer that's entirely inside a
  larger structure (e.g. inside a building), the IBL is baked from
  the parent layer's visibility. This is a natural fit for
  canned offline generation.
- **Layer transitions**: crossfade between two IBL sets over a few
  frames during descent to avoid popping.

## Offline baking flow

```
content/sky_partly_cloudy.hdr (equirect input)
        │
        ▼
  scene_voxelize bake-lightmap
        │
        ▼
content/partly_cloudy.lightmap (skybox + irradiance + prefilter)
                                 [BRDF LUT is shared, baked once]
        │
        ▼
<layer>.vxs references this lightmap by name
```

Each layer's `.vxs` header includes a lightmap reference:
```json
{
  "version": 1,
  "layer_depth": 3,
  "lightmap": "partly_cloudy",
  "palette_entries": 256,
  ...
}
```

## Acceptance criteria

- A metal-ball test scene at max roughness shows a visible blurred
  sky reflection; at min roughness shows a crisp sky reflection.
- A chrome column in Sponza-class scene shows the expected sky
  reflection (match reference screenshot).
- Ambient term on a cloudy day gives soft blue fill matching the
  skybox's average color.
- Primary-ray sky hits look identical to the input skybox.

## Perf target

| Pass | Target |
|---|---|
| Deferred IBL additions | ≤0.5 ms (adds 4 texture samples) |
| Primary ray sky sample | ≤0.2 ms (cache-hot) |
| **Added to Phase 3** | **≤0.7 ms** |
| **Phase 4 frame total** | **≤14 ms** |

## Risks & open questions

- **Lightmap file size.** 30 MB × multiple layers adds up. Mitigate
  by: (a) dedup IBL sets across layers where scenes share skies,
  (b) quantize prefilter mips to RGB9E5 (lossless for HDR, 32→32
  bits, no savings) or BC6H (lossy, 6× smaller but GPU-decoded).
  Pick RGB9E5 to match reference.
- **Sun disk in skybox.** If the skybox contains the sun disk,
  primary rays hitting the sky double-count the sun (once via
  direct lighting in Phase 2, once via sampling). Standard fix:
  occlude the sun disk in the skybox bake; render it as a separate
  pass with analytic disk response.
- **Layer-transition artifacts.** If layer A's IBL is dark (inside
  a cave) and layer B's is bright (outdoors), a fast descent
  flashes. Mitigation: crossfade over 4 frames.

## Scope estimate

~600 LoC (300 Rust, 150 WGSL, 150 baker code, 50 tests). 3–4 days.
Most of the cost is the offline baker; runtime integration is
small.
