# Phase 2 — Deferred G-buffer + PBR + postprocess

**Goal.** Move from inline forward shading to a deferred pipeline
with PBR materials, tonemapping, and FXAA. The frame gets *visibly
better* here — proper material response, no banding, no aliasing —
but still only direct sun + constant ambient.

**Dependencies.** Phase 1 (needs visibility buffer for G-buffer
packing; needs `.vxs` palette for PBR material records).

**Deliverables.**
- Full G-buffer written by the primary raymarch
- PBR palette entry (albedo, metallic, roughness, emission, normal)
- Deferred lighting compute pass
- TonyMcMapface tonemap (LUT-based)
- FXAA pass

## What you see after this phase

A Sponza-class scene renders with correct metallic response (the
columns and ironwork look like metal, not plastic), varying
roughness (polished stone vs rough brick), smooth gradients from
tonemap, and clean edges from FXAA. Still no shadows, no GI — those
are Phases 3–5.

## Architecture

### G-buffer

Four render targets packed into three RGBA16Float textures
(WebGPU-safe; no `Rgba32` required):

```
gbuffer_0: RGBA16F  = [albedo.r, albedo.g, albedo.b, metallic]
gbuffer_1: RGBA16F  = [normal.x, normal.y, normal.z, roughness]
gbuffer_2: RG16F    = [velocity.x, velocity.y]     // for TAA
depth:     Depth32F = linear eye-space depth
```

Alternative packed layout (saves bandwidth, worth measuring):
```
gbuffer_0: RGBA8Unorm = [albedo.rgb, material_id]
gbuffer_1: RGB10A2    = [octahedral-normal.xy, roughness(6), metallic(4)]
gbuffer_2: RG16F      = [velocity.x, velocity.y]
depth:     Depth32F
```
Packed saves ~50% bandwidth but costs an instruction per pack/unpack.
Default to unpacked; profile before switching.

### PBR palette entry

Replace the current 4-byte palette entry with a 32-byte PBR record:

```rust
#[repr(C)]
pub struct PbrMaterial {
    pub albedo: [f32; 3],       // sRGB, linearized on sample
    pub metallic: f32,
    pub roughness: f32,
    pub emission: [f32; 3],     // HDR, pre-multiplied
    pub normal_scale: f32,      // 0 = flat, 1 = full bevel
    pub _pad: [f32; 3],         // 32 B total
}
```

`256` palette entries × 32 B = 8 KB. Fits comfortably in a uniform
buffer if needed.

### Deferred lighting pass

```wgsl
// assets/shaders/deferred.wgsl (new)
@compute @workgroup_size(8, 8)
fn deferred_main(@builtin(global_invocation_id) gid: vec3u) {
    let p = vec2i(gid.xy);
    if (p.x >= screen.x || p.y >= screen.y) { return; }

    let g0 = textureLoad(gbuffer_0, p, 0);
    let g1 = textureLoad(gbuffer_1, p, 0);
    let depth = textureLoad(depth_tex, p, 0);

    let albedo = g0.rgb;
    let metallic = g0.a;
    let normal = oct_decode_or_unpack(g1.xyz);
    let roughness = g1.a;
    let view_dir = reconstruct_view_dir(p, depth, camera);

    // Direct sun (shadow-less for now; Phase 3 adds shadow mask)
    let L = sun.direction;
    let H = normalize(L + view_dir);
    let NdotL = max(dot(normal, L), 0.0);
    let NdotV = max(dot(normal, view_dir), 0.0);
    let NdotH = max(dot(normal, H), 0.0);
    let VdotH = max(dot(view_dir, H), 0.0);

    let F0 = mix(vec3(0.04), albedo, metallic);
    let F = fresnel_schlick(VdotH, F0);
    let D = distribution_ggx(NdotH, roughness);
    let G = geometry_smith(NdotV, NdotL, roughness);

    let specular = (F * D * G) / max(4.0 * NdotV * NdotL, 0.001);
    let kd = (vec3(1.0) - F) * (1.0 - metallic);
    let diffuse = kd * albedo / PI;

    let direct = (diffuse + specular) * sun.color * sun.intensity * NdotL;
    let ambient = albedo * 0.03; // placeholder until Phase 5

    let color = direct + ambient + textureLoad(emission_tex, p, 0).rgb;
    textureStore(lighting_tex, p, vec4(color, 1.0));
}
```

### Tonemap + FXAA + resolve

Three cheap screen-space compute passes, run after TAA:

```
  lighting_tex (HDR linear)
         ↓
     TAA pass (unchanged from Phase 0)
         ↓
  taa_output (HDR linear)
         ↓
     Tonemap pass (TonyMcMapface LUT)
         ↓
  tonemapped (LDR sRGB)
         ↓
     FXAA pass
         ↓
     Swapchain (present)
```

TonyMcMapface is a 48×48×48 RGB9E5 3D LUT. Sampled once per pixel.
The LUT ships with the reference repo under
`external/voxel-raymarching/app/assets/tonemap.dds` — we can reuse
it under its MIT license.

FXAA is fx.wgsl from the reference, essentially unchanged. ~1 ms at
1080p.

## Shaders touched

- **New:** `deferred.wgsl` — deferred lighting compute
- **New:** `tonemap.wgsl` — TonyMcMapface LUT sample
- **New:** `fxaa.wgsl` — FXAA 3.11 port
- **Modified:** `march.wgsl` — writes G-buffer targets instead of
  calling `shade()` inline; samples PbrMaterial from palette
- **Modified:** `main.wgsl` — becomes a thin compositor; lighting
  pass writes to the final color target

## Rust code touched

- **New:** `src/renderer/gbuffer.rs` — G-buffer texture creation,
  resize handling
- **New:** `src/renderer/deferred.rs` — pipeline + bind group for
  deferred pass
- **New:** `src/renderer/postprocess.rs` — tonemap + FXAA pipelines
- **Modified:** `src/world/palette.rs` — `PbrMaterial` replaces `u32`
- **Modified:** `tools/scene_voxelize/` — materialize PBR from GLTF
  material slots (metallic-roughness PBR already in GLTF spec)

## Recursive architecture integration

- G-buffer is screen-space, so layer-agnostic. One set of targets
  for the whole frame.
- PBR palette is per `.vxs` file. A ray that crosses a layer
  boundary reads the current layer's palette (which layer is in
  the chunk pool). No palette-swap overhead — the palette index in
  the G-buffer is local to the `.vxs`, and the deferred pass reads
  the currently-bound palette.
- Open question: if two layers with *different* palettes are
  simultaneously visible (e.g. descending through a layer boundary
  in the frustum), we need multi-palette deferred resolve. Defer
  this until Phase 5 shows it's needed.

## Content pipeline changes

`tools/scene_voxelize/generate` already pulls material data from
GLTF. Extension:

1. GLTF material → `PbrMaterial` record (albedo from baseColorTexture,
   metallic + roughness from metallicRoughnessTexture, emission from
   emissiveTexture). Sample textures at voxel centers during
   voxelization.
2. Deduplicate materials: most scenes have <256 unique PBR records
   even with texture-driven variation. Quantize conservatively.
3. Emission goes into a separate per-voxel buffer only if the scene
   has emissive surfaces (lamps, screens). For now, emission is a
   palette attribute — supports uniform glowing materials but not
   screen-based emission.

## Acceptance criteria

- G-buffer visualization debug mode (pipe each channel to screen).
  Each channel looks correct on a canonical Sponza scene.
- Compared to reference renderer on same scene with same palette:
  albedo channel matches within 2 LSB; normal channel matches within
  1°; roughness matches exactly (same quantization).
- FXAA reduces a canonical "stair-step on voxel edge" test to a
  reference-matching anti-aliased result.
- TonyMcMapface LUT sample matches reference output pixel-for-pixel
  (we're using the same LUT).
- No regression in harness layer-descent test.

## Perf target

| Pass | Target (1080p, M-series) |
|---|---|
| Primary raymarch (G-buffer write) | ≤5.5 ms |
| Deferred lighting | ≤1.5 ms |
| TAA resolve | ≤2.0 ms |
| Tonemap | ≤0.3 ms |
| FXAA | ≤1.0 ms |
| **Phase 2 frame total** | **≤10 ms** |

## Risks & open questions

- **PBR material authoring in scene_voxelize.** Current pipeline
  takes a single color per voxel. Extending to a 32 B record per
  voxel changes the voxel cost from 1 B to 32 B — but dedup via
  palette index keeps on-disk size small. The in-GPU cost is
  unchanged (still 1 palette index per voxel).
- **Normal reconstruction.** G-buffer normal is a combination of
  face normal + per-voxel bevel (from `main.wgsl` current logic).
  Encoding in octahedral requires the full normal; if we pack, we
  lose ~1° precision. Acceptable for voxel geometry.
- **Tonemap licensing.** TonyMcMapface LUT is MIT; reference already
  ships it. Reuse via `include_bytes!`. Alternatively, swap in
  AgX (newer, arguably better). Pick TonyMcMapface for parity with
  the reference benchmark.

## Scope estimate

~1500 LoC net (800 Rust, 500 WGSL, 200 tests). 1 week.
