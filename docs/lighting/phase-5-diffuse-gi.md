# Phase 5 — Diffuse GI (irradiance probes + ambient trace)

**Goal.** Bounce light. A stone column next to a red wall picks up
a red tint. Ceiling gets light from the floor. This is the biggest
perceptual upgrade in the whole roadmap and the most architecturally
involved.

**Dependencies.** Phase 4 (IBL is the fallback when an ambient
trace misses the scene and escapes to sky). Phase 1 (visibility
+ indirect dispatch are used by the trace and denoise).

**Deliverables.**
- Per-layer irradiance probe volume (sparse 3D grid)
- Ambient trace pass (one ray per visible voxel, indirect dispatch)
- Temporal reprojection of ambient
- Spatial denoiser (atrous blur)
- Deferred resolve samples probe irradiance

## What you see after this phase

Color bleed. Soft bounce light in corners. Indirect lighting that
responds correctly to material changes (repainting a wall changes
the bounce color). This is the feature that turns a voxel
screenshot from "tech demo" into "plausible architectural scene".

## Architecture

The reference uses a hybrid approach: a sparse 3D probe grid for
low-frequency irradiance, plus per-pixel ambient trace for
high-frequency detail, denoised temporally + spatially. We copy
this structure.

### Probe volume

```rust
pub struct ProbeVolume {
    pub irradiance:    wgpu::Texture, // 2D array, RGBA16F;
                                       // probe octahedral map × N probes
    pub depth:         wgpu::Texture, // 2D array, RG16F;
                                       // moments for shadow test × N probes
    pub layout:        ProbeLayout,    // spatial mapping probe→grid cell
    pub scale:         u32,            // voxels per probe in this layer
}
```

Each probe stores:
- An octahedral-mapped irradiance map (e.g. 8×8 per probe)
- A moment-depth map (mean + mean-squared) for backface rejection

For a typical layer with `scale = 64`, a 16×16×16 probe grid (4096
probes) covers 1024³ voxels. Storage: 4096 × 8² × 8 B + 4096 × 8² × 4 B
≈ 3 MB per layer.

### Per-layer probe grids

Each `.vxs` carries probe data for its layer, path-coordinate
indexed. This is the crux of recursive integration:

```json
{
  "version": 1,
  "layer_depth": 3,
  "probe_grid": {
    "origin": [0, 0, 0],
    "dims":   [16, 16, 16],
    "scale":  64,
    "data":   "<zlib-compressed irradiance + depth blob>"
  }
}
```

When a ray crosses a layer boundary, probe sampling seamlessly
switches to the other layer's grid — the chunk pool already knows
which layer a voxel belongs to (NodeId tagging from Phase 1).

### Ambient trace

One ray per visible voxel, cosine-weighted hemisphere sample:

```wgsl
// ambient_trace.wgsl (new)
@compute @workgroup_size(64)
fn ambient_trace(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= visibility.hit_count) { return; }

    let entry = hit_list[i];
    let pos = unpack_voxel_position(entry);
    let normal = unpack_voxel_normal(entry);

    // Blue-noise cosine-weighted sample (rotated by frame_id for TAA)
    let jitter = blue_noise_rotation(entry.voxel_slot, frame.id);
    let dir = cosine_hemisphere(normal, jitter);

    let result = march(pos + normal * 0.001, dir, frame.ambient_max_distance);
    var radiance: vec3f;
    if (result.missed) {
        radiance = sample_irradiance(normal); // IBL fallback
    } else {
        // Secondary surface: look up its probe irradiance
        radiance = sample_probe(result.hit_pos, result.hit_normal) * result.albedo;
    }
    hit_list[i].ambient = pack_rgba16f(radiance);
}
```

Indirect dispatch over `visible_voxel_count` — same as shadow.

### Temporal reprojection

Screen-space accumulation buffer. Previous-frame ambient is
reprojected using velocity + depth, blended with current. Rejects
samples where the NodeId at the same pixel changed between frames
(ghost-kill).

Buffers:
```
ambient_accum:  RGBA16F, full-res
ambient_prev:   RGBA16F (previous frame of ambient_accum)
```

### Spatial denoise (atrous)

3-iteration a-trous wavelet. Each iteration is a 5×5 cross-filter
with edge-stopping on depth + normal. Reference uses this exact
pattern.

```wgsl
// atrous.wgsl (new)
@compute @workgroup_size(8, 8)
fn atrous(@builtin(global_invocation_id) gid: vec3u) {
    let p = vec2i(gid.xy);
    let center_depth = textureLoad(depth_tex, p, 0).r;
    let center_normal = textureLoad(gbuffer_1, p, 0).xyz;

    var sum = vec3f(0);
    var weight_sum = 0.0;
    for (var dy = -2; dy <= 2; dy = dy + 1) {
        for (var dx = -2; dx <= 2; dx = dx + 1) {
            let q = p + vec2i(dx, dy) * push.step_size;
            // ... edge-stopping weights on depth + normal ...
            sum = sum + textureLoad(ambient_in, q, 0).rgb * w;
            weight_sum = weight_sum + w;
        }
    }
    textureStore(ambient_out, p, vec4(sum / weight_sum, 1));
}
```

Three iterations at step sizes 1, 2, 4 (standard JBF cascade).

### Deferred resolve integration

The deferred resolve pass, which in Phase 4 samples IBL, now also
samples screen-space ambient:

```wgsl
// deferred.wgsl (modified)
let ss_ambient = textureLoad(ambient_denoised, p, 0).rgb;
let ibl_ambient = sample_ibl(normal, view_dir, roughness, metallic);

// ss_ambient already accounts for near-field bounces and occlusion;
// IBL fills in the sky visibility that the probe grid and ambient
// trace agree escaped the scene.
let ambient = ss_ambient + ibl_ambient * occlusion_estimate;
```

## Shaders touched

- **New:** `ambient_trace.wgsl`, `ambient_reproject.wgsl`,
  `atrous.wgsl`, `probe_sample.wgsl`
- **New (offline):** `probe_bake.wgsl` — bake probes from the scene
- **Modified:** `deferred.wgsl` — samples ambient
- **Modified:** `main.wgsl` — passes first-hit pixel into
  `surface_data` for ambient scatter

## Rust code touched

- **New:** `src/renderer/gi.rs` — probe volume + ambient pipelines
- **New:** `src/world/probe.rs` — probe grid data structure
- **Modified:** `src/renderer/draw.rs` — sequence the ambient
  passes: trace → scatter → reproject → atrous → resolve
- **Modified:** `tools/scene_voxelize/generate/src/` — offline probe
  bake, one grid per layer

## Recursive architecture integration

This is the hard part. Three cases:

1. **Same-layer ambient trace.** The trace ray stays within the
   current layer's bounding box. Sample that layer's probe grid.
   Trivial.

2. **Ray crosses into parent layer.** The ribbon pops up to a
   parent node. Sample the parent layer's probe grid at the ray's
   hit position (path-coordinate determines which layer's grid).
   Requires sampling a probe grid not currently in the primary
   chunk pool — we keep parent-layer probes resident one depth up.

3. **Ray escapes the scene.** IBL fallback. Already handled.

**Layer-transition smoothing.** When the camera descends through a
layer boundary, probes from both layers are simultaneously visible.
The deferred resolve reads whichever layer the hit voxel belongs to.
Because probe grids are per-layer and path-coordinate indexed,
there's no ambiguity — but we need both grids resident. Add a
second probe-grid bind group for the parent layer during transitions.

**Offline probe baking.** For each layer, run the ambient trace
10k+ samples per probe offline (without denoiser) during
`scene_voxelize`. Runtime then does 1 sample/voxel/frame + denoise
for dynamic updates. With static content, runtime traces are
mostly redundant — turn them off at release and use baked probes
directly. Memory rule "canned structures" applies.

## Acceptance criteria

- A red-wall-next-to-white-wall test scene shows red bounce on the
  white wall.
- Cornell-box-like corner darkening.
- No visible probe grid seams (bilinear probe sampling + depth
  moments reject through-wall leaks).
- Temporal stability: static camera, static scene → ambient
  converges in ~8 frames and stays stable.
- Moving camera → no blurry trails from ghosted samples.

## Perf target

| Pass | Target |
|---|---|
| Ambient trace (indirect) | ≤3.5 ms (50k rays × 70 ns) |
| Ambient scatter | ≤0.2 ms |
| Reprojection | ≤0.5 ms |
| A-trous × 3 | ≤1.2 ms |
| Probe sample in deferred | ≤0.3 ms |
| **Added to Phase 4** | **≤5.7 ms** |
| **Phase 5 frame total** | **≤20 ms** |

Reference achieves 5.7 ms for its ambient trace + 2.5 ms for its
ambient chain. Our targets are slightly higher because we lack a
fused trace+resolve pass (they shave 1 ms with tiled dispatch).

## Risks & open questions

- **Per-layer probe grids + transitions.** Two grids resident at
  layer boundaries costs 2× GI memory temporarily. Mitigation: LRU
  eviction of non-current-layer grids after descent stabilizes.
- **Dynamic content (edits).** An edit changes a voxel's albedo;
  bounce lighting should update. With baked probes, edits are
  stale until a rebake. Mitigation: runtime 1-sample/voxel ambient
  trace is always on, and denoiser converges in ~8 frames. Baked
  probes are the steady-state prior, not the final answer.
- **Secondary ray probe sampling.** When an ambient ray hits a
  surface in another layer, we need that layer's probe grid.
  Requires probe grids to be bindless-addressable, or a second
  bind group. WebGPU bindless is flaky; use a second bind group
  for at most the 2 adjacent layers.
- **Fractal self-similarity.** Repeating subtrees share NodeIds,
  but their probe grids are *spatial* (keyed by world position),
  not content-addressed. So probe grids don't dedup across
  identical subtrees at different locations. Acceptable — probe
  data is small enough.

## Scope estimate

~1800 LoC (800 Rust, 600 WGSL, 400 offline bake). 1.5 weeks.
Largest phase after Phase 1.
