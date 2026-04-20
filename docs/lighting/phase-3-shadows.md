# Phase 3 — Sun shadows

**Goal.** A single directional sun casts hard shadows through the
voxel tree. Ray-traced, not shadow mapped. Uses the visibility
buffer + indirect dispatch from Phase 1.

**Dependencies.** Phase 2 (needs G-buffer depth + normal to spawn
shadow rays from first-hit positions).

**Deliverables.**
- Shadow-ray compute pass dispatched over visible voxels
- Per-voxel shadow mask stored in visibility buffer
- Deferred lighting samples the mask

## Why ray-traced, not shadow maps

Shadow maps assume a static mesh with a projection. Our scene is a
fractal tree with dynamic layer transitions — projecting from a
single sun frustum across all layers is awkward and wastes samples
on off-camera geometry. Ray tracing through the existing `march()`
primitive reuses the chunk pool we already built.

With indirect dispatch over `visible_voxel_count` (~50k), shadow
tracing is ~50k rays per frame, not ~2M.

## Architecture

### Shadow mask in visibility buffer

Extend `HitEntry`:

```rust
#[repr(C)]
pub struct HitEntry {
    pub node_id: u64,
    pub voxel_slot: u32,
    pub surface_data: u32,
    pub shadow_mask: u32,   // NEW: 1 bit lit / 0 bit shadowed, +
                            // 24 bits for future 4-tap shadow PCF
}
```

### Shadow pass

```wgsl
// assets/shaders/shadow.wgsl (new)
@compute @workgroup_size(64)
fn shadow_main(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= visibility.hit_count) { return; }

    let entry = hit_list[i];
    let pos = unpack_voxel_position(entry);
    let normal = unpack_voxel_normal(entry);

    // Bias toward normal to avoid self-shadow on the hit voxel
    let origin = pos + normal * sun.shadow_bias;
    let dir = sun.direction;

    let result = march(origin, dir, 1000.0);
    let shadow_mask = select(0u, 1u, result.missed);

    hit_list[i].shadow_mask = shadow_mask;
}
```

Dispatched indirect:
```rust
pass.set_pipeline(&shadow_pipeline);
pass.set_bind_group(0, &shadow_bind_group, &[]);
pass.dispatch_workgroups_indirect(&indirect_args, 0);
```

### Scatter mask back to screen

Two options:

**Option A — scatter during deferred.** Deferred lighting reads
`visibility.hit_map` at each pixel's first-hit NodeId to look up
shadow_mask. One hashmap probe per pixel.

**Option B — scatter to a screen-space shadow texture.** Separate
compute pass reads the visibility hit_list and writes to a full-res
`shadow_tex` using the first-hit pixel coordinates saved in
`surface_data`. Deferred lighting reads `shadow_tex[pixel]`.

Option B has better cache behavior for deferred lighting. Go with B.

```wgsl
// Scatter pass
@compute @workgroup_size(64)
fn scatter_shadow(@builtin(global_invocation_id) gid: vec3u) {
    let i = gid.x;
    if (i >= visibility.hit_count) { return; }
    let entry = hit_list[i];
    let pixel = unpack_first_hit_pixel(entry.surface_data);
    textureStore(shadow_tex, pixel, vec4(f32(entry.shadow_mask), 0, 0, 1));
}
```

Note: a voxel hit by multiple pixels gets written multiple times (to
different pixel coords), which is correct.

### Deferred lighting integration

```wgsl
// deferred.wgsl (modified)
let shadow = textureLoad(shadow_tex, p, 0).r;
let direct = (diffuse + specular) * sun.color * sun.intensity
           * NdotL * shadow;  // <-- new term
```

## Shaders touched

- **New:** `shadow.wgsl`, `scatter_shadow.wgsl` (or merged)
- **Modified:** `deferred.wgsl` — apply shadow mask
- **Modified:** `bindings.wgsl` — shadow_tex bind group

## Rust code touched

- **New:** `src/renderer/shadow.rs` — shadow pipelines, scatter pipeline
- **Modified:** `src/renderer/draw.rs` — dispatch shadow pass
  indirect between primary raymarch and deferred resolve
- **Modified:** `src/renderer/gbuffer.rs` — adds `shadow_tex` target

## Recursive architecture integration

Shadow rays march the same chunk pool, through the same `march()`
primitive, crossing layer boundaries the same way primary rays do.
No special-case code per layer.

Edge case: a voxel hit in layer L might have its shadow occluded by
geometry in layer L-1 (the parent layer). Because `march()` walks
the ribbon up to world root, shadow rays naturally see ancestor
geometry. This is correct — the sun is a global light.

**Exception for spheres.** Shadow rays across a cubed-sphere body
use the existing sphere DDA. `march()` dispatches on NodeKind as
usual; no new code here.

## Acceptance criteria

- A pillar test scene shows correct shadow direction.
- Self-shadowing on voxel steps: no shadow acne (bias is tuned), no
  Peter-Panning at grazing angles.
- Shadow pass cost scales with visible voxel count, not screen area.
  Verify by changing resolution and observing flat shadow time.

## Perf target

| Pass | Target |
|---|---|
| Shadow indirect dispatch | ≤2.5 ms (50k rays × ~50 ns) |
| Scatter | ≤0.1 ms |
| **Added to Phase 2 total** | **≤2.6 ms** |
| **Phase 3 frame total** | **≤13 ms** |

Compared to the reference's shadow_occlusion + shadow passes: theirs
uses a screen-space shadow occlusion buffer with Option-B-style
scatter. We're doing the same thing.

## Risks & open questions

- **Shadow bias.** Per `project_atmosphere_known_issues.md` memory,
  AcesFitted + shadows caused acne before. With TonyMcMapface and
  careful normal-biased bias we should avoid this. If acne appears,
  switch to receiver-plane depth bias.
- **Grazing angles.** Voxel face geometry has hard 90° transitions.
  A sun at low altitude can produce aliased shadow edges. Phase 4
  (IBL ambient) softens the problem; Phase 5 PCF would fix it
  properly. Defer PCF to a future pass.
- **Moving sun.** Sun direction in `sun.direction` uniform; no cached
  shadow maps to invalidate. Shadows update every frame.

## Scope estimate

~400 LoC (200 Rust, 150 WGSL, 50 tests). 2–3 days.
