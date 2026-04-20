// GPU entity Y-clamp.
//
// Runs once per frame, after `heightmap_gen` and before the raster
// entity pass. Samples the heightmap at each instance's XZ and
// overwrites `translate.y` with the ground height. Instances whose
// XZ falls outside the heightmap extent, or whose texel holds the
// GROUND_NONE sentinel, are left untouched (they stay at whatever
// Y the CPU set them to — typically the spawn Y or some fallback).
//
// Instance layout matches `renderer::entity_raster::InstanceData`:
//   translate: vec3<f32>    (bytes 0..12)
//   scale:     f32          (bytes 12..16)
//   tint:      vec4<f32>    (bytes 16..32)
//
// Total stride 32 B. We can treat each instance as `array<vec4<f32>, 2>`
// for convenient load/store.
//
// Workgroup 64: one-dimensional dispatch. At 100k entities that's
// 1563 workgroups, fits comfortably.

struct ClampUniforms {
    /// How many entries in the instance buffer are live.
    entity_count: u32,
    /// `3^delta` — matches the allocated heightmap texture side.
    heightmap_side: u32,
    /// Sentinel: values < this are treated as "no ground", see
    /// `heightmap_gen.wgsl::GROUND_NONE`.
    no_ground_threshold: f32,
    /// Frame XZ origin in world coords (for future non-[0, 3)
    /// frames; currently always 0).
    frame_xz_origin_x: f32,
    frame_xz_origin_z: f32,
    /// Frame XZ extent in world coords (currently WORLD_SIZE = 3).
    frame_xz_size: f32,
    _pad0: u32,
    _pad1: u32,
}

struct Instance {
    // Row 0: translate.xyz + scale
    row0: vec4<f32>,
    // Row 1: tint
    row1: vec4<f32>,
}

@group(0) @binding(0) var<storage, read_write> instances: array<Instance>;
@group(0) @binding(1) var<uniform> clamp_uniforms: ClampUniforms;
// Sampled texture — WebGPU baseline forbids read-only storage
// textures, so we read via `textureLoad` (integer coords, no
// filtering) from a standard `texture_2d` instead.
@group(0) @binding(2) var heightmap: texture_2d<f32>;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= clamp_uniforms.entity_count {
        return;
    }

    let inst = instances[idx];
    let translate = inst.row0.xyz;

    // XZ → heightmap texel coords. Each texel covers
    // `frame_xz_size / heightmap_side` world units.
    let cell_xz = clamp_uniforms.frame_xz_size / f32(clamp_uniforms.heightmap_side);
    let tex_x = i32(floor((translate.x - clamp_uniforms.frame_xz_origin_x) / cell_xz));
    let tex_z = i32(floor((translate.z - clamp_uniforms.frame_xz_origin_z) / cell_xz));
    let side = i32(clamp_uniforms.heightmap_side);
    if tex_x < 0 || tex_x >= side || tex_z < 0 || tex_z >= side {
        return;
    }

    let ground_y = textureLoad(heightmap, vec2<i32>(tex_x, tex_z), 0).r;
    if ground_y < clamp_uniforms.no_ground_threshold {
        return;
    }

    // Snap the entity's BBOX MIN to ground_y. The instance's
    // `translate` is the min corner of the entity's anchor cell in
    // frame coords (see `entity_raster.wgsl`); the entity's visual
    // bottom sits there.
    instances[idx].row0.y = ground_y;
}
