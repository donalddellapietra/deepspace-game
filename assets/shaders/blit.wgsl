// Fullscreen-triangle blit from a compute storage texture to the
// surface. Identity sample: the compute output already has crosshair,
// gamma, highlight baked in, so this is just a texel-for-texel copy.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    // Flip Y so the top screen pixel samples texel row 0, matching the
    // compute shader's top-down gid.y convention.
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

@group(0) @binding(0) var source: texture_2d<f32>;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let dims = textureDimensions(source);
    let px = vec2<i32>(
        clamp(i32(in.uv.x * f32(dims.x)), 0, i32(dims.x) - 1),
        clamp(i32(in.uv.y * f32(dims.y)), 0, i32(dims.y) - 1),
    );
    return textureLoad(source, px, 0);
}
