// Instanced raster pass for entity meshes.
//
// Runs after the ray-march pass. Reads the same depth buffer the
// ray-march wrote (via @builtin(frag_depth) derived from view_proj),
// so entity triangles z-test against terrain correctly.
//
// Per-instance data:
//   translation (vec3) + uniform_scale (f32) in frame-local coords.
//   color_tint (vec4) — multiplied into the baked vertex color.
//
// Per-vertex data:
//   position in the subtree's [0, 3)^3 local space.
//   normal (flat per-face).
//   color (rgb baked from palette at extract time).

struct Uniforms {
    view_proj: mat4x4<f32>,
}

@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
    // Per-instance
    @location(3) i_translate: vec3<f32>,
    @location(4) i_scale: f32,
    @location(5) i_tint: vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) color: vec3<f32>,
}

@vertex
fn vs_main(in: VertexIn) -> VertexOut {
    // Mesh lives in [0,3)^3. Multiply by scale/3 to bring into the
    // entity's anchor cell (which has side length `scale` in frame
    // coords). Then translate to the entity's bbox_min.
    let world = vec3<f32>(
        in.i_translate.x + in.position.x * (in.i_scale / 3.0),
        in.i_translate.y + in.position.y * (in.i_scale / 3.0),
        in.i_translate.z + in.position.z * (in.i_scale / 3.0),
    );
    var out: VertexOut;
    out.clip_position = u.view_proj * vec4<f32>(world, 1.0);
    out.normal = in.normal;
    out.color = in.color * in.i_tint.rgb;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
    let n = normalize(in.normal);
    let diffuse = max(dot(n, sun_dir), 0.0);
    let ambient = 0.3;
    let lit = in.color * (ambient + diffuse * 0.7);
    let gamma = pow(lit, vec3<f32>(1.0 / 2.2));
    return vec4<f32>(gamma, 1.0);
}
