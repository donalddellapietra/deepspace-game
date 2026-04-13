// GPU-instanced NPC part rendering.
//
// Matches BSL voxel lighting as closely as possible without the full
// PBR pipeline. NPC parts don't have vertex AO or subsurface
// scattering, so those are omitted. The key visual elements are:
// - Directional light with shadow approximation
// - BSL ambient tinting in shadowed regions
// - Reinhard-style tone mapping

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Per-instance transform matrix (4 columns)
    @location(10) i_transform_0: vec4<f32>,
    @location(11) i_transform_1: vec4<f32>,
    @location(12) i_transform_2: vec4<f32>,
    @location(13) i_transform_3: vec4<f32>,
    // Per-instance base color
    @location(14) i_color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    // Reconstruct the instance transform matrix.
    let instance_tf = mat4x4<f32>(
        vertex.i_transform_0,
        vertex.i_transform_1,
        vertex.i_transform_2,
        vertex.i_transform_3,
    );

    // Transform vertex position by instance matrix.
    let world_pos = instance_tf * vec4<f32>(vertex.position, 1.0);

    // Transform normal by the instance rotation (upper-left 3x3).
    let normal_matrix = mat3x3<f32>(
        instance_tf[0].xyz,
        instance_tf[1].xyz,
        instance_tf[2].xyz,
    );
    let world_normal = normalize(normal_matrix * vertex.normal);

    var out: VertexOutput;
    out.clip_position = mesh_position_local_to_clip(
        get_world_from_local(0u),
        world_pos,
    );
    out.world_position = world_pos.xyz;
    out.world_normal = world_normal;
    out.color = vertex.i_color;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    let base_color = in.color.rgb;
    let normal = normalize(in.world_normal);

    // Directional light matching setup_environment() in main.rs:
    // Quat::from_euler(EulerRot::XYZ, -0.7, 0.4, 0.0)
    // This gives a light direction roughly (-0.38, -0.64, -0.26) after rotation.
    let light_dir = normalize(vec3<f32>(-0.38, -0.64, -0.26));
    let ndotl = max(dot(normal, -light_dir), 0.0);

    // Approximate scene lighting to match BSL output:
    // ambient_color = (0.9, 0.95, 1.0) at intensity 0.3
    let ambient_color = vec3<f32>(0.9, 0.95, 1.0);
    let ambient_strength = 0.3;
    let light_intensity = 0.7; // complement of ambient

    let ambient = ambient_color * ambient_strength * base_color;
    let diffuse = base_color * ndotl * light_intensity;

    var lit_color = ambient + diffuse;

    // BSL ambient tinting: push shadowed regions toward ambient color.
    let luminance = dot(lit_color, vec3<f32>(0.299, 0.587, 0.114));
    let ambient_tint = ambient_color * ambient_strength;
    let shadow_blend = saturate(1.0 - luminance * 2.0);
    lit_color = lit_color + ambient_tint * shadow_blend * 0.5;

    // Reinhard tone mapping (matches Tonemapping::Reinhard in camera setup).
    lit_color = lit_color / (lit_color + vec3<f32>(1.0));

    return vec4<f32>(lit_color, 1.0);
}
