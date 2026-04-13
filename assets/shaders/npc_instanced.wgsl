// GPU-instanced NPC part rendering.
//
// Each instance is one NPC body part positioned via a 4x4 transform
// matrix passed as vertex attributes (locations 2-5). The base color
// is also per-instance (location 6).
//
// Uses Bevy's mesh view bindings for camera matrices and a simple
// directional light approximation for shading.

#import bevy_pbr::mesh_functions::{get_world_from_local, mesh_position_local_to_clip}

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // Per-instance: 4x4 transform matrix (4 columns)
    // Locations 10-14 avoid conflicts with Bevy's mesh attributes
    // (Position=0, Normal=1, UV=2, Tangent=3, Color=4, Joints=5-6, etc.)
    @location(10) i_transform_0: vec4<f32>,
    @location(11) i_transform_1: vec4<f32>,
    @location(12) i_transform_2: vec4<f32>,
    @location(13) i_transform_3: vec4<f32>,
    // Per-instance: base color
    @location(14) i_color: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec4<f32>,
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
    // Use Bevy's view projection to go from world to clip space.
    // get_world_from_local(0u) gives us the entity's own transform,
    // but we've already applied instance_tf, so we use it to handle
    // the view/projection part.
    out.clip_position = mesh_position_local_to_clip(
        get_world_from_local(0u),
        world_pos,
    );
    out.world_normal = world_normal;
    out.color = vertex.i_color;
    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple directional lighting.
    let light_dir = normalize(vec3<f32>(0.4, -0.7, 0.3));
    let ndotl = max(dot(in.world_normal, -light_dir), 0.0);

    let ambient = 0.35;
    let diffuse = ndotl * 0.65;
    let lit = in.color.rgb * (ambient + diffuse);

    return vec4<f32>(lit, in.color.a);
}
