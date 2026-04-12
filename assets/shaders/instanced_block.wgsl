// Instanced block shader for the world renderer.
//
// Each instance carries a position+scale (vec4) and a block-type
// colour (vec4). The vertex colour attribute (location 5) holds
// per-vertex AO brightness baked by the mesher.

#import bevy_pbr::{
    mesh_functions,
    forward_io::VertexOutput,
    view_transformations::position_world_to_clip,
}

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // location 5 = vertex colour (AO brightness from the mesher)
    @location(5) color: vec4<f32>,
    // Instance data (second vertex buffer, step_mode = Instance)
    @location(3) i_pos_scale: vec4<f32>,
    @location(4) i_color: vec4<f32>,
};

@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;

    // Scale the mesh-local position, then offset by the instance's
    // world-space origin.  The entity's own Transform is identity, so
    // `get_world_from_local` is effectively the identity matrix — but
    // we still go through it so that the mesh bind-group is valid.
    // Pass 0u — we have one entity per group, so only mesh instance 0
    // is valid.  The entity's Transform is identity; all positioning
    // comes from the per-instance data below.
    let mesh_world_from_local = mesh_functions::get_world_from_local(0u);
    let scale = vertex.i_pos_scale.w;
    let world_pos = vec4<f32>(
        vertex.position * scale + vertex.i_pos_scale.xyz,
        1.0,
    );
    out.world_position = mesh_world_from_local * world_pos;
    out.position = position_world_to_clip(out.world_position.xyz);

    out.world_normal = mesh_functions::mesh_normal_local_to_world(
        vertex.normal,
        0u,
    );

    // Mix the per-instance block colour with the per-vertex AO
    // brightness. The mesher stores AO as a grey [b, b, b, 1] in
    // vertex.color; multiplying modulates the block colour.
    out.color = vertex.i_color * vertex.color;

    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple directional + ambient lighting.
    let light_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
    let n = normalize(in.world_normal);
    let ndotl = max(dot(n, light_dir), 0.0);
    let ambient = 0.35;
    let diffuse = ndotl * 0.65;
    let lit = in.color.rgb * (ambient + diffuse);
    return vec4<f32>(lit, in.color.a);
}
