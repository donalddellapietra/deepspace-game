// BSL-inspired voxel lighting shader.
//
// Extends StandardMaterial's PBR output with:
// - Vertex-baked AO integration (squared skylight curve from BSL)
// - BSL-style ambient/lit blending
// - Subsurface scattering for translucent blocks (leaves, water, glass)

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
}

#ifdef PREPASS_PIPELINE
#import bevy_pbr::{
    prepass_io::{VertexOutput, FragmentOutput},
    pbr_deferred_functions::deferred_output,
}
#else
#import bevy_pbr::{
    forward_io::{VertexOutput, FragmentOutput},
    pbr_functions::{apply_pbr_lighting, main_pass_post_lighting_processing},
}
#endif

struct BslParams {
    ambient_color: vec4<f32>,
    subsurface_strength: f32,
    ao_strength: f32,
    _padding: vec2<f32>,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> bsl: BslParams;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;

    // Standard PBR lighting (includes shadow sampling, cascade blending, etc.)
    var lit_color = apply_pbr_lighting(pbr_input);

    // --- BSL-style post-processing ---

    // Vertex AO from the greedy mesher (greyscale in vertex color).
    // Recomputed from the full 125³ grid in merge_child_faces so it's
    // continuous across child boundaries.
#ifdef VERTEX_COLORS
    let ao_raw = in.color.r;
#else
    let ao_raw = 1.0;
#endif
    let ao = mix(1.0, ao_raw * ao_raw, bsl.ao_strength);
    lit_color = vec4(lit_color.rgb * ao, lit_color.a);

    // BSL ambient tinting: push shadowed regions toward ambient color.
    let luminance = dot(lit_color.rgb, vec3(0.299, 0.587, 0.114));
    let ambient_tint = bsl.ambient_color.rgb * bsl.ambient_color.a;
    let shadow_blend = saturate(1.0 - luminance * 2.0);
    lit_color = vec4(
        lit_color.rgb + ambient_tint * shadow_blend * 0.5,
        lit_color.a,
    );

    // Subsurface scattering for translucent blocks.
    // BSL: VoL = dot(viewDir, lightDir) * 0.5 + 0.5, scattering = pow(VoL, 16)
    if (bsl.subsurface_strength > 0.0) {
        let world_normal = normalize(in.world_normal);
        // Approximate: back-facing surfaces relative to the main light
        // get a soft glow. We use the normal as a proxy for the
        // light-to-view transmission direction.
        let transmission = saturate(-world_normal.y * 0.5 + 0.5);
        let sss = pow(transmission, 4.0) * bsl.subsurface_strength * 0.3;
        lit_color = vec4(lit_color.rgb + pbr_input.material.base_color.rgb * sss, lit_color.a);
    }

    out.color = main_pass_post_lighting_processing(pbr_input, lit_color);
#endif

    return out;
}
