// BSL-inspired voxel lighting shader.
//
// Extends StandardMaterial's PBR output with:
// - SSAO integration with BSL's squared skylight curve
// - Warm sun / cool shadow ambient blending (BSL signature look)
// - Subsurface scattering for translucent blocks (leaves, water, glass)
// - Distance fog that fades terrain into atmosphere at render edge

#import bevy_pbr::{
    pbr_fragment::pbr_input_from_standard_material,
    pbr_functions::alpha_discard,
    mesh_view_bindings::view,
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
    fog_start: f32,
    fog_end: f32,
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

    // SSAO occlusion: Bevy's SSAO only darkens indirect/ambient light —
    // invisible under direct sunlight. Read the SSAO value and apply
    // BSL's squared skylight curve to the full lighting result.
    let ssao = pbr_input.diffuse_occlusion.r;
    let ao = mix(1.0, ssao * ssao, bsl.ao_strength);
    lit_color = vec4(lit_color.rgb * ao, lit_color.a);

    // --- Warm sun / cool shadow blending ---
    let luminance = dot(lit_color.rgb, vec3(0.299, 0.587, 0.114));
    let ambient_tint = bsl.ambient_color.rgb * bsl.ambient_color.a;
    let shadow_blend = 1.0 - smoothstep(0.05, 0.5, luminance);
    let base_lum = max(dot(pbr_input.material.base_color.rgb, vec3(0.299, 0.587, 0.114)), 0.05);
    lit_color = vec4(
        lit_color.rgb + ambient_tint * shadow_blend * base_lum * 0.4,
        lit_color.a,
    );

    // --- Subsurface scattering for translucent blocks ---
    if (bsl.subsurface_strength > 0.0) {
        let world_normal = normalize(in.world_normal);
        let transmission = saturate(-world_normal.y * 0.5 + 0.5);
        let sss = pow(transmission, 6.0) * bsl.subsurface_strength * 0.35;
        let sss_color = pbr_input.material.base_color.rgb * vec3(1.1, 1.0, 0.9);
        lit_color = vec4(lit_color.rgb + sss_color * sss, lit_color.a);
    }

    // --- Distance fog ---
    // Fade terrain toward the atmosphere haze color at the render edge.
    // Uses a smooth hermite curve so the transition is gradual, not a
    // hard line. The fog color approximates the horizon haze.
    if (bsl.fog_end > 0.0) {
        let frag_pos = in.world_position.xyz;
        let cam_pos = view.world_position;
        let dist = length(frag_pos - cam_pos);
        let fog_factor = smoothstep(bsl.fog_start, bsl.fog_end, dist);
        // Atmosphere haze color — matches the horizon look.
        let fog_color = vec3(0.7, 0.78, 0.72);
        lit_color = vec4(mix(lit_color.rgb, fog_color, fog_factor), lit_color.a);
    }

    out.color = main_pass_post_lighting_processing(pbr_input, lit_color);
#endif

    return out;
}
