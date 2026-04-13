// BSL-inspired voxel lighting shader.
//
// Extends StandardMaterial's PBR output with:
// - SSAO integration with BSL's squared skylight curve
// - Vertex AO fallback when SSAO is unavailable
// - Warm sun / cool shadow ambient blending (BSL signature look)
// - Subsurface scattering for translucent blocks (leaves, water, glass)

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
    /// XZ distance from camera beyond which fragments are discarded,
    /// creating a smooth circular terrain boundary. 0 = no clipping.
    clip_radius: f32,
    _padding: f32,
}

@group(#{MATERIAL_BIND_GROUP}) @binding(100)
var<uniform> bsl: BslParams;

@fragment
fn fragment(
    in: VertexOutput,
    @builtin(front_facing) is_front: bool,
) -> FragmentOutput {
    // Clip terrain to a smooth circle and compute distance for AO fade.
    var clip_dist_sq = 0.0;
    if (bsl.clip_radius > 0.0) {
        let dx = in.world_position.x - view.world_position.x;
        let dz = in.world_position.z - view.world_position.z;
        clip_dist_sq = dx * dx + dz * dz;
        if (clip_dist_sq > bsl.clip_radius * bsl.clip_radius) {
            discard;
        }
    }

    var pbr_input = pbr_input_from_standard_material(in, is_front);
    pbr_input.material.base_color = alpha_discard(pbr_input.material, pbr_input.material.base_color);

    // Near the clip boundary, disable SSAO in PBR lighting too
    // (not just the BSL AO layer). Without this, apply_pbr_lighting
    // still applies the dark SSAO halo via diffuse_occlusion.
    if (bsl.clip_radius > 0.0) {
        let fade_start_sq = bsl.clip_radius * bsl.clip_radius * 0.64;
        let fade_end_sq = bsl.clip_radius * bsl.clip_radius;
        let pbr_ao_fade = 1.0 - smoothstep(fade_start_sq, fade_end_sq, clip_dist_sq);
        pbr_input.diffuse_occlusion = mix(vec3(1.0), pbr_input.diffuse_occlusion, vec3(pbr_ao_fade));
    }

#ifdef PREPASS_PIPELINE
    let out = deferred_output(in, pbr_input);
#else
    var out: FragmentOutput;

    var lit_color = apply_pbr_lighting(pbr_input);

    // --- Ambient occlusion ---
    // Use SSAO (diffuse_occlusion) as the primary AO source. Falls
    // back to vertex-color AO from the greedy mesher when SSAO is
    // unavailable (e.g. WASM). Both use BSL's squared skylight curve.
    let ssao = pbr_input.diffuse_occlusion.r;
#ifdef VERTEX_COLORS
    let ao_raw = min(ssao, in.color.r);
#else
    let ao_raw = ssao;
#endif
    // Fade AO to zero near the clip boundary so SSAO depth-discontinuity
    // artifacts (dark block outlines at the terrain edge) disappear.
    var ao_fade = 1.0;
    if (bsl.clip_radius > 0.0) {
        let fade_start_sq = bsl.clip_radius * bsl.clip_radius * 0.64; // 80% radius
        let fade_end_sq = bsl.clip_radius * bsl.clip_radius;
        ao_fade = 1.0 - smoothstep(fade_start_sq, fade_end_sq, clip_dist_sq);
    }
    let ao = mix(1.0, ao_raw * ao_raw, bsl.ao_strength * ao_fade);
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

    out.color = main_pass_post_lighting_processing(pbr_input, lit_color);
#endif

    return out;
}
