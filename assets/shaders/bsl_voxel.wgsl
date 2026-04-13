// BSL-inspired voxel lighting shader.
//
// Extends StandardMaterial's PBR output with:
// - SSAO integration with BSL's squared skylight curve
// - Warm sun / cool shadow ambient blending (BSL signature look)
// - Subsurface scattering for translucent blocks (leaves, water, glass)
// - Soft shadow-edge color bleeding

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

    // SSAO occlusion: Bevy's SSAO only darkens indirect/ambient light —
    // invisible under direct sunlight. Read the SSAO value and apply
    // BSL's squared skylight curve to the full lighting result.
    let ssao = pbr_input.diffuse_occlusion.r;
    let ao = mix(1.0, ssao * ssao, bsl.ao_strength);
    lit_color = vec4(lit_color.rgb * ao, lit_color.a);

    // --- Warm sun / cool shadow blending ---
    // BSL's signature: shadowed regions tint toward cool blue ambient,
    // while lit regions keep the warm sun color. This creates depth
    // and atmosphere even with flat-colored voxels.
    let luminance = dot(lit_color.rgb, vec3(0.299, 0.587, 0.114));
    let ambient_tint = bsl.ambient_color.rgb * bsl.ambient_color.a;

    // Shadow blend: how much this fragment is in shadow (low luminance).
    // The smoothstep creates a soft transition at the shadow edge rather
    // than a hard ambient cutoff.
    let shadow_blend = 1.0 - smoothstep(0.05, 0.5, luminance);

    // Apply cool ambient tint in shadowed regions. The base_color
    // multiplication ensures the tint respects the block's own color
    // rather than adding a flat blue wash.
    let base_lum = max(dot(pbr_input.material.base_color.rgb, vec3(0.299, 0.587, 0.114)), 0.05);
    lit_color = vec4(
        lit_color.rgb + ambient_tint * shadow_blend * base_lum * 0.4,
        lit_color.a,
    );

    // --- Subsurface scattering for translucent blocks ---
    // BSL: VoL = dot(viewDir, lightDir) * 0.5 + 0.5, scattering = pow(VoL, 16)
    // We approximate using the world normal as a proxy for the
    // light-to-view transmission direction.
    if (bsl.subsurface_strength > 0.0) {
        let world_normal = normalize(in.world_normal);
        // Back-facing surfaces relative to the downward-angled sun
        // get a soft warm glow — light transmitting through leaves/water.
        let transmission = saturate(-world_normal.y * 0.5 + 0.5);
        // Sharper falloff (power 6) for more concentrated glow spots,
        // tinted slightly warm to match the sun color.
        let sss = pow(transmission, 6.0) * bsl.subsurface_strength * 0.35;
        let sss_color = pbr_input.material.base_color.rgb * vec3(1.1, 1.0, 0.9);
        lit_color = vec4(lit_color.rgb + sss_color * sss, lit_color.a);
    }

    out.color = main_pass_post_lighting_processing(pbr_input, lit_color);
#endif

    return out;
}
