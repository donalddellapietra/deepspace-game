// Ray march shader for base-3 recursive voxel tree.
//
// One unified tree walker. When it descends into a Node child whose
// NodeKind is CubedSphereBody, it switches to the cubed-sphere DDA
// running in that body cell's local frame — no parallel uniforms,
// no separate face_root buffers, no absolute world coords.

#include "bindings.wgsl"
#include "ray_prim.wgsl"
#include "sphere.wgsl"
#include "march.wgsl"

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    let uv = vec2<f32>(f32((idx << 1u) & 2u), f32(idx & 2u));
    var out: VertexOutput;
    out.position = vec4<f32>(uv * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(uv.x, 1.0 - uv.y);
    return out;
}

/// Compute the ray direction for `uv` under the current camera,
/// with the per-frame sub-pixel jitter baked in. Used by both
/// fragment entry points so the ray-march runs on the jittered
/// sample regardless of whether TAAU is the active resolve path.
/// Jitter is zero when TAAU is off, so the non-TAA path behaves
/// exactly like before.
fn jittered_ray_dir(uv: vec2<f32>) -> vec3<f32> {
    let aspect = uniforms.screen_width / uniforms.screen_height;
    let half_fov_tan = tan(camera.fov * 0.5);
    // Jitter is specified in output-texel units; convert to NDC.
    let jitter_ndc_x = camera.jitter_x_px / uniforms.screen_width * 2.0 * aspect * half_fov_tan;
    let jitter_ndc_y = camera.jitter_y_px / uniforms.screen_height * 2.0 * half_fov_tan;
    let ndc = vec2<f32>(
        (uv.x - 0.5) * 2.0 * aspect * half_fov_tan + jitter_ndc_x,
        (0.5 - uv.y) * 2.0 * half_fov_tan + jitter_ndc_y,
    );
    return camera.forward + camera.right * ndc.x + camera.up * ndc.y;
}

/// Shared pixel-shading kernel. Returns both the (gamma-corrected)
/// RGB color that fs_main writes to the swapchain and the raw
/// HitResult so callers that need the hit t (fs_main_taa) can pass
/// it through without re-running the march.
///
/// Color is the same value `fs_main` used to return — including the
/// manual `pow(1/2.2)` gamma correction it applies before write. TAAU
/// uses this same value so visual output stays consistent with the
/// non-TAAU path; the gamma-space clamp + blend in the resolve shader
/// is a small precision hit but not a visible one.
fn shade_pixel(uv: vec2<f32>) -> vec4<f32> {
    let ray_dir = jittered_ray_dir(uv);
    let result = march(camera.pos, ray_dir);

    var color: vec3<f32>;
    if result.hit {
        let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
        let diffuse = max(dot(result.normal, sun_dir), 0.0);
        let ambient = 0.3;
        let hit_pos = camera.pos + ray_dir * result.t;
        let local = clamp((hit_pos - result.cell_min) / result.cell_size, vec3<f32>(0.0), vec3<f32>(1.0));
        let bevel = cube_face_bevel(local, result.normal);
        let lit = result.color * (ambient + diffuse * 0.7) * (0.7 + 0.3 * bevel);
        color = pow(lit, vec3<f32>(1.0 / 2.2));
    } else {
        let sky_t = ray_dir.y * 0.5 + 0.5;
        color = mix(vec3<f32>(0.7, 0.8, 0.95), vec3<f32>(0.3, 0.5, 0.85), sky_t);
    }

    if uniforms.highlight_active != 0u {
        let h_min = uniforms.highlight_min.xyz;
        let h_max = uniforms.highlight_max.xyz;
        let h_size = h_max - h_min;
        if result.hit {
            let hit_pos = camera.pos + ray_dir * result.t;
            let pad_local = max_component(h_size) * 0.03;
            let inside = all(hit_pos >= (h_min - vec3<f32>(pad_local))) &&
                         all(hit_pos <= (h_max + vec3<f32>(pad_local)));
            if inside {
                let local_h = clamp((hit_pos - h_min) / max(h_size, vec3<f32>(1e-6)), vec3<f32>(0.0), vec3<f32>(1.0));
                let edge = min(
                    min(min(local_h.x, 1.0 - local_h.x), min(local_h.y, 1.0 - local_h.y)),
                    min(local_h.z, 1.0 - local_h.z)
                );
                let glow = 1.0 - smoothstep(0.02, 0.12, edge);
                color = mix(color, vec3<f32>(1.0, 0.92, 0.18), glow * 0.85);
            }
        }
    }

    // The crosshair reticle lives in the HTML overlay — see
    // `ui/src/components/Crosshair.tsx` and
    // `src/app/edit_actions/highlight.rs`. Rendering it in the shader
    // would bake it into the ray-march framebuffer, which either
    // sits at half-res under TAAU (blurring the 1-pixel strokes) or
    // aliases against the jitter sequence (pixel crawl on slow
    // motion). DOM overlay is always at physical resolution and
    // composites cleanly on top; it's the SOTA separation.

    // Emit per-ray stats to the shader_stats buffer. Gated behind
    // the `ENABLE_STATS` override so the off-state has zero runtime
    // cost.
    if ENABLE_STATS {
        atomicAdd(&shader_stats.ray_count, 1u);
        if result.hit {
            atomicAdd(&shader_stats.hit_count, 1u);
        } else {
            atomicAdd(&shader_stats.miss_count, 1u);
        }
        if ray_steps >= 2048u {
            atomicAdd(&shader_stats.max_iter_count, 1u);
        }
        atomicAdd(&shader_stats.sum_steps_div4, (ray_steps + 3u) >> 2u);
        atomicMax(&shader_stats.max_steps, ray_steps);
        atomicAdd(&shader_stats.sum_steps_oob_div4, (ray_steps_oob + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_steps_empty_div4, (ray_steps_empty + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_steps_node_descend_div4, (ray_steps_node_descend + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_steps_lod_terminal_div4, (ray_steps_lod_terminal + 3u) >> 2u);
    }

    // Stash t in the alpha channel so fs_main_taa can route it into
    // a second render attachment without re-running the march.
    return vec4<f32>(color, result.t);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let shaded = shade_pixel(in.uv);
    return vec4<f32>(shaded.rgb, 1.0);
}

/// TAAU entry point. Writes color to `@location(0)` and hit t to
/// `@location(1)` (R32Float), both at the half-res render target.
/// The resolve pass reconstructs world-space hit positions from
/// `(camera.pos, ray_dir, t)` to reproject history across frames.
struct TaaFragOut {
    @location(0) color: vec4<f32>,
    @location(1) t: f32,
}

@fragment
fn fs_main_taa(in: VertexOutput) -> TaaFragOut {
    let shaded = shade_pixel(in.uv);
    var out: TaaFragOut;
    out.color = vec4<f32>(shaded.rgb, 1.0);
    out.t = shaded.a;
    return out;
}
