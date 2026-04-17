// Compute-shader ray-march entry. One thread per pixel at
// @workgroup_size(8, 8, 1); each thread owns a row of the workgroup-
// memory stack used by `march_cartesian`. Writes the final RGBA color
// to an rgba16float storage texture that `blit.wgsl` copies to the
// surface.

#include "bindings.wgsl"
#include "ray_prim.wgsl"
#include "sphere.wgsl"
#include "march_cartesian.wgsl"
#include "march.wgsl"

@group(1) @binding(0) var output_color: texture_storage_2d<rgba16float, write>;

@compute @workgroup_size(8, 8, 1)
fn cs_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let width_u = u32(uniforms.screen_width);
    let height_u = u32(uniforms.screen_height);
    if gid.x >= width_u || gid.y >= height_u { return; }

    let uv = vec2<f32>(
        (f32(gid.x) + 0.5) / uniforms.screen_width,
        (f32(gid.y) + 0.5) / uniforms.screen_height,
    );
    let aspect = uniforms.screen_width / uniforms.screen_height;
    let half_fov_tan = tan(camera.fov * 0.5);
    let ndc = vec2<f32>(
        (uv.x - 0.5) * 2.0 * aspect * half_fov_tan,
        (0.5 - uv.y) * 2.0 * half_fov_tan,
    );
    let ray_dir = camera.forward + camera.right * ndc.x + camera.up * ndc.y;

    let result = march(camera.pos, ray_dir, lid);

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

    let pixel = vec2<f32>(f32(gid.x) + 0.5, f32(gid.y) + 0.5);
    let center = vec2<f32>(uniforms.screen_width * 0.5, uniforms.screen_height * 0.5);
    let d = abs(pixel - center);
    let cross_size = 12.0;
    let cross_thickness = 1.5;
    let gap = 3.0;
    let is_crosshair = (d.x < cross_thickness && d.y >= gap && d.y < cross_size)
                    || (d.y < cross_thickness && d.x >= gap && d.x < cross_size);
    if is_crosshair {
        let cross_color = select(
            vec3<f32>(0.95, 0.95, 0.98),
            vec3<f32>(1.0, 0.92, 0.18),
            result.hit,
        );
        color = mix(color, cross_color, 0.95);
    }

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

    textureStore(output_color, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(color, 1.0));
}
