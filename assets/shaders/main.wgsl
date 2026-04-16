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

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let aspect = uniforms.screen_width / uniforms.screen_height;
    let half_fov_tan = tan(camera.fov * 0.5);
    let ndc = vec2<f32>(
        (in.uv.x - 0.5) * 2.0 * aspect * half_fov_tan,
        (0.5 - in.uv.y) * 2.0 * half_fov_tan,
    );
    let ray_dir = camera.forward + camera.right * ndc.x + camera.up * ndc.y;

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
        // Highlight AABB is always in the camera's frame-local system.
        // t is invariant across ribbon pop transforms (origin and dir
        // both ÷3 per pop), so hit_pos = camera.pos + ray_dir * t is
        // always in the original frame regardless of which ribbon
        // level the DDA actually found the hit.
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

    let pixel = vec2<f32>(in.uv.x * uniforms.screen_width, in.uv.y * uniforms.screen_height);
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

    // Emit per-ray stats to the shader_stats buffer. Gated behind
    // the `ENABLE_STATS` override so the off-state has zero runtime
    // cost — per-pixel atomic contention on a 32-byte buffer can
    // add ~0.5–1ms at 1280x720 and would distort baseline perf
    // measurements. Enabled only when the harness passes
    // `--shader-stats`.
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
        atomicAdd(&shader_stats.sum_steps_div4, ray_steps >> 2u);
        atomicMax(&shader_stats.max_steps, ray_steps);
    }

    return vec4<f32>(color, 1.0);
}
