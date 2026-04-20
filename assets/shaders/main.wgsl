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
    let ray_dir = normalize(camera.forward + camera.right * ndc.x + camera.up * ndc.y);

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

    if uniforms.highlight_active != 0u && result.hit {
        // Highlight-cell test.
        //
        // The CPU ships `highlight_path` as the anchor-depth slot
        // sequence (truncated to `edit_depth`), matching the cell
        // that break/place would edit. The walker, however, often
        // terminates on a PACKED-UNIFORM collapsed block — a single
        // GPU-side child entry that represents a deep subtree of
        // identical-content cells. Its `result.cell_min` /
        // `result.cell_size` cover that whole collapsed chunk. Naive
        // prefix matching then glowed the entire chunk instead of
        // the anchor-depth sub-cell the user is targeting.
        //
        // Correct test: the anchor cell is inside the walker cell
        // iff `highlight_path` starts with the walker's full world-
        // root path. When that holds, reconstruct the anchor cell's
        // bounds by extending `result.cell_min/cell_size` one slot
        // at a time through `highlight_path[full_hit_depth..]` — the
        // walker's cell is the precise reference frame, so each
        // extra subdivision stays representable until we're roughly
        // 14 levels below it. Then `hit_pos ∈ anchor_sub_cell` gates
        // the glow to the anchor-depth footprint of the hit.
        let h_depth = uniforms.highlight_path_depth;
        let r_depth = uniforms.render_path_depth;
        let pop_level = result.frame_level;
        let frame_prefix_len = select(0u, r_depth - pop_level, r_depth >= pop_level);
        let walker_depth = result.hit_path_depth;
        let full_hit_depth = frame_prefix_len + walker_depth;
        if h_depth > 0u && full_hit_depth > 0u && full_hit_depth <= h_depth {
            var match_ok: bool = true;
            for (var i: u32 = 0u; i < full_hit_depth; i = i + 1u) {
                var hit_slot: u32;
                if i < frame_prefix_len {
                    hit_slot = unpack_slot_from_path(uniforms.render_path, i);
                } else {
                    hit_slot = unpack_slot_from_path(result.hit_path, i - frame_prefix_len);
                }
                let hl_slot = unpack_slot_from_path(uniforms.highlight_path, i);
                if hit_slot != hl_slot {
                    match_ok = false;
                    break;
                }
            }
            if match_ok {
                // Walker's cell contains the anchor cell. Walk extra
                // slots to locate the anchor sub-cell bounds within
                // the walker's frame, then ray-AABB to decide which
                // pixels fall on the anchor-cell footprint.
                //
                // Why ray-AABB, not `hit_pos ∈ sub_cell`: when the
                // walker terminates on a packed uniform chunk, its
                // `result.t` is the chunk's outer-face entry, so
                // `hit_pos = camera + dir * t` sits on the chunk's
                // surface — outside the anchor sub-cell that lives
                // deeper inside. Using the ray's intersection with
                // the sub-cell box instead means every pixel whose
                // ray actually crosses the sub-cell lights up,
                // giving a correctly-sized glow regardless of where
                // the walker's LOD terminal landed.
                var sub_min = result.cell_min;
                var sub_size = result.cell_size;
                for (var i: u32 = full_hit_depth; i < h_depth; i = i + 1u) {
                    let slot = unpack_slot_from_path(uniforms.highlight_path, i);
                    let sx = slot % 3u;
                    let sy = (slot / 3u) % 3u;
                    let sz = slot / 9u;
                    sub_size = sub_size * (1.0 / 3.0);
                    sub_min = sub_min + vec3<f32>(
                        f32(sx) * sub_size,
                        f32(sy) * sub_size,
                        f32(sz) * sub_size,
                    );
                }
                let sub_max = sub_min + vec3<f32>(sub_size);
                let inv_dir = vec3<f32>(
                    select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
                    select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
                    select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
                );
                let anchor_hit = ray_box(camera.pos, inv_dir, sub_min, sub_max);
                // The ray just needs to traverse the anchor sub-cell
                // somewhere along its path — if it does, this pixel
                // is visualising a chunk of the scene that contains
                // the anchor cell (even if the anchor cell sits
                // deeper inside a uniform block than the visible
                // surface the walker hit).  That projection is what
                // the glow conveys.
                if anchor_hit.t_enter < anchor_hit.t_exit
                    && anchor_hit.t_exit > 0.0
                {
                    color = mix(color, vec3<f32>(1.0, 0.92, 0.18), 0.55);
                }
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
        // Round up on the /4 divide so single-step rays are
        // detectable (otherwise ray_steps=1 shifts to 0 and we lose
        // the signal at high LOD thresholds). u32 sum at 2560x1440
        // with avg_steps=170 = 630M, fits with room to spare.
        atomicAdd(&shader_stats.sum_steps_div4, (ray_steps + 3u) >> 2u);
        atomicMax(&shader_stats.max_steps, ray_steps);
        atomicAdd(&shader_stats.sum_steps_oob_div4, (ray_steps_oob + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_steps_empty_div4, (ray_steps_empty + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_steps_node_descend_div4, (ray_steps_node_descend + 3u) >> 2u);
        atomicAdd(&shader_stats.sum_steps_lod_terminal_div4, (ray_steps_lod_terminal + 3u) >> 2u);
    }

    return vec4<f32>(color, 1.0);
}
