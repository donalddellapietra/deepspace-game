// -------------- Frame dispatch + ancestor pop --------------

/// Top-level march. Dispatches the current frame's DDA on its
/// NodeKind (Cartesian or sphere body), then on miss pops to the
/// next ancestor in the ribbon and continues. When ribbon is
/// exhausted, returns sky (hit=false).
///
/// Each pop transforms the ray into the parent's frame coords:
/// `parent_pos = slot_xyz + frame_pos / 3`, `parent_dir = frame_dir / 3`.
/// The parent's frame cell still spans `[0, 3)³` in its own
/// coords, so the inner DDA is unchanged — only the ray is
/// rescaled and the buffer node_idx swapped.
fn march(world_ray_origin: vec3<f32>, world_ray_dir: vec3<f32>) -> HitResult {
    var ray_origin = world_ray_origin;
    var ray_dir = world_ray_dir;
    var current_idx = uniforms.root_index;
    var current_kind = uniforms.root_kind;
    var inner_r = uniforms.root_radii.x;
    var outer_r = uniforms.root_radii.y;
    var cur_face_bounds = uniforms.root_face_bounds;
    var ribbon_level: u32 = 0u;
    var cur_hmin = uniforms.highlight_min.xyz;
    var cur_hmax = uniforms.highlight_max.xyz;
    var cur_scale: f32 = 1.0;
    var current_cartesian_max_depth: u32 = cartesian_shell_depth_limit(0u);

    var hops: u32 = 0u;
    loop {
        if hops > 80u { break; }
        hops = hops + 1u;

        var r: HitResult;
        if current_kind == ROOT_KIND_BODY {
            let body_origin = vec3<f32>(0.0);
            let body_size = 3.0;
            r = sphere_in_cell(
                current_idx, body_origin, body_size,
                inner_r, outer_r, ray_origin, ray_dir,
            );
        } else if current_kind == ROOT_KIND_FACE {
            r = march_face_root(current_idx, ray_origin, ray_dir, cur_face_bounds);
        } else {
            r = march_cartesian(current_idx, ray_origin, ray_dir, current_cartesian_max_depth);
        }
        if r.hit {
            r.frame_level = ribbon_level;
            r.highlight_min = cur_hmin;
            r.highlight_max = cur_hmax;
            r.frame_scale = cur_scale;
            return r;
        }

        if current_kind == ROOT_KIND_CARTESIAN {
            let target_ribbon_level = next_cartesian_shell_ribbon(ribbon_level);
            if target_ribbon_level > ribbon_level {
                var level = ribbon_level;
                loop {
                    if level >= target_ribbon_level || level >= uniforms.ribbon_count {
                        break;
                    }
                    let entry = ribbon[level];
                    let s = entry.slot;
                    let sx = i32(s % 3u);
                    let sy = i32((s / 3u) % 3u);
                    let sz = i32(s / 9u);
                    let slot_off = vec3<f32>(f32(sx), f32(sy), f32(sz));
                    ray_origin = slot_off + ray_origin / 3.0;
                    ray_dir = ray_dir / 3.0;
                    if uniforms.highlight_active != 0u {
                        cur_hmin = slot_off + cur_hmin / 3.0;
                        cur_hmax = slot_off + cur_hmax / 3.0;
                    }
                    cur_scale = cur_scale * (1.0 / 3.0);
                    current_idx = entry.node_idx;
                    level = level + 1u;
                }
                ribbon_level = target_ribbon_level;
                current_cartesian_max_depth = cartesian_shell_depth_limit(ribbon_level);
                let k = node_kinds[current_idx].kind;
                if k == 1u {
                    current_kind = ROOT_KIND_BODY;
                    inner_r = node_kinds[current_idx].inner_r;
                    outer_r = node_kinds[current_idx].outer_r;
                } else {
                    current_kind = ROOT_KIND_CARTESIAN;
                }
                continue;
            }
        }

        // Ray exited the current frame. Try popping to ancestor.
        if ribbon_level >= uniforms.ribbon_count {
            break;
        }
        if current_kind == ROOT_KIND_FACE {
            let body_pop_level = uniforms.root_face_meta.y;
            if ribbon_level < body_pop_level {
                let entry = ribbon[ribbon_level];
                let s = entry.slot;
                let sx = i32(s % 3u);
                let sy = i32((s / 3u) % 3u);
                let sz = i32(s / 9u);
                let slot_off = vec3<f32>(f32(sx), f32(sy), f32(sz));
                let old_size = cur_face_bounds.w;
                cur_face_bounds = vec4<f32>(
                    cur_face_bounds.x - slot_off.x * old_size,
                    cur_face_bounds.y - slot_off.y * old_size,
                    cur_face_bounds.z - slot_off.z * old_size,
                    old_size * 3.0,
                );
                cur_scale = cur_scale * (1.0 / 3.0);
                current_idx = entry.node_idx;
                ribbon_level = ribbon_level + 1u;
                continue;
            }
            if body_pop_level >= uniforms.ribbon_count {
                break;
            }
            let body_entry = ribbon[body_pop_level];
            current_idx = body_entry.node_idx;
            current_kind = ROOT_KIND_BODY;
            inner_r = node_kinds[current_idx].inner_r;
            outer_r = node_kinds[current_idx].outer_r;
            ribbon_level = body_pop_level + 1u;
        } else {
            let entry = ribbon[ribbon_level];
            let s = entry.slot;
            let sx = i32(s % 3u);
            let sy = i32((s / 3u) % 3u);
            let sz = i32(s / 9u);
            let slot_off = vec3<f32>(f32(sx), f32(sy), f32(sz));
            ray_origin = slot_off + ray_origin / 3.0;
            ray_dir = ray_dir / 3.0;
            if uniforms.highlight_active != 0u {
                cur_hmin = slot_off + cur_hmin / 3.0;
                cur_hmax = slot_off + cur_hmax / 3.0;
            }
            cur_scale = cur_scale * (1.0 / 3.0);
            current_idx = entry.node_idx;
            let k = node_kinds[current_idx].kind;
            if k == 1u {
                current_kind = ROOT_KIND_BODY;
                inner_r = node_kinds[current_idx].inner_r;
                outer_r = node_kinds[current_idx].outer_r;
            } else {
                current_kind = ROOT_KIND_CARTESIAN;
                current_cartesian_max_depth = cartesian_shell_depth_limit(ribbon_level + 1u);
            }
            ribbon_level = ribbon_level + 1u;
        }
    }

    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.highlight_min = cur_hmin;
    result.highlight_max = cur_hmax;
    result.frame_scale = cur_scale;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;
    return result;
}

// -------------- Vertex / Fragment shaders --------------

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
    let ray_metric = max(length(ray_dir), 1e-6);

    let result = march(camera.pos, ray_dir);

    var color: vec3<f32>;
    if result.hit {
        let sun_dir = normalize(vec3<f32>(0.4, 0.7, 0.3));
        let diffuse = max(dot(result.normal, sun_dir), 0.0);
        let axis_tint = abs(result.normal.y) * 1.0
                      + (abs(result.normal.x) + abs(result.normal.z)) * 0.82;
        let ambient = 0.22;
        let hit_pos = camera.pos + ray_dir * result.t;
        let local = clamp((hit_pos - result.cell_min) / result.cell_size, vec3<f32>(0.0), vec3<f32>(1.0));
        let uv = face_uv_for_normal(local, result.normal);
        let face_edge = min(min(uv.x, 1.0 - uv.x), min(uv.y, 1.0 - uv.y));
        let bevel = cube_face_bevel(local, result.normal);
        let border = 1.0 - smoothstep(0.03, 0.09, face_edge);
        let face_shape = 0.72 + 0.28 * bevel;
        var lit = result.color * (ambient + diffuse * 0.78) * axis_tint * face_shape;
        lit = mix(lit, lit * 0.45, border * 0.95);
        color = pow(lit, vec3<f32>(1.0 / 2.2));
    } else {
        let sky_t = ray_dir.y * 0.5 + 0.5;
        color = mix(vec3<f32>(0.7, 0.8, 0.95), vec3<f32>(0.3, 0.5, 0.85), sky_t);
    }

    if uniforms.highlight_active != 0u {
        let h_min = select(uniforms.highlight_min.xyz, result.highlight_min, result.hit);
        let h_max = select(uniforms.highlight_max.xyz, result.highlight_max, result.hit);
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
        let pad = max(h_size.x * 0.02, 0.002);
        let box_min = h_min - vec3<f32>(pad);
        let box_max = h_max + vec3<f32>(pad);
        let h_inv_dir = vec3<f32>(
            select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
            select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
            select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
        );
        let hb = ray_box(camera.pos, h_inv_dir, box_min, box_max);
        if hb.t_enter < hb.t_exit && hb.t_exit > 0.0 {
            let t = max(hb.t_enter, 0.0);
            let t_local = t * ray_metric;
            let result_local = result.t * ray_metric;
            if t_local <= result_local + h_size.x * 0.05 {
                let hit_pos = camera.pos + ray_dir * t;
                let from_min = hit_pos - box_min;
                let from_max = box_max - hit_pos;
                let pixel_world = max(t_local, 0.001) * 2.0 * tan(camera.fov * 0.5) / uniforms.screen_height;
                let ew = max(pixel_world * 2.25, h_size.x * 0.02);
                let near_x = from_min.x < ew || from_max.x < ew;
                let near_y = from_min.y < ew || from_max.y < ew;
                let near_z = from_min.z < ew || from_max.z < ew;
                let edge_count = u32(near_x) + u32(near_y) + u32(near_z);
                if edge_count >= 2u {
                    color = mix(color, vec3<f32>(1.0, 0.92, 0.18), 0.92);
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

    return vec4<f32>(color, 1.0);
}
