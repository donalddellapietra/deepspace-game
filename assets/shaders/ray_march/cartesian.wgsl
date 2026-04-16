// -------------- Stack-based Cartesian tree DDA --------------

/// Cartesian DDA in a single frame rooted at `root_node_idx`. The
/// frame's cell spans `[0, 3)³` in `ray_origin/ray_dir` coords.
/// Returns hit on cell terminal; on miss (ray exits the frame),
/// returns hit=false so the caller can pop to the ancestor ribbon.
fn march_cartesian(
    root_node_idx: u32, ray_origin: vec3<f32>, ray_dir: vec3<f32>, depth_limit: u32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.highlight_min = vec3<f32>(0.0);
    result.highlight_max = vec3<f32>(0.0);
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let inv_dir = vec3<f32>(
        select(1e10, 1.0 / ray_dir.x, abs(ray_dir.x) > 1e-8),
        select(1e10, 1.0 / ray_dir.y, abs(ray_dir.y) > 1e-8),
        select(1e10, 1.0 / ray_dir.z, abs(ray_dir.z) > 1e-8),
    );
    let ray_metric = max(length(ray_dir), 1e-6);
    let step = vec3<i32>(
        select(-1, 1, ray_dir.x >= 0.0),
        select(-1, 1, ray_dir.y >= 0.0),
        select(-1, 1, ray_dir.z >= 0.0),
    );
    let delta_dist = abs(inv_dir);

    var s_node_idx: array<u32, MAX_STACK_DEPTH>;
    var s_cell: array<vec3<i32>, MAX_STACK_DEPTH>;
    var s_side_dist: array<vec3<f32>, MAX_STACK_DEPTH>;
    var s_node_origin: array<vec3<f32>, MAX_STACK_DEPTH>;
    var s_cell_size: array<f32, MAX_STACK_DEPTH>;

    var normal = vec3<f32>(0.0, 1.0, 0.0);
    var depth: u32 = 0u;

    s_node_idx[0] = root_node_idx;
    s_node_origin[0] = vec3<f32>(0.0);
    s_cell_size[0] = 1.0;

    let root_hit = ray_box(ray_origin, inv_dir, vec3<f32>(0.0), vec3<f32>(3.0));
    if root_hit.t_enter >= root_hit.t_exit || root_hit.t_exit < 0.0 {
        return result;
    }

    let t_start = max(root_hit.t_enter, 0.0) + 0.001;
    let entry_pos = ray_origin + ray_dir * t_start;

    s_cell[0] = vec3<i32>(
        clamp(i32(floor(entry_pos.x)), 0, 2),
        clamp(i32(floor(entry_pos.y)), 0, 2),
        clamp(i32(floor(entry_pos.z)), 0, 2),
    );
    let cell_f = vec3<f32>(s_cell[0]);
    s_side_dist[0] = vec3<f32>(
        select((cell_f.x - entry_pos.x) * inv_dir.x,
               (cell_f.x + 1.0 - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
        select((cell_f.y - entry_pos.y) * inv_dir.y,
               (cell_f.y + 1.0 - entry_pos.y) * inv_dir.y, ray_dir.y >= 0.0),
        select((cell_f.z - entry_pos.z) * inv_dir.z,
               (cell_f.z + 1.0 - entry_pos.z) * inv_dir.z, ray_dir.z >= 0.0),
    );

    var iterations = 0u;
    let max_iterations = 2048u;

    loop {
        if iterations >= max_iterations { break; }
        iterations += 1u;

        let cell = s_cell[depth];

        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            if depth == 0u { break; }
            depth -= 1u;

            if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                s_cell[depth].x += step.x;
                s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
            } else if s_side_dist[depth].y < s_side_dist[depth].z {
                s_cell[depth].y += step.y;
                s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                normal = vec3<f32>(0.0, f32(-step.y), 0.0);
            } else {
                s_cell[depth].z += step.z;
                s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                normal = vec3<f32>(0.0, 0.0, f32(-step.z));
            }
            continue;
        }

        let slot = slot_from_xyz(cell.x, cell.y, cell.z);
        let packed = child_packed(s_node_idx[depth], slot);
        let tag = child_tag(packed);

        if tag == 0u {
            // Empty — DDA advance.
            if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                s_cell[depth].x += step.x;
                s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
            } else if s_side_dist[depth].y < s_side_dist[depth].z {
                s_cell[depth].y += step.y;
                s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                normal = vec3<f32>(0.0, f32(-step.y), 0.0);
            } else {
                s_cell[depth].z += step.z;
                s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                normal = vec3<f32>(0.0, 0.0, f32(-step.z));
            }
        } else if tag == 1u {
            let cell_min_h = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
            let cell_max_h = cell_min_h + vec3<f32>(s_cell_size[depth]);
            let cell_box_h = ray_box(ray_origin, inv_dir, cell_min_h, cell_max_h);
            result.hit = true;
            result.t = max(cell_box_h.t_enter, 0.0);
            result.color = palette.colors[child_block_type(packed)].rgb;
            result.normal = normal;
            result.cell_min = cell_min_h;
            result.cell_size = s_cell_size[depth];
            return result;
        } else {
            // tag == 2u: Node child. Look up its kind.
            let child_idx = child_node_index(s_node_idx[depth], slot);
            let kind = node_kinds[child_idx].kind;

            if kind == 1u {
                // CubedSphereBody: dispatch sphere DDA in this body's cell.
                let body_origin = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
                let body_size = s_cell_size[depth];
                let inner_r = node_kinds[child_idx].inner_r;
                let outer_r = node_kinds[child_idx].outer_r;
                let sph = sphere_in_cell(
                    child_idx, body_origin, body_size,
                    inner_r, outer_r, ray_origin, ray_dir,
                );
                if sph.hit {
                    return sph;
                }
                // Sphere missed — advance Cartesian DDA past this cell.
                if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                    s_cell[depth].x += step.x;
                    s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if s_side_dist[depth].y < s_side_dist[depth].z {
                    s_cell[depth].y += step.y;
                    s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    s_cell[depth].z += step.z;
                    s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
                continue;
            }
            if false {
                // Real path (re-enable after diagnostic confirms dispatch):
                let body_origin = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
                let body_size = s_cell_size[depth];
                let inner_r = node_kinds[child_idx].inner_r;
                let outer_r = node_kinds[child_idx].outer_r;
                let sph = sphere_in_cell(
                    child_idx, body_origin, body_size,
                    inner_r, outer_r, ray_origin, ray_dir,
                );
                if sph.hit {
                    return sph;
                }
                // Sphere missed — advance Cartesian DDA past this cell.
                if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                    s_cell[depth].x += step.x;
                    s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                    normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                } else if s_side_dist[depth].y < s_side_dist[depth].z {
                    s_cell[depth].y += step.y;
                    s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                    normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                } else {
                    s_cell[depth].z += step.z;
                    s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                    normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                }
                continue;
            }

            // Cartesian Node: depth/LOD check, then descend.
            let at_max = depth + 1u >= depth_limit || depth + 1u >= MAX_STACK_DEPTH;
            let child_cell_size = s_cell_size[depth] / 3.0;
            let cell_world_size = child_cell_size;
            let min_side = min(s_side_dist[depth].x, min(s_side_dist[depth].y, s_side_dist[depth].z));
            let ray_dist = max(min_side * ray_metric, 0.001);
            let lod_pixels = cell_world_size / ray_dist * uniforms.screen_height / (2.0 * tan(camera.fov * 0.5));
            let at_lod = lod_pixels < 1.0;

            if at_max || at_lod {
                let bt = child_block_type(packed);
                if bt == 255u {
                    if s_side_dist[depth].x < s_side_dist[depth].y && s_side_dist[depth].x < s_side_dist[depth].z {
                        s_cell[depth].x += step.x;
                        s_side_dist[depth].x += delta_dist.x * s_cell_size[depth];
                        normal = vec3<f32>(f32(-step.x), 0.0, 0.0);
                    } else if s_side_dist[depth].y < s_side_dist[depth].z {
                        s_cell[depth].y += step.y;
                        s_side_dist[depth].y += delta_dist.y * s_cell_size[depth];
                        normal = vec3<f32>(0.0, f32(-step.y), 0.0);
                    } else {
                        s_cell[depth].z += step.z;
                        s_side_dist[depth].z += delta_dist.z * s_cell_size[depth];
                        normal = vec3<f32>(0.0, 0.0, f32(-step.z));
                    }
                } else {
                    let cell_min_l = s_node_origin[depth] + vec3<f32>(cell) * s_cell_size[depth];
                    let cell_max_l = cell_min_l + vec3<f32>(s_cell_size[depth]);
                    let cell_box_l = ray_box(ray_origin, inv_dir, cell_min_l, cell_max_l);
                    result.hit = true;
                    result.t = max(cell_box_l.t_enter, 0.0);
                    result.color = palette.colors[bt].rgb;
                    result.normal = normal;
                    result.cell_min = cell_min_l;
                    result.cell_size = s_cell_size[depth];
                    return result;
                }
            } else {
                let parent_origin = s_node_origin[depth];
                let parent_cell_size = s_cell_size[depth];
                let child_origin = parent_origin + vec3<f32>(cell) * parent_cell_size;

                let child_max = child_origin + vec3<f32>(parent_cell_size);
                let child_hit = ray_box(ray_origin, inv_dir, child_origin, child_max);
                let ct_start = max(child_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
                let child_entry = ray_origin + ray_dir * ct_start;
                let local_entry = (child_entry - child_origin) / child_cell_size;

                depth += 1u;
                s_node_idx[depth] = child_idx;
                s_node_origin[depth] = child_origin;
                s_cell_size[depth] = child_cell_size;
                s_cell[depth] = vec3<i32>(
                    clamp(i32(floor(local_entry.x)), 0, 2),
                    clamp(i32(floor(local_entry.y)), 0, 2),
                    clamp(i32(floor(local_entry.z)), 0, 2),
                );
                let lc = vec3<f32>(s_cell[depth]);
                s_side_dist[depth] = vec3<f32>(
                    select((child_origin.x + lc.x * child_cell_size - ray_origin.x) * inv_dir.x,
                           (child_origin.x + (lc.x + 1.0) * child_cell_size - ray_origin.x) * inv_dir.x, ray_dir.x >= 0.0),
                    select((child_origin.y + lc.y * child_cell_size - ray_origin.y) * inv_dir.y,
                           (child_origin.y + (lc.y + 1.0) * child_cell_size - ray_origin.y) * inv_dir.y, ray_dir.y >= 0.0),
                    select((child_origin.z + lc.z * child_cell_size - ray_origin.z) * inv_dir.z,
                           (child_origin.z + (lc.z + 1.0) * child_cell_size - ray_origin.z) * inv_dir.z, ray_dir.z >= 0.0),
                );
            }
        }
    }

    return result;
}

fn cartesian_shell_depth_limit(ribbon_level: u32) -> u32 {
    var depth_limit = uniforms.max_depth;
    let shell_count = min(uniforms.cartesian_shell_meta.x, MAX_CARTESIAN_SHELLS);
    var idx: u32 = 0u;
    loop {
        if idx >= shell_count {
            break;
        }
        let pair = uniforms.cartesian_shell_pairs[idx / 2u];
        let use_second = idx % 2u == 1u;
        let shell_ribbon = select(pair.x, pair.z, use_second);
        let shell_depth = select(pair.y, pair.w, use_second);
        if ribbon_level >= shell_ribbon {
            depth_limit = shell_depth;
        }
        idx = idx + 1u;
    }
    return depth_limit;
}

fn next_cartesian_shell_ribbon(current_ribbon_level: u32) -> u32 {
    let shell_count = min(uniforms.cartesian_shell_meta.x, MAX_CARTESIAN_SHELLS);
    var idx: u32 = 0u;
    loop {
        if idx >= shell_count {
            break;
        }
        let pair = uniforms.cartesian_shell_pairs[idx / 2u];
        let use_second = idx % 2u == 1u;
        let shell_ribbon = select(pair.x, pair.z, use_second);
        if shell_ribbon > current_ribbon_level {
            return min(shell_ribbon, uniforms.ribbon_count);
        }
        idx = idx + 1u;
    }
    return uniforms.ribbon_count;
}
