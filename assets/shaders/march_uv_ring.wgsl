
fn march_uv_ring(
    ring_idx: u32,
    body_origin: vec3<f32>,
    body_size: f32,
    ray_origin: vec3<f32>,
    ray_dir_in: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let dims_x = i32(uniforms.slab_dims.x);
    let slab_depth = uniforms.slab_dims.w;
    if dims_x <= 0 { return result; }

    let center = body_origin + vec3<f32>(body_size * 0.5);
    let pi = 3.14159265;
    let angle_step = 2.0 * pi / f32(dims_x);
    let radius = body_size * 0.38;
    let side = max((2.0 * pi * radius / f32(dims_x)) * 0.95, body_size / 27.0);

    var best_t = 1e20;
    var best: HitResult = result;
    for (var cell_x: i32 = 0; cell_x < dims_x; cell_x = cell_x + 1) {
        let angle = -pi + (f32(cell_x) + 0.5) * angle_step;
        let sa = sin(angle);
        let ca = cos(angle);
        let radial = vec3<f32>(ca, 0.0, sa);
        let tangent = vec3<f32>(-sa, 0.0, ca);
        let up = vec3<f32>(0.0, 1.0, 0.0);
        let cube_origin = center + radial * radius;
        let scale = 3.0 / side;
        let d_origin = ray_origin - cube_origin;
        let local_origin = vec3<f32>(
            dot(tangent, d_origin) * scale + 1.5,
            dot(radial, d_origin) * scale + 1.5,
            dot(up, d_origin) * scale + 1.5,
        );
        let local_dir = vec3<f32>(
            dot(tangent, ray_dir_in) * scale,
            dot(radial, ray_dir_in) * scale,
            dot(up, ray_dir_in) * scale,
        );

        let inv_local = vec3<f32>(
            select(1e10, 1.0 / local_dir.x, abs(local_dir.x) > 1e-8),
            select(1e10, 1.0 / local_dir.y, abs(local_dir.y) > 1e-8),
            select(1e10, 1.0 / local_dir.z, abs(local_dir.z) > 1e-8),
        );
        let cube_box = ray_box(local_origin, inv_local, vec3<f32>(0.0), vec3<f32>(3.0));
        if cube_box.t_enter >= cube_box.t_exit || cube_box.t_exit <= 0.0 {
            continue;
        }

        let sample = sample_slab_cell(ring_idx, slab_depth, cell_x, 0, 0);
        if sample.block_type == 0xFFFEu {
            continue;
        }

        if sample.tag == 2u {
            let aabb_bits = aabbs[sample.child_idx] & 0xFFFu;
            if aabb_bits == 0u {
                continue;
            }
            let amin = vec3<f32>(
                f32(aabb_bits & 3u),
                f32((aabb_bits >> 2u) & 3u),
                f32((aabb_bits >> 4u) & 3u),
            );
            let amax = vec3<f32>(
                f32(((aabb_bits >> 6u) & 3u) + 1u),
                f32(((aabb_bits >> 8u) & 3u) + 1u),
                f32(((aabb_bits >> 10u) & 3u) + 1u),
            );
            let content_box = ray_box(local_origin, inv_local, amin, amax);
            if content_box.t_enter >= content_box.t_exit || content_box.t_exit <= 0.0 {
                continue;
            }

            let sub = march_in_tangent_cube(sample.child_idx, local_origin, local_dir);
            if sub.hit && sub.t < best_t {
                let local_hit = local_origin + local_dir * sub.t;
                let local_in_cell = clamp(
                    (local_hit - sub.cell_min) / sub.cell_size,
                    vec3<f32>(0.0), vec3<f32>(1.0),
                );
                let local_bevel = cube_face_bevel(local_in_cell, sub.normal);
                var out: HitResult;
                out.hit = true;
                out.t = sub.t;
                out.color = sub.color * (0.7 + 0.3 * local_bevel);
                out.normal = tangent * sub.normal.x
                           + radial * sub.normal.y
                           + up * sub.normal.z;
                out.frame_level = 0u;
                out.frame_scale = 1.0;
                let hit_world = ray_origin + ray_dir_in * sub.t;
                out.cell_min = hit_world - vec3<f32>(0.5);
                out.cell_size = 1.0;
                best_t = sub.t;
                best = out;
            }
        } else if sample.tag == 1u {
            if cube_box.t_enter < cube_box.t_exit && cube_box.t_exit > 0.0 {
                let t_local = max(cube_box.t_enter, 0.0);
                if t_local < best_t {
                    let entry_local = local_origin + local_dir * t_local;
                    let dx_lo = abs(entry_local.x - 0.0);
                    let dx_hi = abs(entry_local.x - 3.0);
                    let dy_lo = abs(entry_local.y - 0.0);
                    let dy_hi = abs(entry_local.y - 3.0);
                    let dz_lo = abs(entry_local.z - 0.0);
                    let dz_hi = abs(entry_local.z - 3.0);
                    var best_face = dx_lo;
                    var local_normal = vec3<f32>(-1.0, 0.0, 0.0);
                    if dx_hi < best_face { best_face = dx_hi; local_normal = vec3<f32>(1.0, 0.0, 0.0); }
                    if dy_lo < best_face { best_face = dy_lo; local_normal = vec3<f32>(0.0, -1.0, 0.0); }
                    if dy_hi < best_face { best_face = dy_hi; local_normal = vec3<f32>(0.0, 1.0, 0.0); }
                    if dz_lo < best_face { best_face = dz_lo; local_normal = vec3<f32>(0.0, 0.0, -1.0); }
                    if dz_hi < best_face { best_face = dz_hi; local_normal = vec3<f32>(0.0, 0.0, 1.0); }
                    let local_in_cell = clamp(entry_local / 3.0, vec3<f32>(0.0), vec3<f32>(1.0));
                    let local_bevel = cube_face_bevel(local_in_cell, local_normal);
                    var out: HitResult;
                    out.hit = true;
                    out.t = t_local;
                    out.color = palette[sample.block_type].rgb * (0.7 + 0.3 * local_bevel);
                    out.normal = tangent * local_normal.x
                               + radial * local_normal.y
                               + up * local_normal.z;
                    out.frame_level = 0u;
                    out.frame_scale = 1.0;
                    let hit_world = ray_origin + ray_dir_in * t_local;
                    out.cell_min = hit_world - vec3<f32>(0.5);
                    out.cell_size = 1.0;
                    best_t = t_local;
                    best = out;
                }
            }
        }
    }

    return best;
}

// Top-level march. Dispatches the current frame's Cartesian DDA,
// then on miss pops to the next ancestor in the ribbon and
// continues. When ribbon is exhausted, returns sky (hit=false).
//
// Each pop transforms the ray into the parent's frame coords:
// `parent_pos = slot_xyz + frame_pos / 3`, `parent_dir = frame_dir / 3`.
// The parent's frame cell still spans `[0, 3)³` in its own
// coords, so the inner DDA is unchanged — only the ray is
// rescaled and the buffer node_idx swapped.
