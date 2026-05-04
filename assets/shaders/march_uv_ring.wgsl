fn make_uv_ring_hit(
    pos: vec3<f32>, t_param: f32, inv_norm: f32,
    block_type: u32,
    r: f32, lat_p: f32, lon_p: f32,
    lon_lo: f32, lon_hi: f32,
    lat_lo: f32, lat_hi: f32,
    r_lo: f32, r_hi: f32,
) -> HitResult {
    var result: HitResult;
    let lon_step = lon_hi - lon_lo;
    let lat_step = lat_hi - lat_lo;
    let r_step = r_hi - r_lo;
    let cos_lat = max(cos(lat_p), 1e-3);
    let arc_lon_lo = r * cos_lat * abs(lon_p - lon_lo);
    let arc_lon_hi = r * cos_lat * abs(lon_p - lon_hi);
    let arc_lat_lo = r * abs(lat_p - lat_lo);
    let arc_lat_hi = r * abs(lat_p - lat_hi);
    let arc_r_lo = abs(r - r_lo);
    let arc_r_hi = abs(r - r_hi);
    var best = arc_lon_lo;
    var axis: u32 = 0u;
    if arc_lon_hi < best { best = arc_lon_hi; axis = 0u; }
    if arc_lat_lo < best { best = arc_lat_lo; axis = 1u; }
    if arc_lat_hi < best { best = arc_lat_hi; axis = 1u; }
    if arc_r_lo < best { best = arc_r_lo; axis = 2u; }
    if arc_r_hi < best { best = arc_r_hi; axis = 2u; }

    let lon_in_cell = clamp((lon_p - lon_lo) / lon_step, 0.0, 1.0);
    let lat_in_cell = clamp((lat_p - lat_lo) / lat_step, 0.0, 1.0);
    let r_in_cell = clamp((r - r_lo) / r_step, 0.0, 1.0);
    var u_in_face: f32;
    var v_in_face: f32;
    if axis == 0u {
        u_in_face = lat_in_cell;
        v_in_face = r_in_cell;
    } else if axis == 1u {
        u_in_face = lon_in_cell;
        v_in_face = r_in_cell;
    } else {
        u_in_face = lon_in_cell;
        v_in_face = lat_in_cell;
    }
    let face_edge = min(
        min(u_in_face, 1.0 - u_in_face),
        min(v_in_face, 1.0 - v_in_face),
    );
    let bevel = 0.7 + 0.3 * smoothstep(0.02, 0.14, face_edge);
    let normal = normalize(pos - vec3<f32>(1.5));

    result.hit = true;
    result.t = t_param * inv_norm;
    result.color = palette[block_type].rgb * bevel;
    result.normal = normal;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = pos - vec3<f32>(0.5);
    result.cell_size = 1.0;
    return result;
}

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

    let ray_len = length(ray_dir_in);
    if ray_len <= 1e-8 { return result; }
    let ray_dir = ray_dir_in / ray_len;
    let inv_norm = 1.0 / ray_len;

    let center = body_origin + vec3<f32>(body_size * 0.5);
    let pi = 3.14159265;
    let lon_step = 2.0 * pi / f32(dims_x);
    let radius = body_size * 0.38;
    let shell = max(radius * lon_step, body_size / 27.0);
    let r_lo = radius - shell * 0.5;
    let r_hi = radius + shell * 0.5;
    let lat_lo = -lon_step * 0.5;
    let lat_hi = lon_step * 0.5;
    let oc = ray_origin - center;

    var best = result;
    var best_t = 1e20;
    for (var cell_x: i32 = 0; cell_x < dims_x; cell_x = cell_x + 1) {
        let sample = sample_slab_cell(ring_idx, slab_depth, cell_x, 0, 0);
        if sample.block_type == 0xFFFEu {
            continue;
        }

        let lon_lo = -pi + f32(cell_x) * lon_step;
        let lon_hi = lon_lo + lon_step;
        var candidates: array<f32, 6>;
        candidates[0] = ray_sphere_after(ray_origin, ray_dir, center, r_hi, 0.0);
        candidates[1] = ray_sphere_after(ray_origin, ray_dir, center, r_lo, 0.0);
        candidates[2] = ray_meridian_t(oc, ray_dir, lon_lo, 0.0);
        candidates[3] = ray_meridian_t(oc, ray_dir, lon_hi, 0.0);
        candidates[4] = ray_parallel_t(oc, ray_dir, lat_lo, 0.0);
        candidates[5] = ray_parallel_t(oc, ray_dir, lat_hi, 0.0);

        for (var i: u32 = 0u; i < 6u; i = i + 1u) {
            let t = candidates[i];
            if t <= 0.0 || t >= best_t {
                continue;
            }
            let pos = ray_origin + ray_dir * t;
            let off = pos - center;
            let r = length(off);
            if r <= 1e-6 {
                continue;
            }
            let n = off / r;
            let lat = asin(clamp(n.y, -1.0, 1.0));
            let lon = atan2(n.z, n.x);
            let eps = 1e-4;
            if lon < lon_lo - eps || lon > lon_hi + eps {
                continue;
            }
            if lat < lat_lo - eps || lat > lat_hi + eps {
                continue;
            }
            if r < r_lo - eps || r > r_hi + eps {
                continue;
            }
            best_t = t;
            best = make_uv_ring_hit(
                pos, t, inv_norm, sample.block_type,
                r, lat, lon,
                lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
            );
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
