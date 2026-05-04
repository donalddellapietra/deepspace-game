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

fn uv_ring_point_in_cell(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    t: f32,
    lon_lo: f32,
    lon_hi: f32,
    lat_lo: f32,
    lat_hi: f32,
    r_lo: f32,
    r_hi: f32,
) -> bool {
    let pos = ray_origin + ray_dir * t;
    let off = pos - center;
    let r = length(off);
    if r <= 1e-6 {
        return false;
    }
    let n = off / r;
    let lon = atan2(n.z, n.x);
    let lat = asin(clamp(n.y, -1.0, 1.0));
    let eps = 1e-4;
    return lon >= lon_lo - eps && lon <= lon_hi + eps
        && lat >= lat_lo - eps && lat <= lat_hi + eps
        && r >= r_lo - eps && r <= r_hi + eps;
}

fn uv_ring_cell_boundary_t(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    oc: vec3<f32>,
    after: f32,
    lon_lo: f32,
    lon_hi: f32,
    lat_lo: f32,
    lat_hi: f32,
    r_lo: f32,
    r_hi: f32,
) -> f32 {
    var best = 1e20;
    var candidates: array<f32, 6>;
    candidates[0] = ray_sphere_after(ray_origin, ray_dir, center, r_hi, after);
    candidates[1] = ray_sphere_after(ray_origin, ray_dir, center, r_lo, after);
    candidates[2] = ray_meridian_t(oc, ray_dir, lon_lo, after);
    candidates[3] = ray_meridian_t(oc, ray_dir, lon_hi, after);
    candidates[4] = ray_parallel_t(oc, ray_dir, lat_lo, after);
    candidates[5] = ray_parallel_t(oc, ray_dir, lat_hi, after);
    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        let t = candidates[i];
        if t > after && t < best && uv_ring_point_in_cell(
            ray_origin, ray_dir, center, t,
            lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
        ) {
            best = t;
        }
    }
    return best;
}

fn uv_ring_descend_anchor(
    anchor_idx: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    oc: vec3<f32>,
    center: vec3<f32>,
    inv_norm: f32,
    t_in: f32,
    t_exit: f32,
    slab_lon_lo: f32,
    slab_lon_step: f32,
    slab_lat_lo: f32,
    slab_lat_step: f32,
    slab_r_lo: f32,
    slab_r_step: f32,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    var s_node_idx: array<u32, TANGENT_STACK_DEPTH>;
    var s_cell: array<u32, TANGENT_STACK_DEPTH>;
    var depth: u32 = 0u;
    s_node_idx[0] = anchor_idx;

    var cur_lon_org = slab_lon_lo;
    var cur_lat_org = slab_lat_lo;
    var cur_r_org = slab_r_lo;
    var cur_lon_step = slab_lon_step / 3.0;
    var cur_lat_step = slab_lat_step / 3.0;
    var cur_r_step = slab_r_step / 3.0;
    var t = t_in;

    {
        let pos0 = ray_origin + ray_dir * t;
        let off0 = pos0 - center;
        let r0 = max(length(off0), 1e-9);
        let n0 = off0 / r0;
        let lon0 = atan2(n0.z, n0.x);
        let lat0 = asin(clamp(n0.y, -1.0, 1.0));
        s_cell[0] = pack_cell(vec3<i32>(
            clamp(i32(floor((lon0 - cur_lon_org) / cur_lon_step)), 0, 2),
            clamp(i32(floor((r0 - cur_r_org) / cur_r_step)), 0, 2),
            clamp(i32(floor((lat0 - cur_lat_org) / cur_lat_step)), 0, 2),
        ));
    }

    var iters: u32 = 0u;
    loop {
        if iters >= 2048u { break; }
        iters = iters + 1u;
        if t > t_exit { break; }

        let cell = unpack_cell(s_cell[depth]);
        if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
            if depth == 0u { break; }
            depth = depth - 1u;
            cur_lon_step = cur_lon_step * 3.0;
            cur_lat_step = cur_lat_step * 3.0;
            cur_r_step = cur_r_step * 3.0;
            let popped = unpack_cell(s_cell[depth]);
            cur_lon_org = cur_lon_org - f32(popped.x) * cur_lon_step;
            cur_r_org = cur_r_org - f32(popped.y) * cur_r_step;
            cur_lat_org = cur_lat_org - f32(popped.z) * cur_lat_step;

            let pos_p = ray_origin + ray_dir * t;
            let off_p = pos_p - center;
            let r_p = max(length(off_p), 1e-9);
            let n_p = off_p / r_p;
            let lon_p = atan2(n_p.z, n_p.x);
            let lat_p = asin(clamp(n_p.y, -1.0, 1.0));
            s_cell[depth] = pack_cell(vec3<i32>(
                i32(floor((lon_p - cur_lon_org) / cur_lon_step)),
                i32(floor((r_p - cur_r_org) / cur_r_step)),
                i32(floor((lat_p - cur_lat_org) / cur_lat_step)),
            ));
            continue;
        }

        let lon_lo = cur_lon_org + f32(cell.x) * cur_lon_step;
        let lon_hi = lon_lo + cur_lon_step;
        let r_lo = cur_r_org + f32(cell.y) * cur_r_step;
        let r_hi = r_lo + cur_r_step;
        let lat_lo = cur_lat_org + f32(cell.z) * cur_lat_step;
        let lat_hi = lat_lo + cur_lat_step;

        var t_next = t_exit + 1.0;
        let t_lon_lo = ray_meridian_t(oc, ray_dir, lon_lo, t);
        if t_lon_lo > t && t_lon_lo < t_next { t_next = t_lon_lo; }
        let t_lon_hi = ray_meridian_t(oc, ray_dir, lon_hi, t);
        if t_lon_hi > t && t_lon_hi < t_next { t_next = t_lon_hi; }
        let t_lat_lo = ray_parallel_t(oc, ray_dir, lat_lo, t);
        if t_lat_lo > t && t_lat_lo < t_next { t_next = t_lat_lo; }
        let t_lat_hi = ray_parallel_t(oc, ray_dir, lat_hi, t);
        if t_lat_hi > t && t_lat_hi < t_next { t_next = t_lat_hi; }
        let t_r_lo = ray_sphere_after(ray_origin, ray_dir, center, r_lo, t);
        if t_r_lo > t && t_r_lo < t_next { t_next = t_r_lo; }
        let t_r_hi = ray_sphere_after(ray_origin, ray_dir, center, r_hi, t);
        if t_r_hi > t && t_r_hi < t_next { t_next = t_r_hi; }

        let slot = u32(cell.x + cell.y * 3 + cell.z * 9);
        let header_off = node_offsets[s_node_idx[depth]];
        let occ = tree[header_off];
        let bit = 1u << slot;
        if (occ & bit) == 0u {
            t = t_next + max(cur_r_step * 1e-4, 1e-6);
            if t > t_exit { break; }
            let pos_a = ray_origin + ray_dir * t;
            let off_a = pos_a - center;
            let r_a = max(length(off_a), 1e-9);
            let n_a = off_a / r_a;
            let lon_a = atan2(n_a.z, n_a.x);
            let lat_a = asin(clamp(n_a.y, -1.0, 1.0));
            s_cell[depth] = pack_cell(vec3<i32>(
                i32(floor((lon_a - cur_lon_org) / cur_lon_step)),
                i32(floor((r_a - cur_r_org) / cur_r_step)),
                i32(floor((lat_a - cur_lat_org) / cur_lat_step)),
            ));
            continue;
        }

        let first_child = tree[header_off + 1u];
        let rank = countOneBits(occ & (bit - 1u));
        let child_base = first_child + rank * 2u;
        let packed = tree[child_base];
        let tag = packed & 0xFFu;
        let block_type = (packed >> 8u) & 0xFFFFu;

        if tag == 1u {
            let pos_h = ray_origin + ray_dir * t;
            let off_h = pos_h - center;
            let r_h = max(length(off_h), 1e-9);
            let n_h = off_h / r_h;
            let lon_h = atan2(n_h.z, n_h.x);
            let lat_h = asin(clamp(n_h.y, -1.0, 1.0));
            return make_uv_ring_hit(
                pos_h, t, inv_norm, block_type,
                r_h, lat_h, lon_h,
                lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
            );
        }

        if tag != 2u || block_type == 0xFFFEu {
            t = t_next + max(cur_r_step * 1e-4, 1e-6);
            if t > t_exit { break; }
            let pos_e = ray_origin + ray_dir * t;
            let off_e = pos_e - center;
            let r_e = max(length(off_e), 1e-9);
            let n_e = off_e / r_e;
            let lon_e = atan2(n_e.z, n_e.x);
            let lat_e = asin(clamp(n_e.y, -1.0, 1.0));
            s_cell[depth] = pack_cell(vec3<i32>(
                i32(floor((lon_e - cur_lon_org) / cur_lon_step)),
                i32(floor((r_e - cur_r_org) / cur_r_step)),
                i32(floor((lat_e - cur_lat_org) / cur_lat_step)),
            ));
            continue;
        }

        if depth + 1u >= TANGENT_STACK_DEPTH {
            let pos_h = ray_origin + ray_dir * t;
            let off_h = pos_h - center;
            let r_h = max(length(off_h), 1e-9);
            let n_h = off_h / r_h;
            let lon_h = atan2(n_h.z, n_h.x);
            let lat_h = asin(clamp(n_h.y, -1.0, 1.0));
            return make_uv_ring_hit(
                pos_h, t, inv_norm, block_type,
                r_h, lat_h, lon_h,
                lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
            );
        }

        let child_idx = tree[child_base + 1u];
        depth = depth + 1u;
        s_node_idx[depth] = child_idx;
        cur_lon_org = lon_lo;
        cur_lat_org = lat_lo;
        cur_r_org = r_lo;
        cur_lon_step = cur_lon_step / 3.0;
        cur_lat_step = cur_lat_step / 3.0;
        cur_r_step = cur_r_step / 3.0;

        let pos_c = ray_origin + ray_dir * t;
        let off_c = pos_c - center;
        let r_c = max(length(off_c), 1e-9);
        let n_c = off_c / r_c;
        let lon_c = atan2(n_c.z, n_c.x);
        let lat_c = asin(clamp(n_c.y, -1.0, 1.0));
        s_cell[depth] = pack_cell(vec3<i32>(
            clamp(i32(floor((lon_c - cur_lon_org) / cur_lon_step)), 0, 2),
            clamp(i32(floor((r_c - cur_r_org) / cur_r_step)), 0, 2),
            clamp(i32(floor((lat_c - cur_lat_org) / cur_lat_step)), 0, 2),
        ));
    }

    return result;
}

struct UvRingLocalCellRay {
    origin: vec3<f32>,
    dir: vec3<f32>,
    inv_dir: vec3<f32>,
}

fn uv_ring_local_cell_ray(
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
    center: vec3<f32>,
    lon: f32,
    radius: f32,
    shell: f32,
) -> UvRingLocalCellRay {
    let sa = sin(lon);
    let ca = cos(lon);
    let radial = vec3<f32>(ca, 0.0, sa);
    let tangent = vec3<f32>(-sa, 0.0, ca);
    let up = vec3<f32>(0.0, 1.0, 0.0);
    let cube_origin = center + radial * radius;
    let scale = 3.0 / shell;
    let d_origin = ray_origin - cube_origin;
    var out: UvRingLocalCellRay;
    out.origin = vec3<f32>(
        dot(tangent, d_origin) * scale + 1.5,
        dot(radial, d_origin) * scale + 1.5,
        dot(up, d_origin) * scale + 1.5,
    );
    out.dir = vec3<f32>(
        dot(tangent, ray_dir) * scale,
        dot(radial, ray_dir) * scale,
        dot(up, ray_dir) * scale,
    );
    out.inv_dir = vec3<f32>(
        select(1e10, 1.0 / out.dir.x, abs(out.dir.x) > 1e-8),
        select(1e10, 1.0 / out.dir.y, abs(out.dir.y) > 1e-8),
        select(1e10, 1.0 / out.dir.z, abs(out.dir.z) > 1e-8),
    );
    return out;
}

fn uv_ring_content_aabb_hit(local: UvRingLocalCellRay, child_idx: u32) -> bool {
    let aabb_bits = aabbs[child_idx] & 0xFFFu;
    if aabb_bits == 0u {
        return false;
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
    let hit = ray_box(local.origin, local.inv_dir, amin, amax);
    return hit.t_enter < hit.t_exit && hit.t_exit > 0.0;
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
        let lon_center = -pi + (f32(cell_x) + 0.5) * lon_step;
        let local = uv_ring_local_cell_ray(ray_origin, ray_dir, center, lon_center, radius, shell);
        let broad = ray_box(
            local.origin,
            local.inv_dir,
            vec3<f32>(-0.25),
            vec3<f32>(3.25),
        );
        if broad.t_enter >= broad.t_exit || broad.t_exit <= 0.0 || broad.t_enter >= best_t {
            continue;
        }

        let sample = sample_slab_cell(ring_idx, slab_depth, cell_x, 0, 0);
        if sample.block_type == 0xFFFEu {
            continue;
        }

        let lon_lo = -pi + f32(cell_x) * lon_step;
        let lon_hi = lon_lo + lon_step;

        var t_in = 0.0;
        if !uv_ring_point_in_cell(
            ray_origin, ray_dir, center, 0.0,
            lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
        ) {
            t_in = uv_ring_cell_boundary_t(
                ray_origin, ray_dir, center, oc, 0.0,
                lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
            );
        }
        if t_in >= 1e19 || t_in >= best_t {
            continue;
        }
        let t_exit = uv_ring_cell_boundary_t(
            ray_origin, ray_dir, center, oc, t_in + 1e-4,
            lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
        );
        if t_exit <= t_in {
            continue;
        }

        if sample.tag == 2u {
            if !uv_ring_content_aabb_hit(local, sample.child_idx) {
                if ENABLE_STATS { ray_steps_would_cull = ray_steps_would_cull + 1u; }
                continue;
            }
            let sub = uv_ring_descend_anchor(
                sample.child_idx,
                ray_origin,
                ray_dir,
                oc,
                center,
                inv_norm,
                t_in + 1e-4,
                t_exit,
                lon_lo,
                lon_step,
                lat_lo,
                lat_hi - lat_lo,
                r_lo,
                r_hi - r_lo,
            );
            let sub_t_norm = sub.t / inv_norm;
            if sub.hit && sub_t_norm < best_t {
                best_t = sub_t_norm;
                best = sub;
            }
        } else if sample.tag == 1u {
            if t_in < best_t {
                let pos = ray_origin + ray_dir * t_in;
                let off = pos - center;
                let r = max(length(off), 1e-6);
                let n = off / r;
                let lat = asin(clamp(n.y, -1.0, 1.0));
                let lon = atan2(n.z, n.x);
                best_t = t_in;
                best = make_uv_ring_hit(
                    pos, t_in, inv_norm, sample.block_type,
                    r, lat, lon,
                    lon_lo, lon_hi, lat_lo, lat_hi, r_lo, r_hi,
                );
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
