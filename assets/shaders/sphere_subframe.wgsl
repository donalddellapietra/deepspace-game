// Sphere sub-frame DDA — sphere render scoped to a sub-frame inside
// a `WrappedPlane` subtree. The active frame's "node" is a
// Cartesian Node (a sub-cell of the WP at some depth past
// slab_depth); the camera is projected into the sub-frame's local
// rotated+translated coords by the renderer
// (`world::sphere_geom::camera_in_sphere_subframe`), so the ray
// arrives here with bounded magnitudes. Sphere math then operates
// at precision proportional to the sub-frame extent — layer-
// agnostic.
//
// Frame conventions (set by the renderer per-frame in uniforms):
//   * Origin    = sub-frame center.
//   * +x axis   = lon-tangent at sub-frame center.
//   * +y axis   = lat-tangent at sub-frame center.
//   * +z axis   = radial direction (out from sphere center).
//   * Sphere center is at `(0, 0, -r_c)` with `r_c =
//     uniforms.subframe_r.z`; sphere radius is the WP's intrinsic
//     `body_size / (2π)` (body_size = 3 in standard architecture).
//   * Sub-frame's absolute (lat, lon, r) range =
//     `uniforms.subframe_lat_lon` + `uniforms.subframe_r.xy`.
//
// The DDA mirrors `sphere_descend_anchor`'s structure: stack-based
// descent over 27-children grids, each level a 1/3 refinement of
// the parent's (lat, lon, r) range. Boundaries use the sub-frame
// versions of meridian/parallel/sphere helpers
// (`ray_meridian_subframe_t` etc.) which agree with the body-
// rooted DDA's partition (= absolute (lat, lon, r) values).

fn sphere_uv_in_subframe(
    sub_node_idx: u32,
    ray_origin: vec3<f32>, ray_dir_in: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 1.0;

    let ray_dir = normalize(ray_dir_in);
    let inv_norm = 1.0 / max(length(ray_dir_in), 1e-6);

    // Sub-frame geometry from uniforms.
    let lat_lo = uniforms.subframe_lat_lon.x;
    let lat_hi = uniforms.subframe_lat_lon.y;
    let lon_lo = uniforms.subframe_lat_lon.z;
    let lon_hi = uniforms.subframe_lat_lon.w;
    let r_lo   = uniforms.subframe_r.x;
    let r_hi   = uniforms.subframe_r.y;
    let r_c    = uniforms.subframe_r.z;
    let lat_c  = (lat_lo + lat_hi) * 0.5;
    let lon_c  = (lon_lo + lon_hi) * 0.5;

    // Sphere center in sub-frame local coords: along -z by r_c.
    let cs_center = vec3<f32>(0.0, 0.0, -r_c);
    // Body sphere radius = body_size / (2π). body_size = 3.
    let r_sphere = 3.0 / (2.0 * 3.14159265);

    // Ray-sphere intersect (against the OUTER sphere shell at r =
    // r_sphere; same as the body's outer surface).
    let oc = ray_origin - cs_center;
    let b = dot(oc, ray_dir);
    let c = dot(oc, oc) - r_sphere * r_sphere;
    let disc = b * b - c;
    if disc <= 0.0 { return result; }
    let sq = sqrt(disc);
    let t_enter_sphere = max(-b - sq, 0.0);
    let t_exit_sphere = -b + sq;
    if t_exit_sphere <= 0.0 { return result; }

    // Stack-based DDA over the sub-frame's children grid. Each level
    // is a 27-children Cartesian split: cell coord (cx, cy, cz) in
    // [0, 3)^3 picks one of the parent cell's 27 sub-cells, with
    // axis convention slot.x → lon, slot.y → r, slot.z → lat (matches
    // sphere_dda.wgsl + cpu_raycast_sphere_uv).
    //
    // Stack state per depth d:
    //   s_node_idx[d]  : tree node we're picking children from.
    //   s_cell[d]      : current child coord in [0, 3)^3.
    //   s_lat_lo[d], s_lon_lo[d], s_r_lo[d] : parent cell's
    //                    absolute (lat, lon, r) lower bounds.
    // Steps per axis derived from depth (parent_step / 3 per descent).
    var s_node_idx: array<u32, MAX_STACK_DEPTH>;
    var s_cell: array<u32, MAX_STACK_DEPTH>;
    var s_lat_lo: array<f32, MAX_STACK_DEPTH>;
    var s_lon_lo: array<f32, MAX_STACK_DEPTH>;
    var s_r_lo: array<f32, MAX_STACK_DEPTH>;

    s_node_idx[0] = sub_node_idx;
    s_lat_lo[0] = lat_lo;
    s_lon_lo[0] = lon_lo;
    s_r_lo[0] = r_lo;
    var cur_lat_step = (lat_hi - lat_lo) / 3.0;
    var cur_lon_step = (lon_hi - lon_lo) / 3.0;
    var cur_r_step   = (r_hi   - r_lo)   / 3.0;

    var t = t_enter_sphere + 1e-7;

    // Initial cell from the ray's t. Compute (lat_p, lon_p, r) by
    // projecting the in-sub-frame ray pos through the sphere-center-
    // shifted basis. Bounded magnitudes → ULP scales with sub-frame
    // extent.
    {
        let pos0 = ray_origin + ray_dir * t;
        let v = pos0 - cs_center;  // = pos0 + (0, 0, r_c) since cs = (0, 0, -r_c)
        let r0 = max(length(v), 1e-9);
        // World-Y direction in sub-frame local = (0, cl, sl) (derived
        // in sphere_geom.rs basis). World lat = asin(world Y / r),
        // world lon = atan2(world Z, world X). In sub-frame local:
        //   world Y  = cos(lat_c) · v.y + sin(lat_c) · v.z
        //   world X  = cos(lon_c) · radial_dot - sin(lon_c) · ?? — too
        //              messy to derive in closed form. Use the
        //              direct projection: world basis vectors ARE
        //              the sub-frame basis applied IN REVERSE. For
        //              cell pick, lat_p and lon_p both use world
        //              axes, so we need the v components in WORLD
        //              basis, not sub-frame basis.
        let cl = cos(lat_c); let sl = sin(lat_c);
        let co = cos(lon_c); let so = sin(lon_c);
        // R^T (sub-frame to world) columns = (lon_tan, lat_tan, radial).
        // v_world = lon_tan · v.x + lat_tan · v.y + radial · v.z.
        let vw_x = -so * v.x - sl * co * v.y + cl * co * v.z;
        let vw_y =                 cl * v.y + sl      * v.z;
        let vw_z =  co * v.x - sl * so * v.y + cl * so * v.z;
        let lat_p = asin(clamp(vw_y / r0, -1.0, 1.0));
        let lon_p = atan2(vw_z, vw_x);
        if abs(lat_p) > uniforms.planet_render.y { return result; }
        let cx0 = clamp(i32(floor((lon_p - s_lon_lo[0]) / cur_lon_step)), 0, 2);
        let cy0 = clamp(i32(floor((r0    - s_r_lo[0])   / cur_r_step)),   0, 2);
        let cz0 = clamp(i32(floor((lat_p - s_lat_lo[0]) / cur_lat_step)), 0, 2);
        s_cell[0] = pack_cell(vec3<i32>(cx0, cy0, cz0));
    }

    var depth: u32 = 0u;
    var iters: u32 = 0u;
    let pop_eps = max(cur_r_step * 1e-7, 1e-9);
    loop {
        if iters > 1024u { break; }
        iters = iters + 1u;

        let cell = unpack_cell(s_cell[depth]);
        // Cell bounds (absolute lat, lon, r) at this depth.
        let cell_lat_lo = s_lat_lo[depth] + f32(cell.z) * cur_lat_step;
        let cell_lat_hi = cell_lat_lo + cur_lat_step;
        let cell_lon_lo = s_lon_lo[depth] + f32(cell.x) * cur_lon_step;
        let cell_lon_hi = cell_lon_lo + cur_lon_step;
        let cell_r_lo   = s_r_lo[depth]   + f32(cell.y) * cur_r_step;
        let cell_r_hi   = cell_r_lo + cur_r_step;

        // Sample tree at this cell.
        let slot = u32(cell.x + cell.y * 3 + cell.z * 9);
        let header_off = node_offsets[s_node_idx[depth]];
        let occ = tree[header_off];
        let bit = 1u << slot;
        var bt: u32 = 0xFFFEu;
        var nu_child: u32 = 0u;
        if (occ & bit) != 0u {
            let first_child = tree[header_off + 1u];
            let rank = countOneBits(occ & (bit - 1u));
            let child_base = first_child + rank * 2u;
            let packed = tree[child_base];
            let tag = packed & 0xFFu;
            bt = (packed >> 8u) & 0xFFFFu;
            if tag == 2u && bt != 0xFFFEu {
                nu_child = tree[child_base + 1u];
            }
        }

        // t to 6 boundaries (in sub-frame local) + winning axis.
        var t_next = t_exit_sphere + 1.0;
        var winning: u32 = 6u;
        let t_lon_lo = ray_meridian_subframe_t(oc, ray_dir, cell_lon_lo, lat_c, lon_c, r_c, t);
        if t_lon_lo > 0.0 && t_lon_lo < t_next { t_next = t_lon_lo; winning = 0u; }
        let t_lon_hi = ray_meridian_subframe_t(oc, ray_dir, cell_lon_hi, lat_c, lon_c, r_c, t);
        if t_lon_hi > 0.0 && t_lon_hi < t_next { t_next = t_lon_hi; winning = 1u; }
        let t_lat_lo = ray_parallel_subframe_t(oc, ray_dir, cell_lat_lo, lat_c, r_c, t);
        if t_lat_lo > 0.0 && t_lat_lo < t_next { t_next = t_lat_lo; winning = 2u; }
        let t_lat_hi = ray_parallel_subframe_t(oc, ray_dir, cell_lat_hi, lat_c, r_c, t);
        if t_lat_hi > 0.0 && t_lat_hi < t_next { t_next = t_lat_hi; winning = 3u; }
        let t_r_lo = ray_radius_subframe_t(oc, ray_dir, cell_r_lo, r_c, t);
        if t_r_lo > 0.0 && t_r_lo < t_next { t_next = t_r_lo; winning = 4u; }
        let t_r_hi = ray_radius_subframe_t(oc, ray_dir, cell_r_hi, r_c, t);
        if t_r_hi > 0.0 && t_r_hi < t_next { t_next = t_r_hi; winning = 5u; }

        if bt != 0xFFFEu {
            // Hit. Build (pos, lat, lon, r) for the bevel call.
            let pos = ray_origin + ray_dir * t;
            let v = pos - cs_center;
            let r_h = max(length(v), 1e-9);
            let cl = cos(lat_c); let sl = sin(lat_c);
            let co = cos(lon_c); let so = sin(lon_c);
            let vw_x = -so * v.x - sl * co * v.y + cl * co * v.z;
            let vw_y =                 cl * v.y + sl      * v.z;
            let vw_z =  co * v.x - sl * so * v.y + cl * so * v.z;
            let lat_h = asin(clamp(vw_y / r_h, -1.0, 1.0));
            let lon_h = atan2(vw_z, vw_x);
            // Outward normal in sub-frame local = sub-frame radial
            // direction at hit (= unit vec from sphere center to hit
            // point, projected back into sub-frame basis = v / r_h).
            let n_step = v / r_h;
            // Render (push if non-uniform; cap at MAX_STACK_DEPTH).
            if nu_child == 0u || depth + 1u >= MAX_STACK_DEPTH {
                return make_sphere_hit(
                    pos, n_step, t, inv_norm, bt,
                    r_h, lat_h, lon_h,
                    cell_lon_lo, cell_lon_hi,
                    cell_lat_lo, cell_lat_hi,
                    cell_r_lo,   cell_r_hi,
                    cur_lon_step, cur_lat_step, cur_r_step,
                );
            }

            // Push to descend. Initial sub-cell from the same hit
            // pos's (lat, lon, r) — bounded relative to this cell's
            // range, so the 1-of-3 floor pick is precision-stable.
            let nd = depth + 1u;
            s_node_idx[nd] = nu_child;
            s_lat_lo[nd] = cell_lat_lo;
            s_lon_lo[nd] = cell_lon_lo;
            s_r_lo[nd]   = cell_r_lo;
            cur_lat_step = cur_lat_step / 3.0;
            cur_lon_step = cur_lon_step / 3.0;
            cur_r_step   = cur_r_step   / 3.0;
            let cx2 = clamp(i32(floor((lon_h - s_lon_lo[nd]) / cur_lon_step)), 0, 2);
            let cy2 = clamp(i32(floor((r_h   - s_r_lo[nd])   / cur_r_step)),   0, 2);
            let cz2 = clamp(i32(floor((lat_h - s_lat_lo[nd]) / cur_lat_step)), 0, 2);
            s_cell[nd] = pack_cell(vec3<i32>(cx2, cy2, cz2));
            depth = nd;
            continue;
        }

        // Empty cell — advance ±1 on winning axis, cascade pop on
        // OOB (mirror of sphere_descend_anchor's cascading-pop).
        if winning == 6u { break; }  // tangent ray
        var dx: i32 = 0;
        var dy: i32 = 0;
        var dz: i32 = 0;
        switch winning {
            case 0u: { dx = -1; }
            case 1u: { dx =  1; }
            case 2u: { dz = -1; }
            case 3u: { dz =  1; }
            case 4u: { dy = -1; }
            case 5u: { dy =  1; }
            default: {}
        }
        t = t_next + pop_eps;

        var nx = cell.x + dx;
        var ny = cell.y + dy;
        var nz = cell.z + dz;
        loop {
            if nx >= 0 && nx <= 2 && ny >= 0 && ny <= 2 && nz >= 0 && nz <= 2 {
                s_cell[depth] = pack_cell(vec3<i32>(nx, ny, nz));
                break;
            }
            if depth == 0u { return result; }
            depth = depth - 1u;
            cur_lat_step = cur_lat_step * 3.0;
            cur_lon_step = cur_lon_step * 3.0;
            cur_r_step   = cur_r_step   * 3.0;
            let pcell = unpack_cell(s_cell[depth]);
            nx = pcell.x + dx;
            ny = pcell.y + dy;
            nz = pcell.z + dz;
        }
    }

    return result;
}
