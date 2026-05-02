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
//     `body_size / (2π)` (body_size = 2 in standard architecture).
//   * Sub-frame's absolute (lat, lon, r) range =
//     `uniforms.subframe_lat_lon` + `uniforms.subframe_r.xy`.
//
// The DDA mirrors `sphere_descend_anchor`'s structure: stack-based
// descent over 8-children grids, each level a 1/2 refinement of
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

    // Sub-frame basis (camera projection + ray boundary helpers).
    // Drives the basis (lat_c, lon_c, r_c) used to interpret the
    // sub-frame local ray. Precision-stable when rays interact with
    // cells near these center values.
    let sub_lat_lo = uniforms.subframe_lat_lon.x;
    let sub_lat_hi = uniforms.subframe_lat_lon.y;
    let sub_lon_lo = uniforms.subframe_lat_lon.z;
    let sub_lon_hi = uniforms.subframe_lat_lon.w;
    let r_c    = uniforms.subframe_r.z;
    let lat_c  = (sub_lat_lo + sub_lat_hi) * 0.5;
    let lon_c  = (sub_lon_lo + sub_lon_hi) * 0.5;
    // Node range — what the dispatched GPU node literally partitions
    // into 8 children. May be wider than the sub-frame range when
    // the GPU tree is shallower than the camera's logical depth
    // (e.g. above the slab — node = WP, sub-frame = a thin patch).
    let lat_lo = uniforms.node_lat_lon.x;
    let lat_hi = uniforms.node_lat_lon.y;
    let lon_lo = uniforms.node_lat_lon.z;
    let lon_hi = uniforms.node_lat_lon.w;
    let r_lo   = uniforms.node_r.x;
    let r_hi   = uniforms.node_r.y;

    // Sphere center in sub-frame local coords: along -z by r_c.
    let cs_center = vec3<f32>(0.0, 0.0, -r_c);
    // Body sphere radius = body_size / (2π). body_size = 2.
    let r_sphere = 2.0 / (2.0 * 3.14159265);

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
    // is a 8-children Cartesian split: cell coord (cx, cy, cz) in
    // [0, 2)^3 picks one of the parent cell's 8 sub-cells, with
    // axis convention slot.x → lon, slot.y → r, slot.z → lat (matches
    // sphere_dda.wgsl + cpu_raycast_sphere_uv).
    //
    // Stack state per depth d:
    //   s_node_idx[d]  : tree node we're picking children from.
    //   s_cell[d]      : current child coord in [0, 2)^3.
    //   s_lat_lo[d], s_lon_lo[d], s_r_lo[d] : parent cell's
    //                    absolute (lat, lon, r) lower bounds.
    // Steps per axis derived from depth (parent_step / 2 per descent).
    var s_node_idx: array<u32, MAX_STACK_DEPTH>;
    var s_cell: array<u32, MAX_STACK_DEPTH>;
    var s_lat_lo: array<f32, MAX_STACK_DEPTH>;
    var s_lon_lo: array<f32, MAX_STACK_DEPTH>;
    var s_r_lo: array<f32, MAX_STACK_DEPTH>;

    s_node_idx[0] = sub_node_idx;
    s_lat_lo[0] = lat_lo;
    s_lon_lo[0] = lon_lo;
    s_r_lo[0] = r_lo;
    var cur_lat_step = (lat_hi - lat_lo) / 2.0;
    var cur_lon_step = (lon_hi - lon_lo) / 2.0;
    var cur_r_step   = (r_hi   - r_lo)   / 2.0;

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
        let cx0 = clamp(i32(floor((lon_p - s_lon_lo[0]) / cur_lon_step)), 0, 1);
        let cy0 = clamp(i32(floor((r0    - s_r_lo[0])   / cur_r_step)),   0, 1);
        let cz0 = clamp(i32(floor((lat_p - s_lat_lo[0]) / cur_lat_step)), 0, 1);
        s_cell[0] = pack_cell(vec3<i32>(cx0, cy0, cz0));
    }

    var depth: u32 = 0u;
    var iters: u32 = 0u;
    // Boundary-advance epsilon. Mirrors `sphere_dda.wgsl`'s proven
    // value: `cur_r_step * 1e-7` was 3 orders of magnitude below
    // the f32 ULP at ray t≈1, so `t = t_next + eps` did not
    // actually advance the ray past the boundary — the next
    // iteration re-classified the same cell as empty and the loop
    // bailed on the iter cap, returning sky on patches that should
    // be filled.
    let pop_eps = max(cur_r_step * 1e-4, 1e-6);
    // Pre-compute sub-frame → world basis. Reused on every cascade
    // pop / empty-cell advance to recover (lat, lon, r) from the
    // ray's current position; mirrors the same derivation used at
    // the initial-cell pick (above) and the hit-build (below).
    let cl_b = cos(lat_c); let sl_b = sin(lat_c);
    let co_b = cos(lon_c); let so_b = sin(lon_c);
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
        let slot = u32(cell.x + cell.y * 2 + cell.z * 4);
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
            // range, so the 1-of-2 floor pick is precision-stable.
            let nd = depth + 1u;
            s_node_idx[nd] = nu_child;
            s_lat_lo[nd] = cell_lat_lo;
            s_lon_lo[nd] = cell_lon_lo;
            s_r_lo[nd]   = cell_r_lo;
            cur_lat_step = cur_lat_step / 2.0;
            cur_lon_step = cur_lon_step / 2.0;
            cur_r_step   = cur_r_step   / 2.0;
            let cx2 = clamp(i32(floor((lon_h - s_lon_lo[nd]) / cur_lon_step)), 0, 1);
            let cy2 = clamp(i32(floor((r_h   - s_r_lo[nd])   / cur_r_step)),   0, 1);
            let cz2 = clamp(i32(floor((lat_h - s_lat_lo[nd]) / cur_lat_step)), 0, 1);
            s_cell[nd] = pack_cell(vec3<i32>(cx2, cy2, cz2));
            depth = nd;
            continue;
        }

        // Empty cell — advance ray past the winning boundary, then
        // recompute (lat, lon, r) from the ray's NEW position to
        // pick the next cell. Cascade up on OOB. Mirrors
        // `sphere_dda.wgsl::sphere_descend_anchor`'s recipe — the
        // prior implementation incremented `cell + dx` blindly and
        // then re-applied `dx` at the parent level on cascade pops,
        // which overshoots by O(parent_step) and lands the ray in
        // the wrong sibling cell. That produced diagonal seams
        // (when wrong-sibling crossings stack on multiple axes) and
        // wrong-material splats (sub-cell content sampled with the
        // wrong block_type) at deep zoom.
        if winning == 6u { break; }  // tangent ray
        t = t_next + pop_eps;
        let pos_a = ray_origin + ray_dir * t;
        let v_a = pos_a - cs_center;
        let r_a = max(length(v_a), 1e-9);
        let vw_x_a = -so_b * v_a.x - sl_b * co_b * v_a.y + cl_b * co_b * v_a.z;
        let vw_y_a =                 cl_b * v_a.y + sl_b      * v_a.z;
        let vw_z_a =  co_b * v_a.x - sl_b * so_b * v_a.y + cl_b * so_b * v_a.z;
        let lat_a = asin(clamp(vw_y_a / r_a, -1.0, 1.0));
        let lon_a = atan2(vw_z_a, vw_x_a);
        loop {
            let cx_a = i32(floor((lon_a - s_lon_lo[depth]) / cur_lon_step));
            let cy_a = i32(floor((r_a   - s_r_lo[depth])   / cur_r_step));
            let cz_a = i32(floor((lat_a - s_lat_lo[depth]) / cur_lat_step));
            if cx_a >= 0 && cx_a <= 1 && cy_a >= 0 && cy_a <= 1 && cz_a >= 0 && cz_a <= 1 {
                s_cell[depth] = pack_cell(vec3<i32>(cx_a, cy_a, cz_a));
                break;
            }
            if depth == 0u { return result; }
            depth = depth - 1u;
            cur_lat_step = cur_lat_step * 2.0;
            cur_lon_step = cur_lon_step * 2.0;
            cur_r_step   = cur_r_step   * 2.0;
        }
    }

    return result;
}
