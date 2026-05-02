// `march_uv_sphere`: stack-based DDA for the UV-sphere body.
//
// Precision discipline (matches the cartesian DDA's design):
//
// 1. We compute world (φ, θ, r) ONCE — at the body-shell entry — to
//    get the body-level cell-local fractions `un_*`. Every subsequent
//    iteration evolves `cur_un_*` cell-locally; we never recompute it
//    from the world ray position.
//
// 2. `cur_un_*` evolves by `cur_un += Δt · d_un` per advance, where Δt
//    is the DDA's per-axis step and `d_un_*` is the per-axis rate
//    derived from the cell-centre Jacobian. On a boundary crossing
//    the just-crossed axis snaps to `0` or `1` exactly — no rounding
//    accumulation along the crossed axis.
//
// 3. Descent into a child cell refines `cur_un *= 3 − tier` (one
//    `*3` of error). Pop-up reverses with `(cur_un + popped_tier) / 3`
//    (precision recovers). Across mixed descent / pop / advance, error
//    stays bounded — it doesn't compound `3^K` per iteration like the
//    previous shader did.
//
// 4. Cell origin `(cell_phi_min, cell_theta_min, cell_r_min)` is
//    tracked incrementally on descend/pop. Each update is exact in
//    f32 (integer tier × power-of-3 fraction of body's range).
//
// 5. Per-cell basis is recomputed at each cell entry. This is the
//    shallow-depth fix the original code lacked; at depths 0–4 the
//    basis varies sharply across the body, so a frame-constant basis
//    would distort the rendering. At deep depths the basis varies
//    by `< 1°` per cell — the recompute is correct everywhere.

const MAX_UV_STACK: u32 = 16u;

fn march_uv_sphere(
    body_node_idx: u32,
    ray_origin: vec3<f32>,
    ray_dir: vec3<f32>,
) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 3.0;
    result.color = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0, 1.0, 0.0);

    // ---- Body params ----
    let body = node_kinds[body_node_idx];
    let inner_r_local = bitcast<f32>(body.param_a);
    let outer_r_local = bitcast<f32>(body.param_b);
    let theta_cap     = bitcast<f32>(body.param_c);
    let body_size = 3.0;
    let center = vec3<f32>(body_size * 0.5);
    let outer_r = outer_r_local * body_size;
    let inner_r = inner_r_local * body_size;
    let body_dphi = UV_TWO_PI;
    let body_dth = 2.0 * theta_cap;
    let body_dr = outer_r - inner_r;

    // ---- Body-shell entry ----
    let oc = ray_origin - center;
    let outer_t = uv_ray_sphere(oc, ray_dir, outer_r);
    if outer_t.y < 0.0001 || outer_t.x > outer_t.y { return result; }
    let inside_outer = dot(oc, oc) <= outer_r * outer_r;
    let t_enter = select(max(outer_t.x, 0.0001), 0.0001, inside_outer);
    let t_exit = outer_t.y;

    // Spherical coords at entry — used ONCE to seed body-level un.
    let entry_pos = ray_origin + ray_dir * t_enter;
    let entry_off = entry_pos - center;
    let entry_r = length(entry_off);

    if entry_r < inner_r * 0.9999 {
        // Camera below inner shell (e.g. embedded). Render core.
        result.hit = true;
        result.t = t_enter;
        result.normal = entry_off / max(entry_r, 1e-6);
        result.color = palette[0u].rgb;
        result.cell_min = entry_pos - vec3<f32>(0.5);
        result.cell_size = 1.0;
        return result;
    }
    let entry_theta = asin(clamp(entry_off.y / max(entry_r, 1e-6), -1.0, 1.0));
    if abs(entry_theta) > theta_cap { return result; }
    var entry_phi = atan2(entry_off.z, entry_off.x);
    if entry_phi < 0.0 { entry_phi += UV_TWO_PI; }

    // ---- DDA stack state ----
    var s_node: array<u32, MAX_UV_STACK>;
    var s_pt: array<u32, MAX_UV_STACK>;
    var s_tt: array<u32, MAX_UV_STACK>;
    var s_rt: array<u32, MAX_UV_STACK>;
    s_node[0] = body_node_idx;

    var depth: u32 = 0u;

    // Per-depth state.
    var cur_un_phi   = clamp(entry_phi / body_dphi, 0.0, 1.0 - 1e-7);
    var cur_un_theta = clamp((entry_theta + theta_cap) / body_dth, 0.0, 1.0 - 1e-7);
    var cur_un_r     = clamp((entry_r - inner_r) / body_dr, 0.0, 1.0 - 1e-7);
    var cur_dphi = body_dphi;
    var cur_dth  = body_dth;
    var cur_dr   = body_dr;
    var cell_phi_min:   f32 = 0.0;
    var cell_theta_min: f32 = -theta_cap;
    var cell_r_min:     f32 = inner_r;

    // Per-cell basis + DDA rates. Refreshed by `recompute_cell_state`.
    var d_un_phi: f32;
    var d_un_theta: f32;
    var d_un_r: f32;
    var step_phi: i32;
    var step_theta: i32;
    var step_r: i32;
    var inv_d_un_phi: f32;
    var inv_d_un_theta: f32;
    var inv_d_un_r: f32;
    var side_phi: f32;
    var side_theta: f32;
    var side_r: f32;
    var t: f32 = 0.0;  // cell-local time relative to entry into THIS cell.

    // --- Cell-state recompute (inlined; WGSL has no closures) ---
    // Computes basis at cell centre, per-axis d_un, step direction,
    // and per-axis time-to-next-boundary from `cur_un_*`. Resets `t`
    // to 0 — the new cell's clock starts when we enter it.
    {
        let phi_c = cell_phi_min + cur_dphi * 0.5;
        let theta_c = cell_theta_min + cur_dth * 0.5;
        let r_c = cell_r_min + cur_dr * 0.5;
        let basis = uv_basis_at(phi_c, theta_c, r_c);
        d_un_phi   = (dot(ray_dir, basis.phi_hat)   * basis.inv_r_cos_theta) / cur_dphi;
        d_un_theta = (dot(ray_dir, basis.theta_hat) * basis.inv_r)           / cur_dth;
        d_un_r     =  dot(ray_dir, basis.r_hat)                              / cur_dr;
        step_phi   = select(-1, 1, d_un_phi   >= 0.0);
        step_theta = select(-1, 1, d_un_theta >= 0.0);
        step_r     = select(-1, 1, d_un_r     >= 0.0);
        inv_d_un_phi   = 1.0 / max(abs(d_un_phi),   1e-30);
        inv_d_un_theta = 1.0 / max(abs(d_un_theta), 1e-30);
        inv_d_un_r     = 1.0 / max(abs(d_un_r),     1e-30);
        side_phi   = select(cur_un_phi   * inv_d_un_phi,   (1.0 - cur_un_phi)   * inv_d_un_phi,   d_un_phi   > 0.0);
        side_theta = select(cur_un_theta * inv_d_un_theta, (1.0 - cur_un_theta) * inv_d_un_theta, d_un_theta > 0.0);
        side_r     = select(cur_un_r     * inv_d_un_r,     (1.0 - cur_un_r)     * inv_d_un_r,     d_un_r     > 0.0);
        t = 0.0;
    }

    var iter: u32 = 0u;
    var last_axis: u32 = 0u;  // axis last crossed — used for hit normal
    var t_global: f32 = t_enter;  // total t along ray, for HitResult.t

    loop {
        if iter >= UV_MAX_ITER { break; }
        iter += 1u;

        // ---- Look up the cell at current depth ----
        let pt = u32(clamp(floor(cur_un_phi   * 3.0), 0.0, 2.0));
        let tt = u32(clamp(floor(cur_un_theta * 3.0), 0.0, 2.0));
        let rt = u32(clamp(floor(cur_un_r     * 3.0), 0.0, 2.0));
        s_pt[depth] = pt;
        s_tt[depth] = tt;
        s_rt[depth] = rt;
        let slot = pt + tt * 3u + rt * 9u;

        let header_off = node_offsets[s_node[depth]];
        let occupancy = tree[header_off];
        let occupied = (occupancy & (1u << slot)) != 0u;

        if occupied {
            let mask = (1u << slot) - 1u;
            let rank = countOneBits(occupancy & mask);
            let first_child = tree[header_off + 1u];
            let child_off = first_child + rank * 2u;
            let packed = tree[child_off];
            let tag = packed & 0xFFu;

            if tag == 1u {
                // Block hit. The leaf cell is the (pt, tt, rt) child of
                // node[depth]; its un and dphi are the refined values.
                let leaf_un_phi   = clamp(cur_un_phi   * 3.0 - f32(pt), 0.0, 1.0 - 1e-7);
                let leaf_un_theta = clamp(cur_un_theta * 3.0 - f32(tt), 0.0, 1.0 - 1e-7);
                let leaf_un_r     = clamp(cur_un_r     * 3.0 - f32(rt), 0.0, 1.0 - 1e-7);
                let leaf_dphi = cur_dphi / 3.0;
                let leaf_dth  = cur_dth  / 3.0;
                let leaf_dr   = cur_dr   / 3.0;

                // Compute world hit position and (φ, θ, r) — used by
                // `uv_hit_face` for the smooth surface normal.
                let pos = ray_origin + ray_dir * t_global;
                let off = pos - center;
                let r_w = length(off);
                let theta_w = asin(clamp(off.y / max(r_w, 1e-6), -1.0, 1.0));
                var phi_w = atan2(off.z, off.x);
                if phi_w < 0.0 { phi_w += UV_TWO_PI; }

                let face = uv_hit_face(
                    off, r_w, theta_w, phi_w,
                    leaf_un_phi, leaf_un_theta, leaf_un_r,
                    leaf_dphi, leaf_dth, leaf_dr,
                );
                let bevel = uv_cell_bevel(
                    leaf_un_phi, leaf_un_theta, leaf_un_r, face.axis,
                );
                result.hit = true;
                result.t = t_global;
                result.normal = face.normal;
                result.color = palette[(packed >> 8u) & 0xFFFFu].rgb
                    * (0.7 + 0.3 * bevel);
                result.cell_min = pos - vec3<f32>(0.5);
                result.cell_size = 1.0;
                return result;
            }

            if tag == 2u && depth + 1u < MAX_UV_STACK {
                // Descend into Node child.
                let child_idx = tree[child_off + 1u];
                let new_dphi = cur_dphi / 3.0;
                let new_dth  = cur_dth  / 3.0;
                let new_dr   = cur_dr   / 3.0;
                cell_phi_min   = cell_phi_min   + f32(pt) * new_dphi;
                cell_theta_min = cell_theta_min + f32(tt) * new_dth;
                cell_r_min     = cell_r_min     + f32(rt) * new_dr;
                cur_dphi = new_dphi;
                cur_dth  = new_dth;
                cur_dr   = new_dr;
                cur_un_phi   = clamp(cur_un_phi   * 3.0 - f32(pt), 0.0, 1.0 - 1e-7);
                cur_un_theta = clamp(cur_un_theta * 3.0 - f32(tt), 0.0, 1.0 - 1e-7);
                cur_un_r     = clamp(cur_un_r     * 3.0 - f32(rt), 0.0, 1.0 - 1e-7);
                depth = depth + 1u;
                s_node[depth] = child_idx;

                // Recompute cell state (basis, rates, side_dist, t).
                let phi_c = cell_phi_min + cur_dphi * 0.5;
                let theta_c = cell_theta_min + cur_dth * 0.5;
                let r_c = cell_r_min + cur_dr * 0.5;
                let basis = uv_basis_at(phi_c, theta_c, r_c);
                d_un_phi   = (dot(ray_dir, basis.phi_hat)   * basis.inv_r_cos_theta) / cur_dphi;
                d_un_theta = (dot(ray_dir, basis.theta_hat) * basis.inv_r)           / cur_dth;
                d_un_r     =  dot(ray_dir, basis.r_hat)                              / cur_dr;
                step_phi   = select(-1, 1, d_un_phi   >= 0.0);
                step_theta = select(-1, 1, d_un_theta >= 0.0);
                step_r     = select(-1, 1, d_un_r     >= 0.0);
                inv_d_un_phi   = 1.0 / max(abs(d_un_phi),   1e-30);
                inv_d_un_theta = 1.0 / max(abs(d_un_theta), 1e-30);
                inv_d_un_r     = 1.0 / max(abs(d_un_r),     1e-30);
                side_phi   = select(cur_un_phi   * inv_d_un_phi,   (1.0 - cur_un_phi)   * inv_d_un_phi,   d_un_phi   > 0.0);
                side_theta = select(cur_un_theta * inv_d_un_theta, (1.0 - cur_un_theta) * inv_d_un_theta, d_un_theta > 0.0);
                side_r     = select(cur_un_r     * inv_d_un_r,     (1.0 - cur_un_r)     * inv_d_un_r,     d_un_r     > 0.0);
                t = 0.0;
                continue;
            }
            // tag == 3 (EntityRef) or stack full — fall through to advance.
        }

        // ---- DDA advance: pick min-side axis and step ----
        var min_sd = side_phi;
        var advance_axis: u32 = 0u;
        if side_theta < min_sd { min_sd = side_theta; advance_axis = 1u; }
        if side_r     < min_sd { min_sd = side_r;     advance_axis = 2u; }
        let dt = min_sd - t;
        t = min_sd;
        t_global = t_global + dt;
        last_axis = advance_axis;

        // Evolve un on the OFF axes; snap the advance axis to the
        // boundary just crossed.
        cur_un_phi   = cur_un_phi   + dt * d_un_phi;
        cur_un_theta = cur_un_theta + dt * d_un_theta;
        cur_un_r     = cur_un_r     + dt * d_un_r;
        if advance_axis == 0u {
            cur_un_phi = select(1.0 - 1e-7, 0.0, step_phi > 0);
        } else if advance_axis == 1u {
            cur_un_theta = select(1.0 - 1e-7, 0.0, step_theta > 0);
        } else {
            cur_un_r = select(1.0 - 1e-7, 0.0, step_r > 0);
        }

        // Pre-compute the would-be tier post-advance for the just-
        // crossed axis. If it overflows [0, 2], we need to pop.
        var new_pt = i32(s_pt[depth]);
        var new_tt = i32(s_tt[depth]);
        var new_rt = i32(s_rt[depth]);
        if advance_axis == 0u { new_pt = new_pt + step_phi; }
        else if advance_axis == 1u { new_tt = new_tt + step_theta; }
        else { new_rt = new_rt + step_r; }
        let overflow = (new_pt < 0 || new_pt > 2)
                    || (new_tt < 0 || new_tt > 2)
                    || (new_rt < 0 || new_rt > 2);

        if !overflow {
            // Stay at this depth — advance side_dist and tier in the
            // crossed axis.
            if advance_axis == 0u {
                side_phi = side_phi + inv_d_un_phi;
                s_pt[depth] = u32(new_pt);
            } else if advance_axis == 1u {
                side_theta = side_theta + inv_d_un_theta;
                s_tt[depth] = u32(new_tt);
            } else {
                side_r = side_r + inv_d_un_r;
                s_rt[depth] = u32(new_rt);
            }
            continue;
        }

        // Pop up. Repeat until either we've exited body-root (depth 0)
        // or the post-advance tier at the new depth is in range.
        loop {
            if depth == 0u {
                // Ray exits body. Bail.
                return result;
            }
            depth = depth - 1u;
            // Restore parent geometry.
            cur_dphi = cur_dphi * 3.0;
            cur_dth  = cur_dth  * 3.0;
            cur_dr   = cur_dr   * 3.0;
            cell_phi_min   = cell_phi_min   - f32(s_pt[depth]) * (cur_dphi / 3.0);
            cell_theta_min = cell_theta_min - f32(s_tt[depth]) * (cur_dth  / 3.0);
            cell_r_min     = cell_r_min     - f32(s_rt[depth]) * (cur_dr   / 3.0);
            // Lift cur_un to parent's level.
            cur_un_phi   = (cur_un_phi   + f32(s_pt[depth])) / 3.0;
            cur_un_theta = (cur_un_theta + f32(s_tt[depth])) / 3.0;
            cur_un_r     = (cur_un_r     + f32(s_rt[depth])) / 3.0;
            // Rebuild cell state at the parent.
            let phi_c = cell_phi_min + cur_dphi * 0.5;
            let theta_c = cell_theta_min + cur_dth * 0.5;
            let r_c = cell_r_min + cur_dr * 0.5;
            let basis = uv_basis_at(phi_c, theta_c, r_c);
            d_un_phi   = (dot(ray_dir, basis.phi_hat)   * basis.inv_r_cos_theta) / cur_dphi;
            d_un_theta = (dot(ray_dir, basis.theta_hat) * basis.inv_r)           / cur_dth;
            d_un_r     =  dot(ray_dir, basis.r_hat)                              / cur_dr;
            step_phi   = select(-1, 1, d_un_phi   >= 0.0);
            step_theta = select(-1, 1, d_un_theta >= 0.0);
            step_r     = select(-1, 1, d_un_r     >= 0.0);
            inv_d_un_phi   = 1.0 / max(abs(d_un_phi),   1e-30);
            inv_d_un_theta = 1.0 / max(abs(d_un_theta), 1e-30);
            inv_d_un_r     = 1.0 / max(abs(d_un_r),     1e-30);
            // Side_dist at parent: distance from CURRENT un to next boundary.
            side_phi   = select(cur_un_phi   * inv_d_un_phi,   (1.0 - cur_un_phi)   * inv_d_un_phi,   d_un_phi   > 0.0);
            side_theta = select(cur_un_theta * inv_d_un_theta, (1.0 - cur_un_theta) * inv_d_un_theta, d_un_theta > 0.0);
            side_r     = select(cur_un_r     * inv_d_un_r,     (1.0 - cur_un_r)     * inv_d_un_r,     d_un_r     > 0.0);
            t = 0.0;

            // Advance the parent's tier on the same axis.
            var p_pt = i32(s_pt[depth]);
            var p_tt = i32(s_tt[depth]);
            var p_rt = i32(s_rt[depth]);
            if advance_axis == 0u { p_pt = p_pt + step_phi; }
            else if advance_axis == 1u { p_tt = p_tt + step_theta; }
            else { p_rt = p_rt + step_r; }
            let p_overflow = (p_pt < 0 || p_pt > 2)
                          || (p_tt < 0 || p_tt > 2)
                          || (p_rt < 0 || p_rt > 2);
            if !p_overflow {
                // Snap un on advance axis to entry-of-next-sibling.
                if advance_axis == 0u {
                    cur_un_phi = select(1.0 - 1e-7, 0.0, step_phi > 0);
                    side_phi = side_phi + inv_d_un_phi;
                    s_pt[depth] = u32(p_pt);
                } else if advance_axis == 1u {
                    cur_un_theta = select(1.0 - 1e-7, 0.0, step_theta > 0);
                    side_theta = side_theta + inv_d_un_theta;
                    s_tt[depth] = u32(p_tt);
                } else {
                    cur_un_r = select(1.0 - 1e-7, 0.0, step_r > 0);
                    side_r = side_r + inv_d_un_r;
                    s_rt[depth] = u32(p_rt);
                }
                break;
            }
            // Else continue popping.
        }
    }
    return result;
}
