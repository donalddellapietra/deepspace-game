// Cell-local primitives shared by the body-root and sub-cell marchers:
//
// - `uv_ray_sphere`: ray-sphere centred at `oc` for body-shell entry.
// - `uv_descend_cell`: walk the tree from a node + cell-local `un_*`
//   to the deepest non-empty cell, returning the cell's resolved
//   `(dphi, dth, dr)` and the descent's terminal `un_*`.
// - `uv_cell_step`: cell-local DDA — given the per-axis rates
//   `d_un_*` (in world-time units) at the cell's depth, return the
//   shortest `t` to an axis-plane crossing.
// - `uv_hit_face`: pick the closest cell face for a hit.
// - `uv_cell_bevel`: 2D smoothstep darkening at cell edges.
//
// Boundary-fudge ε convention: every advance of the ray time `t` past
// a cell boundary uses an ε expressed in CELL-LOCAL FRACTION units,
// converted back to `t` via the cell's per-axis rate. A fixed
// world-distance ε (e.g. `1e-5`) cannot be reused across depths
// because cells shrink as `1/3^N` — at depth 12 a `1e-5` ε already
// overshoots multiple cells, which is the bug that plagued the
// previous shader.

const UV_BOUNDARY_FRAC: f32 = 1e-4;

fn uv_ray_sphere(oc: vec3<f32>, dir: vec3<f32>, r: f32) -> vec2<f32> {
    let aa = dot(dir, dir);
    let bb = dot(oc, dir);
    let cc = dot(oc, oc) - r * r;
    let disc = bb * bb - aa * cc;
    if disc <= 0.0 { return vec2<f32>(1e30, -1e30); }
    let sq = sqrt(disc);
    let inv_a = 1.0 / aa;
    return vec2<f32>((-bb - sq) * inv_a, (-bb + sq) * inv_a);
}

// Walk down the tree from `start_node_idx` along the cell-local
// fractions `un_phi/un_theta/un_r` ∈ `[0, 1]³` (relative to a
// "starting cell" whose extents are `dphi/dth/dr`). Returns the
// deepest non-empty cell's `un_*` and `dphi/dth/dr` plus a
// `found_block` flag for callers.
//
// `max_descent` caps how many levels we descend below the start. The
// sub-cell marcher uses `UV_SUBCELL_DESCENT` to stop before
// `un *= 3 - tier` quantization eats the precision; the body-root
// marcher passes `UV_MAX_DEPTH` because precision at the root is
// fresh per iteration.
fn uv_descend_cell(
    start_node_idx: u32,
    dphi: f32, dth: f32, dr: f32,
    un_phi: f32, un_theta: f32, un_r: f32,
    max_descent: u32,
) -> UvDescend {
    var d: UvDescend;
    d.found_block = false;
    d.block_type = 0u;
    d.dphi = dphi;
    d.dth = dth;
    d.dr = dr;
    d.un_phi = clamp(un_phi, 0.0, 1.0 - 1e-7);
    d.un_theta = clamp(un_theta, 0.0, 1.0 - 1e-7);
    d.un_r = clamp(un_r, 0.0, 1.0 - 1e-7);
    d.depth = 0u;

    var node_idx = start_node_idx;
    var u_phi = d.un_phi;
    var u_theta = d.un_theta;
    var u_r = d.un_r;
    var c_dphi = dphi;
    var c_dth = dth;
    var c_dr = dr;
    var depth: u32 = 0u;

    loop {
        if depth >= max_descent { break; }

        let pt = u32(clamp(floor(u_phi * 3.0), 0.0, 2.0));
        let tt = u32(clamp(floor(u_theta * 3.0), 0.0, 2.0));
        let rt = u32(clamp(floor(u_r * 3.0), 0.0, 2.0));
        let slot = pt + tt * 3u + rt * 9u;

        let header_off = node_offsets[node_idx];
        let occupancy = tree[header_off];
        if (occupancy & (1u << slot)) == 0u {
            // Empty slot — descent stops at the current cell. Return
            // the cell's pre-descent `un_*` and `dphi/dth/dr`.
            d.un_phi = u_phi;
            d.un_theta = u_theta;
            d.un_r = u_r;
            d.dphi = c_dphi;
            d.dth = c_dth;
            d.dr = c_dr;
            d.depth = depth;
            return d;
        }

        let mask = (1u << slot) - 1u;
        let rank = countOneBits(occupancy & mask);
        let first_child = tree[header_off + 1u];
        let child_off = first_child + rank * 2u;
        let packed = tree[child_off];
        let tag = packed & 0xFFu;

        // Refine to the child cell.
        u_phi   = clamp(u_phi   * 3.0 - f32(pt), 0.0, 1.0 - 1e-7);
        u_theta = clamp(u_theta * 3.0 - f32(tt), 0.0, 1.0 - 1e-7);
        u_r     = clamp(u_r     * 3.0 - f32(rt), 0.0, 1.0 - 1e-7);
        c_dphi = c_dphi / 3.0;
        c_dth = c_dth / 3.0;
        c_dr = c_dr / 3.0;
        depth = depth + 1u;

        if tag == 1u {
            d.found_block = true;
            d.block_type = (packed >> 8u) & 0xFFFFu;
            d.un_phi = u_phi;
            d.un_theta = u_theta;
            d.un_r = u_r;
            d.dphi = c_dphi;
            d.dth = c_dth;
            d.dr = c_dr;
            d.depth = depth;
            return d;
        }
        if tag == 2u {
            node_idx = tree[child_off + 1u];
            continue;
        }
        // EntityRef / Empty leaf — treat as empty.
        d.un_phi = u_phi;
        d.un_theta = u_theta;
        d.un_r = u_r;
        d.dphi = c_dphi;
        d.dth = c_dth;
        d.dr = c_dr;
        d.depth = depth;
        return d;
    }

    d.un_phi = u_phi;
    d.un_theta = u_theta;
    d.un_r = u_r;
    d.dphi = c_dphi;
    d.dth = c_dth;
    d.dr = c_dr;
    d.depth = depth;
    return d;
}

fn uv_cell_step(
    un_phi: f32, un_theta: f32, un_r: f32,
    d_un_phi: f32, d_un_theta: f32, d_un_r: f32,
) -> UvCellStep {
    var t_phi = 1e30;
    if d_un_phi > 1e-30 { t_phi = (1.0 - un_phi) / d_un_phi; }
    else if d_un_phi < -1e-30 { t_phi = -un_phi / d_un_phi; }
    var t_theta = 1e30;
    if d_un_theta > 1e-30 { t_theta = (1.0 - un_theta) / d_un_theta; }
    else if d_un_theta < -1e-30 { t_theta = -un_theta / d_un_theta; }
    var t_r = 1e30;
    if d_un_r > 1e-30 { t_r = (1.0 - un_r) / d_un_r; }
    else if d_un_r < -1e-30 { t_r = -un_r / d_un_r; }

    var out: UvCellStep;
    out.t = t_phi;
    out.axis = 0u;
    if t_theta < out.t { out.t = t_theta; out.axis = 1u; }
    if t_r < out.t { out.t = t_r; out.axis = 2u; }
    return out;
}

// Cell-local boundary nudge in world-time units. The fudge is
// `UV_BOUNDARY_FRAC` of a cell along the axis that's currently the
// fastest-advancing — converting to world time via that axis's rate.
// As cells shrink at deeper descent the rates grow proportionally,
// so the world-time nudge auto-adapts.
fn uv_boundary_eps(d_un_phi: f32, d_un_theta: f32, d_un_r: f32) -> f32 {
    let m = max(abs(d_un_phi), max(abs(d_un_theta), abs(d_un_r)));
    return UV_BOUNDARY_FRAC / max(m, 1e-30);
}

fn uv_hit_face(
    off: vec3<f32>, r_w: f32, theta_w: f32, phi_w: f32,
    un_phi: f32, un_theta: f32, un_r: f32,
    dphi: f32, dth: f32, dr: f32,
) -> UvHitFace {
    let cos_t = cos(theta_w);
    let arc_phi_lo = r_w * cos_t * dphi * un_phi;
    let arc_phi_hi = r_w * cos_t * dphi * (1.0 - un_phi);
    let arc_th_lo = r_w * dth * un_theta;
    let arc_th_hi = r_w * dth * (1.0 - un_theta);
    let arc_r_lo = dr * un_r;
    let arc_r_hi = dr * (1.0 - un_r);

    let n_radial = off / max(r_w, 1e-6);
    let s_t = sin(theta_w);
    let s_p = sin(phi_w);
    let c_p = cos(phi_w);
    let n_theta = vec3<f32>(-s_t * c_p, cos_t, -s_t * s_p);
    let n_phi = vec3<f32>(-s_p, 0.0, c_p);

    var best = arc_phi_lo;
    var n = -n_phi;
    var ax: u32 = 0u;
    if arc_phi_hi < best { best = arc_phi_hi; n = n_phi; ax = 0u; }
    if arc_th_lo < best { best = arc_th_lo; n = -n_theta; ax = 1u; }
    if arc_th_hi < best { best = arc_th_hi; n = n_theta; ax = 1u; }
    if arc_r_lo < best { best = arc_r_lo; n = -n_radial; ax = 2u; }
    if arc_r_hi < best { best = arc_r_hi; n = n_radial; ax = 2u; }
    var out: UvHitFace;
    out.normal = normalize(n);
    out.axis = ax;
    return out;
}

// Soft-edge bevel: pick the 2D in-face coord pair perpendicular to
// the hit-face axis, then darken at the cell edges so each voxel
// reads as a discrete `(φ, θ, r)` cell.
fn uv_cell_bevel(un_phi: f32, un_theta: f32, un_r: f32, axis: u32) -> f32 {
    var u: f32; var v: f32;
    if axis == 0u { u = un_theta; v = un_r; }
    else if axis == 1u { u = un_phi; v = un_r; }
    else { u = un_phi; v = un_theta; }
    let edge = min(min(u, 1.0 - u), min(v, 1.0 - v));
    return smoothstep(0.02, 0.14, edge);
}
