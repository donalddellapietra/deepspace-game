// Cell-local primitives shared by the UV-sphere marchers.
//
// Architecture matches the CPU walker (`src/world/raycast/uvsphere.rs`):
// - Track ABSOLUTE bounds `(phi_lo, phi_hi, theta_lo, theta_hi,
//   r_lo, r_hi)` during descent. Each level refines bounds via
//   `phi_lo += pt * dphi` (exact in f32 since `pt` is integer and
//   `dphi` is a power-of-1/3 fraction of the body's range).
// - Step the ray to the next cell boundary via ray-vs-{sphere, cone,
//   phi-plane} intersection in body-frame cartesian coords. This is
//   numerically stable at any descent depth — `t` is in O(1) world
//   units, no `3^K` cell-local-fraction amplification anywhere.
// - The previous shader's `uv_cell_step` advanced via cell-local
//   `(1 - un)/d_un` arithmetic, where the descent's
//   `un *= 3 − tier` chain amplified a `1` ULP `phi_w` change to
//   `3^K` ULPs of leaf cell — fatal at depth 10+ once a break has
//   materialized cells that deep.

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

// Ray vs constant-`θ` cone (apex at body centre, half-angle = `θ`).
// Returns the two t intersections; sentinel `(1e30, -1e30)` on miss
// or degenerate (ray along cone axis).
fn uv_ray_cone(oc: vec3<f32>, dir: vec3<f32>, theta: f32) -> vec2<f32> {
    let s = sin(theta);
    let c = cos(theta);
    let s2 = s * s;
    let c2 = c * c;
    let aa = c2 * dir.y * dir.y - s2 * (dir.x * dir.x + dir.z * dir.z);
    let bb = c2 * oc.y * dir.y - s2 * (oc.x * dir.x + oc.z * dir.z);
    let cc = c2 * oc.y * oc.y - s2 * (oc.x * oc.x + oc.z * oc.z);
    if abs(aa) < 1e-10 {
        if abs(bb) < 1e-10 {
            return vec2<f32>(1e30, -1e30);
        }
        let t_lin = -cc / (2.0 * bb);
        return vec2<f32>(t_lin, t_lin);
    }
    let disc = bb * bb - aa * cc;
    if disc < 0.0 {
        return vec2<f32>(1e30, -1e30);
    }
    let sq = sqrt(disc);
    let inv_a = 1.0 / aa;
    return vec2<f32>((-bb - sq) * inv_a, (-bb + sq) * inv_a);
}

// Ray vs constant-`φ` half-plane (radial out from body axis at
// azimuth `φ`). Returns the single intersection `t`, or `1e30` on
// miss / degenerate. The half-plane filter rejects intersections on
// the antipodal half-plane (where `φ` is exact-but-180°-off).
fn uv_ray_phi_plane(oc: vec3<f32>, dir: vec3<f32>, phi: f32) -> f32 {
    let s = sin(phi);
    let c = cos(phi);
    let denom = -s * dir.x + c * dir.z;
    if abs(denom) < 1e-10 { return 1e30; }
    let num = s * oc.x - c * oc.z;
    let t = num / denom;
    let xp = c * (oc.x + dir.x * t) + s * (oc.z + dir.z * t);
    if xp < -1e-5 { return 1e30; }
    return t;
}

// Advance a ray to the smallest `t > t_min` at which it crosses any
// of the cell's six bounding surfaces — two `φ`-half-planes, two
// `θ`-cones, two `r`-spheres. Mirrors `next_boundary` in the CPU
// walker. All intersections are in O(1) body-frame world coords;
// no cell-local amplification.
fn uv_next_boundary(
    oc: vec3<f32>, dir: vec3<f32>, t_min: f32,
    phi_lo: f32, phi_hi: f32,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
) -> f32 {
    var best: f32 = 1e30;

    let t_pl = uv_ray_phi_plane(oc, dir, phi_lo);
    if t_pl > t_min && t_pl < best { best = t_pl; }
    let t_ph = uv_ray_phi_plane(oc, dir, phi_hi);
    if t_ph > t_min && t_ph < best { best = t_ph; }

    let cl = uv_ray_cone(oc, dir, theta_lo);
    if cl.x > t_min && cl.x < best { best = cl.x; }
    if cl.y > t_min && cl.y < best { best = cl.y; }
    let ch = uv_ray_cone(oc, dir, theta_hi);
    if ch.x > t_min && ch.x < best { best = ch.x; }
    if ch.y > t_min && ch.y < best { best = ch.y; }

    let sl = uv_ray_sphere(oc, dir, r_lo);
    if sl.x > t_min && sl.x < best { best = sl.x; }
    if sl.y > t_min && sl.y < best { best = sl.y; }
    let sh = uv_ray_sphere(oc, dir, r_hi);
    if sh.x > t_min && sh.x < best { best = sh.x; }
    if sh.y > t_min && sh.y < best { best = sh.y; }

    return best;
}

// Walk down the tree from `start_node_idx` using ABSOLUTE world
// `(phi_w, theta_w, r_w)`. Returns the deepest reached cell as
// absolute `(phi_lo, phi_hi, theta_lo, theta_hi, r_lo, r_hi)` plus
// `found_block` and the leaf's `dphi/dth/dr` for face/bevel.
//
// Delta-tracked: holds `delta_X = X_w − X_lo` directly through the
// loop instead of accumulating `X_lo` and recomputing the
// difference each level. The straightforward
// `(phi_w − phi_lo) / dphi` form fails at depth 12+ because
// `phi_lo` accumulates `K · ULP(2π)` rounding (~`1e-6` by `K=12`),
// and `dphi = 2π/3^K ≈ 1.2e-5` is the same scale — tier picking
// goes ~50% wrong, breaks land in random cells. Holding `delta`
// keeps the arithmetic in O(`dphi`) magnitude all the way down,
// so tier-picking precision stays at sub-ULP at every depth. See
// `src/world/raycast/uvsphere.rs::descend` for the matching CPU
// implementation.
fn uv_descend_cell(
    start_node_idx: u32,
    start_phi_lo: f32, start_phi_hi: f32,
    start_theta_lo: f32, start_theta_hi: f32,
    start_r_lo: f32, start_r_hi: f32,
    phi_w: f32, theta_w: f32, r_w: f32,
    max_descent: u32,
) -> UvDescend {
    var d: UvDescend;
    d.found_block = false;
    d.block_type = 0u;

    var node_idx = start_node_idx;
    var delta_phi   = phi_w   - start_phi_lo;
    var delta_theta = theta_w - start_theta_lo;
    var delta_r     = r_w     - start_r_lo;
    var dphi    = start_phi_hi   - start_phi_lo;
    var dth     = start_theta_hi - start_theta_lo;
    var dr_axis = start_r_hi     - start_r_lo;
    var depth: u32 = 0u;

    loop {
        if depth >= max_descent { break; }

        // Step down: child cell is `1/3` of parent on every axis.
        dphi    = dphi    / 3.0;
        dth     = dth     / 3.0;
        dr_axis = dr_axis / 3.0;

        let pt = u32(clamp(floor(delta_phi   / max(dphi,    1e-30)), 0.0, 2.0));
        let tt = u32(clamp(floor(delta_theta / max(dth,     1e-30)), 0.0, 2.0));
        let rt = u32(clamp(floor(delta_r     / max(dr_axis, 1e-30)), 0.0, 2.0));
        let slot = pt + tt * 3u + rt * 9u;

        // Refine deltas — subtraction of similar-magnitude values
        // keeps result-magnitude precision; never falls off the
        // tier-picking precision cliff.
        delta_phi   = delta_phi   - f32(pt) * dphi;
        delta_theta = delta_theta - f32(tt) * dth;
        delta_r     = delta_r     - f32(rt) * dr_axis;

        let header_off = node_offsets[node_idx];
        let occupancy = tree[header_off];
        if (occupancy & (1u << slot)) == 0u {
            // Empty slot — return the resolved cell. Recover absolute
            // bounds via `phi_lo = phi_w − delta_phi` etc; ULP is
            // bounded by `phi_w`'s, not `K · ULP(phi_lo)`.
            d.dphi = dphi;
            d.dth = dth;
            d.dr = dr_axis;
            d.phi_lo   = phi_w   - delta_phi;
            d.phi_hi   = d.phi_lo   + dphi;
            d.theta_lo = theta_w - delta_theta;
            d.theta_hi = d.theta_lo + dth;
            d.r_lo     = r_w     - delta_r;
            d.r_hi     = d.r_lo     + dr_axis;
            d.depth = depth + 1u;
            return d;
        }

        let mask = (1u << slot) - 1u;
        let rank = countOneBits(occupancy & mask);
        let first_child = tree[header_off + 1u];
        let child_off = first_child + rank * 2u;
        let packed = tree[child_off];
        let tag = packed & 0xFFu;

        depth = depth + 1u;

        if tag == 1u {
            d.found_block = true;
            d.block_type = (packed >> 8u) & 0xFFFFu;
            d.dphi = dphi;
            d.dth = dth;
            d.dr = dr_axis;
            d.phi_lo   = phi_w   - delta_phi;
            d.phi_hi   = d.phi_lo   + dphi;
            d.theta_lo = theta_w - delta_theta;
            d.theta_hi = d.theta_lo + dth;
            d.r_lo     = r_w     - delta_r;
            d.r_hi     = d.r_lo     + dr_axis;
            d.depth = depth;
            return d;
        }
        if tag == 2u {
            node_idx = tree[child_off + 1u];
            continue;
        }
        // EntityRef / Empty leaf — treat as empty at this resolved cell.
        d.dphi = dphi;
        d.dth = dth;
        d.dr = dr_axis;
        d.phi_lo   = phi_w   - delta_phi;
        d.phi_hi   = d.phi_lo   + dphi;
        d.theta_lo = theta_w - delta_theta;
        d.theta_hi = d.theta_lo + dth;
        d.r_lo     = r_w     - delta_r;
        d.r_hi     = d.r_lo     + dr_axis;
        d.depth = depth;
        return d;
    }

    // Hit `max_descent`. Return the deepest committed cell.
    d.dphi = dphi / 3.0;
    d.dth = dth / 3.0;
    d.dr = dr_axis / 3.0;
    d.phi_lo   = phi_w   - delta_phi;
    d.phi_hi   = d.phi_lo   + dphi;
    d.theta_lo = theta_w - delta_theta;
    d.theta_hi = d.theta_lo + dth;
    d.r_lo     = r_w     - delta_r;
    d.r_hi     = d.r_lo     + dr_axis;
    d.depth = depth;
    return d;
}

// Closest cell-face descriptor — uses ABSOLUTE bounds + world
// `(phi_w, theta_w, r_w)` directly (no un-fractions, no cell-local
// amplification).
fn uv_hit_face(
    off: vec3<f32>, r_w: f32, theta_w: f32, phi_w: f32,
    phi_lo: f32, phi_hi: f32,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
) -> UvHitFace {
    let cos_t = cos(theta_w);
    let arc_phi_lo = r_w * cos_t * abs(phi_w - phi_lo);
    let arc_phi_hi = r_w * cos_t * abs(phi_w - phi_hi);
    let arc_th_lo  = r_w * abs(theta_w - theta_lo);
    let arc_th_hi  = r_w * abs(theta_w - theta_hi);
    let arc_r_lo   = abs(r_w - r_lo);
    let arc_r_hi   = abs(r_w - r_hi);

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
    if arc_th_lo  < best { best = arc_th_lo;  n = -n_theta; ax = 1u; }
    if arc_th_hi  < best { best = arc_th_hi;  n = n_theta;  ax = 1u; }
    if arc_r_lo   < best { best = arc_r_lo;   n = -n_radial; ax = 2u; }
    if arc_r_hi   < best { best = arc_r_hi;   n = n_radial;  ax = 2u; }
    var out: UvHitFace;
    out.normal = normalize(n);
    out.axis = ax;
    return out;
}

// Soft-edge bevel from absolute bounds. The 2D in-face coord pair
// perpendicular to the hit-face axis darkens at cell edges.
fn uv_cell_bevel_abs(
    phi_w: f32, theta_w: f32, r_w: f32,
    phi_lo: f32, phi_hi: f32,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
    axis: u32,
) -> f32 {
    let dphi = max(phi_hi - phi_lo, 1e-30);
    let dth  = max(theta_hi - theta_lo, 1e-30);
    let dr   = max(r_hi - r_lo, 1e-30);
    let un_phi   = clamp((phi_w   - phi_lo)   / dphi, 0.0, 1.0);
    let un_theta = clamp((theta_w - theta_lo) / dth,  0.0, 1.0);
    let un_r     = clamp((r_w     - r_lo)     / dr,   0.0, 1.0);
    var u: f32; var v: f32;
    if axis == 0u { u = un_theta; v = un_r; }
    else if axis == 1u { u = un_phi; v = un_r; }
    else { u = un_phi; v = un_theta; }
    let edge = min(min(u, 1.0 - u), min(v, 1.0 - v));
    return smoothstep(0.02, 0.14, edge);
}
