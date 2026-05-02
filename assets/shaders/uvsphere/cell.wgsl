// Cell-local primitives shared by the UV-sphere marcher.
//
// Architecture mirrors the CPU walker (`src/world/raycast/uvsphere.rs`):
// - Track ABSOLUTE bounds `(phi_lo, phi_hi, theta_lo, theta_hi,
//   r_lo, r_hi)` during descent, but propagated as DELTAS
//   (`delta_X = X_w − X_lo`) to keep tier-picking precision at
//   sub-ULP regardless of descent depth.
// - Step the ray to the next cell boundary via ray-vs-{sphere,
//   cone, φ-plane} intersection in body-frame cartesian coords;
//   numerically stable at any descent depth.
// - The boundary-step result records WHICH face the ray crossed
//   to enter the next cell. The renderer uses that axis directly
//   for the hit's face normal + bevel — never the closest-face
//   arc-distance heuristic, which mis-picks near cell edges and
//   produces triangular sub-cell artifacts.

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

// Ray vs constant-`φ` half-plane (radial out from body axis).
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

// Boundary-step result. `t` is world-distance to the closest cell
// face crossing > `t_min`; `axis` ∈ {0=φ, 1=θ, 2=r} is which face
// won; `side` is 0 for the cell's `_lo` face, 1 for `_hi`.
struct UvBoundaryHit {
    t: f32,
    axis: u32,
    side: u32,
}

// Closest of the cell's six bounding surfaces. Each ray-vs-surface
// candidate has a known axis+side, so the returned `axis/side`
// describes WHICH face the ray will cross next — the renderer uses
// this for the next cell's face-normal lookup, bypassing
// arc-distance auto-picking that flips axis near cell edges and
// produces sub-cell triangular artifacts in the bevel.
fn uv_next_boundary(
    oc: vec3<f32>, dir: vec3<f32>, t_min: f32,
    phi_lo: f32, phi_hi: f32,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
) -> UvBoundaryHit {
    var out: UvBoundaryHit;
    out.t = 1e30;
    out.axis = 2u;
    out.side = 1u;

    let t_pl = uv_ray_phi_plane(oc, dir, phi_lo);
    if t_pl > t_min && t_pl < out.t { out.t = t_pl; out.axis = 0u; out.side = 0u; }
    let t_ph = uv_ray_phi_plane(oc, dir, phi_hi);
    if t_ph > t_min && t_ph < out.t { out.t = t_ph; out.axis = 0u; out.side = 1u; }

    let cl = uv_ray_cone(oc, dir, theta_lo);
    if cl.x > t_min && cl.x < out.t { out.t = cl.x; out.axis = 1u; out.side = 0u; }
    if cl.y > t_min && cl.y < out.t { out.t = cl.y; out.axis = 1u; out.side = 0u; }
    let ch = uv_ray_cone(oc, dir, theta_hi);
    if ch.x > t_min && ch.x < out.t { out.t = ch.x; out.axis = 1u; out.side = 1u; }
    if ch.y > t_min && ch.y < out.t { out.t = ch.y; out.axis = 1u; out.side = 1u; }

    let sl = uv_ray_sphere(oc, dir, r_lo);
    if sl.x > t_min && sl.x < out.t { out.t = sl.x; out.axis = 2u; out.side = 0u; }
    if sl.y > t_min && sl.y < out.t { out.t = sl.y; out.axis = 2u; out.side = 0u; }
    let sh = uv_ray_sphere(oc, dir, r_hi);
    if sh.x > t_min && sh.x < out.t { out.t = sh.x; out.axis = 2u; out.side = 1u; }
    if sh.y > t_min && sh.y < out.t { out.t = sh.y; out.axis = 2u; out.side = 1u; }

    return out;
}

// Walk down the tree from `start_node_idx` against world
// `(phi_w, theta_w, r_w)`. Returns the deepest reached cell as
// absolute bounds + `dphi/dth/dr` and a `found_block` flag.
//
// Delta-tracked: holds `delta_X = X_w − X_lo` directly through the
// loop. The straightforward `(phi_w − phi_lo) / dphi` form fails at
// depth 12+ because `phi_lo` accumulates `K · ULP(2π)` rounding
// (~1e-6 by `K=12`), and `dphi = 2π/3^K ≈ 1.2e-5` is the same
// scale — tier picking goes ~50% wrong, breaks land in random
// cells. Holding `delta` keeps the arithmetic in O(`dphi`)
// magnitude all the way down. See `src/world/raycast/uvsphere.rs::descend`.
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

        dphi    = dphi    / 3.0;
        dth     = dth     / 3.0;
        dr_axis = dr_axis / 3.0;

        let pt = u32(clamp(floor(delta_phi   / max(dphi,    1e-30)), 0.0, 2.0));
        let tt = u32(clamp(floor(delta_theta / max(dth,     1e-30)), 0.0, 2.0));
        let rt = u32(clamp(floor(delta_r     / max(dr_axis, 1e-30)), 0.0, 2.0));
        let slot = pt + tt * 3u + rt * 9u;

        delta_phi   = delta_phi   - f32(pt) * dphi;
        delta_theta = delta_theta - f32(tt) * dth;
        delta_r     = delta_r     - f32(rt) * dr_axis;

        let header_off = node_offsets[node_idx];
        let occupancy = tree[header_off];
        if (occupancy & (1u << slot)) == 0u {
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
        // EntityRef / Empty leaf — treat as empty.
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

// Face normal in body-frame world coords, given the axis+side of
// the face the ray crossed to enter the current cell. The basis
// `{r̂, θ̂, φ̂}` is evaluated at the world position; for radial
// faces this is exactly the surface normal, for cone/φ-plane
// faces it's a tangent-plane normal — same approximation as the
// previous `uv_hit_face` returned, but without the arc-distance
// auto-pick that flipped axis near cell edges.
fn uv_face_normal(
    off: vec3<f32>, r_w: f32, theta_w: f32, phi_w: f32,
    axis: u32, side: u32,
) -> vec3<f32> {
    let cos_t = cos(theta_w);
    let sin_t = sin(theta_w);
    let cos_p = cos(phi_w);
    let sin_p = sin(phi_w);
    var n: vec3<f32>;
    if axis == 0u {
        // φ-plane normal = ±φ̂.
        let phi_hat = vec3<f32>(-sin_p, 0.0, cos_p);
        n = select(-phi_hat, phi_hat, side == 1u);
    } else if axis == 1u {
        // θ-cone normal = ±θ̂.
        let theta_hat = vec3<f32>(-sin_t * cos_p, cos_t, -sin_t * sin_p);
        n = select(-theta_hat, theta_hat, side == 1u);
    } else {
        // r-sphere normal = ±r̂. Use the actual offset for
        // numerical stability (avoids re-deriving from sin/cos).
        let r_hat = off / max(r_w, 1e-6);
        n = select(-r_hat, r_hat, side == 1u);
    }
    return normalize(n);
}

// Soft-edge bevel from absolute bounds. The 2D in-face coord pair
// is determined by `axis` — the orthogonal pair to the face's
// out-of-plane direction. With the entry-axis recorded by
// `uv_next_boundary` (rather than auto-picked), the (u, v) basis
// is consistent across the cell's visible face — no mid-cell
// axis flip, no triangular bevel artifact.
fn uv_cell_bevel(
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
