// UV-sphere body marcher — layer-agnostic cell-local DDA in
// `(un_phi, un_theta, un_r)` parameter-space coords.
//
// Architecture (mirrors `march_cartesian`'s precision discipline):
//
// - Each cell at any descent depth is `[0, 1)³` in cell-local
//   `(un_phi, un_theta, un_r)`. Ray-axis-plane intersections happen
//   at `un = 0` and `un = 1` — same cancellation-free arithmetic as
//   cartesian's plane crossings, regardless of how deep we descend.
//
// - The world ray direction is converted to a per-cell parameter-
//   space direction via the local Jacobian — the orthonormal
//   `{r̂, θ̂, φ̂}` basis at the current ray position, scaled by the
//   cell-axis sizes `(d.dphi, d.dth, d.dr)`. The Jacobian is
//   re-evaluated at the ENTRY of each cell.
//
// - Curvature collapses to 0 automatically: the Jacobian's
//   variation across a cell is O(`cell_size / body_radius`), which
//   shrinks like 3⁻ᴺ. Past depth ~5 it's <1% of cell size, by
//   depth 10 it's < 0.001%. The single-Jacobian-per-cell
//   approximation is exact in the deep-zoom limit; at shallow
//   depth it's an approximation but still smooth.
//
// - No mode switch. Same DDA at every depth. Same algorithm
//   handles a body filling the frame at the root and a cell at
//   depth 60 — the mathematics is identical, just the cell-axis
//   sizes shrink.
//
// Cell topology:
// - The body root (a `NodeKind::UvSphereBody` node) has 27 children
//   indexed by `(φ-tier, θ-tier, r-tier)` ∈ {0,1,2}³.
// - Body bounds: `φ ∈ [0, 2π)` (wraps), `θ ∈ [-θ_cap, +θ_cap]`,
//   `r ∈ [inner_r, outer_r]`. Each descent shrinks each axis
//   range by 1/3.
// - Body params (inner_r, outer_r, theta_cap) read from
//   `node_kinds[body_node_idx]` packed by
//   `GpuNodeKind::from_node_kind`.

#include "bindings.wgsl"

const UV_TWO_PI: f32 = 6.2831853;
// The descent terminates on a Block/Empty slot — the cap is just a
// hardware safety. Match the storage tree's `MAX_DEPTH` so the walker
// can reach any valid leaf regardless of worldgen chain length.
const UV_MAX_DEPTH: u32 = 63u;
const UV_MAX_ITER: u32 = 256u;

// Ray–sphere centered at body-frame center, returning both roots.
// Used ONCE per ray for body-shell entry (radii are O(1) at body
// scale so precision is fine here).
fn uv_ray_sphere(oc: vec3<f32>, dir: vec3<f32>, r: f32) -> vec2<f32> {
    let aa = dot(dir, dir);
    let bb = dot(oc, dir);
    let cc = dot(oc, oc) - r * r;
    let disc = bb * bb - aa * cc;
    if disc <= 0.0 {
        return vec2<f32>(1e30, -1e30);
    }
    let sq = sqrt(disc);
    let inv_a = 1.0 / aa;
    return vec2<f32>((-bb - sq) * inv_a, (-bb + sq) * inv_a);
}

// Outcome of descending from the body root to the deepest cell
// containing the point at `t`.
//
// `un_phi / un_theta / un_r` are cell-local fractions in `[0, 1)³`,
// tracked via the precision-stable `un = un · 3 − tier` recursion.
// `dphi / dth / dr` are cell-axis sizes at the resolved depth,
// tracked via `/= 3` per descent — both stay precise at any depth.
struct UvDescend {
    found_block: bool,
    block_type: u32,
    dphi: f32,
    dth: f32,
    dr: f32,
    un_phi: f32,
    un_theta: f32,
    un_r: f32,
    depth: u32,
}

fn uv_descend(
    body_node_idx: u32,
    body_inner_r: f32, body_outer_r: f32, body_theta_cap: f32,
    phi_w: f32, theta_w: f32, r_w: f32,
) -> UvDescend {
    return uv_descend_from_frame(
        body_node_idx,
        0.0, -body_theta_cap, body_inner_r,
        UV_TWO_PI, 2.0 * body_theta_cap, body_outer_r - body_inner_r,
        phi_w, theta_w, r_w,
    );
}

/// Descend from an arbitrary UV frame whose `(φ, θ, r)` range is
/// `(phi_min, phi_min+frame_dphi) × (theta_min, theta_min+frame_dth) × (r_min, r_min+frame_dr)`.
/// `frame_node_idx` is the BFS index of the frame's tree node — its
/// 27 children are addressed by `pt + tt*3 + rt*9` exactly as the
/// body root's children are. Operates entirely in cell-local
/// fractions: at each level `un_*` rolls forward via
/// `un = un·3 − tier`, and `dphi/dth/dr /= 3`. Both stay precise
/// regardless of how deep the frame itself is.
fn uv_descend_from_frame(
    frame_node_idx: u32,
    phi_min: f32, theta_min: f32, r_min: f32,
    frame_dphi: f32, frame_dth: f32, frame_dr: f32,
    phi_w: f32, theta_w: f32, r_w: f32,
) -> UvDescend {
    var d: UvDescend;
    d.found_block = false;
    d.block_type = 0u;
    d.dphi = frame_dphi;
    d.dth = frame_dth;
    d.dr = frame_dr;
    d.un_phi = 0.0;
    d.un_theta = 0.0;
    d.un_r = 0.0;
    d.depth = 0u;

    var un_phi = clamp((phi_w - phi_min) / frame_dphi, 0.0, 1.0 - 1e-7);
    var un_theta = clamp((theta_w - theta_min) / frame_dth, 0.0, 1.0 - 1e-7);
    var un_r = clamp((r_w - r_min) / frame_dr, 0.0, 1.0 - 1e-7);

    var node_idx = frame_node_idx;
    var depth: u32 = 0u;

    loop {
        if depth >= UV_MAX_DEPTH { break; }

        let dphi = d.dphi / 3.0;
        let dth = d.dth / 3.0;
        let dr = d.dr / 3.0;

        let pt = u32(clamp(floor(un_phi * 3.0), 0.0, 2.0));
        let tt = u32(clamp(floor(un_theta * 3.0), 0.0, 2.0));
        let rt = u32(clamp(floor(un_r * 3.0), 0.0, 2.0));
        let slot = pt + tt * 3u + rt * 9u;

        let header_off = node_offsets[node_idx];
        let occupancy = tree[header_off];
        let bit = (occupancy >> slot) & 1u;
        if bit == 0u {
            // Empty slot: descent stops here. Return the empty cell's
            // dimensions so the caller can step the ray through it.
            d.dphi = dphi;
            d.dth = dth;
            d.dr = dr;
            d.un_phi = clamp(un_phi * 3.0 - f32(pt), 0.0, 1.0 - 1e-7);
            d.un_theta = clamp(un_theta * 3.0 - f32(tt), 0.0, 1.0 - 1e-7);
            d.un_r = clamp(un_r * 3.0 - f32(rt), 0.0, 1.0 - 1e-7);
            d.depth = depth + 1u;
            return d;
        }

        let mask = (1u << slot) - 1u;
        let rank = countOneBits(occupancy & mask);
        let first_child = tree[header_off + 1u];
        let child_off = first_child + rank * 2u;
        let packed = tree[child_off];
        // Tag/block_type bit-layout matches `gpu::pack`: tag in low
        // byte, block_type in bits 8..23.
        let tag = packed & 0xFFu;

        d.dphi = dphi;
        d.dth = dth;
        d.dr = dr;
        un_phi = clamp(un_phi * 3.0 - f32(pt), 0.0, 1.0 - 1e-7);
        un_theta = clamp(un_theta * 3.0 - f32(tt), 0.0, 1.0 - 1e-7);
        un_r = clamp(un_r * 3.0 - f32(rt), 0.0, 1.0 - 1e-7);
        d.un_phi = un_phi;
        d.un_theta = un_theta;
        d.un_r = un_r;
        d.depth = depth + 1u;

        if tag == 1u {
            d.found_block = true;
            d.block_type = (packed >> 8u) & 0xFFFFu;
            return d;
        }
        if tag == 2u {
            node_idx = tree[child_off + 1u];
            depth = depth + 1u;
            continue;
        }
        // EntityRef or Empty leaf: treat as empty.
        return d;
    }

    return d;
}

// Hit-face descriptor: closest cell-boundary face the ray was
// approaching, world-space normal, and the axis tag (0=φ, 1=θ, 2=r)
// so callers can pick the in-face 2D pair for the bevel.
struct UvHitFace {
    normal: vec3<f32>,
    axis: u32,
}

// Pick the closest cell-boundary face from precision-stable `un_*`
// (cell-local fractions ∈ [0, 1]) and `dphi/dth/dr` (cell-axis sizes).
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

    // Outward normals at the hit point in world coords.
    // {r̂, θ̂, φ̂} are mutually orthogonal in spherical coordinates.
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

// Soft-edge bevel for UV cells. Picks the 2D in-face coord pair
// perpendicular to the hit-face axis, then darkens at the cell
// edges so each voxel reads as a discrete (φ, θ, r) cell.
fn uv_cell_bevel(
    un_phi: f32, un_theta: f32, un_r: f32,
    axis: u32,
) -> f32 {
    var u: f32; var v: f32;
    if axis == 0u {
        u = un_theta;
        v = un_r;
    } else if axis == 1u {
        u = un_phi;
        v = un_r;
    } else {
        u = un_phi;
        v = un_theta;
    }
    let edge = min(min(u, 1.0 - u), min(v, 1.0 - v));
    return smoothstep(0.02, 0.14, edge);
}

// Cell-local DDA step. Given the current `un_*` position and the
// per-cell parameter-space direction `d_un_*`, return the smallest
// positive `t_advance` (in world ray-distance units, since `d_un_*`
// already includes the world→param Jacobian) at which the ray
// crosses an axis-plane at `un = 0` or `un = 1`. Returns sentinel
// on no crossing.
//
// This is the cartesian-style stepping: the math operates entirely
// on values in `[0, 1)³`, so precision is bounded by f32 ULPs of
// O(1) values regardless of how deep the descent is.
struct UvCellStep {
    t: f32,
    axis: u32,  // 0=φ, 1=θ, 2=r
}

fn uv_cell_step(
    un_phi: f32, un_theta: f32, un_r: f32,
    d_un_phi: f32, d_un_theta: f32, d_un_r: f32,
) -> UvCellStep {
    var t_phi = 1e30;
    if d_un_phi > 1e-12 { t_phi = (1.0 - un_phi) / d_un_phi; }
    else if d_un_phi < -1e-12 { t_phi = -un_phi / d_un_phi; }
    var t_theta = 1e30;
    if d_un_theta > 1e-12 { t_theta = (1.0 - un_theta) / d_un_theta; }
    else if d_un_theta < -1e-12 { t_theta = -un_theta / d_un_theta; }
    var t_r = 1e30;
    if d_un_r > 1e-12 { t_r = (1.0 - un_r) / d_un_r; }
    else if d_un_r < -1e-12 { t_r = -un_r / d_un_r; }

    var out: UvCellStep;
    out.t = t_phi;
    out.axis = 0u;
    if t_theta < out.t { out.t = t_theta; out.axis = 1u; }
    if t_r < out.t { out.t = t_r; out.axis = 2u; }
    return out;
}

// Top-level UV-sphere DDA. `body_node_idx` is the BFS index of the
// body root; `ray_origin` and `ray_dir` are in the body's local
// `[0, 3)³` frame.
fn march_uv_sphere(body_node_idx: u32, ray_origin: vec3<f32>, ray_dir: vec3<f32>) -> HitResult {
    var result: HitResult;
    result.hit = false;
    result.t = 1e20;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_min = vec3<f32>(0.0);
    result.cell_size = 3.0;
    result.color = vec3<f32>(0.0);
    result.normal = vec3<f32>(0.0, 1.0, 0.0);

    let body = node_kinds[body_node_idx];
    let inner_r_local = bitcast<f32>(body.param_a);
    let outer_r_local = bitcast<f32>(body.param_b);
    let theta_cap     = bitcast<f32>(body.param_c);
    let body_size = 3.0;
    let center = vec3<f32>(body_size * 0.5);
    let outer_r = outer_r_local * body_size;
    let inner_r = inner_r_local * body_size;

    let oc_init = ray_origin - center;

    // Find ray entry into the outer shell (precision is fine here
    // because radii are O(1) at body-frame scale).
    let outer_t = uv_ray_sphere(oc_init, ray_dir, outer_r);
    if outer_t.y < 0.0001 || outer_t.x > outer_t.y {
        return result;
    }
    let inside_outer = dot(oc_init, oc_init) <= outer_r * outer_r;
    var t: f32 = select(max(outer_t.x, 0.0001), 0.0001, inside_outer);
    let t_exit_outer = outer_t.y;

    var iter: u32 = 0u;

    loop {
        if iter >= UV_MAX_ITER { break; }
        iter += 1u;

        if t > t_exit_outer + 1e-4 { break; }

        let pos = ray_origin + ray_dir * t;
        let off = pos - center;
        let r_w = length(off);

        if r_w > outer_r * 1.0001 { break; }
        if r_w < inner_r * 0.9999 {
            // Inner-core hit: ray reached the inner shell. Render
            // as stone with smooth radial normal.
            result.hit = true;
            result.t = t;
            result.normal = off / max(r_w, 1e-6);
            result.color = palette[0u].rgb;
            result.cell_min = pos - vec3<f32>(0.5);
            result.cell_size = 1.0;
            return result;
        }
        let theta_w = asin(clamp(off.y / max(r_w, 1e-6), -1.0, 1.0));
        if abs(theta_w) > theta_cap { break; }

        var phi_w = atan2(off.z, off.x);
        if phi_w < 0.0 { phi_w += UV_TWO_PI; }

        let d = uv_descend(
            body_node_idx,
            inner_r, outer_r, theta_cap,
            phi_w, theta_w, r_w,
        );

        if d.found_block {
            let face = uv_hit_face(
                off, r_w, theta_w, phi_w,
                d.un_phi, d.un_theta, d.un_r,
                d.dphi, d.dth, d.dr,
            );
            let bevel = uv_cell_bevel(d.un_phi, d.un_theta, d.un_r, face.axis);

            result.hit = true;
            result.t = t;
            result.normal = face.normal;
            result.color = palette[d.block_type].rgb * (0.7 + 0.3 * bevel);
            result.cell_min = pos - vec3<f32>(0.5);
            result.cell_size = 1.0;
            return result;
        }

        // EMPTY cell. Step ray to the next cell-local axis crossing.
        //
        // Convert world ray dir to cell-local parameter direction
        // via the local Jacobian:
        //
        //   d_phi_param   = (ray_dir · φ̂) / (r·cos θ)
        //   d_theta_param = (ray_dir · θ̂) / r
        //   d_r_param     = (ray_dir · r̂)
        //
        // where {r̂, θ̂, φ̂} is the local orthonormal spherical basis
        // at the current ray pos. Then per-axis cell-local rates:
        //
        //   d_un_phi   = d_phi_param   / dphi
        //   d_un_theta = d_theta_param / dth
        //   d_un_r     = d_r_param     / dr
        //
        // The Jacobian is evaluated ONCE per cell (curvature-as-
        // constant within cell). At deep depth it's essentially
        // constant — curvature collapses to 0; at shallow depth it
        // varies and we accept the per-cell approximation.
        let cos_t = cos(theta_w);
        let sin_t_w = sin(theta_w);
        let cos_p = cos(phi_w);
        let sin_p = sin(phi_w);
        let r_hat = vec3<f32>(cos_t * cos_p, sin_t_w, cos_t * sin_p);
        let theta_hat = vec3<f32>(-sin_t_w * cos_p, cos_t, -sin_t_w * sin_p);
        let phi_hat = vec3<f32>(-sin_p, 0.0, cos_p);
        let inv_r = 1.0 / max(r_w, 1e-6);
        let inv_r_cos = 1.0 / max(r_w * cos_t, 1e-6);

        let d_un_phi = dot(ray_dir, phi_hat) * inv_r_cos / max(d.dphi, 1e-12);
        let d_un_theta = dot(ray_dir, theta_hat) * inv_r / max(d.dth, 1e-12);
        let d_un_r = dot(ray_dir, r_hat) / max(d.dr, 1e-12);

        let step = uv_cell_step(
            d.un_phi, d.un_theta, d.un_r,
            d_un_phi, d_un_theta, d_un_r,
        );
        if step.t > 1e20 {
            break;
        }
        // Advance with a small ε to land inside the neighbor cell
        // rather than exactly on the shared boundary.
        t = t + step.t + max(step.t * 1e-4, 1e-5);
    }

    return result;
}

// UV-sub-cell DDA. Same Jacobian-per-cell stepping as
// `march_uv_sphere`, but the descent starts at a `frame_node_idx`
// nested inside the body — every iteration's `un_*` lands in the
// FRAME's `[0, 1]³`, not the body root's, so cell-local resolution
// stays at f32 ULPs of the frame regardless of how deep the frame
// itself is.
//
// Inputs:
// - `frame_node_idx`: BFS index of the sub-cell node (its 27 children
//   are the next UV tiers).
// - `body_inner_r`, `body_outer_r`, `body_theta_cap`: body params,
//   in body-frame `[0, 3)³` units (already body_size-scaled).
// - `phi_min`, `theta_min`, `r_min`: frame origin in body's spherical
//   coords.
// - `frame_dphi`, `frame_dth`, `frame_dr`: frame extents in same.
// - `ray_origin`, `ray_dir`: ray in body-frame `[0, 3)³` cartesian.
//   The renderer's `gpu_camera_for_frame` writes the camera in this
//   frame for sub-cell dispatch via `cartesian_path()`.
//
// When the ray exits the FRAME's `(φ, θ, r)` range we terminate
// (no ribbon-pop yet — that's a follow-up). Most rays in
// inside-the-body gameplay terminate on the surface block well
// before exiting the frame, so the practical visual is correct;
// rays threading the body without hitting are a known follow-up.
fn march_uv_subcell(
    frame_node_idx: u32,
    body_inner_r: f32, body_outer_r: f32, body_theta_cap: f32,
    phi_min: f32, theta_min: f32, r_min: f32,
    frame_dphi: f32, frame_dth: f32, frame_dr: f32,
    ray_origin: vec3<f32>, ray_dir: vec3<f32>,
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

    let body_size = 3.0;
    let center = vec3<f32>(body_size * 0.5);
    let inner_r = body_inner_r;
    let outer_r = body_outer_r;
    let theta_cap = body_theta_cap;
    let phi_max = phi_min + frame_dphi;
    let theta_max = theta_min + frame_dth;
    let r_max = r_min + frame_dr;

    let oc_init = ray_origin - center;

    // Body-shell entry test (same as the body-root marcher). Even
    // though the frame is a sub-cell, the ray's entry into the
    // overall body is what bounds the descent's t-range.
    let outer_t = uv_ray_sphere(oc_init, ray_dir, outer_r);
    if outer_t.y < 0.0001 || outer_t.x > outer_t.y {
        return result;
    }
    let inside_outer = dot(oc_init, oc_init) <= outer_r * outer_r;
    var t: f32 = select(max(outer_t.x, 0.0001), 0.0001, inside_outer);
    let t_exit_outer = outer_t.y;

    var iter: u32 = 0u;
    loop {
        if iter >= UV_MAX_ITER { break; }
        iter += 1u;

        if t > t_exit_outer + 1e-4 { break; }

        let pos = ray_origin + ray_dir * t;
        let off = pos - center;
        let r_w = length(off);

        if r_w > outer_r * 1.0001 { break; }
        if r_w < inner_r * 0.9999 {
            result.hit = true;
            result.t = t;
            result.normal = off / max(r_w, 1e-6);
            result.color = palette[0u].rgb;
            result.cell_min = pos - vec3<f32>(0.5);
            result.cell_size = 1.0;
            return result;
        }
        let theta_w = asin(clamp(off.y / max(r_w, 1e-6), -1.0, 1.0));
        if abs(theta_w) > theta_cap { break; }
        var phi_w = atan2(off.z, off.x);
        if phi_w < 0.0 { phi_w += UV_TWO_PI; }

        // Frame-bound check: if the ray has left the sub-cell's
        // `(φ, θ, r)` range, we have no descent context here. This
        // diff terminates; ribbon-pop into the frame's parent (and
        // sibling sub-cells) is a follow-up.
        let in_frame =
            phi_w >= phi_min - 1e-6 && phi_w <= phi_max + 1e-6
            && theta_w >= theta_min - 1e-6 && theta_w <= theta_max + 1e-6
            && r_w >= r_min - 1e-6 && r_w <= r_max + 1e-6;
        if !in_frame {
            break;
        }

        let d = uv_descend_from_frame(
            frame_node_idx,
            phi_min, theta_min, r_min,
            frame_dphi, frame_dth, frame_dr,
            phi_w, theta_w, r_w,
        );

        if d.found_block {
            let face = uv_hit_face(
                off, r_w, theta_w, phi_w,
                d.un_phi, d.un_theta, d.un_r,
                d.dphi, d.dth, d.dr,
            );
            let bevel = uv_cell_bevel(d.un_phi, d.un_theta, d.un_r, face.axis);

            result.hit = true;
            result.t = t;
            result.normal = face.normal;
            result.color = palette[d.block_type].rgb * (0.7 + 0.3 * bevel);
            result.cell_min = pos - vec3<f32>(0.5);
            result.cell_size = 1.0;
            return result;
        }

        // EMPTY cell. Step via cell-local Jacobian DDA — same as
        // `march_uv_sphere`, just the per-cell `dphi/dth/dr` come
        // from the frame-rooted descent so they're already cell-
        // local at the sub-cell scale.
        let cos_t = cos(theta_w);
        let sin_t_w = sin(theta_w);
        let cos_p = cos(phi_w);
        let sin_p = sin(phi_w);
        let r_hat = vec3<f32>(cos_t * cos_p, sin_t_w, cos_t * sin_p);
        let theta_hat = vec3<f32>(-sin_t_w * cos_p, cos_t, -sin_t_w * sin_p);
        let phi_hat = vec3<f32>(-sin_p, 0.0, cos_p);
        let inv_r = 1.0 / max(r_w, 1e-6);
        let inv_r_cos = 1.0 / max(r_w * cos_t, 1e-6);

        let d_un_phi = dot(ray_dir, phi_hat) * inv_r_cos / max(d.dphi, 1e-12);
        let d_un_theta = dot(ray_dir, theta_hat) * inv_r / max(d.dth, 1e-12);
        let d_un_r = dot(ray_dir, r_hat) / max(d.dr, 1e-12);

        let step = uv_cell_step(
            d.un_phi, d.un_theta, d.un_r,
            d_un_phi, d_un_theta, d_un_r,
        );
        if step.t > 1e20 {
            break;
        }
        t = t + step.t + max(step.t * 1e-4, 1e-5);
    }

    return result;
}

// Stone block-type index — read from CPU-side `block::STONE`.
fn crate_stone_index() -> u32 {
    return 0u;
}
