// UV-sphere body marcher — full voxel DDA in (φ, θ, r) parameter
// space.
//
// Cell topology:
// - The body root (a `NodeKind::UvSphereBody` node) has 27 children
//   indexed by `(φ-tier, θ-tier, r-tier)` ∈ {0,1,2}³, slot index
//   `pt + 3·tt + 9·rt` matching `slot_index(x, y, z)`.
// - Body bounds: `φ ∈ [0, 2π)` (wraps), `θ ∈ [-θ_cap, +θ_cap]`,
//   `r ∈ [inner_r, outer_r]`. Each descent shrinks each axis
//   range by 1/3.
// - Descendants are `NodeKind::Cartesian` storage; slot semantics
//   keep the (φ, θ, r) interpretation contagiously while inside
//   the body subtree.
//
// Algorithm per ray step:
// 1. Compute (φ, θ, r) at the current ray-frame position.
// 2. Descend from body root to the deepest cell whose bounds contain
//    the position; stop on a Block child (HIT) or an Empty slot
//    (continue).
// 3. On Empty: pick the smallest t exceeding `t_current` among the
//    six cell-boundary intersections (2 φ-half-planes, 2 θ-cones,
//    2 r-spheres). Advance the ray by that t and repeat.
// 4. Exit when the ray leaves the body's outer shell or passes a
//    polar cap.
//
// Body params: read from `node_kinds[body_node_idx]` packed by
// `GpuNodeKind::from_node_kind`.
//
// Storage access: identical to `march_cartesian`. The slot is
// reinterpreted but the tree[] / node_offsets[] / aabbs[] reads are
// the same primitives.

#include "bindings.wgsl"

const UV_TWO_PI: f32 = 6.2831853;
// The descent terminates on a Block/Empty slot — the cap is just a
// hardware safety. Match the storage tree's `MAX_DEPTH` (defined in
// `src/world/tree.rs`) so the walker can reach any valid leaf
// regardless of worldgen chain length.
const UV_MAX_DEPTH: u32 = 63u;
const UV_MAX_ITER: u32 = 256u;

// Ray–sphere centered at body-frame center, returning both roots
// (or sentinel pair when the ray misses).
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

// Ray–cone (axis = +Y, apex at body-frame center, half-angle from
// the y axis = π/2 - |θ|; equivalently `cos²θ·y² = sin²θ·(x² + z²)`).
// Returns BOTH t roots (caller picks the half-cone with sign(y) =
// sign(θ)).
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

// Ray–half-plane (constant φ; plane through body Y axis with outward
// normal `(-sin φ, 0, cos φ)`). Single t root; caller filters for
// "correct half" (radial-out from Y axis).
fn uv_ray_phi_plane(oc: vec3<f32>, dir: vec3<f32>, phi: f32) -> f32 {
    let s = sin(phi);
    let c = cos(phi);
    let denom = -s * dir.x + c * dir.z;
    if abs(denom) < 1e-10 {
        return 1e30;
    }
    let num = s * oc.x - c * oc.z;
    return num / denom;
}

// Filter: cone solution `t` lies on the half-cone matching sign(θ).
fn uv_cone_half_ok(oc: vec3<f32>, dir: vec3<f32>, t: f32, theta: f32) -> bool {
    let y = oc.y + dir.y * t;
    if theta > 1e-6 {
        return y >= -1e-5;
    } else if theta < -1e-6 {
        return y <= 1e-5;
    } else {
        // θ = 0: cone degenerates to the equatorial plane; both signs ok.
        return true;
    }
}

// Filter: phi-plane solution `t` lies on the radial-out half (the
// other half is the antipodal φ + π plane).
fn uv_phi_half_ok(oc: vec3<f32>, dir: vec3<f32>, t: f32, phi: f32) -> bool {
    let s = sin(phi);
    let c = cos(phi);
    let xp = c * (oc.x + dir.x * t) + s * (oc.z + dir.z * t);
    return xp > -1e-5;
}

// Update `best` with `cand` when `cand` is greater than `t_min` and
// smaller than the current best. Track the axis tag in `best_axis`
// so the hit normal can be derived from which boundary was crossed.
struct UvBoundary {
    t: f32,
    axis: u32, // 0 = phi, 1 = theta, 2 = r
}
fn uv_consider(best: ptr<function, UvBoundary>, cand: f32, t_min: f32, axis: u32) {
    if cand > t_min && cand < (*best).t {
        (*best).t = cand;
        (*best).axis = axis;
    }
}

// Compute the smallest t > t_min where the ray crosses any of the 6
// boundaries of a cell with bounds (phi_lo, phi_hi, theta_lo,
// theta_hi, r_lo, r_hi). `oc` is `ray_origin - body_center`.
fn uv_next_boundary(
    oc: vec3<f32>, dir: vec3<f32>, t_min: f32,
    phi_lo: f32, phi_hi: f32,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
) -> UvBoundary {
    var best: UvBoundary;
    best.t = 1e30;
    best.axis = 99u;

    // φ planes
    let t_pl = uv_ray_phi_plane(oc, dir, phi_lo);
    if uv_phi_half_ok(oc, dir, t_pl, phi_lo) { uv_consider(&best, t_pl, t_min, 0u); }
    let t_ph = uv_ray_phi_plane(oc, dir, phi_hi);
    if uv_phi_half_ok(oc, dir, t_ph, phi_hi) { uv_consider(&best, t_ph, t_min, 0u); }

    // θ cones
    let cl = uv_ray_cone(oc, dir, theta_lo);
    if uv_cone_half_ok(oc, dir, cl.x, theta_lo) { uv_consider(&best, cl.x, t_min, 1u); }
    if uv_cone_half_ok(oc, dir, cl.y, theta_lo) { uv_consider(&best, cl.y, t_min, 1u); }
    let ch = uv_ray_cone(oc, dir, theta_hi);
    if uv_cone_half_ok(oc, dir, ch.x, theta_hi) { uv_consider(&best, ch.x, t_min, 1u); }
    if uv_cone_half_ok(oc, dir, ch.y, theta_hi) { uv_consider(&best, ch.y, t_min, 1u); }

    // r spheres
    let sl = uv_ray_sphere(oc, dir, r_lo);
    uv_consider(&best, sl.x, t_min, 2u);
    uv_consider(&best, sl.y, t_min, 2u);
    let sh = uv_ray_sphere(oc, dir, r_hi);
    uv_consider(&best, sh.x, t_min, 2u);
    uv_consider(&best, sh.y, t_min, 2u);

    return best;
}

// Outcome of descending from the body root to the deepest cell
// containing the point at `t`. The descend reads node children
// using the standard tree[] layout (occupancy mask + first_child +
// 2-u32 entries).
struct UvDescend {
    found_block: bool,
    block_type: u32,
    // Bounds of the cell we landed in (deepest non-traversed).
    phi_lo: f32, phi_hi: f32,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
    depth: u32,
}

fn uv_descend(
    body_node_idx: u32,
    body_inner_r: f32, body_outer_r: f32, body_theta_cap: f32,
    phi_w: f32, theta_w: f32, r_w: f32,
) -> UvDescend {
    var d: UvDescend;
    d.found_block = false;
    d.block_type = 0u;
    d.phi_lo = 0.0;
    d.phi_hi = UV_TWO_PI;
    d.theta_lo = -body_theta_cap;
    d.theta_hi = body_theta_cap;
    d.r_lo = body_inner_r;
    d.r_hi = body_outer_r;
    d.depth = 0u;

    var node_idx = body_node_idx;
    var depth: u32 = 0u;

    loop {
        if depth >= UV_MAX_DEPTH { break; }

        let dphi = (d.phi_hi - d.phi_lo) / 3.0;
        let dth = (d.theta_hi - d.theta_lo) / 3.0;
        let dr = (d.r_hi - d.r_lo) / 3.0;

        let pt_f = floor((phi_w - d.phi_lo) / dphi);
        let tt_f = floor((theta_w - d.theta_lo) / dth);
        let rt_f = floor((r_w - d.r_lo) / dr);
        let pt = u32(clamp(pt_f, 0.0, 2.0));
        let tt = u32(clamp(tt_f, 0.0, 2.0));
        let rt = u32(clamp(rt_f, 0.0, 2.0));
        let slot = pt + tt * 3u + rt * 9u;

        let header_off = node_offsets[node_idx];
        let occupancy = tree[header_off];
        let bit = (occupancy >> slot) & 1u;
        if bit == 0u {
            // Empty slot: descent stops here. Refine bounds to the
            // empty cell (so caller can step to its next boundary).
            d.phi_lo = d.phi_lo + f32(pt) * dphi;
            d.phi_hi = d.phi_lo + dphi;
            d.theta_lo = d.theta_lo + f32(tt) * dth;
            d.theta_hi = d.theta_lo + dth;
            d.r_lo = d.r_lo + f32(rt) * dr;
            d.r_hi = d.r_lo + dr;
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

        // Refine bounds to the child cell.
        d.phi_lo = d.phi_lo + f32(pt) * dphi;
        d.phi_hi = d.phi_lo + dphi;
        d.theta_lo = d.theta_lo + f32(tt) * dth;
        d.theta_hi = d.theta_lo + dth;
        d.r_lo = d.r_lo + f32(rt) * dr;
        d.r_hi = d.r_lo + dr;
        d.depth = depth + 1u;

        if tag == 1u {
            // BLOCK child: terminal, return for hit.
            d.found_block = true;
            d.block_type = (packed >> 8u) & 0xFFFFu;
            return d;
        }
        if tag == 2u {
            // NODE child: descend further.
            node_idx = tree[child_off + 1u];
            depth = depth + 1u;
            continue;
        }
        // EntityRef or Empty leaf: treat as empty.
        return d;
    }

    return d;
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

    let oc = ray_origin - center;

    // Find ray entry into the outer shell.
    let outer_t = uv_ray_sphere(oc, ray_dir, outer_r);
    if outer_t.y < 0.0001 || outer_t.x > outer_t.y {
        return result;
    }
    let inside_outer = dot(oc, oc) <= outer_r * outer_r;
    var t: f32 = select(max(outer_t.x, 0.0001), 0.0001, inside_outer);

    // Outer-shell exit time — the loop terminates when t exceeds
    // this (or when we step out the cap, hit the inner core, etc.).
    let t_exit_outer = outer_t.y;

    var iter: u32 = 0u;

    loop {
        if iter >= UV_MAX_ITER { break; }
        iter += 1u;

        if t > t_exit_outer + 1e-4 { break; }

        let pos = ray_origin + ray_dir * t;
        let off = pos - center;
        let r_w = length(off);

        // Body-shell exits.
        if r_w > outer_r * 1.0001 { break; }
        if r_w < inner_r * 0.9999 {
            // Inner-core hit: ray reached the inner shell boundary
            // without finding a voxel block. The body's worldgen
            // sometimes leaves rays pointed straight through the
            // core in regions that dedup to long uniform-empty
            // chains; treating the inner shell as solid stone keeps
            // the surface continuous in that case. Render with the
            // standard (palette + cube_face_bevel) shading so it
            // looks identical to a regular stone block.
            let pos_core = ray_origin + ray_dir * t;
            result.hit = true;
            result.t = t;
            result.normal = off / max(r_w, 1e-6);
            result.color = palette[0u].rgb;  // STONE
            // Anchor cell_min/size around the hit so shade_pixel's
            // cube_face_bevel returns 1.0 (no edge darkening at the
            // smooth inner shell).
            result.cell_min = pos_core - vec3<f32>(0.5);
            result.cell_size = 1.0;
            return result;
        }
        let theta_w = asin(clamp(off.y / max(r_w, 1e-6), -1.0, 1.0));
        if abs(theta_w) > theta_cap { break; }

        var phi_w_raw = atan2(off.z, off.x);
        var phi_w = phi_w_raw;
        if phi_w < 0.0 { phi_w += UV_TWO_PI; }

        // Descend from body root to the deepest cell containing this
        // (φ, θ, r). Returns either a block (HIT) or empty-cell
        // bounds (continue stepping).
        let d = uv_descend(
            body_node_idx,
            inner_r, outer_r, theta_cap,
            phi_w, theta_w, r_w,
        );

        if d.found_block {
            // HIT. Pick the closest cell-boundary face for the
            // normal + axis tag, then bake the UV bevel into the
            // hit color so cell edges darken like 2-3-2-2's cubed-
            // sphere voxels.
            let face = uv_hit_face(off, r_w, theta_w, phi_w,
                d.phi_lo, d.phi_hi, d.theta_lo, d.theta_hi, d.r_lo, d.r_hi);
            let un_phi = clamp((phi_w - d.phi_lo) / max(d.phi_hi - d.phi_lo, 1e-12), 0.0, 1.0);
            let un_theta = clamp((theta_w - d.theta_lo) / max(d.theta_hi - d.theta_lo, 1e-12), 0.0, 1.0);
            let un_r = clamp((r_w - d.r_lo) / max(d.r_hi - d.r_lo, 1e-12), 0.0, 1.0);
            let bevel = uv_cell_bevel(un_phi, un_theta, un_r, face.axis);

            result.hit = true;
            result.t = t;
            result.normal = face.normal;
            // `(0.7 + 0.3·bevel)` matches the multiplier
            // `shade_pixel` applies for Cartesian cells, so the UV
            // body's bevel intensity is consistent with the rest of
            // the world.
            result.color = palette[d.block_type].rgb * (0.7 + 0.3 * bevel);
            // Anchor cell_min/size around the hit so `shade_pixel`'s
            // own `cube_face_bevel(local, normal)` lands at local =
            // (0.5, 0.5, 0.5) and returns 1.0 → no double-darken on
            // top of the UV bevel we just baked in.
            let pos = ray_origin + ray_dir * t;
            result.cell_min = pos - vec3<f32>(0.5);
            result.cell_size = 1.0;
            return result;
        }

        // Empty cell at depth d.depth. Step ray to the smallest t
        // exceeding the current t at any of the 6 cell boundaries.
        let bd = uv_next_boundary(
            oc, ray_dir, t,
            d.phi_lo, d.phi_hi, d.theta_lo, d.theta_hi, d.r_lo, d.r_hi,
        );
        if bd.t > 1e20 {
            // No forward boundary — ray exits, end of body.
            break;
        }
        // Advance by a small ε proportional to the step magnitude so
        // the next iteration's `pos` lands inside the neighbor cell.
        let step = bd.t - t;
        t = bd.t + max(step * 1e-4, 1e-5);
    }

    return result;
}

// Hit-face descriptor: closest cell-boundary face the ray was
// approaching, world-space normal pointing along that face's
// outward direction, and the axis tag (0=φ, 1=θ, 2=r) so callers
// can pick the in-face 2D pair for the bevel.
struct UvHitFace {
    normal: vec3<f32>,
    axis: u32,
}
fn uv_hit_face(
    off: vec3<f32>, r_w: f32, theta_w: f32, phi_w: f32,
    phi_lo: f32, phi_hi: f32,
    theta_lo: f32, theta_hi: f32,
    r_lo: f32, r_hi: f32,
) -> UvHitFace {
    // Convert parameter-space distances into world-space arc-length
    // so the comparison picks the GEOMETRICALLY closest face.
    // Δφ arc ≈ r·cos(θ)·Δφ; Δθ arc ≈ r·Δθ; Δr is already linear.
    let cos_t = cos(theta_w);
    let arc_phi_lo = r_w * cos_t * abs(phi_w - phi_lo);
    let arc_phi_hi = r_w * cos_t * abs(phi_w - phi_hi);
    let arc_th_lo = r_w * abs(theta_w - theta_lo);
    let arc_th_hi = r_w * abs(theta_w - theta_hi);
    let arc_r_lo = abs(r_w - r_lo);
    let arc_r_hi = abs(r_w - r_hi);

    // Outward normals at the hit point for each face direction.
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

// Soft-edge bevel for UV cells, analogous to `cube_face_bevel` for
// Cartesian cubes. Picks the 2D in-face coord pair perpendicular
// to the hit-face axis, then darkens at the cell edges so each
// voxel reads as a discrete (φ, θ, r) cell with right-angled edges.
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

// Stone block-type index — read from CPU-side `block::STONE`.
// Hardcoded for MVP. See `src/world/palette.rs::block` for the
// canonical assignment (STONE = 0).
fn crate_stone_index() -> u32 {
    return 0u;
}
