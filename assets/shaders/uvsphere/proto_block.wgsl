// Prototype: replace ONE specific UV cell with a cartesian voxel.
//
// Target = a real cell in the body's tree at a tier-aligned path.
// The OBB's centre / basis / extents are derived from the cell's
// EXACT bounds, so it sits in the same volume the UV cell would
// occupy — no floating off the surface, no clipping into the
// sphere. If this looks right, the same primitive generalises:
// every Block with `d.depth ≥ N` uses its own bounds for centre /
// basis / extents and replaces its UV rendering pixel-for-pixel.

// --- Target cell selection -------------------------------------
//
// Path = [14, 21, 23] (depth 3) — slot encoding `pt + tt*3 + rt*9`:
//   d=1: pt=2 (φ ∈ [4π/3, 2π]), tt=1 (θ ∈ [-θcap/3, θcap/3]),
//        rt=1 (r ∈ [inner_r + dr/3, inner_r + 2dr/3]).
//   d=2: pt=0 (φ ∈ [4π/3, 4π/3 + 2π/9]), tt=1, rt=2.
//   d=3: pt=2, tt=1, rt=2 — outermost r-slot at depth 3.
//
// Cell bounds (in body-marcher local frame, where `body_size=3.0`,
// inner_r=0.15, outer_r=0.60, θcap=80°≈1.396):
//   φ ∈ [4.654, 4.887]      (≈ south face, slightly east of -z)
//   θ ∈ [-0.052, 0.052]     (equator-centred, depth-3 width)
//   r ∈ [0.4333, 0.45]      (top of grass band — visible on surface)
//
// Why this cell and not the original mid-shell `[14, 21, 5]`:
// the body's grass shell at r ∈ [0.4275, 0.45] sits in FRONT of the
// dirt band at r ∈ [0.315, 0.4275]. Targeting a dirt cell hides
// the OBB behind grass. The outermost r-slot (rt=2 at depth 3) lies
// IN the grass band, on the body's silhouette — the OBB replaces a
// real grass cell pixel-for-pixel.
//
// Cell centre in body-frame world coords ≈ (1.526, 1.5, 1.059).
// Camera at (1.5, 1.5, 0.5) looking +z (test setup) puts the target
// ~1° off forward.
const PROTO_TARGET_PHI:        f32 = 4.7705;       // (4.654 + 4.887)/2
const PROTO_TARGET_THETA:      f32 = 0.0;
const PROTO_TARGET_R:          f32 = 0.4417;       // (0.4333 + 0.45)/2

const PROTO_TARGET_HALF_DPHI:  f32 = 0.1164;       // (4.887 − 4.654)/2
const PROTO_TARGET_HALF_DTH:   f32 = 0.052;        // (0.052 − (−0.052))/2
const PROTO_TARGET_HALF_DR:    f32 = 0.0083;       // (0.45 − 0.4333)/2

// Bright magenta — unmistakable against the body's green/brown.
const PROTO_TARGET_COLOR: vec3<f32> = vec3<f32>(1.0, 0.0, 1.0);

// Ray-vs-OBB intersection (slab method projected onto the OBB
// axes). Returns the closer of `(t_enter, axis, side)` on hit, or
// `t = 1e30` on miss.
fn proto_ray_vs_obb(
    ray_origin: vec3<f32>, ray_dir: vec3<f32>,
    body_center: vec3<f32>,
) -> UvBoundaryHit {
    var out: UvBoundaryHit;
    out.t = 1e30;
    out.axis = 0u;
    out.side = 0u;

    let cos_p = cos(PROTO_TARGET_PHI);
    let sin_p = sin(PROTO_TARGET_PHI);
    let cos_t = cos(PROTO_TARGET_THETA);
    let sin_t = sin(PROTO_TARGET_THETA);

    // Local basis at target — matches the cell's tangent + radial frame.
    let r_hat     = vec3<f32>( cos_t * cos_p,  sin_t,  cos_t * sin_p);
    let theta_hat = vec3<f32>(-sin_t * cos_p,  cos_t, -sin_t * sin_p);
    let phi_hat   = vec3<f32>(-sin_p,           0.0,    cos_p);

    let center = body_center + r_hat * PROTO_TARGET_R;

    // Half-extents in WORLD units. Tangential extents are arc lengths
    // `r · cos θ · Δφ` and `r · Δθ`; radial extent is just `Δr`.
    let h_phi = PROTO_TARGET_HALF_DPHI * PROTO_TARGET_R * cos_t;
    let h_th  = PROTO_TARGET_HALF_DTH  * PROTO_TARGET_R;
    let h_r   = PROTO_TARGET_HALF_DR;

    // Ray in OBB-local: project (origin − centre) onto the basis,
    // and the ray dir similarly.
    let to_origin = ray_origin - center;
    let q0 = dot(to_origin, phi_hat);
    let q1 = dot(to_origin, theta_hat);
    let q2 = dot(to_origin, r_hat);
    let d0 = dot(ray_dir, phi_hat);
    let d1 = dot(ray_dir, theta_hat);
    let d2 = dot(ray_dir, r_hat);

    var t_min: f32 = -1e30;
    var t_max: f32 =  1e30;
    var enter_axis: u32 = 0u;
    var enter_side: u32 = 0u;

    // Slab axis 0 (φ̂).
    if abs(d0) < 1e-12 {
        if abs(q0) > h_phi { return out; }
    } else {
        let inv_d = 1.0 / d0;
        var t_a = (-h_phi - q0) * inv_d;
        var t_b = ( h_phi - q0) * inv_d;
        var sa: u32 = 0u;
        if t_a > t_b { let tmp = t_a; t_a = t_b; t_b = tmp; sa = 1u; }
        if t_a > t_min { t_min = t_a; enter_axis = 0u; enter_side = sa; }
        if t_b < t_max { t_max = t_b; }
        if t_min > t_max { return out; }
    }
    // Slab axis 1 (θ̂).
    if abs(d1) < 1e-12 {
        if abs(q1) > h_th { return out; }
    } else {
        let inv_d = 1.0 / d1;
        var t_a = (-h_th - q1) * inv_d;
        var t_b = ( h_th - q1) * inv_d;
        var sa: u32 = 0u;
        if t_a > t_b { let tmp = t_a; t_a = t_b; t_b = tmp; sa = 1u; }
        if t_a > t_min { t_min = t_a; enter_axis = 1u; enter_side = sa; }
        if t_b < t_max { t_max = t_b; }
        if t_min > t_max { return out; }
    }
    // Slab axis 2 (r̂).
    if abs(d2) < 1e-12 {
        if abs(q2) > h_r { return out; }
    } else {
        let inv_d = 1.0 / d2;
        var t_a = (-h_r - q2) * inv_d;
        var t_b = ( h_r - q2) * inv_d;
        var sa: u32 = 0u;
        if t_a > t_b { let tmp = t_a; t_a = t_b; t_b = tmp; sa = 1u; }
        if t_a > t_min { t_min = t_a; enter_axis = 2u; enter_side = sa; }
        if t_b < t_max { t_max = t_b; }
        if t_min > t_max { return out; }
    }

    if t_max < 0.0001 { return out; }
    out.t = max(t_min, 0.0001);
    out.axis = enter_axis;
    out.side = enter_side;
    return out;
}

// Render the prototype OBB at hit time. The OBB axes are
// `(φ̂, θ̂, r̂)` at the target's centre — same basis the slab test
// used. Bevel is in OBB-local `(u, v) ∈ [-1, 1]` of the two axes
// orthogonal to the entered face.
fn proto_obb_render(
    ray_origin: vec3<f32>, ray_dir: vec3<f32>,
    body_center: vec3<f32>,
    bd: UvBoundaryHit,
) -> HitResult {
    var result: HitResult;
    result.hit = true;
    result.t = bd.t;
    result.frame_level = 0u;
    result.frame_scale = 1.0;
    result.cell_size = 1.0;

    let cos_p = cos(PROTO_TARGET_PHI);
    let sin_p = sin(PROTO_TARGET_PHI);
    let cos_t = cos(PROTO_TARGET_THETA);
    let sin_t = sin(PROTO_TARGET_THETA);
    let r_hat     = vec3<f32>( cos_t * cos_p,  sin_t,  cos_t * sin_p);
    let theta_hat = vec3<f32>(-sin_t * cos_p,  cos_t, -sin_t * sin_p);
    let phi_hat   = vec3<f32>(-sin_p,           0.0,    cos_p);
    let center = body_center + r_hat * PROTO_TARGET_R;
    let h_phi = PROTO_TARGET_HALF_DPHI * PROTO_TARGET_R * cos_t;
    let h_th  = PROTO_TARGET_HALF_DTH  * PROTO_TARGET_R;
    let h_r   = PROTO_TARGET_HALF_DR;

    var n: vec3<f32>;
    if bd.axis == 0u {
        n = select(-phi_hat, phi_hat, bd.side == 1u);
    } else if bd.axis == 1u {
        n = select(-theta_hat, theta_hat, bd.side == 1u);
    } else {
        n = select(-r_hat, r_hat, bd.side == 1u);
    }
    result.normal = normalize(n);

    let pos = ray_origin + ray_dir * bd.t;
    let p = pos - center;
    let u = clamp(dot(p, phi_hat)   / max(h_phi, 1e-12), -1.0, 1.0);
    let v = clamp(dot(p, theta_hat) / max(h_th,  1e-12), -1.0, 1.0);
    let w = clamp(dot(p, r_hat)     / max(h_r,   1e-12), -1.0, 1.0);
    let un_u = u * 0.5 + 0.5;
    let un_v = v * 0.5 + 0.5;
    let un_w = w * 0.5 + 0.5;
    var a: f32; var b: f32;
    if bd.axis == 0u { a = un_v; b = un_w; }
    else if bd.axis == 1u { a = un_u; b = un_w; }
    else { a = un_u; b = un_v; }
    let edge = min(min(a, 1.0 - a), min(b, 1.0 - b));
    let bevel = smoothstep(0.02, 0.14, edge);

    result.color = PROTO_TARGET_COLOR * (0.7 + 0.3 * bevel);
    result.cell_min = pos - vec3<f32>(0.5);
    return result;
}
