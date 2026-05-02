// Prototype: render ONE cartesian voxel embedded in the UV sphere.
//
// At the target spherical position `(φ_c, θ_c, r_c)`, place an
// oriented axis-aligned box whose axes are the local `(φ̂, θ̂, r̂)`
// basis at that position and whose extents match the half-widths
// of a UV cell at some chosen depth. The marcher tries this OBB
// every ray; if the ray hits the box closer than (or at all when)
// the UV march finds something, the OBB renders instead. Pixels
// outside the box keep the UV-sphere rendering.
//
// This is the smallest version of the hybrid scheme: a single
// cartesian voxel "in place of" a UV cell. If the placement +
// orientation + face shading look correct, the same machinery
// generalises to every cell at depth ≥ N.
//
// Body-frame coords: body centre at `(1.5, 1.5, 1.5)`, body radius
// in `[inner_r, outer_r]` body-frame units.

// Target — south face of the body (visible from default spawn at
// `[1.5, 1.5, 1.0]` looking +z), at the outer-shell grass band.
const PROTO_TARGET_PHI: f32   = 4.712389;   // 3π/2
const PROTO_TARGET_THETA: f32 = 0.0;        // equator
const PROTO_TARGET_R: f32     = 0.59;       // outer-shell band
// Box half-extents in spherical-tangent coords (rad / rad / radial
// world). At depth 5: dphi ≈ 0.026 rad → half ≈ 0.013. Pick a hair
// larger so the box visibly straddles the surface.
const PROTO_TARGET_HALF_DPHI: f32 = 0.025;
const PROTO_TARGET_HALF_DTH:  f32 = 0.025;
const PROTO_TARGET_HALF_DR:   f32 = 0.012;

// Box block colour — palette index 2 = grass on the demo body.
const PROTO_TARGET_BLOCK_TYPE: u32 = 2u;

// Ray-vs-OBB intersection in body-frame world coords. Returns the
// closer of (`t_enter`, axis, side) on hit, or `t = 1e30` on miss.
// Slab method projected onto the OBB's axes.
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

    // Local basis at target.
    let r_hat     = vec3<f32>( cos_t * cos_p,  sin_t,  cos_t * sin_p);
    let theta_hat = vec3<f32>(-sin_t * cos_p,  cos_t, -sin_t * sin_p);
    let phi_hat   = vec3<f32>(-sin_p,           0.0,    cos_p);

    // Box centre in body-frame world coords.
    let center = body_center + r_hat * PROTO_TARGET_R;

    // Half-extents in WORLD units along each axis.
    // Tangential extents are arc lengths: `r · cos θ · Δφ` and `r · Δθ`.
    let h_phi = PROTO_TARGET_HALF_DPHI * PROTO_TARGET_R * cos_t;
    let h_th  = PROTO_TARGET_HALF_DTH  * PROTO_TARGET_R;
    let h_r   = PROTO_TARGET_HALF_DR;

    // Ray in OBB-local coords (origin shifted, projected onto basis).
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
        var sb: u32 = 1u;
        if t_a > t_b { let tmp = t_a; t_a = t_b; t_b = tmp; sa = 1u; sb = 0u; }
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
        var sb: u32 = 1u;
        if t_a > t_b { let tmp = t_a; t_a = t_b; t_b = tmp; sa = 1u; sb = 0u; }
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
        var sb: u32 = 1u;
        if t_a > t_b { let tmp = t_a; t_a = t_b; t_b = tmp; sa = 1u; sb = 0u; }
        if t_a > t_min { t_min = t_a; enter_axis = 2u; enter_side = sa; }
        if t_b < t_max { t_max = t_b; }
        if t_min > t_max { return out; }
    }

    // Camera inside the box (`t_min < 0`): clamp to 0 — the inside
    // face entered "now" is the camera's containing one.
    if t_max < 0.0001 { return out; }
    out.t = max(t_min, 0.0001);
    out.axis = enter_axis;
    out.side = enter_side;
    return out;
}

// Render the prototype OBB at hit time. Uses the SAME
// `(r̂, θ̂, φ̂)` basis as the OBB axes. Bevel is in OBB-local
// `(u, v) ∈ [-1, 1]` of the TWO axes orthogonal to the entry face.
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

    // Face normal: ±basis along the entered axis.
    var n: vec3<f32>;
    if bd.axis == 0u {
        n = select(-phi_hat, phi_hat, bd.side == 1u);
    } else if bd.axis == 1u {
        n = select(-theta_hat, theta_hat, bd.side == 1u);
    } else {
        n = select(-r_hat, r_hat, bd.side == 1u);
    }
    result.normal = normalize(n);

    // Bevel from OBB-local (u, v, w). Each component ∈ [-1, 1] for
    // points inside the box; the entered axis is at ±1.
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

    result.color = palette[PROTO_TARGET_BLOCK_TYPE].rgb * (0.7 + 0.3 * bevel);
    result.cell_min = pos - vec3<f32>(0.5);
    return result;
}
