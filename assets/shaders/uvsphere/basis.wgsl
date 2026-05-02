// Spherical basis math.
//
// At a point on the body's spherical coords `(φ, θ, r)`, the local
// orthonormal basis is
//
//     r̂ = ( cos θ cos φ,  sin θ,  cos θ sin φ )
//     θ̂ = (-sin θ cos φ,  cos θ, -sin θ sin φ )
//     φ̂ = (-sin φ,        0,      cos φ )
//
// `r̂ ⊥ θ̂ ⊥ φ̂`; together they map cell-local UV displacement to
// body-frame cartesian. The Jacobian inverse — used to convert a
// world-space ray dir into per-axis cell-local rates — is
//
//     d_phi/dt   = (ray · φ̂) / (r · cos θ)
//     d_theta/dt = (ray · θ̂) /  r
//     d_r/dt     = (ray · r̂)
//
// Both are evaluated at the cell's centre; the basis is essentially
// constant within a deep cell (variation `O(dphi)` per cell, < 1°
// at depth 5+).

struct UvBasis {
    r_hat: vec3<f32>,
    theta_hat: vec3<f32>,
    phi_hat: vec3<f32>,
    inv_r: f32,
    inv_r_cos_theta: f32,
}

fn uv_basis_at(phi: f32, theta: f32, r: f32) -> UvBasis {
    let cos_p = cos(phi);
    let sin_p = sin(phi);
    let cos_t = cos(theta);
    let sin_t = sin(theta);
    var b: UvBasis;
    b.r_hat     = vec3<f32>( cos_t * cos_p,  sin_t,  cos_t * sin_p);
    b.theta_hat = vec3<f32>(-sin_t * cos_p,  cos_t, -sin_t * sin_p);
    b.phi_hat   = vec3<f32>(-sin_p,           0.0,    cos_p);
    b.inv_r = 1.0 / max(r, 1e-6);
    b.inv_r_cos_theta = 1.0 / max(r * cos_t, 1e-6);
    return b;
}

// Convert a body-frame cartesian ray direction to per-axis cell-local
// rates `(d_un_phi, d_un_theta, d_un_r)` at a cell whose extents are
// `(dphi, dth, dr)`. `un_*` is the cell-local fraction in `[0, 1]`,
// so `d_un_*` is the rate at which `un_*` advances per unit world
// time `t` (where `world_pos = ray_origin + ray_dir * t`).
fn uv_d_un(
    basis: UvBasis,
    ray_dir: vec3<f32>,
    dphi: f32, dth: f32, dr: f32,
) -> vec3<f32> {
    let d_phi   = dot(ray_dir, basis.phi_hat)   * basis.inv_r_cos_theta;
    let d_theta = dot(ray_dir, basis.theta_hat) * basis.inv_r;
    let d_r     = dot(ray_dir, basis.r_hat);
    return vec3<f32>(
        d_phi   / max(dphi, 1e-30),
        d_theta / max(dth,  1e-30),
        d_r     / max(dr,   1e-30),
    );
}
