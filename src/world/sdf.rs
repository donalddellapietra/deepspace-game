//! Vector helpers used by camera/raycast/edit code.
//!
//! Originally hosted SDF primitives (notably the displaced-sphere
//! `Planet` SDF used by the cubed-sphere worldgen). The cubed-sphere
//! stack was deleted; this module retains only the small Vec3
//! helpers and `tangent_basis`, which several non-sphere callers use.

pub type Vec3 = [f32; 3];

#[inline]
pub fn add(a: Vec3, b: Vec3) -> Vec3 { [a[0]+b[0], a[1]+b[1], a[2]+b[2]] }
#[inline]
pub fn sub(a: Vec3, b: Vec3) -> Vec3 { [a[0]-b[0], a[1]-b[1], a[2]-b[2]] }
#[inline]
pub fn scale(a: Vec3, s: f32) -> Vec3 { [a[0]*s, a[1]*s, a[2]*s] }
#[inline]
pub fn dot(a: Vec3, b: Vec3) -> f32 { a[0]*b[0]+a[1]*b[1]+a[2]*b[2] }
#[inline]
pub fn length(a: Vec3) -> f32 { dot(a, a).sqrt() }
#[inline]
pub fn normalize(a: Vec3) -> Vec3 {
    let l = length(a);
    if l > 1e-12 { scale(a, 1.0/l) } else { [0.0, 1.0, 0.0] }
}

/// Orthonormal basis in the plane perpendicular to `up`. Picks a
/// stable "forward" by projecting world -Z onto that plane (falls
/// back to world +X if degenerate when `up` is ±Z). Returns
/// `(right, forward)`.
pub fn tangent_basis(up: Vec3) -> (Vec3, Vec3) {
    let ref_fwd = [0.0, 0.0, -1.0];
    let d = dot(ref_fwd, up);
    let fwd_unnorm = sub(ref_fwd, scale(up, d));
    let fwd = if length(fwd_unnorm) < 0.01 {
        let alt = [1.0, 0.0, 0.0];
        let d2 = dot(alt, up);
        normalize(sub(alt, scale(up, d2)))
    } else {
        normalize(fwd_unnorm)
    };
    // right = fwd × up
    let right = [
        fwd[1] * up[2] - fwd[2] * up[1],
        fwd[2] * up[0] - fwd[0] * up[2],
        fwd[0] * up[1] - fwd[1] * up[0],
    ];
    (normalize(right), fwd)
}
