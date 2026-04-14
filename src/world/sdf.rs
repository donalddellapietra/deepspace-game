//! Vec3 helpers + 3D value noise used by worldgen.

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

/// Orthonormal basis in the plane perpendicular to `up`. Returns `(right, forward)`.
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
    let right = [
        fwd[1] * up[2] - fwd[2] * up[1],
        fwd[2] * up[0] - fwd[0] * up[2],
        fwd[0] * up[1] - fwd[1] * up[0],
    ];
    (normalize(right), fwd)
}

/// Trilinear-interpolated value noise in [-1, 1]. Deterministic in seed.
pub fn noise3d(p: Vec3, seed: u32) -> f32 {
    let (xi, xf) = floor_frac(p[0]);
    let (yi, yf) = floor_frac(p[1]);
    let (zi, zf) = floor_frac(p[2]);
    let ux = smoothstep(xf);
    let uy = smoothstep(yf);
    let uz = smoothstep(zf);
    let h = |ix: i32, iy: i32, iz: i32| -> f32 { hash_lattice(ix, iy, iz, seed) };
    let c000 = h(xi,   yi,   zi);
    let c100 = h(xi+1, yi,   zi);
    let c010 = h(xi,   yi+1, zi);
    let c110 = h(xi+1, yi+1, zi);
    let c001 = h(xi,   yi,   zi+1);
    let c101 = h(xi+1, yi,   zi+1);
    let c011 = h(xi,   yi+1, zi+1);
    let c111 = h(xi+1, yi+1, zi+1);
    let x00 = lerp(c000, c100, ux);
    let x10 = lerp(c010, c110, ux);
    let x01 = lerp(c001, c101, ux);
    let x11 = lerp(c011, c111, ux);
    let y0 = lerp(x00, x10, uy);
    let y1 = lerp(x01, x11, uy);
    lerp(y0, y1, uz)
}

fn floor_frac(x: f32) -> (i32, f32) {
    let f = x.floor();
    (f as i32, x - f)
}
fn smoothstep(t: f32) -> f32 { t * t * (3.0 - 2.0 * t) }
fn lerp(a: f32, b: f32, t: f32) -> f32 { a + (b - a) * t }

fn hash_lattice(ix: i32, iy: i32, iz: i32, seed: u32) -> f32 {
    let mut h = seed;
    h = h.wrapping_mul(374761393).wrapping_add(ix as u32);
    h = h.wrapping_mul(668265263).wrapping_add(iy as u32);
    h = h.wrapping_mul(1274126177).wrapping_add(iz as u32);
    h ^= h >> 13;
    h = h.wrapping_mul(1274126177);
    h ^= h >> 16;
    ((h & 0xFFFF) as f32) / 32768.0 - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noise_in_range() {
        for i in 0..1000 {
            let x = i as f32 * 0.137;
            let n = noise3d([x, x * 1.3, x * 0.7], 42);
            assert!(n >= -1.0 && n <= 1.0);
        }
    }

    #[test]
    fn noise_deterministic() {
        assert_eq!(noise3d([1.0, 2.0, 3.0], 42), noise3d([1.0, 2.0, 3.0], 42));
    }

    #[test]
    fn tangent_basis_is_orthonormal() {
        let up = normalize([0.3, 0.8, 0.5]);
        let (r, f) = tangent_basis(up);
        assert!(dot(r, up).abs() < 1e-5);
        assert!(dot(f, up).abs() < 1e-5);
        assert!(dot(r, f).abs() < 1e-5);
        assert!((length(r) - 1.0).abs() < 1e-5);
        assert!((length(f) - 1.0).abs() < 1e-5);
    }
}
