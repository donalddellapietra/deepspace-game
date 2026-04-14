//! Signed-distance fields for procedural generation.
//!
//! `Planet` is a sphere displaced by 3D value noise — the same shape
//! the Godot GDVoxelTerrain plugin uses for planets. Content here is
//! pure (no Godot deps): sampled at block-center positions by the
//! worldgen to decide solid/empty, and used at runtime to compute
//! radial gravity fields for collision.

use super::palette::block;

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

// ───────────────────────────────────────────────────────── Planet SDF

/// A planet: a sphere of `radius` centered at `center`, with its
/// surface displaced by 3D value noise of amplitude `noise_scale`.
///
/// All lengths are in world space. Gravity pulls toward `center` with
/// acceleration `gravity` at the surface, falling off linearly to
/// zero at `influence_radius`. Outside `influence_radius` this planet
/// contributes no gravity (allowing flight / space outside).
#[derive(Clone, Debug)]
pub struct Planet {
    pub center: Vec3,
    pub radius: f32,
    pub noise_scale: f32,
    pub noise_freq: f32,
    pub noise_seed: u32,
    pub gravity: f32,
    pub influence_radius: f32,
    pub surface_block: u8,
    pub core_block: u8,
}

impl Planet {
    /// SDF: <0 inside, 0 on surface, >0 outside. Negative value ≈
    /// how deep the point sits below the displaced sphere surface.
    pub fn distance(&self, p: Vec3) -> f32 {
        let to = sub(p, self.center);
        let base = length(to) - self.radius;
        if base > self.noise_scale * 1.25 { return base; }
        base - self.displacement(p)
    }

    /// Displacement of the surface at p (world space). Noise is
    /// evaluated in world space and scaled. Asymmetric: peaks go up
    /// higher than valleys go down, like the Godot plugin's planet.
    pub fn displacement(&self, p: Vec3) -> f32 {
        let n = noise3d(scale(p, self.noise_freq), self.noise_seed);
        if n < 0.0 { 0.5 * n * self.noise_scale } else { n * self.noise_scale }
    }

    /// Block type at point p. Top of surface = surface block,
    /// deep interior = core block. Smooth biome transition by depth.
    pub fn block_at(&self, p: Vec3) -> u8 {
        let to = sub(p, self.center);
        let r = length(to);
        // Depth below the undisplaced radius.
        let under = self.radius - r;
        let grass_band = self.noise_scale * 0.15;
        let dirt_band = self.noise_scale * 0.6;
        if under < grass_band { self.surface_block }
        else if under < dirt_band { block::DIRT }
        else { self.core_block }
    }

    /// Gravity acceleration vector at p (world units / s²).
    /// Falls off linearly from full `gravity` at surface to 0 at
    /// `influence_radius`. Returns [0;3] beyond influence.
    pub fn gravity_at(&self, p: Vec3) -> Vec3 {
        let to = sub(self.center, p);
        let r = length(to);
        if r > self.influence_radius || r < 1e-8 { return [0.0, 0.0, 0.0]; }
        let up_to_center = scale(to, 1.0 / r);
        // Full gravity from surface outward to influence_radius, ramps down.
        let t = if r < self.radius {
            1.0
        } else {
            let x = (r - self.radius) / (self.influence_radius - self.radius).max(1e-6);
            (1.0 - x).clamp(0.0, 1.0)
        };
        scale(up_to_center, self.gravity * t)
    }

    /// Unit "up" at p (away from planet center). For player orientation.
    pub fn up_at(&self, p: Vec3) -> Vec3 {
        let to = sub(p, self.center);
        let l = length(to);
        if l > 1e-8 { scale(to, 1.0 / l) } else { [0.0, 1.0, 0.0] }
    }
}

// ───────────────────────────────────────────────────────── 3D value noise

/// Trilinear-interpolated value noise in [-1, 1]. Deterministic in
/// seed. One lattice per unit of p — scale the input to change
/// spatial frequency.
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
    // Map to [-1, 1].
    ((h & 0xFFFF) as f32) / 32768.0 - 1.0
}

// ───────────────────────────────────────────────────────── tests

#[cfg(test)]
mod tests {
    use super::*;

    fn test_planet() -> Planet {
        Planet {
            center: [1.5, 1.5, 1.5],
            radius: 0.4,
            noise_scale: 0.05,
            noise_freq: 50.0,
            noise_seed: 42,
            gravity: 20.0,
            influence_radius: 0.8,
            surface_block: block::GRASS,
            core_block: block::STONE,
        }
    }

    #[test]
    fn sdf_inside_center_is_negative() {
        let p = test_planet();
        assert!(p.distance(p.center) < 0.0);
    }

    #[test]
    fn sdf_far_outside_is_positive() {
        let p = test_planet();
        assert!(p.distance([2.9, 1.5, 1.5]) > 0.0);
    }

    #[test]
    fn sdf_near_surface_small() {
        let p = test_planet();
        // A point exactly at radius from center (before displacement).
        let d = p.distance([p.center[0] + p.radius, p.center[1], p.center[2]]);
        assert!(d.abs() < p.noise_scale * 1.1);
    }

    #[test]
    fn sdf_early_exit_matches_base() {
        // Well outside noise influence, distance should equal base.
        let p = test_planet();
        let q = [p.center[0] + p.radius + p.noise_scale * 3.0, p.center[1], p.center[2]];
        let base = length(sub(q, p.center)) - p.radius;
        assert!((p.distance(q) - base).abs() < 1e-5);
    }

    #[test]
    fn gravity_points_toward_center_on_surface() {
        let p = test_planet();
        let q = [p.center[0] + p.radius, p.center[1], p.center[2]];
        let g = p.gravity_at(q);
        // g points in -x (toward center).
        assert!(g[0] < 0.0);
        assert!((length(g) - p.gravity).abs() < 0.01);
    }

    #[test]
    fn gravity_zero_far_away() {
        let p = test_planet();
        let q = [p.center[0] + p.influence_radius + 1.0, p.center[1], p.center[2]];
        let g = p.gravity_at(q);
        assert_eq!(g, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn up_vector_is_unit() {
        let p = test_planet();
        let u = p.up_at([p.center[0] + 0.1, p.center[1] + 0.2, p.center[2] - 0.3]);
        assert!((length(u) - 1.0).abs() < 1e-5);
    }

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
}
