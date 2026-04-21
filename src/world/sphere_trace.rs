//! Curved-space sphere-tracing through the cube-to-sphere remap.
//!
//! A "sphere body" is a volume whose content lives in cube coords
//! `[-1, 1]^3` but renders as the unit ball in world coords via the
//! `sphere_remap::forward` map F. This module ray-marches such a
//! body: world-space rays are transformed into cube coords at each
//! step (Newton on F), the occupancy is queried in cube space, and
//! safe world-space steps are derived from cube-space safe distances
//! scaled by σ_min(J(c)).
//!
//! Occupancy is supplied via a trait so this module is independent of
//! the tree implementation. Tests use analytic occupancies (solid,
//! hollow, shell) to verify correctness against ray-sphere geometry.
//!
//! Scope: this is the Layer-2 CPU reference. The same algorithm ports
//! to WGSL for the renderer — kept pure and short so the port is
//! mechanical.

use super::sphere_remap::{self, V3};

// ---------- occupancy interface ----------

/// Occupancy query for a sphere body. Callers implement this to
/// expose their data structure (analytic field, voxel tree, brickmap,
/// …) to the sphere-tracer.
pub trait CubeOccupancy {
    /// Returns `(is_filled, safe_cube_distance)`.
    ///
    /// `c` is a cube-space coord in `[-1, 1]^3`. `is_filled` is the
    /// occupancy of the cell containing `c`. `safe_cube_distance` is
    /// a conservative cube-space distance to the nearest occupancy
    /// change (i.e. how far along the ray we can advance in cube
    /// space before crossing in/out of occupancy).
    ///
    /// Non-negative. Returning zero will make the trace stall.
    fn query(&self, c: V3) -> (bool, f32);
}

// ---------- results ----------

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphereHit {
    /// Distance along normalized ray direction in world space.
    pub t_world: f32,
    /// World-space hit position.
    pub pos_world: V3,
    /// Cube-space position at hit.
    pub c_cube: V3,
    /// World-space surface normal (outward).
    pub world_normal: V3,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct TraceStats {
    pub steps: u32,
    pub newton_fails: u32,
}

// ---------- the trace ----------

pub struct TraceConfig {
    /// Upper bound on march steps per ray.
    pub max_steps: u32,
    /// Minimum world-space step taken when the Jacobian pinches.
    /// Prevents stalls near cube corners/edges.
    pub min_world_step: f32,
    /// Epsilon added to ray-ball entry to avoid starting on the
    /// cube's exact surface (where σ_min(J) = 0).
    pub entry_offset: f32,
    /// Floor on σ_min to avoid divide-by-small when multiplying a
    /// safe cube distance by it. Any point with σ_min below this gets
    /// treated as σ_min = this value for step sizing.
    pub sigma_floor: f32,
    /// Newton iteration budget for F^-1 per step. 4 is comfortable.
    pub newton_iters: u32,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            max_steps: 512,
            min_world_step: 1e-4,
            entry_offset: 1e-3,
            sigma_floor: 0.02,
            newton_iters: 4,
        }
    }
}

/// Ray-march a sphere body. The body is the unit ball in world space.
///
/// `ray_origin_world` and `ray_dir_world` define a world-space ray;
/// `ray_dir_world` is normalized internally.
pub fn trace<O: CubeOccupancy>(
    occupancy: &O,
    ray_origin_world: V3,
    ray_dir_world: V3,
    cfg: &TraceConfig,
) -> (Option<SphereHit>, TraceStats) {
    let mut stats = TraceStats::default();

    // Normalize ray so t parameterizes world-space arc length.
    let dmag2 = ray_dir_world[0] * ray_dir_world[0]
        + ray_dir_world[1] * ray_dir_world[1]
        + ray_dir_world[2] * ray_dir_world[2];
    if dmag2 < 1e-20 {
        return (None, stats);
    }
    let inv_dmag = 1.0 / dmag2.sqrt();
    let d = [
        ray_dir_world[0] * inv_dmag,
        ray_dir_world[1] * inv_dmag,
        ray_dir_world[2] * inv_dmag,
    ];

    // 1) Ray vs. unit ball in world space.
    let Some((t_enter, t_exit)) = ray_unit_ball(ray_origin_world, d) else {
        return (None, stats);
    };
    if t_exit < 0.0 {
        return (None, stats);
    }

    let mut t = t_enter.max(0.0) + cfg.entry_offset;
    let t_max = t_exit;

    // 2) March. Warm-start Newton from previous cube coord.
    // Seed with the cube coord at the ball entry: for rays starting
    // outside, that's approximately the entry w itself (since F is
    // identity on axes and near-identity near the origin).
    let w_entry = [
        ray_origin_world[0] + d[0] * t,
        ray_origin_world[1] + d[1] * t,
        ray_origin_world[2] + d[2] * t,
    ];
    let mut c_warm = w_entry;

    // Track previous step's world-space t so when we hit, we can
    // refine via a single-step back-off for the world-space hit t.
    let mut prev_t = t;
    let mut prev_filled = false;

    while stats.steps < cfg.max_steps {
        stats.steps += 1;

        if t > t_max + cfg.min_world_step {
            return (None, stats);
        }

        let w = [
            ray_origin_world[0] + d[0] * t,
            ray_origin_world[1] + d[1] * t,
            ray_origin_world[2] + d[2] * t,
        ];

        // F^-1(w), warm-started.
        let c = match sphere_remap::inverse_from(w, c_warm, cfg.newton_iters) {
            Some(c) => c,
            None => {
                // Newton diverged (e.g. near cube edge) — take a tiny
                // fallback step and retry.
                stats.newton_fails += 1;
                prev_t = t;
                prev_filled = false;
                t += cfg.min_world_step;
                continue;
            }
        };
        c_warm = c;

        let (filled, safe_cube) = occupancy.query(c);

        if filled && !prev_filled {
            // Crossed empty → filled between prev_t and t.
            // Refine via linear interpolation on world-space t.
            // (Better: bisect — cheap to add later.)
            let t_hit = if stats.steps == 1 {
                t // first step already inside occupancy
            } else {
                0.5 * (prev_t + t)
            };
            let w_hit = [
                ray_origin_world[0] + d[0] * t_hit,
                ray_origin_world[1] + d[1] * t_hit,
                ray_origin_world[2] + d[2] * t_hit,
            ];
            // Refine cube coord at the hit.
            let c_hit = sphere_remap::inverse_from(w_hit, c_warm, cfg.newton_iters).unwrap_or(c);
            // Outward world-space normal: for now use the radial
            // direction (world position normalized). This is correct
            // for the body's outer surface; for interior walls we'd
            // reconstruct from the cube face normal via J^-T.
            let wmag = (w_hit[0] * w_hit[0] + w_hit[1] * w_hit[1] + w_hit[2] * w_hit[2])
                .sqrt()
                .max(1e-12);
            let normal = [w_hit[0] / wmag, w_hit[1] / wmag, w_hit[2] / wmag];
            return (
                Some(SphereHit {
                    t_world: t_hit,
                    pos_world: w_hit,
                    c_cube: c_hit,
                    world_normal: normal,
                }),
                stats,
            );
        }

        if filled {
            // Started inside; keep advancing until we exit and re-enter,
            // or back off — for simplicity treat as hit here if it's the
            // first step (ray origin is inside occupancy).
            if stats.steps == 1 {
                let wmag = (w[0] * w[0] + w[1] * w[1] + w[2] * w[2])
                    .sqrt()
                    .max(1e-12);
                let normal = [w[0] / wmag, w[1] / wmag, w[2] / wmag];
                return (
                    Some(SphereHit {
                        t_world: t,
                        pos_world: w,
                        c_cube: c,
                        world_normal: normal,
                    }),
                    stats,
                );
            }
        }

        // Safe step: σ_min(J(c)) · safe_cube, floored both by a
        // per-step minimum and a σ_min floor.
        let sigma_raw = sphere_remap::sigma_min(c);
        let sigma = sigma_raw.max(cfg.sigma_floor);
        let step = (safe_cube * sigma).max(cfg.min_world_step);

        prev_t = t;
        prev_filled = filled;

        t += step;
    }

    (None, stats)
}

// ---------- analytic ray-ball ----------

/// Ray-vs-unit-ball. `dir` is assumed normalized.
pub fn ray_unit_ball(origin: V3, dir: V3) -> Option<(f32, f32)> {
    // |origin + t·dir|² = 1, with |dir|² = 1.
    let od = origin[0] * dir[0] + origin[1] * dir[1] + origin[2] * dir[2];
    let oo = origin[0] * origin[0] + origin[1] * origin[1] + origin[2] * origin[2];
    let disc = od * od - (oo - 1.0);
    if disc < 0.0 {
        return None;
    }
    let s = disc.sqrt();
    Some((-od - s, -od + s))
}

// ---------- test occupancies ----------

#[cfg(test)]
pub mod test_occupancies {
    use super::*;

    /// Everywhere filled. The ray just needs to enter the ball to hit.
    pub struct Solid;
    impl CubeOccupancy for Solid {
        fn query(&self, _c: V3) -> (bool, f32) {
            (true, 1.0)
        }
    }

    /// Nowhere filled. Ray always misses.
    pub struct Hollow;
    impl CubeOccupancy for Hollow {
        fn query(&self, _c: V3) -> (bool, f32) {
            // Safe distance = large (ray can skip the whole interior),
            // but σ_min will still throttle near corners.
            (false, 2.0)
        }
    }

    /// Filled when `|c|_infty >= inner`, empty otherwise. This is a
    /// "shell" whose inner boundary is the cube `[-inner, inner]^3`
    /// (remapped to a warped inner surface in world space).
    pub struct Shell {
        pub inner: f32,
    }
    impl CubeOccupancy for Shell {
        fn query(&self, c: V3) -> (bool, f32) {
            let m = c[0].abs().max(c[1].abs()).max(c[2].abs());
            let filled = m >= self.inner;
            // Safe cube distance = |m − inner| (distance to boundary
            // in cube L∞). Conservative: always non-negative.
            let safe = (m - self.inner).abs().max(1e-4);
            (filled, safe)
        }
    }
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use test_occupancies::*;

    fn normalize(v: V3) -> V3 {
        let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt().max(1e-12);
        [v[0] / m, v[1] / m, v[2] / m]
    }

    // ---- ray-ball helper ----

    #[test]
    fn ray_unit_ball_hits_along_axis() {
        // ray from (-2, 0, 0) along +x: enters at t=1, exits at t=3.
        let (a, b) = ray_unit_ball([-2.0, 0.0, 0.0], [1.0, 0.0, 0.0]).unwrap();
        assert!((a - 1.0).abs() < 1e-5, "enter={a}");
        assert!((b - 3.0).abs() < 1e-5, "exit={b}");
    }

    #[test]
    fn ray_unit_ball_misses() {
        assert!(ray_unit_ball([0.0, 2.0, 0.0], [1.0, 0.0, 0.0]).is_none());
    }

    // ---- solid: parity with ray-ball ----

    #[test]
    fn solid_occupancy_hits_at_ball_entry() {
        let cfg = TraceConfig::default();
        // Along +x axis (avoids corner singularity in Newton).
        let (hit, stats) = trace(&Solid, [-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], &cfg);
        let h = hit.expect("solid sphere should hit from outside");
        eprintln!("solid axis: hit.t={:.4}, steps={}", h.t_world, stats.steps);
        assert!(
            (h.t_world - 1.0).abs() < 5e-3,
            "hit.t = {} (expected ≈1.0)",
            h.t_world
        );
        // normal points outward along -x (toward camera)
        let n = h.world_normal;
        assert!(n[0] < -0.99, "normal not outward radial: {:?}", n);
    }

    #[test]
    fn solid_occupancy_oblique_rays_hit_correct_t() {
        let cfg = TraceConfig::default();
        // Fire rays from a distance-2 sphere toward origin at various
        // directions; they should all hit at t ≈ 1 (unit ball surface).
        for seed in 0..50_u64 {
            // pseudo-random unit direction
            let a = (seed as f32) * 0.137 + 0.1;
            let b = (seed as f32) * 0.071 + 0.23;
            let dir_u = [a.sin() * b.cos(), a.sin() * b.sin(), a.cos()];
            let dir = normalize(dir_u);
            // origin is at -2·dir (distance 2 from origin, pointing inward)
            let origin = [-2.0 * dir[0], -2.0 * dir[1], -2.0 * dir[2]];
            let (hit, stats) = trace(&Solid, origin, dir, &cfg);
            let h = hit.unwrap_or_else(|| {
                panic!("seed={seed} dir={:?}: expected hit, stats={:?}", dir, stats)
            });
            assert!(
                (h.t_world - 1.0).abs() < 0.02,
                "seed={seed}: t_world={} expected ≈1.0 (steps={})",
                h.t_world,
                stats.steps
            );
        }
    }

    // ---- hollow: should miss ----

    #[test]
    fn hollow_occupancy_misses_all_rays() {
        let cfg = TraceConfig {
            max_steps: 1024,
            ..TraceConfig::default()
        };
        // Same axis-aligned ray: should enter, march through, exit without hit.
        let (hit, stats) = trace(&Hollow, [-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], &cfg);
        assert!(hit.is_none(), "hollow should miss: {:?}", hit);
        eprintln!("hollow axis: steps={}", stats.steps);
        assert!(stats.steps < cfg.max_steps, "trace stalled on hollow");
    }

    // ---- shell: inner surface ----

    #[test]
    fn shell_outer_is_unit_sphere() {
        let cfg = TraceConfig::default();
        let occ = Shell { inner: 0.5 };
        // Ray from outside along +x: first hit is outer sphere at t=1.
        let (hit, _) = trace(&occ, [-2.0, 0.0, 0.0], [1.0, 0.0, 0.0], &cfg);
        let h = hit.unwrap();
        assert!(
            (h.t_world - 1.0).abs() < 5e-3,
            "outer shell hit: t={}",
            h.t_world
        );
    }

    #[test]
    fn shell_inner_surface_is_inner_cube_image() {
        // Shell inner boundary is |c|_inf = 0.5. Along the +x axis,
        // F(0.5, 0, 0) = (0.5, 0, 0) (axis identity). So a ray going
        // outward from origin along +x should first hit the SHELL at
        // world x = 0.5.
        let cfg = TraceConfig::default();
        let occ = Shell { inner: 0.5 };
        let (hit, stats) = trace(&occ, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], &cfg);
        let h = hit.expect("ray from inside must hit shell");
        eprintln!("shell inner axis: t={}, steps={}", h.t_world, stats.steps);
        assert!(
            (h.t_world - 0.5).abs() < 0.02,
            "inner shell hit t={} expected ≈0.5",
            h.t_world
        );
    }

    // ---- silhouette regression ----

    /// Rasterize a 2D silhouette using orthographic projection along
    /// +z. Each pixel's ray starts at (x, y, -2) with direction (0,0,1).
    /// Returns a binary mask (true = hit) plus the pixel footprint.
    fn rasterize_silhouette<O: CubeOccupancy>(
        occ: &O,
        w: usize,
        h: usize,
        half_extent: f32,
    ) -> Vec<bool> {
        let cfg = TraceConfig::default();
        let mut mask = vec![false; w * h];
        for py in 0..h {
            for px in 0..w {
                let x = (px as f32 + 0.5) / w as f32 * 2.0 * half_extent - half_extent;
                let y = (py as f32 + 0.5) / h as f32 * 2.0 * half_extent - half_extent;
                let (hit, _) = trace(occ, [x, y, -2.0], [0.0, 0.0, 1.0], &cfg);
                mask[py * w + px] = hit.is_some();
            }
        }
        mask
    }

    /// Fit a circle to a binary silhouette: centroid + max radial
    /// deviation from that centroid. Returns (cx, cy, r_mean, r_stddev).
    fn fit_circle(mask: &[bool], w: usize, h: usize) -> (f32, f32, f32, f32) {
        let mut n = 0.0_f64;
        let mut cx = 0.0_f64;
        let mut cy = 0.0_f64;
        for py in 0..h {
            for px in 0..w {
                if mask[py * w + px] {
                    cx += px as f64 + 0.5;
                    cy += py as f64 + 0.5;
                    n += 1.0;
                }
            }
        }
        if n < 1.0 {
            return (0.0, 0.0, 0.0, f32::INFINITY);
        }
        cx /= n;
        cy /= n;
        // Radial statistics: for each "edge" pixel (boundary of mask),
        // compute distance to centroid.
        let mut rs = Vec::new();
        for py in 0..h {
            for px in 0..w {
                if !mask[py * w + px] {
                    continue;
                }
                let mut boundary = false;
                for (dx, dy) in [(-1_i32, 0_i32), (1, 0), (0, -1), (0, 1)] {
                    let nx = px as i32 + dx;
                    let ny = py as i32 + dy;
                    if nx < 0 || ny < 0 || nx >= w as i32 || ny >= h as i32 {
                        boundary = true;
                        break;
                    }
                    if !mask[ny as usize * w + nx as usize] {
                        boundary = true;
                        break;
                    }
                }
                if boundary {
                    let ex = px as f64 + 0.5 - cx;
                    let ey = py as f64 + 0.5 - cy;
                    rs.push((ex * ex + ey * ey).sqrt());
                }
            }
        }
        if rs.is_empty() {
            return (cx as f32, cy as f32, 0.0, f32::INFINITY);
        }
        let mean: f64 = rs.iter().sum::<f64>() / rs.len() as f64;
        let var: f64 = rs.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / rs.len() as f64;
        (cx as f32, cy as f32, mean as f32, var.sqrt() as f32)
    }

    #[test]
    fn silhouette_is_a_circle() {
        // Solid sphere body: the orthographic silhouette must be a
        // disc. Fit a circle to the boundary; max radial deviation
        // from the fit radius should be subpixel.
        let w = 128;
        let h = 128;
        let mask = rasterize_silhouette(&Solid, w, h, 1.1);
        let (cx, cy, r_mean, r_std) = fit_circle(&mask, w, h);
        let count: usize = mask.iter().filter(|&&b| b).count();
        eprintln!(
            "silhouette: {count} pixels, center=({cx:.2}, {cy:.2}), r_mean={r_mean:.3}, r_std={r_std:.3}"
        );
        assert!(count > 10_000, "silhouette too small: {count} pixels");
        // Center should be near image center.
        assert!(
            (cx - w as f32 / 2.0).abs() < 1.0 && (cy - h as f32 / 2.0).abs() < 1.0,
            "silhouette off-center: ({cx}, {cy})"
        );
        // Boundary pixels should lie on a circle: std-dev of their
        // radial distance is subpixel.
        assert!(
            r_std < 0.8,
            "silhouette is not a circle: r_std={r_std:.3} pixels"
        );
    }

    #[test]
    fn silhouette_has_no_cube_face_seams() {
        // Rotated view: put the camera so a cube corner points at it.
        // In the current setup (ortho along +z), (1,1,1) corner is at
        // cube corner in cube coords; but F maps it to the world point
        // (√⅓, √⅓, √⅓) on the sphere. The silhouette in world space
        // is always a circle regardless of camera orientation — this
        // is the claim we want to verify with a non-axis-aligned
        // "view direction". We test by checking the silhouette
        // rasterized along different directions is still circular.
        let w = 96;
        let h = 96;
        // Along (1, 1, 1)/√3 direction, projected onto the plane
        // perpendicular to that direction. For simplicity, we just
        // rotate the ray origin grid by a constant rotation.
        let cfg = TraceConfig::default();
        let dir = normalize([1.0, 1.0, 1.0]);
        // Build an orthonormal basis (u, v, dir).
        let tmp = if dir[0].abs() < 0.9 {
            [1.0_f32, 0.0, 0.0]
        } else {
            [0.0_f32, 1.0, 0.0]
        };
        let u0 = [
            tmp[1] * dir[2] - tmp[2] * dir[1],
            tmp[2] * dir[0] - tmp[0] * dir[2],
            tmp[0] * dir[1] - tmp[1] * dir[0],
        ];
        let u = normalize(u0);
        let v = [
            dir[1] * u[2] - dir[2] * u[1],
            dir[2] * u[0] - dir[0] * u[2],
            dir[0] * u[1] - dir[1] * u[0],
        ];
        let half = 1.1_f32;
        let mut mask = vec![false; w * h];
        for py in 0..h {
            for px in 0..w {
                let s = (px as f32 + 0.5) / w as f32 * 2.0 * half - half;
                let t = (py as f32 + 0.5) / h as f32 * 2.0 * half - half;
                let origin = [
                    s * u[0] + t * v[0] - 2.0 * dir[0],
                    s * u[1] + t * v[1] - 2.0 * dir[1],
                    s * u[2] + t * v[2] - 2.0 * dir[2],
                ];
                let (hit, _) = trace(&Solid, origin, dir, &cfg);
                mask[py * w + px] = hit.is_some();
            }
        }
        let (cx, cy, r_mean, r_std) = fit_circle(&mask, w, h);
        eprintln!(
            "corner-on silhouette: center=({cx:.2}, {cy:.2}), r_mean={r_mean:.3}, r_std={r_std:.3}"
        );
        let count: usize = mask.iter().filter(|&&b| b).count();
        assert!(count > 5_000, "corner-on silhouette too small: {count}");
        assert!(
            r_std < 1.0,
            "corner-on silhouette shows seams: r_std={r_std:.3}"
        );
    }
}
