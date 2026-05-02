//! Sphere-Mercator render lens — prototype, not wired into the renderer.
//!
//! See `docs/design/sphere-mercator-prototype.md` for the architecture.
//!
//! The world is a flat wrapped Cartesian slab. The "sphere" is a
//! render-time camera lens: per-pixel transform that converts a
//! screen-space ray into a slab-space ray, and a per-hit transform
//! that rotates the slab-space normal back into the apparent UV-sphere
//! tangent frame. Nothing in the world is curved.
//!
//! All depth-dependent geometry stays in slab space. The lens only
//! ever touches bounded inputs (`|θ| ≤ π`, `|φ| ≤ π/2`, anchor offsets
//! bounded by R), so f32 holds at any tree depth.

use super::tangent::TangentFrame;

/// Slab axis convention (matches `wrapped_planet.rs`):
/// - `dims[0]` = X = longitude (wrap)
/// - `dims[1]` = Y = vertical / radial depth
/// - `dims[2]` = Z = latitude (bounded; pole strips at ends)
#[derive(Clone, Copy, Debug)]
pub struct PlanetLens {
    /// Planet center in world coords (relative to camera anchor —
    /// kept f32 because the slab DDA owns the depth-dependent
    /// precision; the lens is bounded by `R` and angle ranges).
    pub center: [f32; 3],
    /// Surface radius (= slab "ground" altitude). Convention used
    /// throughout: `R = body_size / (2π)` so one full longitudinal
    /// wrap covers `body_size` world units, matching `TangentFrame`.
    pub radius: f32,
    /// Slab dims in cells at the slab anchor depth.
    pub dims: [u32; 3],
    /// Which slab Y row's TOP is the "surface" (r = R). Typically
    /// `dims[1] - 1` so the top face of the topmost cell is the
    /// surface. Stored as a float so altitude maps continuously.
    pub surface_slab_y: f32,
    /// World-space height of one slab cell along Y (radial). Used
    /// to convert altitude (`r − R`) to slab Y units. Independent
    /// of the X/Z arc-length scaling.
    pub cell_size_y: f32,
}

impl PlanetLens {
    /// Build a lens whose X/Z scaling matches the worldgen 2:1 aspect.
    /// Y scaling is `body_size / dims[0]` so a slab voxel is roughly
    /// cubic at the equator.
    pub fn from_worldgen(center: [f32; 3], body_size: f32, dims: [u32; 3]) -> Self {
        let radius = body_size / (2.0 * std::f32::consts::PI);
        let cell_size_y = body_size / dims[0] as f32;
        Self {
            center,
            radius,
            dims,
            surface_slab_y: dims[1] as f32 - 1.0,
            cell_size_y,
        }
    }

    /// Convert a slab cell-space point to world coords.
    /// `slab.0 = x` ∈ `[0, dims[0])` (wrapping),
    /// `slab.1 = y` (radial, surface at `surface_slab_y + 1`),
    /// `slab.2 = z` ∈ `[0, dims[2])`.
    pub fn slab_to_world(&self, slab: [f32; 3]) -> [f32; 3] {
        let two_pi = 2.0 * std::f32::consts::PI;
        let pi = std::f32::consts::PI;
        let theta = two_pi * slab[0] / self.dims[0] as f32;
        let phi = pi * (slab[2] / self.dims[2] as f32 - 0.5);
        let r = self.radius + (slab[1] - (self.surface_slab_y + 1.0)) * self.cell_size_y;
        let (sl, cl) = phi.sin_cos();
        let (so, co) = theta.sin_cos();
        [
            self.center[0] + r * cl * co,
            self.center[1] + r * sl,
            self.center[2] + r * cl * so,
        ]
    }

    /// Inverse of `slab_to_world`. Returns slab coords with `slab.0`
    /// wrapped into `[0, dims[0])`. Returns `None` only at the
    /// degenerate singularity `world == center`.
    pub fn world_to_slab(&self, world: [f32; 3]) -> Option<[f32; 3]> {
        let v = [
            world[0] - self.center[0],
            world[1] - self.center[1],
            world[2] - self.center[2],
        ];
        let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if r < 1e-12 {
            return None;
        }
        let n = [v[0] / r, v[1] / r, v[2] / r];
        let phi = n[1].clamp(-1.0, 1.0).asin();
        let theta = n[2].atan2(n[0]);
        let two_pi = 2.0 * std::f32::consts::PI;
        let pi = std::f32::consts::PI;
        let mut slab_x = theta / two_pi * self.dims[0] as f32;
        let w = self.dims[0] as f32;
        slab_x = slab_x.rem_euclid(w);
        let slab_z = (phi / pi + 0.5) * self.dims[2] as f32;
        let slab_y = (self.surface_slab_y + 1.0) + (r - self.radius) / self.cell_size_y;
        Some([slab_x, slab_y, slab_z])
    }

    /// Tangent frame at a slab cell's center. Used by `shade_normal`
    /// and by external callers that need to position render-time
    /// geometry (e.g. block bevels, entity transforms).
    pub fn frame_at_slab(&self, slab: [f32; 3]) -> TangentFrame {
        let two_pi = 2.0 * std::f32::consts::PI;
        let pi = std::f32::consts::PI;
        let theta = two_pi * slab[0] / self.dims[0] as f32;
        let phi = pi * (slab[2] / self.dims[2] as f32 - 0.5);
        TangentFrame::at(self.center, self.radius, phi, theta)
    }

    /// Rotate an axis-aligned slab-space normal into world coords for
    /// shading. `slab_normal` should be one of `(±1, 0, 0)`,
    /// `(0, ±1, 0)`, `(0, 0, ±1)` — the face normal returned by the
    /// flat-cartesian DDA.
    ///
    /// `slab` is the cell's center (or any sample point on the cell
    /// face — the frame varies smoothly). At deep depth the cell's
    /// `θ`/`φ` are bounded inputs to `sin/cos`, so the rotation is
    /// f32-stable regardless of tree depth.
    pub fn shade_normal(&self, slab: [f32; 3], slab_normal: [f32; 3]) -> [f32; 3] {
        let f = self.frame_at_slab(slab);
        f.local_normal_to_world(slab_normal)
    }

    /// Project a world-space camera ray into a slab-space ray.
    ///
    /// Returns `None` if the ray misses the sphere of radius
    /// `radius + max_alt_above_surface` around `center` (i.e. the
    /// camera is looking past the planet entirely). Otherwise returns
    /// `(slab_origin, slab_dir, t_anchor)`:
    ///
    /// - `slab_origin` is the slab coords of the lens anchor (the
    ///   first sphere intersection along the ray).
    /// - `slab_dir` is the ray direction expressed in slab cell-units
    ///   per unit world-`t`. The slab DDA can step on this directly.
    /// - `t_anchor` is the world-`t` from `cam_pos` to the anchor.
    ///
    /// The transformation linearises the sphere at the anchor. For
    /// rays that traverse a wide angular span (grazing horizon shots)
    /// the linearisation drifts; the wiring branch will re-anchor
    /// every K slab cells. For most pixels (looking ± ground), one
    /// anchor per ray is enough.
    pub fn project_ray(
        &self,
        cam_pos: [f32; 3],
        ray_dir: [f32; 3],
    ) -> Option<ProjectedRay> {
        let v = [
            cam_pos[0] - self.center[0],
            cam_pos[1] - self.center[1],
            cam_pos[2] - self.center[2],
        ];
        let dlen = (ray_dir[0] * ray_dir[0] + ray_dir[1] * ray_dir[1] + ray_dir[2] * ray_dir[2]).sqrt();
        if dlen < 1e-12 {
            return None;
        }
        let d = [ray_dir[0] / dlen, ray_dir[1] / dlen, ray_dir[2] / dlen];

        // Sphere intersection at surface radius. If the camera is
        // outside the sphere we want the entry point; if inside, we
        // anchor at the camera's foot on the sphere (t = 0 in the
        // linearisation but slab_y reflects altitude).
        let v_dot_d = v[0] * d[0] + v[1] * d[1] + v[2] * d[2];
        let v_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        let r_sq = self.radius * self.radius;
        let disc = v_dot_d * v_dot_d - (v_sq - r_sq);

        let cam_outside = v_sq > r_sq;
        let t_anchor;
        let anchor_world;
        if cam_outside {
            if disc < 0.0 {
                return None; // ray misses the sphere
            }
            let t = -v_dot_d - disc.sqrt();
            if t < 0.0 {
                return None; // sphere is behind the camera
            }
            // Convert from unit-d t to original-d t.
            t_anchor = t / dlen;
            anchor_world = [
                cam_pos[0] + ray_dir[0] * t_anchor,
                cam_pos[1] + ray_dir[1] * t_anchor,
                cam_pos[2] + ray_dir[2] * t_anchor,
            ];
        } else {
            // Camera is below/at the surface: anchor at the camera's
            // own radial projection on the sphere. t_anchor = 0.
            t_anchor = 0.0;
            anchor_world = cam_pos;
        }

        // Slab coords of the anchor. (For an outside-camera ray this
        // is on the surface, slab_y == surface_slab_y + 1.)
        let slab_anchor = self.world_to_slab(anchor_world)?;

        // Build the tangent frame at the anchor and decompose the
        // ORIGINAL (un-normalised) ray into (east, normal, north).
        let f = TangentFrame::at(
            self.center,
            self.radius,
            (slab_anchor[2] / self.dims[2] as f32 - 0.5) * std::f32::consts::PI,
            slab_anchor[0] / self.dims[0] as f32 * 2.0 * std::f32::consts::PI,
        );
        let de = ray_dir[0] * f.east[0] + ray_dir[1] * f.east[1] + ray_dir[2] * f.east[2];
        let dn = ray_dir[0] * f.normal[0] + ray_dir[1] * f.normal[1] + ray_dir[2] * f.normal[2];
        let dh = ray_dir[0] * f.north[0] + ray_dir[1] * f.north[1] + ray_dir[2] * f.north[2];

        // Convert to slab cell-units / unit world-t.
        // World units per slab-X cell = body_size / dims[0] = 2π·R / dims[0].
        // World units per slab-Y cell = cell_size_y.
        // World units per slab-Z cell = π·R / dims[2].
        let two_pi = 2.0 * std::f32::consts::PI;
        let pi = std::f32::consts::PI;
        let cells_per_unit_x = self.dims[0] as f32 / (two_pi * self.radius);
        let cells_per_unit_y = 1.0 / self.cell_size_y;
        let cells_per_unit_z = self.dims[2] as f32 / (pi * self.radius);

        // The camera's slab origin: the anchor is at t_anchor. To
        // give the slab DDA a ray that parameterises in world-t with
        // the camera at t=0, back up from the anchor by t_anchor in
        // the (already-converted) slab direction.
        let slab_dir = [
            de * cells_per_unit_x,
            dn * cells_per_unit_y,
            dh * cells_per_unit_z,
        ];
        let slab_origin = [
            slab_anchor[0] - slab_dir[0] * t_anchor,
            slab_anchor[1] - slab_dir[1] * t_anchor,
            slab_anchor[2] - slab_dir[2] * t_anchor,
        ];

        Some(ProjectedRay {
            slab_origin,
            slab_dir,
            slab_anchor,
            t_anchor,
        })
    }
}

/// Output of `PlanetLens::project_ray`. The slab DDA consumes
/// `slab_origin + t * slab_dir`; `t` is the same world-`t` as the
/// camera ray, so a slab hit's `t` plugs back into
/// `cam_pos + t * ray_dir` for depth/AO/etc.
#[derive(Clone, Copy, Debug)]
pub struct ProjectedRay {
    pub slab_origin: [f32; 3],
    pub slab_dir: [f32; 3],
    /// Slab coords at the lens anchor (the first sphere intersection
    /// or the camera's own projection if inside). Useful for
    /// re-anchoring along long rays.
    pub slab_anchor: [f32; 3],
    /// World-`t` from camera to anchor.
    pub t_anchor: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, eps: f32, msg: &str) {
        assert!((a - b).abs() < eps, "{}: {} vs {} (Δ={})", msg, a, b, (a - b).abs());
    }

    fn assert_close_v(a: [f32; 3], b: [f32; 3], eps: f32, msg: &str) {
        for i in 0..3 {
            assert!(
                (a[i] - b[i]).abs() < eps,
                "{} axis {}: {} vs {} (Δ={})",
                msg, i, a[i], b[i], (a[i] - b[i]).abs(),
            );
        }
    }

    fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    fn norm(a: [f32; 3]) -> f32 {
        dot(a, a).sqrt()
    }

    fn lens() -> PlanetLens {
        // Match worldgen defaults: dims = [27, 2, 14], body_size 1.0
        // (arbitrary — only ratios matter for the math).
        PlanetLens::from_worldgen([0.5, 0.5, 0.5], 1.0, [27, 2, 14])
    }

    /// `slab_to_world` ∘ `world_to_slab` is the identity on a generic
    /// slab point. This proves the bridge formulas are mutual inverses.
    #[test]
    fn slab_world_round_trip() {
        let l = lens();
        for &cx in &[0.5_f32, 5.3, 13.0, 26.7] {
            for &cy in &[0.5_f32, 1.0, 1.7, 2.5] {
                for &cz in &[0.5_f32, 3.2, 7.0, 12.8] {
                    let slab_in = [cx, cy, cz];
                    let world = l.slab_to_world(slab_in);
                    let slab_out = l.world_to_slab(world).unwrap();
                    assert_close_v(slab_in, slab_out, 5e-4, "round-trip");
                }
            }
        }
    }

    /// Wrap consistency: `slab_x = 0` and `slab_x = W` map to the
    /// same world point. The slab is topologically a cylinder along X.
    #[test]
    fn longitude_wraps() {
        let l = lens();
        let p0 = l.slab_to_world([0.0, 1.5, 7.0]);
        let pw = l.slab_to_world([l.dims[0] as f32, 1.5, 7.0]);
        assert_close_v(p0, pw, 1e-5, "lon wrap");
    }

    /// North pole: `slab_z = L` corresponds to `+Y` in world.
    /// (And `slab_z = 0` to `−Y`.)
    #[test]
    fn poles_align_with_y() {
        let l = lens();
        let north = l.slab_to_world([0.0, l.surface_slab_y + 1.0, l.dims[2] as f32]);
        assert_close(north[1] - l.center[1], l.radius, 1e-4, "north +Y");
        let south = l.slab_to_world([0.0, l.surface_slab_y + 1.0, 0.0]);
        assert_close(south[1] - l.center[1], -l.radius, 1e-4, "south -Y");
    }

    /// Tangent frame is orthonormal at every slab cell sample. Reuses
    /// the same invariants `TangentFrame` already proves; included
    /// here so a regression in `frame_at_slab` shows up directly.
    #[test]
    fn frame_orthonormal_across_slab() {
        let l = lens();
        for cx in 0..27 {
            for cz in 1..13 {
                let f = l.frame_at_slab([cx as f32 + 0.5, l.surface_slab_y + 1.0, cz as f32 + 0.5]);
                assert_close(norm(f.east), 1.0, 1e-5, "east unit");
                assert_close(norm(f.normal), 1.0, 1e-5, "normal unit");
                assert_close(norm(f.north), 1.0, 1e-5, "north unit");
                assert_close(dot(f.east, f.normal), 0.0, 1e-5, "east⊥normal");
                assert_close(dot(f.east, f.north), 0.0, 1e-5, "east⊥north");
                assert_close(dot(f.normal, f.north), 0.0, 1e-5, "normal⊥north");
                let cr = cross(f.east, f.normal);
                assert_close_v(cr, f.north, 1e-5, "east×normal == north");
            }
        }
    }

    /// THE precision test: at a leaf cell sized `1/3^25` of a slab
    /// cell (= layer 25 anchor depth), the tangent frame is still
    /// orthonormal in f32. This is the property the previous
    /// sphere-DDA approach lost — the input to `sin/cos` here is
    /// `θ ∈ [-π, π]`, bounded, so f32 stays sharp.
    #[test]
    fn frame_precision_at_depth_25() {
        let l = lens();
        // Pick a slab cell near the equator and add a deep-depth
        // perturbation: a sub-cell offset of 1/3^25 of a slab cell.
        let base_cx = 13.0_f32;
        let base_cz = 7.0_f32;
        let three_pow_25: f32 = 3.0_f32.powi(25);
        let deep_offset = 1.0 / three_pow_25;
        let f = l.frame_at_slab([
            base_cx + deep_offset,
            l.surface_slab_y + 1.0,
            base_cz + deep_offset,
        ]);
        assert_close(norm(f.east), 1.0, 1e-5, "east unit @d25");
        assert_close(norm(f.normal), 1.0, 1e-5, "normal unit @d25");
        assert_close(norm(f.north), 1.0, 1e-5, "north unit @d25");
        assert_close(dot(f.east, f.normal), 0.0, 1e-5, "east⊥normal @d25");
        assert_close(dot(f.east, f.north), 0.0, 1e-5, "east⊥north @d25");
        // Compare to the same frame WITHOUT the deep-depth offset:
        // they should be effectively identical (the angular delta is
        // below f32 sin/cos precision).
        let f_base = l.frame_at_slab([base_cx, l.surface_slab_y + 1.0, base_cz]);
        assert_close_v(f.east, f_base.east, 1e-5, "east stable @d25");
        assert_close_v(f.normal, f_base.normal, 1e-5, "normal stable @d25");
        assert_close_v(f.north, f_base.north, 1e-5, "north stable @d25");
    }

    /// `shade_normal` rotates the slab-space `+Y` (radial) normal to
    /// the world-space outward direction at the cell's `(θ, φ)`.
    /// This is the per-block rotation a shader would apply to the
    /// returned `slab_normal` before lighting.
    #[test]
    fn shade_normal_rotates_to_outward() {
        let l = lens();
        // A cell on the equator at slab_x=0 should have its +Y
        // (radial) face point along world +X.
        let n = l.shade_normal([0.0, l.surface_slab_y + 1.0, l.dims[2] as f32 / 2.0], [0.0, 1.0, 0.0]);
        assert_close_v(n, [1.0, 0.0, 0.0], 1e-5, "equator slab_x=0 → +X");

        // At slab_x = W/4 (90° east) it should point along +Z.
        let q = l.dims[0] as f32 / 4.0;
        let n = l.shade_normal([q, l.surface_slab_y + 1.0, l.dims[2] as f32 / 2.0], [0.0, 1.0, 0.0]);
        assert_close_v(n, [0.0, 0.0, 1.0], 1e-5, "equator slab_x=W/4 → +Z");
    }

    /// Project a camera ray that grazes the planet from outside.
    /// The slab anchor must be on the surface (`slab_y` == top), and
    /// re-projecting the slab anchor through `slab_to_world` must
    /// land on the original sphere intersection point.
    #[test]
    fn project_ray_anchor_on_surface() {
        let l = lens();
        // Camera 5R away on +X, looking back at the planet.
        let cam = [l.center[0] + 5.0 * l.radius, l.center[1], l.center[2]];
        let dir = [-1.0, 0.0, 0.0];
        let p = l.project_ray(cam, dir).expect("ray should hit planet");

        // Anchor must be on the surface (slab_y == surface + 1 within
        // a hair).
        assert_close(p.slab_anchor[1], l.surface_slab_y + 1.0, 1e-3, "anchor on surface");

        // Re-derive the world point at the anchor and compare to the
        // analytic sphere intersection.
        let world_anchor = l.slab_to_world(p.slab_anchor);
        let analytic = [l.center[0] + l.radius, l.center[1], l.center[2]];
        assert_close_v(world_anchor, analytic, 1e-4, "anchor world matches");

        // The slab origin + t_anchor * slab_dir should land on the
        // anchor (this is what makes `t` parameterise consistently
        // in world and slab space).
        let recovered = [
            p.slab_origin[0] + p.slab_dir[0] * p.t_anchor,
            p.slab_origin[1] + p.slab_dir[1] * p.t_anchor,
            p.slab_origin[2] + p.slab_dir[2] * p.t_anchor,
        ];
        assert_close_v(recovered, p.slab_anchor, 1e-3, "ray param consistent");
    }

    /// A ray that points away from the planet returns `None`.
    #[test]
    fn project_ray_misses() {
        let l = lens();
        let cam = [l.center[0] + 5.0 * l.radius, l.center[1], l.center[2]];
        let dir = [1.0, 0.0, 0.0]; // pointing further away
        assert!(l.project_ray(cam, dir).is_none(), "away ray should miss");
    }
}
