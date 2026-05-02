//! Tangent-frame computation for `NodeKind::TangentBlock`.
//!
//! When the sphere descender enters a TangentBlock node it knows the
//! cell's bounds in `(lon, lat, r)` from its descent context and
//! the implied sphere geometry from the parent WrappedPlane (sphere
//! center + radius). From those it builds the local cube frame the
//! ray transforms into before being handed to `march_cartesian`.
//!
//! Convention (matches WGSL `sphere_uv_in_cell`):
//!
//! - World axes: +Y points to the north pole. Sphere center at
//!   `body_origin + body_size/2`. Radius `R = body_size / (2π)`.
//! - For unit-sphere position `n`: `lat = asin(n.y)`,
//!   `lon = atan2(n.z, n.x)`. Equator on the X–Z plane.
//! - Local cube frame: +X = east tangent (∂/∂lon), +Y = outward
//!   normal, +Z = north tangent (∂/∂lat). Right-handed:
//!   `east × normal == north`.

/// Surface position + orthonormal tangent basis at a sphere point.
///
/// `origin` is the sphere-surface point at `(lat, lon)` in WORLD
/// coords (i.e., already offset from `body_origin`). `east`,
/// `normal`, `north` are the columns of the local→world rotation.
#[derive(Clone, Copy, Debug)]
pub struct TangentFrame {
    pub origin: [f32; 3],
    pub east: [f32; 3],
    pub normal: [f32; 3],
    pub north: [f32; 3],
}

impl TangentFrame {
    /// Build a tangent frame at `(lat, lon)` on a sphere of radius
    /// `r_sphere` centered at `cs_center`.
    #[inline]
    pub fn at(cs_center: [f32; 3], r_sphere: f32, lat: f32, lon: f32) -> Self {
        let (sl, cl) = lat.sin_cos();
        let (so, co) = lon.sin_cos();
        let normal = [cl * co, sl, cl * so];
        let east = [-so, 0.0, co];
        // east × normal — see module-level proof.
        let north = [-sl * co, cl, -sl * so];
        let origin = [
            cs_center[0] + r_sphere * normal[0],
            cs_center[1] + r_sphere * normal[1],
            cs_center[2] + r_sphere * normal[2],
        ];
        Self { origin, east, normal, north }
    }

    /// Transform a world-space point into the cube's local `[0, 3)³`
    /// coordinate system.
    ///
    /// `cube_side` is the world-space length of the cube edge; the
    /// local cube is centered on `self.origin`.
    #[inline]
    pub fn world_to_local(&self, world_pos: [f32; 3], cube_side: f32) -> [f32; 3] {
        let dx = world_pos[0] - self.origin[0];
        let dy = world_pos[1] - self.origin[1];
        let dz = world_pos[2] - self.origin[2];
        let scale = 2.0 / cube_side;
        [
            (self.east[0] * dx + self.east[1] * dy + self.east[2] * dz) * scale + 1.0,
            (self.normal[0] * dx + self.normal[1] * dy + self.normal[2] * dz) * scale + 1.0,
            (self.north[0] * dx + self.north[1] * dy + self.north[2] * dz) * scale + 1.0,
        ]
    }

    /// Transform a world-space direction into the cube's local
    /// frame. Direction is scaled by `2.0 / cube_side` so an
    /// `(origin_local, dir_local)` pair has the same `t`
    /// parameterization as the world ray it came from.
    #[inline]
    pub fn world_dir_to_local(&self, world_dir: [f32; 3], cube_side: f32) -> [f32; 3] {
        let scale = 2.0 / cube_side;
        [
            (self.east[0] * world_dir[0] + self.east[1] * world_dir[1] + self.east[2] * world_dir[2]) * scale,
            (self.normal[0] * world_dir[0] + self.normal[1] * world_dir[1] + self.normal[2] * world_dir[2]) * scale,
            (self.north[0] * world_dir[0] + self.north[1] * world_dir[1] + self.north[2] * world_dir[2]) * scale,
        ]
    }

    /// Transform a local-frame normal back to world coords.
    /// Uses the basis columns directly (no scale, since normals are
    /// unit-length).
    #[inline]
    pub fn local_normal_to_world(&self, local_normal: [f32; 3]) -> [f32; 3] {
        [
            self.east[0] * local_normal[0]
                + self.normal[0] * local_normal[1]
                + self.north[0] * local_normal[2],
            self.east[1] * local_normal[0]
                + self.normal[1] * local_normal[1]
                + self.north[1] * local_normal[2],
            self.east[2] * local_normal[0]
                + self.normal[2] * local_normal[1]
                + self.north[2] * local_normal[2],
        ]
    }
}

/// Choose the cube edge length for a tangent block whose covered
/// region on the sphere spans `(lon_step, lat_step, r_step)` (the
/// cell's angular and radial extents). The cube must contain the
/// cell's worst-case world extent so the cell's voxel content fits.
#[inline]
pub fn cube_side_for_cell(r_sphere: f32, lat_center: f32, lon_step: f32, lat_step: f32, r_step: f32) -> f32 {
    let lon_arc = r_sphere * lat_center.cos().abs() * lon_step;
    let lat_arc = r_sphere * lat_step;
    lon_arc.max(lat_arc).max(r_step)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
    }

    fn norm(a: [f32; 3]) -> f32 {
        dot(a, a).sqrt()
    }

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

    /// At every sample point on the sphere, the basis is orthonormal
    /// and the normal points outward from the sphere center.
    #[test]
    fn frame_is_orthonormal_and_outward() {
        let cs_center = [1.0, 1.0, 1.0];
        let r = 3.0 / (2.0 * std::f32::consts::PI);
        let pi = std::f32::consts::PI;
        // Equator + mid-latitude bands, full longitude sweep.
        for &lat in &[-0.4 * pi, -0.2 * pi, 0.0, 0.2 * pi, 0.4 * pi] {
            for li in 0..12 {
                let lon = -pi + (li as f32 + 0.5) * (2.0 * pi / 12.0);
                let f = TangentFrame::at(cs_center, r, lat, lon);
                assert_close(norm(f.east), 1.0, 1e-5, "east unit");
                assert_close(norm(f.normal), 1.0, 1e-5, "normal unit");
                assert_close(norm(f.north), 1.0, 1e-5, "north unit");
                assert_close(dot(f.east, f.normal), 0.0, 1e-5, "east⊥normal");
                assert_close(dot(f.east, f.north), 0.0, 1e-5, "east⊥north");
                assert_close(dot(f.normal, f.north), 0.0, 1e-5, "normal⊥north");
                // Right-handed: east × normal == north.
                let cr = cross(f.east, f.normal);
                assert_close_v(cr, f.north, 1e-5, "east×normal == north");
                // Origin lies on the sphere surface (distance r from center).
                let off = sub(f.origin, cs_center);
                assert_close(norm(off), r, 1e-5, "origin on sphere");
                // Normal points outward (same direction as origin offset).
                let offn = [off[0] / r, off[1] / r, off[2] / r];
                assert_close_v(offn, f.normal, 1e-5, "normal points outward");
            }
        }
    }

    /// Mapping `world → local → world` recovers the input.
    #[test]
    fn world_to_local_round_trip() {
        let cs_center = [1.0, 1.0, 1.0];
        let r = 3.0 / (2.0 * std::f32::consts::PI);
        let f = TangentFrame::at(cs_center, r, 0.3, -0.7);
        let cube_side = 0.05;
        // A point near the surface in world coords.
        let p_world = [
            f.origin[0] + 0.4 * f.east[0] + 0.1 * f.normal[0] - 0.2 * f.north[0],
            f.origin[1] + 0.4 * f.east[1] + 0.1 * f.normal[1] - 0.2 * f.north[1],
            f.origin[2] + 0.4 * f.east[2] + 0.1 * f.normal[2] - 0.2 * f.north[2],
        ];
        let local = f.world_to_local(p_world, cube_side);
        // local should be 1.0 + (offset_in_local_units * 3 / cube_side).
        // Offsets above are in world-space units along the basis vectors.
        let scale = 2.0 / cube_side;
        assert_close(local[0], 1.0 + 0.4 * scale, 1e-3, "x");
        assert_close(local[1], 1.0 + 0.1 * scale, 1e-3, "y");
        assert_close(local[2], 1.0 - 0.2 * scale, 1e-3, "z");

        // Reverse: world = origin + R * (local - 1.0) * (cube/3)
        let inv_scale = cube_side / 2.0;
        let lx = (local[0] - 1.0) * inv_scale;
        let ly = (local[1] - 1.0) * inv_scale;
        let lz = (local[2] - 1.0) * inv_scale;
        let recovered = [
            f.origin[0] + f.east[0] * lx + f.normal[0] * ly + f.north[0] * lz,
            f.origin[1] + f.east[1] * lx + f.normal[1] * ly + f.north[1] * lz,
            f.origin[2] + f.east[2] * lx + f.normal[2] * ly + f.north[2] * lz,
        ];
        assert_close_v(recovered, p_world, 1e-4, "round-trip");
    }

    /// Rays parameterize identically in world and local frames:
    /// for every `t`, `local_origin + local_dir * t` equals the
    /// local-mapped image of `world_origin + world_dir * t`.
    #[test]
    fn ray_parameterization_preserved() {
        let cs_center = [1.0, 1.0, 1.0];
        let r = 3.0 / (2.0 * std::f32::consts::PI);
        let f = TangentFrame::at(cs_center, r, -0.1, 1.2);
        let cube_side = 0.07;

        let world_origin = [0.3, 1.7, 0.9];
        let world_dir = [0.2, -0.5, 0.4];

        let local_origin = f.world_to_local(world_origin, cube_side);
        let local_dir = f.world_dir_to_local(world_dir, cube_side);

        for &t in &[0.0_f32, 0.13, 0.5, 1.7] {
            let world_at_t = [
                world_origin[0] + world_dir[0] * t,
                world_origin[1] + world_dir[1] * t,
                world_origin[2] + world_dir[2] * t,
            ];
            let local_at_t_via_world = f.world_to_local(world_at_t, cube_side);
            let local_at_t_direct = [
                local_origin[0] + local_dir[0] * t,
                local_origin[1] + local_dir[1] * t,
                local_origin[2] + local_dir[2] * t,
            ];
            assert_close_v(local_at_t_via_world, local_at_t_direct, 1e-4, "ray@t");
        }
    }

    /// Local-frame normal returns to the basis vector it represents.
    #[test]
    fn local_normal_unmaps_to_world() {
        let cs_center = [1.0, 1.0, 1.0];
        let r = 3.0 / (2.0 * std::f32::consts::PI);
        let f = TangentFrame::at(cs_center, r, 0.5, -1.0);
        assert_close_v(f.local_normal_to_world([1.0, 0.0, 0.0]), f.east, 1e-5, "+X");
        assert_close_v(f.local_normal_to_world([0.0, 1.0, 0.0]), f.normal, 1e-5, "+Y");
        assert_close_v(f.local_normal_to_world([0.0, 0.0, 1.0]), f.north, 1e-5, "+Z");
    }
}
