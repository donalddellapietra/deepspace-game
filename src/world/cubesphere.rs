//! Cubed-sphere coordinates: the math that lets a planet be built
//! from voxels that bulge outward at large scales but feel like flat
//! cubes at the surface.
//!
//! A "planet" is six cube faces wrapped around a sphere. Each face
//! carries its own flat grid of cells parameterized by (u, v) ∈
//! [-1, 1]², plus a radial axis r = distance from planet center. A
//! "block" is a cell in (face, u, v, r) space; its 8 world-space
//! corners live on two concentric spheres at radii r and r + Δr,
//! which gives the block its signature bulged-square shape.
//!
//! The six faces index as:
//!   0 = +X   1 = -X   2 = +Y   3 = -Y   4 = +Z   5 = -Z
//! For each face we pick two orthogonal tangent axes (u, v) so that
//! adjacent cells on the same face share exact edges, and the three
//! faces meeting at every cube corner align there as well.
//!
//! There are no poles — one of the main reasons to use a cubed-sphere
//! over a lat/lon parameterization. Cells near cube-face seams are
//! only mildly stretched (at most ~1.5× area ratio between the
//! center of a face and its corners), which is invisible in
//! gameplay.
//!
//! This module is pure math. It touches nothing else in the engine;
//! later passes (tree integration, renderer, collision) will build
//! on it.

use super::palette::block;
use super::sdf::{self, Planet, Vec3};

/// One of the six cube faces.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Face {
    PosX = 0,
    NegX = 1,
    PosY = 2,
    NegY = 3,
    PosZ = 4,
    NegZ = 5,
}

impl Face {
    pub const ALL: [Face; 6] = [
        Face::PosX, Face::NegX,
        Face::PosY, Face::NegY,
        Face::PosZ, Face::NegZ,
    ];

    pub fn from_index(i: u8) -> Face {
        match i {
            0 => Face::PosX, 1 => Face::NegX,
            2 => Face::PosY, 3 => Face::NegY,
            4 => Face::PosZ, 5 => Face::NegZ,
            _ => panic!("invalid face index {i}"),
        }
    }

    /// Unit vector pointing "out" of this face's center.
    pub fn normal(self) -> Vec3 {
        match self {
            Face::PosX => [ 1.0,  0.0,  0.0],
            Face::NegX => [-1.0,  0.0,  0.0],
            Face::PosY => [ 0.0,  1.0,  0.0],
            Face::NegY => [ 0.0, -1.0,  0.0],
            Face::PosZ => [ 0.0,  0.0,  1.0],
            Face::NegZ => [ 0.0,  0.0, -1.0],
        }
    }

    /// Tangent basis (u_axis, v_axis) for this face. Chosen so that
    /// (u_axis × v_axis) = normal — right-handed, consistent winding.
    pub fn tangents(self) -> (Vec3, Vec3) {
        match self {
            Face::PosX => ([ 0.0,  0.0, -1.0], [ 0.0,  1.0,  0.0]),
            Face::NegX => ([ 0.0,  0.0,  1.0], [ 0.0,  1.0,  0.0]),
            Face::PosY => ([ 1.0,  0.0,  0.0], [ 0.0,  0.0, -1.0]),
            Face::NegY => ([ 1.0,  0.0,  0.0], [ 0.0,  0.0,  1.0]),
            Face::PosZ => ([ 1.0,  0.0,  0.0], [ 0.0,  1.0,  0.0]),
            Face::NegZ => ([-1.0,  0.0,  0.0], [ 0.0,  1.0,  0.0]),
        }
    }
}

/// A point in cubed-sphere coordinates, relative to some planet
/// center.  `face` selects one of 6 cube faces; `u, v ∈ [-1, 1]` is
/// the 2D position on that face; `r` is the radial distance from
/// planet center in world units.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CubeSphereCoord {
    pub face: Face,
    pub u: f32,
    pub v: f32,
    pub r: f32,
}

/// Convert (face, u, v) → unit direction pointing outward from the
/// planet center. `u, v` both in [-1, 1]. The cube face is projected
/// onto the sphere by normalizing: cube_point = normal + u·u_axis +
/// v·v_axis; sphere_dir = normalize(cube_point).
///
/// Adjacent cells share edges because the cube points agree at the
/// boundary and normalization is continuous.
pub fn face_uv_to_dir(face: Face, u: f32, v: f32) -> Vec3 {
    let n = face.normal();
    let (ua, va) = face.tangents();
    let cube_pt = [
        n[0] + u * ua[0] + v * va[0],
        n[1] + u * ua[1] + v * va[1],
        n[2] + u * ua[2] + v * va[2],
    ];
    sdf::normalize(cube_pt)
}

/// Given a planet center and a `CubeSphereCoord`, return the
/// world-space position of that point.
pub fn coord_to_world(center: Vec3, c: CubeSphereCoord) -> Vec3 {
    let dir = face_uv_to_dir(c.face, c.u, c.v);
    sdf::add(center, sdf::scale(dir, c.r))
}

/// Convert a world-space position to cubed-sphere coordinates
/// relative to a planet center. Picks the face whose axis is most
/// aligned with (pos - center); ties break toward the face with
/// lower index. Returns `None` if `pos == center` (undefined
/// direction).
pub fn world_to_coord(center: Vec3, pos: Vec3) -> Option<CubeSphereCoord> {
    let d = sdf::sub(pos, center);
    let r = sdf::length(d);
    if r < 1e-12 { return None; }
    let dir = sdf::scale(d, 1.0 / r);

    // Pick the dominant axis.
    let ax = dir[0].abs();
    let ay = dir[1].abs();
    let az = dir[2].abs();
    let (face, u, v) = if ax >= ay && ax >= az {
        // X-dominant.
        if dir[0] > 0.0 { (Face::PosX, -dir[2] / ax,  dir[1] / ax) }
        else            { (Face::NegX,  dir[2] / ax,  dir[1] / ax) }
    } else if ay >= az {
        // Y-dominant.
        if dir[1] > 0.0 { (Face::PosY,  dir[0] / ay, -dir[2] / ay) }
        else            { (Face::NegY,  dir[0] / ay,  dir[2] / ay) }
    } else {
        // Z-dominant.
        if dir[2] > 0.0 { (Face::PosZ,  dir[0] / az,  dir[1] / az) }
        else            { (Face::NegZ, -dir[0] / az,  dir[1] / az) }
    };

    Some(CubeSphereCoord { face, u, v, r })
}

/// The eight world-space corners of a block spanning
/// `[u, u+du] × [v, v+dv] × [r, r+dr]` on `face`, around `center`.
/// Returned order: `[u0,v0,r0], [u1,v0,r0], [u0,v1,r0], [u1,v1,r0],
///                  [u0,v0,r1], [u1,v0,r1], [u0,v1,r1], [u1,v1,r1]`.
/// These corners are NOT coplanar: the top and bottom faces are
/// spherical patches (bulged), and the four side faces are
/// frustum-like walls between them. That bulge is the whole point.
pub fn block_corners(
    center: Vec3,
    face: Face,
    u: f32, v: f32, r: f32,
    du: f32, dv: f32, dr: f32,
) -> [Vec3; 8] {
    let mut out = [[0.0; 3]; 8];
    let coords = [
        (u,      v,      r),
        (u + du, v,      r),
        (u,      v + dv, r),
        (u + du, v + dv, r),
        (u,      v,      r + dr),
        (u + du, v,      r + dr),
        (u,      v + dv, r + dr),
        (u + du, v + dv, r + dr),
    ];
    for (i, &(cu, cv, cr)) in coords.iter().enumerate() {
        out[i] = coord_to_world(center, CubeSphereCoord { face, u: cu, v: cv, r: cr });
    }
    out
}

// ────────────────────────────────────────────────── planet data

/// A planet represented in cubed-sphere coordinates.
///
/// Each of the 6 faces carries an N×N grid of surface cells
/// (single-layer shell for now — altitude stacking comes later).
/// `blocks[f * N*N + j*N + i]` is the palette index of the cell at
/// `(face=f, cell_i=i, cell_j=j)`, or 0 if empty / above surface.
///
/// This flat layout is what the GPU reads to shade the planet per
/// fragment: ray→sphere hit → face picker → (i, j) → block index →
/// palette color. Later phases will lift this to an LOD'd nested
/// tree, but the flat single-shell form is the simplest thing that
/// lets us see real, SDF-driven content on the bulged-square grid.
#[derive(Clone, Debug)]
pub struct CubeSpherePlanet {
    pub center: Vec3,
    pub radius: f32,
    pub cells_per_face_edge: u32,
    pub blocks: Vec<u8>,
}

impl CubeSpherePlanet {
    pub fn empty(center: Vec3, radius: f32, cells_per_face_edge: u32) -> Self {
        let n = cells_per_face_edge as usize;
        Self {
            center,
            radius,
            cells_per_face_edge,
            blocks: vec![0; 6 * n * n],
        }
    }

    #[inline]
    pub fn cell_index(&self, face: Face, i: u32, j: u32) -> usize {
        let n = self.cells_per_face_edge as usize;
        (face as usize) * n * n + (j as usize) * n + (i as usize)
    }

    pub fn set(&mut self, face: Face, i: u32, j: u32, block: u8) {
        let idx = self.cell_index(face, i, j);
        self.blocks[idx] = block;
    }

    pub fn get(&self, face: Face, i: u32, j: u32) -> u8 {
        self.blocks[self.cell_index(face, i, j)]
    }

    /// Center (u, v) of the (i, j) cell on a face of N×N cells.
    /// Cells tile [-1, 1]² uniformly in (u, v) space; their
    /// projections to the sphere are slightly non-uniform in
    /// solid angle, which is the standard cubed-sphere property.
    pub fn cell_center_uv(&self, i: u32, j: u32) -> (f32, f32) {
        let n = self.cells_per_face_edge as f32;
        let u = -1.0 + 2.0 * (i as f32 + 0.5) / n;
        let v = -1.0 + 2.0 * (j as f32 + 0.5) / n;
        (u, v)
    }

    /// Intersect a ray with the planet's outer shell. Returns the
    /// entry-t if the ray hits in front of the camera.
    pub fn ray_t(&self, origin: Vec3, dir: Vec3) -> Option<f32> {
        let oc = sdf::sub(origin, self.center);
        let b = sdf::dot(oc, dir);
        let c = sdf::dot(oc, oc) - self.radius * self.radius;
        let disc = b * b - c;
        if disc <= 0.0 { return None; }
        let sq = disc.sqrt();
        let t0 = -b - sq;
        let t1 = -b + sq;
        if t0 > 0.0 { Some(t0) } else if t1 > 0.0 { Some(t1) } else { None }
    }

    /// Which (face, i, j) cell does this ray hit first on the shell?
    /// Returns the entry-t and the cell it lands in, clamped to the
    /// grid. Does NOT skip empty cells — that's the caller's call
    /// (the highlight wants the shell position, a block raycast
    /// would want a loop).
    pub fn hit_cell(&self, origin: Vec3, dir: Vec3) -> Option<(f32, Face, u32, u32)> {
        let t = self.ray_t(origin, dir)?;
        let hit = sdf::add(origin, sdf::scale(dir, t));
        let coord = world_to_coord(self.center, hit)?;
        let n = self.cells_per_face_edge as f32;
        let ug = ((coord.u + 1.0) * 0.5 * n).floor();
        let vg = ((coord.v + 1.0) * 0.5 * n).floor();
        let i = (ug as i32).clamp(0, self.cells_per_face_edge as i32 - 1) as u32;
        let j = (vg as i32).clamp(0, self.cells_per_face_edge as i32 - 1) as u32;
        Some((t, coord.face, i, j))
    }
}

/// Fill a `CubeSpherePlanet`'s cells by sampling an SDF `Planet`
/// at each cell's world-space center. A cell is solid (gets the
/// planet's surface block) iff the SDF says its center is below
/// the displaced surface. Above-surface cells stay empty.
///
/// This is the single-shell equivalent of the voxel-octree
/// `build_space_subtree`: one radius, one sample per column. Good
/// enough to visualize terrain on the sphere right now; later
/// phases layer altitude columns below/above.
pub fn generate_from_sdf(
    center: Vec3,
    radius: f32,
    cells_per_face_edge: u32,
    sdf_planet: &Planet,
) -> CubeSpherePlanet {
    let mut cs = CubeSpherePlanet::empty(center, radius, cells_per_face_edge);
    for &face in &Face::ALL {
        for j in 0..cells_per_face_edge {
            for i in 0..cells_per_face_edge {
                let (u, v) = cs.cell_center_uv(i, j);
                let world = coord_to_world(center, CubeSphereCoord { face, u, v, r: radius });
                // Cell is "present" if the undisplaced surface at
                // radius `radius` is below the noise-displaced SDF
                // surface at this direction — equivalently, SDF < 0
                // means we're inside the planet, so the surface cell
                // is solid.
                if sdf_planet.distance(world) < 0.0 {
                    cs.set(face, i, j, sdf_planet.block_at(world));
                } else {
                    // Cell is above the displaced surface (valley
                    // where the noise pushes terrain inward). Leave
                    // empty — the raymarch sees through this cell.
                    cs.set(face, i, j, 0);
                }
            }
        }
    }
    // Make sure a completely-empty planet still shows *something* for
    // debugging: fill the 4 cardinal face centers with stone if the
    // whole buffer is empty.
    if cs.blocks.iter().all(|&b| b == 0) {
        for &face in &Face::ALL {
            let mid = cells_per_face_edge / 2;
            cs.set(face, mid, mid, block::STONE);
        }
    }
    cs
}

/// The twelve edges of a cubed-sphere block as pairs of corner
/// indices into `block_corners`'s output. Useful for drawing the
/// bulged wireframe outline that Minecraft's flat-cube outline
/// becomes on a planet.
pub const BLOCK_EDGES: [(usize, usize); 12] = [
    // Bottom square (r = r0).
    (0, 1), (1, 3), (3, 2), (2, 0),
    // Top square (r = r0 + dr).
    (4, 5), (5, 7), (7, 6), (6, 4),
    // Verticals connecting bottom to top.
    (0, 4), (1, 5), (2, 6), (3, 7),
];

// ──────────────────────────────────────────────────────────── tests

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32) -> bool { (a - b).abs() < 1e-5 }
    fn approx_v(a: Vec3, b: Vec3) -> bool {
        approx(a[0], b[0]) && approx(a[1], b[1]) && approx(a[2], b[2])
    }

    #[test]
    fn face_center_matches_normal() {
        for &f in &Face::ALL {
            let dir = face_uv_to_dir(f, 0.0, 0.0);
            assert!(approx_v(dir, f.normal()),
                "face {:?} center should equal its normal", f);
        }
    }

    #[test]
    fn face_corners_are_unit_vectors() {
        // Every (u, v) in [-1, 1]² should project to a unit vector.
        for &f in &Face::ALL {
            for &(u, v) in &[(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),
                              (0.5, -0.3), (0.0, 0.0)] {
                let d = face_uv_to_dir(f, u, v);
                assert!(approx(sdf::length(d), 1.0),
                    "face {:?} at ({}, {}) not unit: {:?}", f, u, v, d);
            }
        }
    }

    #[test]
    fn tangents_are_orthonormal_to_normal() {
        for &f in &Face::ALL {
            let n = f.normal();
            let (ua, va) = f.tangents();
            assert!(approx(sdf::dot(ua, n), 0.0),
                "face {:?} u_axis · normal != 0", f);
            assert!(approx(sdf::dot(va, n), 0.0),
                "face {:?} v_axis · normal != 0", f);
            assert!(approx(sdf::dot(ua, va), 0.0),
                "face {:?} u_axis · v_axis != 0", f);
            assert!(approx(sdf::length(ua), 1.0));
            assert!(approx(sdf::length(va), 1.0));
        }
    }

    #[test]
    fn tangent_basis_is_right_handed() {
        // u × v should equal the outward normal (right-handed).
        for &f in &Face::ALL {
            let n = f.normal();
            let (ua, va) = f.tangents();
            let cross = [
                ua[1] * va[2] - ua[2] * va[1],
                ua[2] * va[0] - ua[0] * va[2],
                ua[0] * va[1] - ua[1] * va[0],
            ];
            assert!(approx_v(cross, n),
                "face {:?}: u × v = {:?}, expected {:?}", f, cross, n);
        }
    }

    #[test]
    fn world_to_coord_inverts_coord_to_world() {
        let center = [1.5, 1.5, 1.5];
        // Try a bunch of face/uv/r combinations.
        let cases = [
            (Face::PosX, 0.0, 0.0, 0.5),
            (Face::NegY, 0.3, -0.7, 1.2),
            (Face::PosZ, -0.9, 0.9, 0.01),
            (Face::NegX, 0.6, 0.4, 2.0),
            (Face::PosY, -0.2, -0.2, 0.8),
            (Face::NegZ, 0.0, 0.0, 1.0),
        ];
        for &(face, u, v, r) in &cases {
            let world = coord_to_world(center, CubeSphereCoord { face, u, v, r });
            let back = world_to_coord(center, world).unwrap();
            assert_eq!(back.face, face, "face mismatch for {:?}", (face, u, v, r));
            assert!(approx(back.u, u), "u mismatch: {} vs {}", back.u, u);
            assert!(approx(back.v, v), "v mismatch: {} vs {}", back.v, v);
            assert!(approx(back.r, r), "r mismatch: {} vs {}", back.r, r);
        }
    }

    #[test]
    fn adjacent_cells_share_edges() {
        // Two neighboring cells on the same face, sharing the
        // edge u = u_shared: their matching corners must be
        // bit-identical in world space (same call → same floats).
        let center = [0.0, 0.0, 0.0];
        let (face, v0, v1, r0, r1) = (Face::PosX, -0.2, -0.1, 1.0, 1.01);
        let shared_u = 0.5;

        let left_right_u1 = coord_to_world(center,
            CubeSphereCoord { face, u: shared_u, v: v0, r: r0 });
        let right_left_u0 = coord_to_world(center,
            CubeSphereCoord { face, u: shared_u, v: v0, r: r0 });
        assert_eq!(left_right_u1, right_left_u0,
            "cells meeting at u={shared_u} on the same face must share corners exactly");

        // And at the outer radius.
        let lr1 = coord_to_world(center,
            CubeSphereCoord { face, u: shared_u, v: v1, r: r1 });
        let rl1 = coord_to_world(center,
            CubeSphereCoord { face, u: shared_u, v: v1, r: r1 });
        assert_eq!(lr1, rl1);
    }

    #[test]
    fn faces_meet_at_cube_edges() {
        // The seam between +X face and +Y face lives on the line
        // x = y = 1 in cube-space. On +X this is v = 1 (since
        // v_axis = +Y); on +Y this is u = 1 (since u_axis = +X).
        // Both parameterize the same seam line with a free z
        // coordinate — varied here by u on +X and v on +Y.
        for &t in &[-0.8, -0.3, 0.0, 0.3, 0.8] {
            let on_posx = face_uv_to_dir(Face::PosX, t, 1.0);
            let on_posy = face_uv_to_dir(Face::PosY, 1.0, t);
            assert!(approx_v(on_posx, on_posy),
                "seam +X/+Y at t={}: {:?} vs {:?}", t, on_posx, on_posy);
        }
    }

    #[test]
    fn block_corners_have_radial_bulge() {
        // Inner-face corners are at radius r, outer at r + dr.
        let center = [0.0, 0.0, 0.0];
        let corners = block_corners(center, Face::PosX, -0.1, -0.1, 1.0, 0.2, 0.2, 0.05);
        for i in 0..4 {
            let len = sdf::length(corners[i]);
            assert!(approx(len, 1.0), "inner corner {} not at r=1: len={}", i, len);
        }
        for i in 4..8 {
            let len = sdf::length(corners[i]);
            assert!(approx(len, 1.05), "outer corner {} not at r=1.05: len={}", i, len);
        }
    }

    #[test]
    fn block_edges_reference_valid_corners() {
        for &(a, b) in &BLOCK_EDGES {
            assert!(a < 8 && b < 8);
            assert_ne!(a, b);
        }
        assert_eq!(BLOCK_EDGES.len(), 12);
    }

    #[test]
    fn world_to_coord_picks_correct_face() {
        let c = [0.0, 0.0, 0.0];
        // Strong +X direction.
        assert_eq!(world_to_coord(c, [1.0, 0.1, 0.1]).unwrap().face, Face::PosX);
        // Strong -Y direction.
        assert_eq!(world_to_coord(c, [0.1, -1.0, 0.1]).unwrap().face, Face::NegY);
        // Strong +Z direction.
        assert_eq!(world_to_coord(c, [0.2, 0.2, 1.0]).unwrap().face, Face::PosZ);
    }

    #[test]
    fn cube_sphere_planet_generates_solid_cells() {
        let sdf = Planet {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
            noise_scale: 0.05,
            noise_freq: 5.0,
            noise_seed: 1,
            gravity: 1.0,
            influence_radius: 2.0,
            surface_block: block::GRASS,
            core_block: block::STONE,
        };
        let planet = generate_from_sdf([0.0, 0.0, 0.0], 1.0, 8, &sdf);
        // With radius = SDF radius and small noise, about half the
        // cells should be solid (noise pushes some above, some below).
        let solid_count = planet.blocks.iter().filter(|&&b| b != 0).count();
        assert!(solid_count > 0, "expected some solid cells");
        assert!(solid_count < planet.blocks.len(), "expected some empty cells");
    }

    #[test]
    fn cube_sphere_planet_fully_solid_when_sampled_below_surface() {
        let sdf = Planet {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
            noise_scale: 0.0,
            noise_freq: 1.0,
            noise_seed: 1,
            gravity: 1.0,
            influence_radius: 2.0,
            surface_block: block::GRASS,
            core_block: block::STONE,
        };
        // Sample well below the undisplaced surface.
        let planet = generate_from_sdf([0.0, 0.0, 0.0], 0.5, 6, &sdf);
        assert!(planet.blocks.iter().all(|&b| b != 0),
            "every cell should be solid at r=0.5 for a radius-1 SDF");
    }

    #[test]
    fn cube_sphere_planet_fully_empty_when_sampled_above_surface() {
        let sdf = Planet {
            center: [0.0, 0.0, 0.0],
            radius: 1.0,
            noise_scale: 0.0,
            noise_freq: 1.0,
            noise_seed: 1,
            gravity: 1.0,
            influence_radius: 2.0,
            surface_block: block::GRASS,
            core_block: block::STONE,
        };
        // Sample well above the surface.
        let planet = generate_from_sdf([0.0, 0.0, 0.0], 2.0, 6, &sdf);
        // The fallback "debug centers" path lights 6 cells. All else empty.
        let solid = planet.blocks.iter().filter(|&&b| b != 0).count();
        assert_eq!(solid, 6, "only the 6 fallback debug cells should be solid");
    }

    #[test]
    fn face_uv_stays_in_unit_square_for_on_face_points() {
        // Any direction on or near a face (tilted less than 45°
        // from the normal) should yield |u|, |v| ≤ 1.
        for &f in &Face::ALL {
            for &tilt in &[0.1, 0.3, 0.7] {
                // Tilt by `tilt` along each tangent.
                let (ua, va) = f.tangents();
                let dir = sdf::normalize(sdf::add(
                    f.normal(),
                    sdf::add(sdf::scale(ua, tilt), sdf::scale(va, tilt)),
                ));
                let c = world_to_coord([0.0, 0.0, 0.0], dir).unwrap();
                assert_eq!(c.face, f, "tilt {} on face {:?} picked {:?}", tilt, f, c.face);
                assert!(c.u.abs() <= 1.0 + 1e-5, "u = {} out of range", c.u);
                assert!(c.v.abs() <= 1.0 + 1e-5, "v = {} out of range", c.v);
            }
        }
    }
}
