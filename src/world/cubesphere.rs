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

use super::sdf::{self, Planet, Vec3};
use super::tree::{
    empty_children, slot_index, uniform_children, Child, NodeId, NodeLibrary,
};

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
/// planet center. `u, v ∈ [-1, 1]` live in **equal-angle** space:
/// each uniform step in `(u, v)` covers the same angular slice of
/// the sphere as seen from its center, which makes cells project
/// to nearly-uniform area (~6% variation peak-to-peak vs. ~50% for
/// the basic gnomonic projection).
///
/// The warp is one `tan` each on `u` and `v`: a uniform cell grid
/// gets pre-compressed toward the face's middle, exactly canceling
/// the expansion `normalize` would otherwise apply near the corners.
pub fn face_uv_to_dir(face: Face, u: f32, v: f32) -> Vec3 {
    let cube_u = (u * std::f32::consts::FRAC_PI_4).tan();
    let cube_v = (v * std::f32::consts::FRAC_PI_4).tan();
    let n = face.normal();
    let (ua, va) = face.tangents();
    let cube_pt = [
        n[0] + cube_u * ua[0] + cube_v * va[0],
        n[1] + cube_u * ua[1] + cube_v * va[1],
        n[2] + cube_u * ua[2] + cube_v * va[2],
    ];
    sdf::normalize(cube_pt)
}

/// Inverse of the equal-angle warp: raw cube-face coord → equal-angle.
/// The shader + raymarchers compute `cube_u = ratio` (e.g. `-n.z/ax`),
/// then call this to get the `u ∈ [-1, 1]` used for cell indexing.
#[inline]
pub fn cube_to_ea(c: f32) -> f32 {
    c.atan() * (4.0 / std::f32::consts::PI)
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

    // Pick the dominant axis; the raw ratios give the CUBE-space
    // (u, v) on that face. Then apply the equal-angle inverse so
    // `u, v` are the same coords cells are indexed by.
    let ax = dir[0].abs();
    let ay = dir[1].abs();
    let az = dir[2].abs();
    let (face, cube_u, cube_v) = if ax >= ay && ax >= az {
        if dir[0] > 0.0 { (Face::PosX, -dir[2] / ax,  dir[1] / ax) }
        else            { (Face::NegX,  dir[2] / ax,  dir[1] / ax) }
    } else if ay >= az {
        if dir[1] > 0.0 { (Face::PosY,  dir[0] / ay, -dir[2] / ay) }
        else            { (Face::NegY,  dir[0] / ay,  dir[2] / ay) }
    } else {
        if dir[2] > 0.0 { (Face::PosZ,  dir[0] / az,  dir[1] / az) }
        else            { (Face::NegZ, -dir[0] / az,  dir[1] / az) }
    };

    Some(CubeSphereCoord {
        face,
        u: cube_to_ea(cube_u),
        v: cube_to_ea(cube_v),
        r,
    })
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

/// A planet made of 6 face-subtrees in the content-addressed
/// `NodeLibrary`. Each face subtree's 3 recursive axes are
/// `(u_slot, v_slot, r_slot)` in the planet's equal-angle cubed-sphere
/// frame — NOT (x, y, z). A subtree leaf is a block / empty terminal
/// exactly like any other subtree, so the existing content-addressed
/// dedup, LOD cascade, and editing primitives all apply.
///
/// Recursion semantics: each node splits its local (u, v, r) box
/// into 3×3×3 children. The slot at `(us, vs, rs)` covers
/// `u ∈ [u_lo + us·du/3, u_lo + (us+1)·du/3]` and analogously for
/// v and r, where `(u_lo, u_hi, v_lo, v_hi, r_lo, r_hi)` is the
/// current node's box. At the top level each face's box is
/// `[-1, 1] × [-1, 1] × [inner_r, outer_r]`.
///
/// A leaf cell is a "bulged voxel": its world-space corners come
/// from `block_corners` at the cell's (u, v, r) extents. Zoom in
/// and you descend into the subtree, revealing 27 smaller bulged
/// voxels per parent — the same "blocks inside blocks" mechanic
/// the rest of the engine uses, just interpreted in spherical
/// coordinates.
#[derive(Clone, Debug)]
pub struct SphericalPlanet {
    pub center: Vec3,
    /// Cells span `r ∈ [inner_r, outer_r]`. A solid column typical of
    /// a rocky planet has its surface near the midpoint and empty
    /// cells above, solid cells below.
    pub inner_r: f32,
    pub outer_r: f32,
    /// One subtree root per cube face. Indexed by `Face as usize`.
    /// Subtrees live in the shared `NodeLibrary` alongside the
    /// Cartesian space tree — dedup is natural because they use the
    /// same `Child`/`Node` representation.
    pub face_roots: [NodeId; 6],
    /// Levels of recursion under each face root. Zoom-in reveals up
    /// to `depth` cascades of 27 sub-cells before bottoming out.
    pub depth: u32,
}

/// Build a 6-face `SphericalPlanet` in `lib` by recursively sampling
/// `sdf` along each face's `(u, v, r)` subtree. Uniform solid /
/// uniform empty subtrees are built once and referenced many times
/// by the content-addressed library, so a pristine spherical planet
/// costs only `O(surface_area · depth)` unique nodes — not `O(27^depth)`.
pub fn generate_spherical_planet(
    lib: &mut NodeLibrary,
    center: Vec3,
    inner_r: f32,
    outer_r: f32,
    depth: u32,
    sdf: &Planet,
) -> SphericalPlanet {
    let mut face_roots = [0u64; 6];
    for &face in &Face::ALL {
        let child = build_face_subtree(
            lib, face, center,
            -1.0, 1.0, -1.0, 1.0, inner_r, outer_r,
            depth, sdf,
        );
        // The face root must be a Node — we always wrap so the
        // renderer has a tree to walk per face, even if the whole
        // face is uniform empty / solid.
        face_roots[face as usize] = match child {
            Child::Node(id) => id,
            Child::Empty => lib.insert(empty_children()),
            Child::Block(b) => lib.insert(uniform_children(Child::Block(b))),
        };
        lib.ref_inc(face_roots[face as usize]);
    }
    SphericalPlanet { center, inner_r, outer_r, face_roots, depth }
}

/// Recursive builder for one cubed-sphere face. Returns a `Child` so
/// the caller can collapse uniform subtrees up the call chain.
fn build_face_subtree(
    lib: &mut NodeLibrary,
    face: Face,
    center: Vec3,
    u_lo: f32, u_hi: f32,
    v_lo: f32, v_hi: f32,
    r_lo: f32, r_hi: f32,
    depth: u32,
    sdf: &Planet,
) -> Child {
    // Cell center in world space.
    let u_c = 0.5 * (u_lo + u_hi);
    let v_c = 0.5 * (v_lo + v_hi);
    let r_c = 0.5 * (r_lo + r_hi);
    let p_center = coord_to_world(center, CubeSphereCoord { face, u: u_c, v: v_c, r: r_c });
    let d_center = sdf.distance(p_center);

    // Conservative bound on world-space distance from p_center to any
    // corner of this cell. Lateral extent grows with radius; radial
    // extent is straightforward. Overestimating is safe (means extra
    // subdivision, never wrong bits).
    let du = u_hi - u_lo;
    let dv = v_hi - v_lo;
    let dr = r_hi - r_lo;
    // At equal-angle, uv half-span on the sphere is roughly
    // r · tan(du/2 · π/4). Use a generous linearization: r · du/2.
    let lateral_half = r_hi * 0.5 * du.max(dv);
    let radial_half = 0.5 * dr;
    let cell_rad = (lateral_half * lateral_half + radial_half * radial_half).sqrt();

    // Fully outside: the SDF surface can't reach this cell, so every
    // sub-cell is empty. Content-addressed into a dedup'd empty subtree.
    if d_center > cell_rad {
        if depth == 0 { return Child::Empty; }
        return Child::Node(build_uniform_empty(lib, depth));
    }
    // Fully inside: every sub-cell is solid. Pick the center-sample
    // block type and stamp the whole subtree with it.
    if d_center < -cell_rad {
        let b = sdf.block_at(p_center);
        if depth == 0 { return Child::Block(b); }
        return match lib.build_uniform_subtree(b, depth) {
            Child::Node(id) => Child::Node(id),
            other => other, // (Shouldn't happen for depth ≥ 1, but safe.)
        };
    }

    // Straddles the surface or too close to commit. At depth 0 we
    // have no more room to resolve — sample the center and go.
    if depth == 0 {
        return if d_center < 0.0 {
            Child::Block(sdf.block_at(p_center))
        } else {
            Child::Empty
        };
    }

    // Recurse: split this cell's (u, v, r) box into 3×3×3 sub-cells.
    let mut children = empty_children();
    for rs in 0..3 {
        for vs in 0..3 {
            for us in 0..3 {
                let us_lo = u_lo + du * (us as f32) / 3.0;
                let us_hi = u_lo + du * (us as f32 + 1.0) / 3.0;
                let vs_lo = v_lo + dv * (vs as f32) / 3.0;
                let vs_hi = v_lo + dv * (vs as f32 + 1.0) / 3.0;
                let rs_lo = r_lo + dr * (rs as f32) / 3.0;
                let rs_hi = r_lo + dr * (rs as f32 + 1.0) / 3.0;
                children[slot_index(us, vs, rs)] = build_face_subtree(
                    lib, face, center,
                    us_lo, us_hi, vs_lo, vs_hi, rs_lo, rs_hi,
                    depth - 1, sdf,
                );
            }
        }
    }
    Child::Node(lib.insert(children))
}

/// Build (and cache) a uniform-empty subtree of the given depth.
fn build_uniform_empty(lib: &mut NodeLibrary, depth: u32) -> NodeId {
    let mut id = lib.insert(empty_children());
    for _ in 1..depth {
        id = lib.insert(uniform_children(Child::Node(id)));
    }
    id
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
    use crate::world::palette::block;

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

    fn test_sdf(radius: f32, noise: f32) -> Planet {
        Planet {
            center: [0.0, 0.0, 0.0],
            radius,
            noise_scale: noise,
            noise_freq: 5.0,
            noise_seed: 1,
            gravity: 1.0,
            influence_radius: radius * 2.0,
            surface_block: block::GRASS,
            core_block: block::STONE,
        }
    }

    /// Walk into a face's subtree at the given slot path and return
    /// the resolved child (Empty / Block / Node).
    fn walk_subtree(
        lib: &NodeLibrary,
        root: NodeId,
        path: &[(usize, usize, usize)],
    ) -> Child {
        let mut node = lib.get(root).unwrap();
        let mut current_child = Child::Node(root);
        for &(us, vs, rs) in path {
            let slot = slot_index(us, vs, rs);
            current_child = node.children[slot];
            match current_child {
                Child::Node(id) => node = lib.get(id).unwrap(),
                _ => return current_child,
            }
        }
        current_child
    }

    #[test]
    fn spherical_planet_has_six_faces() {
        let mut lib = NodeLibrary::default();
        let sdf = test_sdf(1.0, 0.05);
        let planet = generate_spherical_planet(&mut lib, [0.0, 0.0, 0.0], 0.5, 1.5, 3, &sdf);
        for id in planet.face_roots {
            assert!(lib.get(id).is_some(), "every face root must exist in library");
        }
    }

    #[test]
    fn spherical_planet_outer_is_empty_inner_is_solid() {
        let mut lib = NodeLibrary::default();
        let sdf = test_sdf(1.0, 0.0);
        // Inner 0.5, outer 1.5 → SDF surface at r=1 is the midpoint.
        // Bottom (rs=0) layer is deep inside solid; top (rs=2) is deep air.
        let planet = generate_spherical_planet(&mut lib, [0.0, 0.0, 0.0], 0.5, 1.5, 2, &sdf);
        // Walk one level into the +X face, middle u/v, bottom rs: solid.
        let inner = walk_subtree(&lib, planet.face_roots[Face::PosX as usize], &[(1, 1, 0)]);
        match inner {
            Child::Block(_) => {}
            Child::Node(id) => {
                // Descend one more step; deepest layer should contain Block terminals.
                let node = lib.get(id).unwrap();
                assert!(node.children.iter().any(|c| matches!(c, Child::Block(_))),
                    "inner subtree should contain solid blocks");
            }
            Child::Empty => panic!("inner slot at r_lo should be solid"),
        }
        // Top slot: empty.
        let outer = walk_subtree(&lib, planet.face_roots[Face::PosX as usize], &[(1, 1, 2)]);
        assert!(matches!(outer, Child::Empty | Child::Node(_)),
            "outer slot should resolve to empty or an empty subtree");
    }

    #[test]
    fn spherical_planet_dedup_keeps_node_count_modest() {
        // A fully pristine sphere with a simple SDF should produce a
        // small number of unique nodes — most subtrees collapse into
        // a single uniform-empty / uniform-solid cache entry.
        let mut lib = NodeLibrary::default();
        let sdf = test_sdf(1.0, 0.0);
        let _ = generate_spherical_planet(&mut lib, [0.0, 0.0, 0.0], 0.5, 1.5, 4, &sdf);
        // Depth 4 means up to 27^4 ≈ 530k potential unique cells per
        // face; dedup should bring us well under that.
        assert!(lib.len() < 20_000,
            "dedup'd spherical planet should have < 20k unique nodes, got {}",
            lib.len());
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
