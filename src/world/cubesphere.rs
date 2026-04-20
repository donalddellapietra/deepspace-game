//! Cubed-sphere body topology. A spherical body lives inside the
//! voxel tree as a `NodeKind::CubedSphereBody` node whose 27
//! children are laid out as:
//!
//! - **6 face-center slots** (per [`FACE_SLOTS`]) hold the face
//!   subtrees. Each face subtree's root is tagged
//!   `NodeKind::CubedSphereFace { face }` so the walker treats slot
//!   indices as `(u_slot, v_slot, r_slot)` instead of `(x, y, z)`.
//! - **Center slot** (1, 1, 1) = [`INTERIOR_SLOT`] holds the
//!   interior filler (uniform core-block subtree).
//! - The remaining 20 slots are `Empty` — corners and edges of the
//!   containing cube cell that the sphere doesn't fill.
//!
//! Radii `inner_r`, `outer_r` are stored on the body node in its
//! containing cell's local `[0, 1)` frame. The shader multiplies by
//! the body cell's render-frame size at draw time — this is what
//! keeps the sphere precision-safe at deep zoom.

use super::sdf::{self, Planet, Vec3};
use super::tree::{
    empty_children, slot_index, uniform_children, Child, NodeId, NodeKind, NodeLibrary,
};

// ─────────────────────────────────────────────────────────── Face

/// One of the six cube faces.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
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

    /// Outward-pointing unit normal.
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

    /// Orthonormal tangents `(u_axis, v_axis)`. Must match the
    /// shader's `face_u_axis` / `face_v_axis`.
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

// ─────────────────────────────────────────────────── slot layout

/// Body-grid slot index for each face's subtree. Indexed by
/// `Face as usize`. Mirror of the shader's `face_slot()`.
pub const FACE_SLOTS: [usize; 6] = [
    slot_index(2, 1, 1), // PosX
    slot_index(0, 1, 1), // NegX
    slot_index(1, 2, 1), // PosY
    slot_index(1, 0, 1), // NegY
    slot_index(1, 1, 2), // PosZ
    slot_index(1, 1, 0), // NegZ
];

/// Body-grid slot index for the interior filler (body center).
pub const INTERIOR_SLOT: usize = slot_index(1, 1, 1);

// ─────────────────────────────────────────────── projection math

/// A point in cubed-sphere coordinates. `u`/`v` are face-local EA
/// coords in `[-1, 1]²`; `r` is the radial distance from body center.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CubeSphereCoord {
    pub face: Face,
    pub u: f32,
    pub v: f32,
    pub r: f32,
}

/// Equal-area cube-face warp: the cube-tangent coordinate `c` maps
/// to the EA coord `e = atan(c) * 4/π`. Flattens pixel stretching.
#[inline]
pub fn cube_to_ea(c: f32) -> f32 { c.atan() * (4.0 / std::f32::consts::PI) }

#[inline]
pub fn ea_to_cube(e: f32) -> f32 { (e * std::f32::consts::FRAC_PI_4).tan() }

/// Face-local (u, v) ∈ [-1, 1]² -> unit direction on the sphere.
pub fn face_uv_to_dir(face: Face, u: f32, v: f32) -> Vec3 {
    let cu = ea_to_cube(u);
    let cv = ea_to_cube(v);
    let n = face.normal();
    let (ua, va) = face.tangents();
    let cube_pt = [
        n[0] + cu * ua[0] + cv * va[0],
        n[1] + cu * ua[1] + cv * va[1],
        n[2] + cu * ua[2] + cv * va[2],
    ];
    sdf::normalize(cube_pt)
}

/// Pick the dominant cube face for a unit direction.
pub fn pick_face(n: Vec3) -> Face {
    let ax = n[0].abs();
    let ay = n[1].abs();
    let az = n[2].abs();
    if ax >= ay && ax >= az {
        if n[0] > 0.0 { Face::PosX } else { Face::NegX }
    } else if ay >= az {
        if n[1] > 0.0 { Face::PosY } else { Face::NegY }
    } else if n[2] > 0.0 { Face::PosZ } else { Face::NegZ }
}

/// Cube-sphere coord -> world position around `center`.
pub fn coord_to_world(center: Vec3, c: CubeSphereCoord) -> Vec3 {
    let dir = face_uv_to_dir(c.face, c.u, c.v);
    sdf::add(center, sdf::scale(dir, c.r))
}

/// World position -> cube-sphere coord relative to `center`. Returns
/// `None` if the point is exactly at the center (direction undefined).
pub fn world_to_coord(center: Vec3, pos: Vec3) -> Option<CubeSphereCoord> {
    let d = sdf::sub(pos, center);
    let r = sdf::length(d);
    if r < 1e-12 { return None; }
    let dir = sdf::scale(d, 1.0 / r);
    let face = pick_face(dir);
    let n_axis = face.normal();
    let (u_axis, v_axis) = face.tangents();
    let axis_dot = sdf::dot(dir, n_axis);
    if axis_dot.abs() <= 1e-12 { return None; }
    let cu = sdf::dot(dir, u_axis) / axis_dot;
    let cv = sdf::dot(dir, v_axis) / axis_dot;
    Some(CubeSphereCoord {
        face, u: cube_to_ea(cu), v: cube_to_ea(cv), r,
    })
}

/// Eight world-space corners of a UVR block on `face` around `center`.
pub fn block_corners(
    center: Vec3, face: Face,
    u: f32, v: f32, r: f32,
    du: f32, dv: f32, dr: f32,
) -> [Vec3; 8] {
    let mut out = [[0.0; 3]; 8];
    let cs = [
        (u,      v,      r),
        (u + du, v,      r),
        (u,      v + dv, r),
        (u + du, v + dv, r),
        (u,      v,      r + dr),
        (u + du, v,      r + dr),
        (u,      v + dv, r + dr),
        (u + du, v + dv, r + dr),
    ];
    for (i, &(cu, cv, cr)) in cs.iter().enumerate() {
        out[i] = coord_to_world(center, CubeSphereCoord { face, u: cu, v: cv, r: cr });
    }
    out
}

pub const BLOCK_EDGES: [(usize, usize); 12] = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
];

// ───────────────────────────────────────────── body construction

/// Max depth the SDF recursion is allowed to descend into a face
/// subtree. Below this a cell commits to solid-or-empty based on
/// its center sample and wraps the remaining depth in a dedup'd
/// uniform filler. Keeps body-insert wall time bounded independent
/// of requested `depth`.
const SDF_DETAIL_LEVELS: u32 = 4;

/// Build the six face subtrees + interior filler + body node and
/// insert it into `lib`. The caller places the returned body id
/// into some parent cell and bumps its refcount.
///
/// `inner_r`, `outer_r` are in the containing cell's `[0, 1)` frame
/// (so `outer_r <= 0.5` keeps the sphere inside one cell). `depth` is
/// the face subtree depth — also the maximum edit depth inside the
/// face.
pub fn insert_spherical_body(
    lib: &mut NodeLibrary,
    inner_r: f32,
    outer_r: f32,
    depth: u32,
    sdf: &Planet,
) -> NodeId {
    debug_assert!(
        0.0 < inner_r && inner_r < outer_r && outer_r <= 0.5,
        "radii must satisfy 0 < inner_r < outer_r <= 0.5 (cell-local)",
    );

    let body_center: Vec3 = [0.5, 0.5, 0.5];
    let sdf_budget = depth.min(SDF_DETAIL_LEVELS);

    // Build the six face subtrees. Each face's root is tagged
    // `CubedSphereFace { face }` so walkers treat its children's
    // slot coordinates as (u_slot, v_slot, r_slot). Internal nodes
    // stay Cartesian for dedup efficiency — their content semantics
    // are determined by the face-root tag.
    let mut face_roots = [Child::Empty; 6];
    for &face in &Face::ALL {
        let child = build_face_subtree(
            lib, face, body_center,
            inner_r, outer_r,
            -1.0, 1.0, -1.0, 1.0, inner_r, outer_r,
            depth, sdf_budget, sdf,
        );
        let face_root_id = match child {
            Child::Node(id) => {
                let children = lib.get(id).expect("face root just inserted").children;
                lib.insert_with_kind(children, NodeKind::CubedSphereFace { face })
            }
            Child::Empty => {
                lib.insert_with_kind(empty_children(), NodeKind::CubedSphereFace { face })
            }
            Child::Block(b) => {
                lib.insert_with_kind(uniform_children(Child::Block(b)), NodeKind::CubedSphereFace { face })
            }
            Child::EntityRef(_) => unreachable!("sphere worldgen never produces EntityRef"),
        };
        face_roots[face as usize] = Child::Node(face_root_id);
    }

    // Body children: 6 face subtrees, 1 interior, 20 empty.
    let mut children = empty_children();
    for &face in &Face::ALL {
        children[FACE_SLOTS[face as usize]] = face_roots[face as usize];
    }
    children[INTERIOR_SLOT] = lib.build_uniform_subtree(sdf.core_block, depth);

    lib.insert_with_kind(
        children,
        NodeKind::CubedSphereBody { inner_r, outer_r },
    )
}

/// Recursive SDF-driven face subtree builder. Samples the SDF at the
/// current cell's center; if the cell is fully outside or fully
/// inside the influence radius, commits to a uniform subtree at that
/// depth. Otherwise subdivides into 27 children at `(u_slot, v_slot,
/// r_slot)` until `depth == 0` or `sdf_budget == 0`.
#[allow(clippy::too_many_arguments)]
fn build_face_subtree(
    lib: &mut NodeLibrary,
    face: Face,
    body_center: Vec3,
    body_inner_r: f32, body_outer_r: f32,
    u_lo: f32, u_hi: f32,
    v_lo: f32, v_hi: f32,
    r_lo: f32, r_hi: f32,
    depth: u32,
    sdf_budget: u32,
    sdf: &Planet,
) -> Child {
    let uc = 0.5 * (u_lo + u_hi);
    let vc = 0.5 * (v_lo + v_hi);
    let rc = 0.5 * (r_lo + r_hi);
    let p_center = coord_to_world(body_center, CubeSphereCoord { face, u: uc, v: vc, r: rc });
    let d_center = sdf.distance(p_center);

    let du = u_hi - u_lo;
    let dv = v_hi - v_lo;
    let dr = r_hi - r_lo;
    let lateral_half = r_hi * 0.5 * du.max(dv);
    let radial_half = 0.5 * dr;
    let cell_rad = (lateral_half * lateral_half + radial_half * radial_half).sqrt();

    // Fully outside the SDF surface — uniform empty.
    if d_center > cell_rad {
        if depth == 0 { return Child::Empty; }
        return Child::Node(uniform_empty(lib, depth));
    }
    // Fully inside — uniform block.
    if d_center < -cell_rad {
        let b = sdf.block_at(p_center);
        if depth == 0 { return Child::Block(b); }
        return lib.build_uniform_subtree(b, depth);
    }

    if depth == 0 {
        return if d_center < 0.0 {
            Child::Block(sdf.block_at(p_center))
        } else {
            Child::Empty
        };
    }
    if sdf_budget == 0 {
        return if d_center < 0.0 {
            lib.build_uniform_subtree(sdf.block_at(p_center), depth)
        } else {
            Child::Node(uniform_empty(lib, depth))
        };
    }

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
                    lib, face, body_center,
                    body_inner_r, body_outer_r,
                    us_lo, us_hi, vs_lo, vs_hi, rs_lo, rs_hi,
                    depth - 1, sdf_budget - 1, sdf,
                );
            }
        }
    }
    Child::Node(lib.insert(children))
}

fn uniform_empty(lib: &mut NodeLibrary, depth: u32) -> NodeId {
    let mut id = lib.insert(empty_children());
    for _ in 1..depth {
        id = lib.insert(uniform_children(Child::Node(id)));
    }
    id
}

// ──────────────────────────────────────────────────────── tests

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
            assert!(approx_v(face_uv_to_dir(f, 0.0, 0.0), f.normal()));
        }
    }

    #[test]
    fn face_corners_are_unit_vectors() {
        for &f in &Face::ALL {
            for &(u, v) in &[(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0), (0.5, -0.3)] {
                assert!(approx(sdf::length(face_uv_to_dir(f, u, v)), 1.0));
            }
        }
    }

    #[test]
    fn world_coord_round_trip() {
        let center = [0.5, 0.5, 0.5];
        for &(face, u, v, r) in &[
            (Face::PosX, 0.0, 0.0, 0.3),
            (Face::NegY, 0.3, -0.7, 0.45),
            (Face::PosZ, -0.9, 0.9, 0.15),
        ] {
            let w = coord_to_world(center, CubeSphereCoord { face, u, v, r });
            let back = world_to_coord(center, w).unwrap();
            assert_eq!(back.face, face);
            assert!(approx(back.u, u));
            assert!(approx(back.v, v));
            assert!(approx(back.r, r));
        }
    }

    #[test]
    fn insert_body_has_face_and_interior_children() {
        let mut lib = NodeLibrary::default();
        let sdf = Planet {
            center: [0.5, 0.5, 0.5], radius: 0.32,
            noise_scale: 0.0, noise_freq: 1.0, noise_seed: 0,
            gravity: 0.0, influence_radius: 1.0,
            surface_block: block::GRASS, core_block: block::STONE,
        };
        let body = insert_spherical_body(&mut lib, 0.12, 0.45, 6, &sdf);
        let node = lib.get(body).unwrap();
        assert!(matches!(node.kind, NodeKind::CubedSphereBody { .. }));
        for &face in &Face::ALL {
            let slot = FACE_SLOTS[face as usize];
            match node.children[slot] {
                Child::Node(id) => match lib.get(id).unwrap().kind {
                    NodeKind::CubedSphereFace { face: f } => assert_eq!(f, face),
                    _ => panic!("face slot {slot} not a CubedSphereFace"),
                },
                _ => panic!("face slot {slot} not a Node"),
            }
        }
        match node.children[INTERIOR_SLOT] {
            Child::Node(id) => assert_eq!(lib.get(id).unwrap().uniform_type, block::STONE),
            Child::Block(b) => assert_eq!(b, block::STONE),
            _ => panic!("interior slot missing"),
        }
        for slot in 0..27 {
            if slot == INTERIOR_SLOT || FACE_SLOTS.contains(&slot) { continue; }
            assert!(matches!(node.children[slot], Child::Empty));
        }
    }
}
