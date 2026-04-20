//! Cubed-sphere geometry and worldgen helpers.
//!
//! The planet lives **inside the voxel tree** as a `NodeKind::
//! CubedSphereBody` node. Its 27 children carry the planet's
//! structure:
//!
//! - The **6 face-center slots** (per `FACE_SLOTS` below) hold the
//!   face subtrees. Each face subtree's root is tagged
//!   `NodeKind::CubedSphereFace { face }` so the shader / CPU
//!   raycast know to interpret slot indices as `(u_slot, v_slot,
//!   r_slot)` instead of `(x_slot, y_slot, z_slot)`.
//! - The **center slot (1, 1, 1)** holds the uniform interior
//!   filler — a dedup'd chain of `Block(core_block)`.
//! - The **other 20 slots** are `Empty` (corners and edges of the
//!   containing cube cell, which the sphere doesn't fill).
//!
//! Radii (`inner_r`, `outer_r`) live on the body node's `NodeKind`
//! and are expressed in the **containing cell's local `[0, 1)`
//! frame**. The shader scales them by the body cell's render-frame
//! size at draw time.
//!
//! This module is pure geometry + tree construction. It has no
//! `SphericalPlanet` struct, no parallel raycaster, no separate
//! GPU buffers — all of that has been replaced by the unified
//! tree-walk + NodeKind dispatch in the shader and `edit.rs`.

use super::sdf::{self, Planet, Vec3};
use super::tree::{
    empty_children, slot_index, uniform_children, Child, NodeId, NodeKind, NodeLibrary,
};

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

/// Slot in a `CubedSphereBody` node's 27-grid that holds each
/// face's subtree. Indexed by `Face as usize`. The non-face slots
/// (center + 20 corners/edges) are filled with the interior or
/// empty respectively.
pub const FACE_SLOTS: [usize; 6] = [
    slot_index(2, 1, 1), // PosX
    slot_index(0, 1, 1), // NegX
    slot_index(1, 2, 1), // PosY
    slot_index(1, 0, 1), // NegY
    slot_index(1, 1, 2), // PosZ
    slot_index(1, 1, 0), // NegZ
];

/// Slot in a `CubedSphereBody` node's 27-grid that holds the
/// interior filler.
pub const INTERIOR_SLOT: usize = slot_index(1, 1, 1);

/// A point in cubed-sphere coordinates.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct CubeSphereCoord {
    pub face: Face,
    pub u: f32,
    pub v: f32,
    pub r: f32,
}

/// Convert (face, u, v) → unit direction (equal-angle warp).
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

/// Inverse of the equal-angle warp.
#[inline]
pub fn cube_to_ea(c: f32) -> f32 {
    c.atan() * (4.0 / std::f32::consts::PI)
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
    } else {
        if n[2] > 0.0 { Face::PosZ } else { Face::NegZ }
    }
}

/// Given a planet center and a `CubeSphereCoord`, return the
/// world-space position of that point.
pub fn coord_to_world(center: Vec3, c: CubeSphereCoord) -> Vec3 {
    let dir = face_uv_to_dir(c.face, c.u, c.v);
    sdf::add(center, sdf::scale(dir, c.r))
}

/// World-space position → cubed-sphere coords relative to a center.
pub fn world_to_coord(center: Vec3, pos: Vec3) -> Option<CubeSphereCoord> {
    let d = sdf::sub(pos, center);
    let r = sdf::length(d);
    if r < 1e-12 { return None; }
    let dir = sdf::scale(d, 1.0 / r);

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
        face, u: cube_to_ea(cube_u), v: cube_to_ea(cube_v), r,
    })
}

/// The eight world-space corners of a block spanning
/// `[u, u+du] × [v, v+dv] × [r, r+dr]` on `face`, around `center`.
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

/// The twelve edges of a cubed-sphere block as pairs of corner
/// indices into `block_corners`'s output.
pub const BLOCK_EDGES: [(usize, usize); 12] = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
];

// ────────────────────────────────────────────────── body insertion

/// Max levels the SDF recursion is allowed to descend into a face
/// subtree before committing to uniform solid-or-empty based on the
/// cell's center sample. Below this, the remaining tree depth is
/// wrapped in a dedup'd uniform filler.
///
/// Face subtrees are sampled in `(u, v, r)` spherical coordinates —
/// only 6 of them (one per face), each covering the face's full
/// angular range. The straddle cell count at the budget boundary is
/// ~6 × 9^N (2D surface through 3D angular grid), much smaller than
/// a 27-slot Cartesian-indexed worldgen would give at the same
/// resolution. 6 levels gives 3⁶ = 729 angular voxels across each
/// face, enough for smooth surface detail.
const SDF_DETAIL_LEVELS: u32 = 6;

/// Build the body as a cubed-sphere tree: 6 face subtrees carrying
/// SDF-carved content in face-local `(u, v, r)` coords, plus a
/// uniform interior filler at the center slot. The body node carries
/// `NodeKind::CubedSphereBody { inner_r, outer_r }` so the renderer
/// can dispatch sphere-SDF + face-walker; each face subtree root
/// carries `NodeKind::CubedSphereFace { face }` so the camera
/// pipeline can rotate "up" to the face's radial axis.
///
/// Face subtree slot `(us, vs, rs)` stores content sampled at
/// spherical body position `face_uv_to_dir(u, v) * (inner + r * shell)
/// + center`. Critically, the `r` axis of the subtree is the local
/// **radial** direction at any `(u, v)`. Stacking cells along
/// `rs+1` moves radially outward from the planet core, so a tower
/// built at ANY angular position is perpendicular to gravity.
///
/// `inner_r` and `outer_r` are in the containing-cell's local
/// `[0, 1)³` frame (per the spec's §1d). `depth` is the maximum
/// recursion depth of each face subtree.
///
/// The SDF (`sdf`) is sampled in the containing cell's local frame
/// — its `center` should be `(0.5, 0.5, 0.5)` and its `radius` and
/// `noise_scale` should be in cell-local units.
pub fn insert_spherical_body(
    lib: &mut NodeLibrary,
    inner_r: f32,
    outer_r: f32,
    depth: u32,
    sdf: &Planet,
) -> NodeId {
    debug_assert!(0.0 < inner_r && inner_r < outer_r && outer_r <= 0.5,
        "radii must satisfy 0 < inner_r < outer_r <= 0.5 (cell-local)");

    let body_center: Vec3 = [0.5, 0.5, 0.5];
    let sdf_budget = depth.min(SDF_DETAIL_LEVELS);

    // Precompute uniform subtrees so commit-to-uniform in the
    // SDF-carving recursion is O(1) instead of O(depth). See the
    // old Cartesian-indexed worldgen commit for the perf rationale;
    // still ~10× savings even for the narrower 6-face layout.
    let depth_plus_one = depth as usize + 1;
    let mut uniform_empty: Vec<NodeId> = Vec::with_capacity(depth_plus_one);
    let mut uniform_block: std::collections::HashMap<u8, Vec<Child>> =
        std::collections::HashMap::new();
    let block_kinds: [u8; 3] = [
        sdf.surface_block,
        crate::world::palette::block::DIRT,
        sdf.core_block,
    ];
    {
        let empty_leaf = lib.insert(empty_children());
        uniform_empty.push(empty_leaf);
        uniform_empty.push(empty_leaf);
        for _ in 2..=depth {
            let prev_e = *uniform_empty.last().unwrap();
            let next_e = lib.insert(uniform_children(Child::Node(prev_e)));
            uniform_empty.push(next_e);
        }
        for b in block_kinds.iter().copied() {
            if uniform_block.contains_key(&b) { continue; }
            let mut v: Vec<Child> = Vec::with_capacity(depth_plus_one);
            v.push(Child::Block(b));
            let leaf = lib.insert(uniform_children(Child::Block(b)));
            v.push(Child::Node(leaf));
            for _ in 2..=depth {
                let prev = *v.last().unwrap();
                let id = lib.insert(uniform_children(prev));
                v.push(Child::Node(id));
            }
            uniform_block.insert(b, v);
        }
    }

    // Build the 6 face subtrees. Each face's TOP node gets
    // `NodeKind::CubedSphereFace { face }` so downstream walkers
    // know to interpret slot indices as `(u_slot, v_slot, r_slot)`.
    // Internal nodes stay plain Cartesian (slot_index is numerically
    // identical — the face tag propagates semantic meaning, not
    // storage layout).
    let mut face_root_children: [Child; 6] = [Child::Empty; 6];
    for &face in &Face::ALL {
        let child = build_face_subtree(
            lib, face, body_center,
            -1.0, 1.0, -1.0, 1.0, inner_r, outer_r,
            depth, sdf_budget, sdf,
            &uniform_empty, &uniform_block,
        );
        let face_root_id = match child {
            Child::Node(id) => {
                let n = lib.get(id).expect("face root just inserted");
                let children = n.children;
                lib.insert_with_kind(children, NodeKind::CubedSphereFace { face })
            }
            Child::Empty => lib.insert_with_kind(
                empty_children(),
                NodeKind::CubedSphereFace { face },
            ),
            Child::Block(b) => lib.insert_with_kind(
                uniform_children(Child::Block(b)),
                NodeKind::CubedSphereFace { face },
            ),
        };
        face_root_children[face as usize] = Child::Node(face_root_id);
    }

    // Body's 27 children: 6 face subtrees at face-center slots,
    // uniform interior at the center, empty elsewhere.
    let mut body_children = empty_children();
    for &face in &Face::ALL {
        body_children[FACE_SLOTS[face as usize]] =
            face_root_children[face as usize];
    }
    let interior = uniform_block
        .get(&sdf.core_block)
        .and_then(|v| v.get(depth as usize).copied())
        .unwrap_or_else(|| lib.build_uniform_subtree(sdf.core_block, depth));
    body_children[INTERIOR_SLOT] = interior;

    lib.insert_with_kind(
        body_children,
        NodeKind::CubedSphereBody { inner_r, outer_r },
    )
}

/// Recursive builder for a face subtree's content in `(u, v, r)`
/// spherical coords. Each cell's SDF sample point is
/// `face_uv_to_dir(u_c, v_c) * r_c + body_center`, so the cell's
/// `r` axis at that `(u, v)` is the local **radial** direction on
/// the sphere. Entire-in and entire-out cells collapse to uniform
/// subtrees (via precomputed table); straddlers recurse until
/// `sdf_budget` is exhausted.
fn build_face_subtree(
    lib: &mut NodeLibrary,
    face: Face,
    body_center: Vec3,
    u_lo: f32, u_hi: f32,
    v_lo: f32, v_hi: f32,
    r_lo: f32, r_hi: f32,
    depth: u32,
    sdf_budget: u32,
    sdf: &Planet,
    uniform_empty: &[NodeId],
    uniform_block: &std::collections::HashMap<u8, Vec<Child>>,
) -> Child {
    let u_c = 0.5 * (u_lo + u_hi);
    let v_c = 0.5 * (v_lo + v_hi);
    let r_c = 0.5 * (r_lo + r_hi);
    let p_center = coord_to_world(body_center, CubeSphereCoord { face, u: u_c, v: v_c, r: r_c });
    let d_center = sdf.distance(p_center);

    let du = u_hi - u_lo;
    let dv = v_hi - v_lo;
    let dr = r_hi - r_lo;
    // Cell bounding sphere: lateral half-width ≈ `r_hi × 0.5 ×
    // max(du, dv)` because the cube-to-sphere warp scales linearly
    // with radius. Radial half-width = `0.5 × dr`.
    let lateral_half = r_hi * 0.5 * du.max(dv);
    let radial_half = 0.5 * dr;
    let cell_rad = (lateral_half * lateral_half + radial_half * radial_half).sqrt();

    if d_center > cell_rad {
        if depth == 0 { return Child::Empty; }
        return Child::Node(uniform_empty[depth as usize]);
    }
    if d_center < -cell_rad {
        let b = sdf.block_at(p_center);
        if depth == 0 { return Child::Block(b); }
        return uniform_block
            .get(&b)
            .and_then(|v| v.get(depth as usize).copied())
            .unwrap_or_else(|| lib.build_uniform_subtree(b, depth));
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
            let b = sdf.block_at(p_center);
            uniform_block
                .get(&b)
                .and_then(|v| v.get(depth as usize).copied())
                .unwrap_or_else(|| lib.build_uniform_subtree(b, depth))
        } else {
            Child::Node(uniform_empty[depth as usize])
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
                    us_lo, us_hi, vs_lo, vs_hi, rs_lo, rs_hi,
                    depth - 1, sdf_budget - 1, sdf,
                    uniform_empty, uniform_block,
                );
            }
        }
    }
    Child::Node(lib.insert(children))
}

// ────────────────────────────────────────────────────────────── tests

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
            assert!(approx_v(dir, f.normal()));
        }
    }

    #[test]
    fn face_corners_are_unit_vectors() {
        for &f in &Face::ALL {
            for &(u, v) in &[(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0),
                              (0.5, -0.3), (0.0, 0.0)] {
                let d = face_uv_to_dir(f, u, v);
                assert!(approx(sdf::length(d), 1.0));
            }
        }
    }

    #[test]
    fn world_to_coord_inverts_coord_to_world() {
        let center = [0.5, 0.5, 0.5];
        for &(face, u, v, r) in &[
            (Face::PosX, 0.0, 0.0, 0.3),
            (Face::NegY, 0.3, -0.7, 0.45),
            (Face::PosZ, -0.9, 0.9, 0.15),
        ] {
            let world = coord_to_world(center, CubeSphereCoord { face, u, v, r });
            let back = world_to_coord(center, world).unwrap();
            assert_eq!(back.face, face);
            assert!(approx(back.u, u));
            assert!(approx(back.v, v));
            assert!(approx(back.r, r));
        }
    }

    #[test]
    fn insert_body_creates_body_node_with_face_children() {
        let mut lib = NodeLibrary::default();
        let sdf = Planet {
            center: [0.5, 0.5, 0.5],
            radius: 0.32,
            noise_scale: 0.0, noise_freq: 1.0, noise_seed: 0,
            gravity: 0.0, influence_radius: 1.0,
            surface_block: block::GRASS, core_block: block::STONE,
        };
        let body_id = insert_spherical_body(&mut lib, 0.12, 0.45, 6, &sdf);
        let body = lib.get(body_id).expect("body node exists");
        assert!(matches!(body.kind, NodeKind::CubedSphereBody { .. }));
        // Each of the 6 face-center slots holds a face subtree.
        for &face in &Face::ALL {
            let slot = FACE_SLOTS[face as usize];
            match body.children[slot] {
                Child::Node(face_id) => {
                    let face_node = lib.get(face_id).expect("face root exists");
                    match face_node.kind {
                        NodeKind::CubedSphereFace { face: f } => assert_eq!(f, face),
                        _ => panic!("face slot {slot} not a CubedSphereFace"),
                    }
                }
                _ => panic!("face slot {slot} not a Node"),
            }
        }
        // Center slot is interior filler (uniform stone subtree).
        // For a sphere of radius 0.32 at (0.5, 0.5, 0.5), the interior
        // slot's sub-box [1/3, 2/3]³ has half-diagonal ≈ 0.289 and the
        // SDF is -0.32 at its center — entirely inside → uniform stone.
        match body.children[INTERIOR_SLOT] {
            Child::Node(id) => {
                let n = lib.get(id).unwrap();
                assert_eq!(n.uniform_type, block::STONE);
            }
            Child::Block(b) => assert_eq!(b, block::STONE),
            Child::Empty => panic!("interior slot is empty"),
        }
        // Non-face, non-interior slots are always Empty in the
        // u/v/r-indexed layout — the sphere content is entirely
        // routed through the 6 face subtrees.
        for slot in 0..27 {
            if slot == INTERIOR_SLOT { continue; }
            if FACE_SLOTS.contains(&slot) { continue; }
            assert!(matches!(body.children[slot], Child::Empty),
                "slot {slot} should be Empty in the u/v/r-indexed layout");
        }
    }
}
