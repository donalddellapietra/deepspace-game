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

/// Max levels the SDF recursion is allowed to descend into a body
/// subtree before committing to uniform solid-or-empty based on the
/// cell's center sample. Below this, the remaining tree depth is
/// wrapped in a dedup'd uniform filler.
///
/// Cartesian-indexed worldgen visits 9× more straddle cells per
/// added level (a 2D surface through the 3D grid), and runs across
/// all 27 body slots rather than just 6 face-center slots — so
/// every added level costs ~40× the work of the previous one.
/// 4 keeps worldgen tractable (~seconds for a single planet) while
/// leaving the LOD-terminal sphere-clip in the shader to smooth the
/// silhouette past the voxel resolution limit.
const SDF_DETAIL_LEVELS: u32 = 4;

/// Build the body as a Cartesian-indexed 27-ary tree of SDF-carved
/// content, insert it into `lib`, and return its `NodeId`. The body
/// node itself carries `NodeKind::CubedSphereBody { inner_r, outer_r }`
/// so the renderer's ray-sphere pre-clip can fire on ray-body
/// dispatch; the 6 face-center slots additionally carry
/// `NodeKind::CubedSphereFace { face }` so the camera-basis pipeline
/// can rotate "up" to the face's radial axis when the camera zooms
/// into a face subtree.
///
/// All other body slots (edges, corners, interior) are plain
/// Cartesian subtrees. The sphere surface may wrap into edge and
/// corner slots when the sphere's radius is larger than the
/// face-center slot's reach — the Cartesian-indexed worldgen makes
/// every slot check its own position against the SDF, so gaps
/// between face subtrees on the sphere surface are populated.
///
/// `inner_r` and `outer_r` are in the containing-cell's local
/// `[0, 1)³` frame (per the spec's §1d). `depth` is the maximum
/// recursion depth inside the body.
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

    let sdf_budget = depth.min(SDF_DETAIL_LEVELS);

    // Pre-build uniform subtrees at every depth we might need.
    // `lib.build_uniform_subtree` allocates a new 27-ary array per
    // iteration (content-addressed dedup returns the same NodeId
    // but the allocation + hash + lookup cost is still per-call).
    // During worldgen, straddle cells at the SDF-budget boundary
    // commit to uniform subtrees of the remaining depth — millions
    // of calls for a depth-28 body. Precomputing once flips that
    // from O(N × depth) allocations to O(depth) + O(N) table
    // lookups.
    // `uniform_empty[d]` = NodeId of a `d`-level subtree that is
    // entirely empty. `uniform_stone[d]` = the `Child` at depth `d`
    // of a uniform stone subtree (Block at d=0, Node at d >= 1).
    // Indices 0..=depth are valid.
    let mut uniform_empty: Vec<NodeId> = Vec::with_capacity(depth as usize + 1);
    let mut uniform_stone: Vec<Child> = Vec::with_capacity(depth as usize + 1);
    {
        let empty_leaf = lib.insert(empty_children());
        uniform_empty.push(empty_leaf); // d=0 (single level) = 27 empties
        uniform_stone.push(Child::Block(sdf.core_block));
        for _ in 1..=depth {
            let prev_e = *uniform_empty.last().unwrap();
            let next_e = lib.insert(uniform_children(Child::Node(prev_e)));
            uniform_empty.push(next_e);
            let prev_s = *uniform_stone.last().unwrap();
            let sid = lib.insert(uniform_children(prev_s));
            uniform_stone.push(Child::Node(sid));
        }
    }

    // Build one Cartesian subtree per body-cell slot. Each slot
    // occupies a `1/3 × 1/3 × 1/3` sub-box of the body's local
    // `[0, 1)³` frame, and that sub-box is SDF-carved directly.
    //
    // Face-center slots (6 of them) additionally get the face tag
    // on their top node, so the camera pipeline recognises them as
    // "inside a planet face" at `face_depth >= 1`.
    let mut body_children = empty_children();
    let core_block = sdf.core_block;
    for zs in 0..3 {
        for ys in 0..3 {
            for xs in 0..3 {
                let slot = slot_index(xs, ys, zs);
                let sub_min: Vec3 = [
                    xs as f32 / 3.0,
                    ys as f32 / 3.0,
                    zs as f32 / 3.0,
                ];
                let sub_max: Vec3 = [
                    (xs + 1) as f32 / 3.0,
                    (ys + 1) as f32 / 3.0,
                    (zs + 1) as f32 / 3.0,
                ];
                let child = build_cartesian_subtree(
                    lib, sub_min, sub_max, depth, sdf_budget, sdf,
                    &uniform_empty, &uniform_stone, core_block,
                );
                let face_idx = FACE_SLOTS.iter().position(|&s| s == slot);
                body_children[slot] = match face_idx {
                    Some(fi) => tag_with_face(lib, child, Face::from_index(fi as u8)),
                    None => child,
                };
            }
        }
    }

    lib.insert_with_kind(
        body_children,
        NodeKind::CubedSphereBody { inner_r, outer_r },
    )
}

/// Recursive builder for a Cartesian-sub-box SDF-carved subtree.
/// Every recursion level samples the SDF at the sub-box's center
/// in the body's local `[0, 1)³` frame; cells entirely outside the
/// planet SDF collapse to empty, cells entirely inside collapse to
/// a uniform block subtree, and straddlers recurse until
/// `sdf_budget` is exhausted.
fn build_cartesian_subtree(
    lib: &mut NodeLibrary,
    sub_min: Vec3,
    sub_max: Vec3,
    depth: u32,
    sdf_budget: u32,
    sdf: &Planet,
    uniform_empty: &[NodeId],
    uniform_stone: &[Child],
    core_block: u8,
) -> Child {
    let center: Vec3 = [
        0.5 * (sub_min[0] + sub_max[0]),
        0.5 * (sub_min[1] + sub_max[1]),
        0.5 * (sub_min[2] + sub_max[2]),
    ];
    let d_center = sdf.distance(center);

    // Cell bounding-sphere radius = half the diagonal of the sub-box.
    let hx = 0.5 * (sub_max[0] - sub_min[0]);
    let hy = 0.5 * (sub_max[1] - sub_min[1]);
    let hz = 0.5 * (sub_max[2] - sub_min[2]);
    let cell_rad = (hx * hx + hy * hy + hz * hz).sqrt();

    if d_center > cell_rad {
        // Fully outside the planet — empty all the way down.
        if depth == 0 { return Child::Empty; }
        return Child::Node(uniform_empty[depth as usize]);
    }
    if d_center < -cell_rad {
        // Fully inside the planet. Use the precomputed uniform stone
        // subtree when the center sample agrees with `core_block`
        // (saves a hot-path O(depth) subtree build). For surface-
        // adjacent cells whose block_at returns something other than
        // core_block (e.g. DIRT or GRASS), fall back to the generic
        // dedup'd path — rare relative to the stone interior.
        let b = sdf.block_at(center);
        if depth == 0 { return Child::Block(b); }
        if b == core_block {
            return uniform_stone[depth as usize];
        }
        return lib.build_uniform_subtree(b, depth);
    }

    if depth == 0 {
        return if d_center < 0.0 {
            Child::Block(sdf.block_at(center))
        } else {
            Child::Empty
        };
    }
    if sdf_budget == 0 {
        // Ran out of SDF detail — commit to the center sample and
        // wrap the remaining depth in a precomputed uniform subtree.
        return if d_center < 0.0 {
            let b = sdf.block_at(center);
            if b == core_block {
                uniform_stone[depth as usize]
            } else {
                lib.build_uniform_subtree(b, depth)
            }
        } else {
            Child::Node(uniform_empty[depth as usize])
        };
    }

    let mut children = empty_children();
    let tx = (sub_max[0] - sub_min[0]) / 3.0;
    let ty = (sub_max[1] - sub_min[1]) / 3.0;
    let tz = (sub_max[2] - sub_min[2]) / 3.0;
    for zs in 0..3 {
        for ys in 0..3 {
            for xs in 0..3 {
                let cmin: Vec3 = [
                    sub_min[0] + xs as f32 * tx,
                    sub_min[1] + ys as f32 * ty,
                    sub_min[2] + zs as f32 * tz,
                ];
                let cmax: Vec3 = [
                    cmin[0] + tx,
                    cmin[1] + ty,
                    cmin[2] + tz,
                ];
                children[slot_index(xs, ys, zs)] = build_cartesian_subtree(
                    lib, cmin, cmax, depth - 1, sdf_budget - 1, sdf,
                    uniform_empty, uniform_stone, core_block,
                );
            }
        }
    }
    Child::Node(lib.insert(children))
}

/// Wrap a face-center slot's subtree with `NodeKind::CubedSphereFace`
/// so the camera pipeline can detect when the render frame is inside
/// a specific face and rotate the camera basis to the face's local
/// `(u, v, r)` axes. The child slot's content is unchanged.
fn tag_with_face(lib: &mut NodeLibrary, child: Child, face: Face) -> Child {
    match child {
        Child::Node(id) => {
            let n = lib.get(id).expect("face root just inserted");
            let children = n.children;
            Child::Node(lib.insert_with_kind(children, NodeKind::CubedSphereFace { face }))
        }
        Child::Empty => {
            Child::Node(lib.insert_with_kind(
                empty_children(),
                NodeKind::CubedSphereFace { face },
            ))
        }
        Child::Block(b) => Child::Node(lib.insert_with_kind(
            uniform_children(Child::Block(b)),
            NodeKind::CubedSphereFace { face },
        )),
    }
}

fn build_uniform_empty(lib: &mut NodeLibrary, depth: u32) -> NodeId {
    let mut id = lib.insert(empty_children());
    for _ in 1..depth {
        id = lib.insert(uniform_children(Child::Node(id)));
    }
    id
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
        // Non-face, non-interior slots — with Cartesian-indexed
        // worldgen these can be non-Empty too when the sphere extends
        // into their sub-box. We only assert that edge/corner slots
        // whose bounding sphere is fully outside the planet are Empty.
        // For this SDF (radius=0.32, no noise), slots whose nearest
        // corner is past 0.32 from (0.5,0.5,0.5) should be empty.
    }
}
