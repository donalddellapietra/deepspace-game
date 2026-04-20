//! Cubed-sphere geometry + worldgen. One file, pure functions.
//!
//! A spherical body lives inside the Cartesian voxel tree as a
//! `NodeKind::CubedSphereBody` node. Its 27 children are laid out in
//! XYZ like any Cartesian node, but six specific slots (the face-
//! centers) hold face subtrees, one slot (the body center) holds a
//! uniform-stone core, and the 20 edge/corner slots are empty.
//!
//! Inside a face subtree, a node's 27 children are interpreted in
//! `(u, v, r)` axes on that face. The face root carries
//! `NodeKind::CubedSphereFace { face }`; deeper nodes stay Cartesian
//! (slot index semantics follow the face root's convention
//! contagiously along the descent path).
//!
//! Geometry uses the equal-angle cubed-sphere projection:
//! `dir = normalize(n + tan(u·π/4)·u_axis + tan(v·π/4)·v_axis)`.
//! That spreads solid angle evenly across a face, and gives the
//! UVR shell a smooth curved surface with no seams between faces.

use super::sdf::{self, Planet, Vec3};
use super::tree::{empty_children, slot_index, uniform_children, Child, NodeId, NodeKind, NodeLibrary};

// ─────────────────────────────────────────────────────── face enum

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
    pub const ALL: [Face; 6] = [Face::PosX, Face::NegX, Face::PosY, Face::NegY, Face::PosZ, Face::NegZ];

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

/// Body slot holding each face's subtree. Indexed by `Face as usize`.
pub const FACE_SLOTS: [usize; 6] = [
    slot_index(2, 1, 1), // PosX
    slot_index(0, 1, 1), // NegX
    slot_index(1, 2, 1), // PosY
    slot_index(1, 0, 1), // NegY
    slot_index(1, 1, 2), // PosZ
    slot_index(1, 1, 0), // NegZ
];

/// Body slot holding the interior (uniform-stone core).
pub const CORE_SLOT: usize = slot_index(1, 1, 1);

// ─────────────────────────────────────────────── coord conversions

/// Pick the cube face whose outward normal aligns with `n`. `n` need
/// not be unit length; only direction matters.
#[inline]
pub fn pick_face(n: Vec3) -> Face {
    let ax = n[0].abs();
    let ay = n[1].abs();
    let az = n[2].abs();
    if ax >= ay && ax >= az {
        if n[0] >= 0.0 { Face::PosX } else { Face::NegX }
    } else if ay >= az {
        if n[1] >= 0.0 { Face::PosY } else { Face::NegY }
    } else {
        if n[2] >= 0.0 { Face::PosZ } else { Face::NegZ }
    }
}

/// Equal-angle warp: cube-plane coord ↔ EA coord. `ea_to_cube(c) =
/// tan(c·π/4)`; its inverse `cube_to_ea(c) = atan(c)·4/π`. Both map
/// `[-1, 1]` to `[-1, 1]` symmetrically. Spreads solid angle evenly
/// across a face so UVR cells look the same size from the center.
#[inline]
pub fn ea_to_cube(c: f32) -> f32 {
    (c * std::f32::consts::FRAC_PI_4).tan()
}
#[inline]
pub fn cube_to_ea(c: f32) -> f32 {
    c.atan() * (4.0 / std::f32::consts::PI)
}

/// `(face, u ∈ [-1,1], v ∈ [-1,1])` → unit direction from sphere center.
pub fn face_uv_to_dir(face: Face, u: f32, v: f32) -> Vec3 {
    let cu = ea_to_cube(u);
    let cv = ea_to_cube(v);
    let n = face.normal();
    let (ua, va) = face.tangents();
    sdf::normalize([
        n[0] + cu * ua[0] + cv * va[0],
        n[1] + cu * ua[1] + cv * va[1],
        n[2] + cu * ua[2] + cv * va[2],
    ])
}

/// Face-space coordinates relative to a sphere at `center`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct FacePoint {
    pub face: Face,
    /// Normalized face-u ∈ [0, 1).
    pub un: f32,
    /// Normalized face-v ∈ [0, 1).
    pub vn: f32,
    /// Normalized radial ∈ [0, 1): 0 at inner shell, 1 at outer shell.
    pub rn: f32,
}

/// World-point inside body → cubed-sphere `(face, u, v, r)`. Returns
/// `None` if the point is at the sphere's exact center or radii are
/// degenerate.
///
/// `inner_r_local` / `outer_r_local` are in the body cell's local
/// `[0, 1)` frame; `body_size` is the body cell's size in the caller's
/// frame. `point_body` is in the same frame as `body_size`, measured
/// from the body cell's origin (so center is at `body_size * 0.5`).
pub fn body_point_to_face_space(
    point_body: Vec3,
    inner_r_local: f32,
    outer_r_local: f32,
    body_size: f32,
) -> Option<FacePoint> {
    let center = [body_size * 0.5; 3];
    let offset = sdf::sub(point_body, center);
    let r = sdf::length(offset);
    if r <= 1e-12 { return None; }
    let n = sdf::scale(offset, 1.0 / r);
    let face = pick_face(n);
    let n_axis = face.normal();
    let (u_axis, v_axis) = face.tangents();
    let axis_dot = sdf::dot(n, n_axis);
    if axis_dot.abs() <= 1e-12 { return None; }
    let cube_u = sdf::dot(n, u_axis) / axis_dot;
    let cube_v = sdf::dot(n, v_axis) / axis_dot;
    let inner = inner_r_local * body_size;
    let outer = outer_r_local * body_size;
    let shell = outer - inner;
    if shell <= 0.0 { return None; }
    Some(FacePoint {
        face,
        un: ((cube_to_ea(cube_u) + 1.0) * 0.5).clamp(0.0, 0.9999999),
        vn: ((cube_to_ea(cube_v) + 1.0) * 0.5).clamp(0.0, 0.9999999),
        rn: ((r - inner) / shell).clamp(0.0, 0.9999999),
    })
}

/// Inverse of `body_point_to_face_space`: cubed-sphere coords → body-
/// local XYZ.
pub fn face_space_to_body_point(
    face: Face,
    un: f32, vn: f32, rn: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    body_size: f32,
) -> Vec3 {
    let center = [body_size * 0.5; 3];
    let radius = (inner_r_local + rn * (outer_r_local - inner_r_local)) * body_size;
    let dir = face_uv_to_dir(face, un * 2.0 - 1.0, vn * 2.0 - 1.0);
    sdf::add(center, sdf::scale(dir, radius))
}

/// Ray–outer-sphere entry time, in body-frame units. `None` if miss.
pub fn ray_outer_sphere_hit(
    ray_origin_body: Vec3,
    ray_dir: Vec3,
    outer_r_local: f32,
    body_size: f32,
) -> Option<f32> {
    let center = [body_size * 0.5; 3];
    let outer = outer_r_local * body_size;
    let oc = sdf::sub(ray_origin_body, center);
    let b = sdf::dot(oc, ray_dir);
    let c = sdf::dot(oc, oc) - outer * outer;
    let disc = b * b - c;
    if disc <= 0.0 { return None; }
    let sq = disc.sqrt();
    let t_enter = (-b - sq).max(0.0);
    let t_exit = -b + sq;
    let t = if t_enter > 0.0 { t_enter } else { t_exit };
    if t > 0.0 { Some(t) } else { None }
}

/// Scan a hit path for the first `CubedSphereBody` ancestor. Returns
/// `(path_index, inner_r, outer_r)` where `path_index` is the entry
/// whose child is the body node (so `path[index+1]` is the face slot
/// if the hit continues into a face subtree).
pub fn find_body_ancestor_in_path(
    library: &NodeLibrary,
    hit_path: &[(NodeId, usize)],
) -> Option<(usize, f32, f32)> {
    for (index, &(node_id, slot)) in hit_path.iter().enumerate() {
        let Some(node) = library.get(node_id) else { continue };
        let Child::Node(child_id) = node.children[slot] else { continue };
        let Some(child) = library.get(child_id) else { continue };
        if let NodeKind::CubedSphereBody { inner_r, outer_r } = child.kind {
            return Some((index, inner_r, outer_r));
        }
    }
    None
}

// ────────────────────────────────────────────────── worldgen

/// The demo planet. Inner/outer radii in the body cell's local
/// `[0, 1)` frame — `0 < inner_r < outer_r ≤ 0.5` so the sphere fits
/// cleanly inside one Cartesian cell. A smooth surface (no noise),
/// stone core, grass surface.
#[derive(Clone, Debug)]
pub struct PlanetSetup {
    pub inner_r: f32,
    pub outer_r: f32,
    /// Face subtree depth. Internal SDF sampling caps at
    /// `SDF_DETAIL_LEVELS` past which uniform-stone/uniform-empty
    /// fillers extend the subtree via dedup.
    pub depth: u32,
    pub sdf: Planet,
}

pub fn demo_planet() -> PlanetSetup {
    let center: Vec3 = [0.5, 0.5, 0.5];
    let inner_r = 0.12_f32;
    let outer_r = 0.45_f32;
    PlanetSetup {
        inner_r,
        outer_r,
        depth: 28,
        sdf: Planet {
            center,
            radius: 0.30,
            noise_scale: 0.0,
            noise_freq: 1.0,
            noise_seed: 0,
            gravity: 9.8,
            influence_radius: outer_r * 2.0,
            surface_block: crate::world::palette::block::GRASS,
            core_block: crate::world::palette::block::STONE,
        },
    }
}

/// Max levels of SDF recursion into a face subtree. Below this, each
/// cell commits to solid-or-empty from its center sample and extends
/// via uniform dedup. Limits worldgen cost without visibly changing
/// a smooth sphere.
const SDF_DETAIL_LEVELS: u32 = 4;

/// Build a spherical body node and return its `NodeId`. Caller is
/// responsible for placing it inside a parent (e.g., world root's
/// center slot) and bumping its refcount.
pub fn insert_spherical_body(
    lib: &mut NodeLibrary,
    inner_r: f32,
    outer_r: f32,
    depth: u32,
    sdf: &Planet,
) -> NodeId {
    debug_assert!(0.0 < inner_r && inner_r < outer_r && outer_r <= 0.5);

    // Build each face subtree, tagging only the root with
    // CubedSphereFace — internal nodes stay Cartesian for maximal
    // dedup (slot-index UVR convention is established at the root).
    let mut body_children = empty_children();
    for &face in &Face::ALL {
        let child = build_face_subtree(
            lib, face, inner_r, outer_r,
            -1.0, 1.0, -1.0, 1.0, 0.0, 1.0,
            depth, depth.min(SDF_DETAIL_LEVELS), sdf,
        );
        let face_root = match child {
            Child::Node(id) => {
                let children = lib.get(id).expect("face root just inserted").children;
                lib.insert_with_kind(children, NodeKind::CubedSphereFace { face })
            }
            Child::Empty => lib.insert_with_kind(empty_children(), NodeKind::CubedSphereFace { face }),
            Child::Block(b) => {
                lib.insert_with_kind(uniform_children(Child::Block(b)), NodeKind::CubedSphereFace { face })
            }
            Child::EntityRef(_) => unreachable!("worldgen never emits entity refs"),
        };
        body_children[FACE_SLOTS[face as usize]] = Child::Node(face_root);
    }
    body_children[CORE_SLOT] = lib.build_uniform_subtree(sdf.core_block, depth);

    lib.insert_with_kind(body_children, NodeKind::CubedSphereBody { inner_r, outer_r })
}

/// Recursive build of one face subtree. Returns a `Child` so the
/// caller can collapse uniform subtrees. Emit cells by sampling the
/// SDF at the cell center under the equal-angle UVR-to-world map.
#[allow(clippy::too_many_arguments)]
fn build_face_subtree(
    lib: &mut NodeLibrary,
    face: Face,
    inner_r: f32,
    outer_r: f32,
    u_lo: f32, u_hi: f32,
    v_lo: f32, v_hi: f32,
    rn_lo: f32, rn_hi: f32,
    depth: u32,
    sdf_budget: u32,
    sdf: &Planet,
) -> Child {
    let body_size = 1.0f32; // sampled in body-local [0, 1)³
    let u_c = 0.5 * (u_lo + u_hi);
    let v_c = 0.5 * (v_lo + v_hi);
    let rn_c = 0.5 * (rn_lo + rn_hi);
    let p_center = face_space_to_body_point(
        face,
        (u_c + 1.0) * 0.5, (v_c + 1.0) * 0.5, rn_c,
        inner_r, outer_r, body_size,
    );
    let d_center = sdf.distance(p_center);
    let radial_half = 0.5 * (rn_hi - rn_lo) * (outer_r - inner_r);
    let lateral_half = 0.5 * (u_hi - u_lo).max(v_hi - v_lo) * outer_r;
    let cell_rad = (lateral_half * lateral_half + radial_half * radial_half).sqrt();

    if d_center > cell_rad {
        return if depth == 0 { Child::Empty } else { Child::Node(uniform_empty_chain(lib, depth)) };
    }
    if d_center < -cell_rad {
        let b = sdf.block_at(p_center);
        return if depth == 0 { Child::Block(b) } else { lib.build_uniform_subtree(b, depth) };
    }
    if depth == 0 {
        return if d_center < 0.0 { Child::Block(sdf.block_at(p_center)) } else { Child::Empty };
    }
    if sdf_budget == 0 {
        return if d_center < 0.0 {
            lib.build_uniform_subtree(sdf.block_at(p_center), depth)
        } else {
            Child::Node(uniform_empty_chain(lib, depth))
        };
    }

    let mut children = empty_children();
    let du = (u_hi - u_lo) / 3.0;
    let dv = (v_hi - v_lo) / 3.0;
    let drn = (rn_hi - rn_lo) / 3.0;
    for rs in 0..3 {
        for vs in 0..3 {
            for us in 0..3 {
                children[slot_index(us, vs, rs)] = build_face_subtree(
                    lib, face, inner_r, outer_r,
                    u_lo + du * us as f32, u_lo + du * (us + 1) as f32,
                    v_lo + dv * vs as f32, v_lo + dv * (vs + 1) as f32,
                    rn_lo + drn * rs as f32, rn_lo + drn * (rs + 1) as f32,
                    depth - 1, sdf_budget - 1, sdf,
                );
            }
        }
    }
    Child::Node(lib.insert(children))
}

fn uniform_empty_chain(lib: &mut NodeLibrary, depth: u32) -> NodeId {
    let mut id = lib.insert(empty_children());
    for _ in 1..depth {
        id = lib.insert(uniform_children(Child::Node(id)));
    }
    id
}

/// Install a body into the world tree at `host_slots`, returning the
/// new world root and the body's path from the new root.
pub fn install_at_root_center(
    lib: &mut NodeLibrary,
    world_root: NodeId,
    setup: &PlanetSetup,
) -> (NodeId, crate::world::anchor::Path) {
    let body_id = insert_spherical_body(lib, setup.inner_r, setup.outer_r, setup.depth, &setup.sdf);
    let host_slot = slot_index(1, 1, 1) as u8;
    let root_node = lib.get(world_root).expect("world root exists");
    let mut children = root_node.children;
    children[host_slot as usize] = Child::Node(body_id);
    let new_root = lib.insert(children);
    let mut body_path = crate::world::anchor::Path::root();
    body_path.push(host_slot);
    (new_root, body_path)
}

// ─────────────────────────────────────────────────────── tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::palette::block;

    #[test]
    fn face_center_matches_normal() {
        for &f in &Face::ALL {
            let dir = face_uv_to_dir(f, 0.0, 0.0);
            let n = f.normal();
            for i in 0..3 { assert!((dir[i] - n[i]).abs() < 1e-5); }
        }
    }

    #[test]
    fn ea_cube_round_trip() {
        for x in [-0.9_f32, -0.3, 0.0, 0.5, 0.99] {
            assert!((cube_to_ea(ea_to_cube(x)) - x).abs() < 1e-5);
        }
    }

    #[test]
    fn body_face_space_round_trip() {
        for &face in &Face::ALL {
            for &(u, v, r) in &[(0.1_f32, 0.1, 0.1), (0.5, 0.5, 0.5), (0.9, 0.9, 0.9)] {
                let body = face_space_to_body_point(face, u, v, r, 0.12, 0.45, 1.0);
                let back = body_point_to_face_space(body, 0.12, 0.45, 1.0).unwrap();
                assert_eq!(back.face, face);
                assert!((back.un - u).abs() < 1e-4, "un {u} → {}", back.un);
                assert!((back.vn - v).abs() < 1e-4);
                assert!((back.rn - r).abs() < 1e-4);
            }
        }
    }

    #[test]
    fn insert_body_creates_structured_children() {
        let mut lib = NodeLibrary::default();
        let sdf = Planet {
            center: [0.5; 3], radius: 0.30,
            noise_scale: 0.0, noise_freq: 1.0, noise_seed: 0,
            gravity: 0.0, influence_radius: 1.0,
            surface_block: block::GRASS, core_block: block::STONE,
        };
        let body = insert_spherical_body(&mut lib, 0.12, 0.45, 6, &sdf);
        let body_node = lib.get(body).unwrap();
        assert!(matches!(body_node.kind, NodeKind::CubedSphereBody { .. }));
        for &face in &Face::ALL {
            let slot = FACE_SLOTS[face as usize];
            match body_node.children[slot] {
                Child::Node(id) => {
                    let n = lib.get(id).unwrap();
                    assert!(matches!(n.kind, NodeKind::CubedSphereFace { face: f } if f == face));
                }
                _ => panic!("face slot {slot} not a Node"),
            }
        }
        match body_node.children[CORE_SLOT] {
            Child::Node(id) => {
                assert_eq!(lib.get(id).unwrap().uniform_type, block::STONE);
            }
            Child::Block(b) => assert_eq!(b, block::STONE),
            _ => panic!("core slot empty"),
        }
        for s in 0..27 {
            if s == CORE_SLOT || FACE_SLOTS.contains(&s) { continue; }
            assert!(matches!(body_node.children[s], Child::Empty));
        }
    }
}
