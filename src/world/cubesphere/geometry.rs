//! Cubed-sphere geometry primitives: the face enum, face-slot
//! layout, equal-angle warp (`ea_to_cube` / `cube_to_ea`),
//! body↔face-space coordinate conversions, and the ray-outer-sphere
//! entry test.
//!
//! Geometry uses the equal-angle cubed-sphere projection:
//!   dir = normalize(n + tan(u·π/4)·u_axis + tan(v·π/4)·v_axis)
//! which spreads solid angle evenly across a face and gives the UVR
//! shell a smooth curved surface with no seams between faces.
//!
//! Inside a face subtree, a node's 27 children are interpreted in
//! `(u, v, r)` slot order — `slot_index(us, vs, rs)`. Only the face
//! root carries `NodeKind::CubedSphereFace`; deeper nodes stay
//! `Cartesian` (the UVR convention is contagious along the descent
//! path). The body node's 27 children are indexed in XYZ: six
//! specific slots hold face subtrees, one slot holds a uniform-stone
//! core, and the remaining 20 are empty.

use crate::world::sdf::{self, Vec3};
use crate::world::tree::{slot_index, Child, NodeId, NodeKind, NodeLibrary};

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
    } else if n[2] >= 0.0 { Face::PosZ } else { Face::NegZ }
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

// ─────────────────────────────────────────────── body ↔ face space

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
/// `[0, 1)` frame; `body_size` is the body cell's size in the
/// caller's frame. `point_body` is in the same frame as `body_size`,
/// measured from the body cell's origin (so center is at
/// `body_size * 0.5`).
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

/// Inverse of `body_point_to_face_space`: cubed-sphere coords →
/// body-local XYZ.
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

// ─────────────────────────────────────────────────── ray helpers

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

// ────────────────────────────────────────────────────────── tests

#[cfg(test)]
mod tests {
    use super::*;

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
}
