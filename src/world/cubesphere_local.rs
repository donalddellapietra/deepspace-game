//! CPU-side cubed-sphere projection primitives. Mirrors the
//! shader's face_math / face_walk precision contract: all inputs are
//! in a body cell's local frame where the cell spans `[0, body_size)`
//! and the radii are given as fractions of `body_size`. Nothing in
//! this module touches a body-absolute constant — callers pass
//! `body_size` explicitly so the same primitives work at any render
//! frame scale.

use super::cubesphere::{cube_to_ea, face_uv_to_dir, pick_face, Face};
use super::sdf;
use super::tree::{Child, NodeId, NodeKind, NodeLibrary};

/// A point in face-normalized `(un, vn, rn) ∈ [0, 1)³` coords, plus
/// the world-radial distance at which it sits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalFacePoint {
    pub face: Face,
    pub un: f32,
    pub vn: f32,
    pub rn: f32,
    pub radius: f32,
}

/// Body cell center, in body cell-local coords. The body cell spans
/// `[0, body_size)³` and its center is at `body_size * 0.5`.
#[inline]
pub fn body_center(body_size: f32) -> [f32; 3] {
    [body_size * 0.5; 3]
}

/// Ray vs. outer shell in the body cell's frame. Returns the first
/// `t > 0` at which the ray crosses the outer sphere, or `None`.
pub fn ray_outer_sphere_hit(
    ray_origin_body: [f32; 3],
    ray_dir: [f32; 3],
    outer_r_local: f32,
    body_size: f32,
) -> Option<f32> {
    let center = body_center(body_size);
    let outer = outer_r_local * body_size;
    let oc = sdf::sub(ray_origin_body, center);
    let b = sdf::dot(oc, ray_dir);
    let c = sdf::dot(oc, oc) - outer * outer;
    let disc = b * b - c;
    if disc <= 0.0 { return None; }
    let sq = disc.sqrt();
    let t_enter = (-b - sq).max(0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 { return None; }
    let t = if t_enter > 0.0 { t_enter } else { t_exit };
    if t > 0.0 { Some(t) } else { None }
}

/// Project a point inside the body's shell into `(face, un, vn, rn)`
/// face-normalized coords. Returns `None` for degenerate inputs
/// (point exactly at body center, or shell collapsed).
pub fn body_point_to_face_space(
    point_body: [f32; 3],
    inner_r_local: f32,
    outer_r_local: f32,
    body_size: f32,
) -> Option<LocalFacePoint> {
    let center = body_center(body_size);
    let local = sdf::sub(point_body, center);
    let radius = sdf::length(local);
    if radius <= 1e-12 { return None; }
    let n = sdf::scale(local, 1.0 / radius);
    let face = pick_face(n);
    let n_axis = face.normal();
    let (u_axis, v_axis) = face.tangents();
    let axis_dot = sdf::dot(n, n_axis);
    if axis_dot.abs() <= 1e-12 { return None; }
    let cu = sdf::dot(n, u_axis) / axis_dot;
    let cv = sdf::dot(n, v_axis) / axis_dot;
    let inner = inner_r_local * body_size;
    let outer = outer_r_local * body_size;
    let shell = outer - inner;
    if shell <= 0.0 { return None; }
    Some(LocalFacePoint {
        face,
        un: ((cube_to_ea(cu) + 1.0) * 0.5).clamp(0.0, 0.9999999),
        vn: ((cube_to_ea(cv) + 1.0) * 0.5).clamp(0.0, 0.9999999),
        rn: ((radius - inner) / shell).clamp(0.0, 0.9999999),
        radius,
    })
}

/// Inverse of `body_point_to_face_space`: rebuild the body-local
/// world position from face-normalized coords.
pub fn face_space_to_body_point(
    face: Face,
    un: f32, vn: f32, rn: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    body_size: f32,
) -> [f32; 3] {
    let center = body_center(body_size);
    let radius = (inner_r_local + rn * (outer_r_local - inner_r_local)) * body_size;
    let dir = face_uv_to_dir(face, un * 2.0 - 1.0, vn * 2.0 - 1.0);
    sdf::add(center, sdf::scale(dir, radius))
}

/// Scan a hit path for the first entry whose child is a
/// `NodeKind::CubedSphereBody`. Returns the path-index where the body
/// sits plus its radii. Callers extract face slot from `path[i+1]`
/// and UVR slots from `path[i+2..]`.
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
