use super::cubesphere::{cube_to_ea, face_uv_to_dir, pick_face, Face};
use super::sdf;
use super::tree::{Child, NodeId, NodeKind, NodeLibrary};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LocalFacePoint {
    pub face: Face,
    pub un: f32,
    pub vn: f32,
    pub rn: f32,
    pub radius: f32,
}

#[inline]
pub fn body_center(body_size: f32) -> [f32; 3] {
    [body_size * 0.5; 3]
}

pub fn ray_outer_sphere_hit(
    ray_origin_body: [f32; 3],
    ray_dir: [f32; 3],
    outer_r_local: f32,
    body_size: f32,
) -> Option<f32> {
    let center = body_center(body_size);
    let outer_r = outer_r_local * body_size;
    let oc = sdf::sub(ray_origin_body, center);
    let b = sdf::dot(oc, ray_dir);
    let c = sdf::dot(oc, oc) - outer_r * outer_r;
    let disc = b * b - c;
    if disc <= 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t_enter = (-b - sq).max(0.0);
    let t_exit = -b + sq;
    let t = if t_enter > 0.0 { t_enter } else { t_exit };
    if t > 0.0 { Some(t) } else { None }
}

pub fn body_point_to_face_space(
    point_body: [f32; 3],
    inner_r_local: f32,
    outer_r_local: f32,
    body_size: f32,
) -> Option<LocalFacePoint> {
    let center = body_center(body_size);
    let local = sdf::sub(point_body, center);
    let radius = sdf::length(local);
    if radius <= 1e-12 {
        return None;
    }
    let n = sdf::scale(local, 1.0 / radius);
    let face = pick_face(n);
    let n_axis = face.normal();
    let (u_axis, v_axis) = face.tangents();
    let axis_dot = sdf::dot(n, n_axis);
    if axis_dot.abs() <= 1e-12 {
        return None;
    }
    let cube_u = sdf::dot(n, u_axis) / axis_dot;
    let cube_v = sdf::dot(n, v_axis) / axis_dot;
    let inner_r = inner_r_local * body_size;
    let outer_r = outer_r_local * body_size;
    let shell = outer_r - inner_r;
    if shell <= 0.0 {
        return None;
    }
    Some(LocalFacePoint {
        face,
        un: ((cube_to_ea(cube_u) + 1.0) * 0.5).clamp(0.0, 0.9999999),
        vn: ((cube_to_ea(cube_v) + 1.0) * 0.5).clamp(0.0, 0.9999999),
        rn: ((radius - inner_r) / shell).clamp(0.0, 0.9999999),
        radius,
    })
}

pub fn face_space_to_body_point(
    face: Face,
    un: f32,
    vn: f32,
    rn: f32,
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
/// `NodeKind::CubedSphereBody` node. Returns the index in `hit_path`
/// where the body is the child, plus its radii.
///
/// Used by both sphere raycast dispatch and AABB reconstruction:
/// once the body entry is located, callers extract the face slot
/// from `hit_path[index + 1]` and the per-face-subtree slot indices
/// from `hit_path[index + 2..]`.
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
