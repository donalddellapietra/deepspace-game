//! Frame-local cubed-sphere helpers.
//!
//! The renderer walks face subtrees as plain 27-ary Cartesian nodes,
//! so this module no longer carries the body-frame sphere math that
//! used to live here. Two helpers survive:
//!
//! * `world_vec_to_face_axes` — rotates a world-frame vector into a
//!   face's local `(u_axis, v_axis, n_axis)` basis. Used by the
//!   camera-basis pipeline so the rendered "up" on a planet surface
//!   points along the face's radial axis.
//! * `find_body_ancestor_in_path` — scans a hit path for the nearest
//!   ancestor tagged `CubedSphereBody`. Used by highlight / edit
//!   code to recognize when a hit is inside a planet body.

use super::cubesphere::Face;
use super::sdf;
use super::tree::{Child, NodeId, NodeKind, NodeLibrary};

/// Rotate a world-frame vector into a face's local `(u, v, r)` axes.
///
/// Each cubed-sphere face has an orthonormal basis
/// `(u_axis, v_axis, n_axis)` = `(Face::tangents(), Face::normal())`
/// picked so that the face subtree's slot ordering
/// `slot_index(us, vs, rs)` numerically matches the walker's
/// `slot_index(x, y, z)` with `(x, y, z) ≡ (u, v, r)`. Expressing a
/// ray direction in these axes lets the generic Cartesian DDA walk
/// a face subtree correctly.
#[inline]
pub fn world_vec_to_face_axes(world: [f32; 3], face: Face) -> [f32; 3] {
    let (u_axis, v_axis) = face.tangents();
    let n_axis = face.normal();
    [
        sdf::dot(world, u_axis),
        sdf::dot(world, v_axis),
        sdf::dot(world, n_axis),
    ]
}

/// Scan a hit path for the first entry whose child is a
/// `NodeKind::CubedSphereBody` node. Returns the index in `hit_path`
/// where the body is the child, plus its radii.
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
