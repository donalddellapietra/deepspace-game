//! Render-frame helpers: walking the camera path to find the active
//! frame, transforming positions/AABBs into that frame.
//!
//! Cartesian frames are linear `[0, 3)³`. When the render frame
//! descends through a cubed-sphere body, the frame kind becomes
//! `Body` (linear root at the body cell itself). When it descends
//! into a face subtree, it becomes `Sphere` — the LINEAR render root
//! stays at the containing body cell (so the shader can ray-march
//! the sphere shell in body-local coords), but the frame carries an
//! explicit `(face, u_min, v_min, r_min, size)` window telling the
//! sphere DDA which absolute UVR region is being marched.
//!
//! The window bounds are accumulated purely from slot coordinates
//! during Cartesian descents inside the face subtree — no separate
//! `face_depth` counter, no `face_root_id` field. Everything is
//! recoverable from the render path.
//!
//! Pure functions; no `App` state; unit-testable.

use crate::world::anchor::Path;
use crate::world::cubesphere::Face;
use crate::world::tree::{slot_coords, Child, NodeId, NodeKind, NodeLibrary};

/// Sphere frame: the render root is inside a face subtree. The
/// linear `node_id` is the containing body cell; ray-march happens
/// in body-local `[0, 3)³` coords; hits are restricted to the face
/// window.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereFrame {
    /// Path from world root to the containing body cell.
    pub body_path: Path,
    pub face: Face,
    pub inner_r: f32,
    pub outer_r: f32,
    /// Face-window bounds in normalized `[0, 1]` face coords.
    pub face_u_min: f32,
    pub face_v_min: f32,
    pub face_r_min: f32,
    pub face_size: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActiveFrameKind {
    Cartesian,
    /// Render root IS a cubed-sphere body cell. No face window —
    /// the sphere DDA covers all 6 faces.
    Body { inner_r: f32, outer_r: f32 },
    /// Render root is deep inside a face subtree. Linear frame stays
    /// at the containing body; the face window restricts hits.
    Sphere(SphereFrame),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveFrame {
    /// Path used by the linear ribbon / camera-local transforms.
    /// For `Sphere` frames this ends at the containing body, NOT the
    /// deepest face subtree node (camera stays in body-local coords).
    pub render_path: Path,
    /// Logical interaction layer. For `Cartesian` and `Body` equals
    /// `render_path`; for `Sphere` it extends through the face
    /// subtree to the deepest descended node.
    pub logical_path: Path,
    pub node_id: NodeId,
    pub kind: ActiveFrameKind,
}

/// Build a `Path` from the slot prefix the GPU ribbon walker
/// actually reached.
pub fn frame_from_slots(slots: &[u8]) -> Path {
    let mut frame = Path::root();
    for &slot in slots {
        frame.push(slot);
    }
    frame
}

/// Resolve the active render frame. Walks the camera's anchor path
/// down to `desired_depth`, building a sphere window on the fly
/// when the descent crosses a `CubedSphereBody` → `CubedSphereFace`
/// boundary.
pub fn compute_render_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    camera_anchor: &Path,
    desired_depth: u8,
) -> ActiveFrame {
    let mut target = *camera_anchor;
    target.truncate(desired_depth);

    let mut node_id = world_root;
    let mut reached = Path::root();

    // Sphere-frame accumulator: once we enter a body, start tracking
    // `(body_path, radii, face_slot_reached, face_window)`.
    let mut body: Option<(Path, f32, f32)> = None;
    let mut sphere: Option<(Face, f32, f32, f32, f32, Path)> = None;
    //                       face  u_min v_min r_min size logical_path

    for k in 0..target.depth() as usize {
        let Some(node) = library.get(node_id) else { break };
        let slot = target.slot(k) as usize;
        let Child::Node(child_id) = node.children[slot] else { break };
        let Some(child) = library.get(child_id) else { break };
        reached.push(slot as u8);

        match child.kind {
            NodeKind::Cartesian => {
                node_id = child_id;
                // If inside a face subtree, accumulate the slot's
                // UVR contribution into the face window.
                if let Some((
                    _face,
                    ref mut u_min,
                    ref mut v_min,
                    ref mut r_min,
                    ref mut size,
                    ref mut logical,
                )) = sphere {
                    let (us, vs, rs) = slot_coords(slot);
                    let child_size = *size / 3.0;
                    *u_min += us as f32 * child_size;
                    *v_min += vs as f32 * child_size;
                    *r_min += rs as f32 * child_size;
                    *size = child_size;
                    *logical = reached;
                }
            }
            NodeKind::CubedSphereBody { inner_r, outer_r } => {
                node_id = child_id;
                body = Some((reached, inner_r, outer_r));
            }
            NodeKind::CubedSphereFace { face } => {
                node_id = child_id;
                // Entering a face: initialise window to the full face.
                sphere = Some((face, 0.0, 0.0, 0.0, 1.0, reached));
            }
        }
    }

    if let Some((face, u_min, v_min, r_min, size, logical_path)) = sphere {
        let (body_path, inner_r, outer_r) =
            body.expect("sphere frame requires a containing body");
        // Linear render root stays at the body cell; the face
        // window carries the deeper info.
        let body_node_id = library
            .get(world_root)
            .and_then(|_| walk_to_node(library, world_root, &body_path))
            .unwrap_or(node_id);
        return ActiveFrame {
            render_path: body_path,
            logical_path,
            node_id: body_node_id,
            kind: ActiveFrameKind::Sphere(SphereFrame {
                body_path,
                face,
                inner_r,
                outer_r,
                face_u_min: u_min,
                face_v_min: v_min,
                face_r_min: r_min,
                face_size: size,
            }),
        };
    }

    let kind = library.get(node_id).map(|n| n.kind).unwrap_or(NodeKind::Cartesian);
    ActiveFrame {
        render_path: reached,
        logical_path: reached,
        node_id,
        kind: match kind {
            NodeKind::CubedSphereBody { inner_r, outer_r } => ActiveFrameKind::Body { inner_r, outer_r },
            _ => ActiveFrameKind::Cartesian,
        },
    }
}

fn walk_to_node(library: &NodeLibrary, root: NodeId, path: &Path) -> Option<NodeId> {
    let mut node_id = root;
    for k in 0..path.depth() as usize {
        let node = library.get(node_id)?;
        match node.children[path.slot(k) as usize] {
            Child::Node(c) => node_id = c,
            _ => return None,
        }
    }
    Some(node_id)
}

pub fn with_render_margin(
    library: &NodeLibrary,
    world_root: NodeId,
    logical_path: &Path,
    render_margin: u8,
) -> ActiveFrame {
    let logical = compute_render_frame(library, world_root, logical_path, logical_path.depth());
    let min_render_depth = match logical.kind {
        ActiveFrameKind::Sphere(sphere) => (sphere.body_path.depth() + 1).min(logical.logical_path.depth()),
        ActiveFrameKind::Body { .. } | ActiveFrameKind::Cartesian => logical.logical_path.depth(),
    };
    let render_depth = logical
        .logical_path
        .depth()
        .saturating_sub(render_margin)
        .max(min_render_depth);
    if render_depth == logical.logical_path.depth() {
        return logical;
    }

    let mut render_path = logical.logical_path;
    render_path.truncate(render_depth);
    let render = compute_render_frame(library, world_root, &render_path, render_depth);
    ActiveFrame {
        render_path: render.render_path,
        logical_path: logical.logical_path,
        node_id: render.node_id,
        kind: render.kind,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, slot_index, uniform_children};

    fn cartesian_chain(depth: u8) -> (NodeLibrary, NodeId) {
        let mut lib = NodeLibrary::default();
        let mut node = lib.insert(empty_children());
        for _ in 1..depth {
            node = lib.insert(uniform_children(Child::Node(node)));
        }
        lib.ref_inc(node);
        (lib, node)
    }

    #[test]
    fn cartesian_descends_linearly() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..4 { anchor.push(13); }
        let f = compute_render_frame(&lib, root, &anchor, 3);
        assert_eq!(f.render_path.depth(), 3);
        assert!(matches!(f.kind, ActiveFrameKind::Cartesian));
    }

    #[test]
    fn sphere_body_enters_body_kind() {
        let mut lib = NodeLibrary::default();
        let body = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereBody { inner_r: 0.12, outer_r: 0.45 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(13);
        let f = compute_render_frame(&lib, root, &anchor, 1);
        assert!(matches!(f.kind, ActiveFrameKind::Body { .. }));
    }

    #[test]
    fn sphere_face_enters_sphere_kind() {
        let mut lib = NodeLibrary::default();
        let face = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereFace { face: Face::PosX },
        );
        let mut body_children = empty_children();
        body_children[14] = Child::Node(face);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody { inner_r: 0.12, outer_r: 0.45 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(13);
        anchor.push(14);
        let f = compute_render_frame(&lib, root, &anchor, 2);
        match f.kind {
            ActiveFrameKind::Sphere(sphere) => {
                assert_eq!(sphere.face, Face::PosX);
                assert_eq!(sphere.face_size, 1.0);
            }
            _ => panic!("expected Sphere"),
        }
    }
}
