//! Render-frame helpers: walking the camera path to find the
//! active frame, transforming positions/AABBs into that frame.
//!
//! The "render frame" is the GPU's view of the world. The shader
//! starts ray marching at a frame root, with the camera expressed
//! in that frame's coordinates. Cartesian frames are linear
//! `[0, 3)³`; sphere frames stay rooted at the containing body
//! cell but carry an explicit cubed-sphere face-cell window.
//!
//! All functions here are **pure** — no `App` state — for direct
//! unit testing.

use crate::world::anchor::Path;
use crate::world::cubesphere::Face;
use crate::world::tree::{Child, NodeId, NodeKind, NodeLibrary};

/// Metadata for a render frame whose camera is positioned inside a
/// cubed-sphere face subtree. The sphere silhouette and voxel content
/// are handled by the unified Cartesian walker; this struct carries
/// only the face tag (needed to rotate the camera basis so "up" points
/// radially) plus body-lookup info for highlight / edit code that
/// wants to recognize a planet hit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereFrame {
    pub body_path: Path,
    pub body_node_id: NodeId,
    pub face: Face,
    pub inner_r: f32,
    pub outer_r: f32,
    /// Depth of the render frame *inside* the face subtree — 0 when
    /// the render frame is the face root, N when the camera has
    /// descended N slots further into the face subtree. Only
    /// `face_depth >= 1` triggers the face-axis-rotated camera basis;
    /// at face_depth == 0 the body cell itself is the frame and world
    /// axes are fine.
    pub face_depth: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActiveFrameKind {
    Cartesian,
    Body { inner_r: f32, outer_r: f32 },
    Sphere(SphereFrame),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveFrame {
    /// Path used by the linear ribbon / camera-local transforms.
    pub render_path: Path,
    /// Logical interaction/render layer path. For Cartesian this is
    /// identical to `render_path`; for sphere frames it continues
    /// through the face subtree while the linear render root stays
    /// at the containing body.
    pub logical_path: Path,
    pub node_id: NodeId,
    pub kind: ActiveFrameKind,
}

/// Build a `Path` from the slot prefix the GPU ribbon walker
/// actually reached. This is the renderer's effective frame.
pub fn frame_from_slots(slots: &[u8]) -> Path {
    let mut frame = Path::root();
    for &slot in slots {
        frame.push(slot);
    }
    frame
}

/// Resolve the active frame. Cartesian zones use a normal linear
/// frame root. Sphere zones keep the linear root at the containing
/// body cell but continue the logical frame through the face
/// subtree, carrying explicit face-cell bounds so render/edit can
/// operate at the same layer depth without absolute-depth caps.
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
    let mut body_info: Option<(Path, NodeId, f32, f32)> = None;
    let mut face_info: Option<(Face, NodeId)> = None;
    for k in 0..target.depth() as usize {
        let Some(node) = library.get(node_id) else { break };
        let slot = target.slot(k) as usize;
        match node.children[slot] {
            Child::Node(child_id) => {
                let Some(child) = library.get(child_id) else { break };
                reached.push(slot as u8);
                match child.kind {
                    NodeKind::Cartesian => {
                        node_id = child_id;
                    }
                    NodeKind::CubedSphereBody { inner_r, outer_r } => {
                        node_id = child_id;
                        body_info = Some((reached, child_id, inner_r, outer_r));
                    }
                    NodeKind::CubedSphereFace { face } => {
                        node_id = child_id;
                        if body_info.is_some() {
                            face_info = Some((face, child_id));
                        }
                    }
                }
            }
            Child::Block(_) | Child::Empty => break,
        }
    }
    if let Some((face, _face_root_id)) = face_info {
        let (body_path, body_node_id, inner_r, outer_r) =
            body_info.expect("sphere frame requires containing body");
        let face_depth = reached.depth().saturating_sub(body_path.depth() + 1) as u32;
        if face_depth == 0 {
            // Camera is exactly at the face root — descend the render
            // frame back up to the body so the shader's root is the
            // body node. That lets the sphere-SDF pre-clip in `march`
            // cull rays missing the outer sphere and produce a round
            // silhouette. The `Body` kind signals the camera-basis
            // pipeline that no face-axis rotation is needed yet.
            ActiveFrame {
                render_path: body_path,
                logical_path: reached,
                node_id: body_node_id,
                kind: ActiveFrameKind::Body { inner_r, outer_r },
            }
        } else {
            ActiveFrame {
                render_path: reached,
                logical_path: reached,
                node_id,
                kind: ActiveFrameKind::Sphere(SphereFrame {
                    body_path,
                    body_node_id,
                    face,
                    inner_r,
                    outer_r,
                    face_depth,
                }),
            }
        }
    } else {
        let kind = library.get(node_id).map(|n| n.kind).unwrap_or(NodeKind::Cartesian);
        ActiveFrame {
            render_path: reached,
            logical_path: reached,
            node_id,
            kind: match kind {
                NodeKind::CubedSphereBody { inner_r, outer_r } => {
                    ActiveFrameKind::Body { inner_r, outer_r }
                }
                _ => ActiveFrameKind::Cartesian,
            },
        }
    }
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
        ActiveFrameKind::Body { .. } => logical.logical_path.depth(),
        // Shell architecture: the render frame IS the innermost
        // shell root. The shader pops outward via the ribbon for
        // coarser context. No render_margin needed — each shell
        // has a bounded depth budget.
        ActiveFrameKind::Cartesian => logical.logical_path.depth(),
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

    fn cartesian_frame(path: Path, node_id: NodeId) -> ActiveFrame {
        ActiveFrame {
            render_path: path,
            logical_path: path,
            node_id,
            kind: ActiveFrameKind::Cartesian,
        }
    }

    fn cartesian_chain(depth: u8) -> (NodeLibrary, NodeId) {
        let mut lib = NodeLibrary::default();
        let mut node = lib.insert(empty_children());
        for _ in 1..depth {
            node = lib.insert(uniform_children(Child::Node(node)));
        }
        lib.ref_inc(node);
        (lib, node)
    }

    // --------- compute_render_frame ---------

    #[test]
    fn render_frame_root_when_desired_depth_zero() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..3 { anchor.push(13); }
        let frame = compute_render_frame(&lib, root, &anchor, 0);
        assert_eq!(frame.render_path.depth(), 0);
        assert_eq!(frame.node_id, root);
    }

    #[test]
    fn render_frame_descends_through_cartesian() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..4 { anchor.push(13); }
        let frame = compute_render_frame(&lib, root, &anchor, 3);
        assert_eq!(frame.render_path.depth(), 3);
    }

    #[test]
    fn render_frame_enters_sphere_face_logically() {
        let mut lib = NodeLibrary::default();
        let face = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereFace {
                face: crate::world::cubesphere::Face::PosX,
            },
        );
        let mut body_children = empty_children();
        body_children[14] = Child::Node(face);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(13);
        anchor.push(14);
        let frame = compute_render_frame(&lib, root, &anchor, 2);
        assert_eq!(frame.render_path.depth(), 2);
        assert_eq!(frame.logical_path.depth(), 2);
        assert_eq!(frame.node_id, face);
        assert!(matches!(frame.kind, ActiveFrameKind::Sphere(_)));
    }

    #[test]
    fn render_frame_descends_into_body() {
        let mut lib = NodeLibrary::default();
        let body = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(13);
        let frame = compute_render_frame(&lib, root, &anchor, 1);
        assert_eq!(frame.render_path.depth(), 1);
        assert_eq!(frame.node_id, body);
    }

    #[test]
    fn render_frame_truncates_when_camera_anchor_shallow() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        anchor.push(13);
        let frame = compute_render_frame(&lib, root, &anchor, 5);
        assert!(frame.render_path.depth() <= 1);
    }

    #[test]
    fn render_frame_stops_when_path_misses_node() {
        // Build root with a Block at slot 5 (not a Node).
        let mut lib = NodeLibrary::default();
        let mut root_children = empty_children();
        root_children[5] = Child::Block(crate::world::palette::block::STONE);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        let mut anchor = Path::root();
        anchor.push(5);
        anchor.push(0);
        let frame = compute_render_frame(&lib, root, &anchor, 2);
        assert_eq!(frame.render_path.depth(), 0, "Block child terminates descent");
    }

    #[test]
    fn frame_from_slots_builds_exact_prefix() {
        let slots = [13u8, 16u8, 4u8];
        let p = frame_from_slots(&slots);
        assert_eq!(p.depth(), slots.len() as u8);
        assert_eq!(p.as_slice(), &slots);
    }
}
