//! Render-frame resolution.
//!
//! `compute_render_frame(world, camera, desired_depth)` returns the
//! frame the shader + CPU raycast operate in. Two kinds:
//!
//!   * `Cartesian` — slot-XYZ descent through Cartesian nodes.
//!   * `Body` — render root IS a `CubedSphereBody` cell; shader
//!     dispatches the whole-sphere body march.
//!
//! Deep-m rendering goes through the body march with walker
//! precision preserved via the cubed-sphere DDA in `sphere.rs`.
//!
//! Pure functions; no `App` state.

use crate::world::anchor::{Path, WorldPos};
use crate::world::tree::{Child, NodeId, NodeKind, NodeLibrary};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActiveFrameKind {
    Cartesian,
    /// Render root is a `CubedSphereBody` cell. Shader runs the full
    /// whole-sphere march in body-local coords.
    Body { inner_r: f32, outer_r: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveFrame {
    /// Path from world root to the render frame's root node.
    pub render_path: Path,
    /// Logical interaction layer — the user's edit-depth anchor.
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

/// Resolve the active render frame from a camera WorldPos + desired
/// anchor depth.
///
/// * If the anchor lands on a `CubedSphereBody` cell → `Body`.
/// * Else `Cartesian`.
pub fn compute_render_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    camera: &WorldPos,
    desired_depth: u8,
) -> ActiveFrame {
    // Walk the anchor slot by slot, checking each descent resolves
    // to a real Node. Stop early at non-Node children. If we hit a
    // CubedSphereBody, pick Body kind.
    let mut target = camera.anchor;
    target.truncate(desired_depth);

    let mut node_id = world_root;
    let mut reached = Path::root();
    let mut body_meta: Option<(f32, f32)> = None;
    for k in 0..target.depth() as usize {
        let Some(node) = library.get(node_id) else { break };
        let slot = target.slot(k) as usize;
        let Child::Node(child_id) = node.children[slot] else { break };
        let Some(child) = library.get(child_id) else { break };
        match child.kind {
            NodeKind::Cartesian => {
                node_id = child_id;
                reached.push(slot as u8);
            }
            NodeKind::CubedSphereBody { inner_r, outer_r } => {
                node_id = child_id;
                reached.push(slot as u8);
                body_meta = Some((inner_r, outer_r));
                break;
            }
            NodeKind::CubedSphereFace { .. } => {
                break;
            }
        }
    }

    let kind = match body_meta {
        Some((inner_r, outer_r)) => ActiveFrameKind::Body { inner_r, outer_r },
        None => ActiveFrameKind::Cartesian,
    };
    ActiveFrame {
        render_path: reached,
        logical_path: reached,
        node_id,
        kind,
    }
}

pub fn with_render_margin(
    library: &NodeLibrary,
    world_root: NodeId,
    camera: &WorldPos,
    logical_depth: u8,
    render_margin: u8,
) -> ActiveFrame {
    let target_depth = logical_depth.saturating_sub(render_margin);
    let logical = compute_render_frame(library, world_root, camera, logical_depth);
    if target_depth >= logical_depth {
        return logical;
    }
    let render = compute_render_frame(library, world_root, camera, target_depth);
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
        for _ in 0..4 {
            anchor.push(13);
        }
        let camera = WorldPos::new(anchor, [0.5; 3]);
        let f = compute_render_frame(&lib, root, &camera, 3);
        assert_eq!(f.render_path.depth(), 3);
        assert!(matches!(f.kind, ActiveFrameKind::Cartesian));
    }

    #[test]
    fn body_kind_when_anchor_ends_on_body() {
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
        anchor.push(slot_index(1, 1, 1) as u8);
        let camera = WorldPos::new(anchor, [0.5; 3]);
        let f = compute_render_frame(&lib, root, &camera, 1);
        assert!(matches!(f.kind, ActiveFrameKind::Body { .. }));
    }
}
