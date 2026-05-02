//! Render-frame helpers: walking the camera path to find the
//! active frame, transforming positions/AABBs into that frame.
//!
//! The "render frame" is the GPU's view of the world. The shader
//! starts ray marching at a frame root, with the camera expressed
//! in that frame's coordinates. All frames are linear Cartesian
//! `[0, 3)³`.
//!
//! All functions here are **pure** — no `App` state — for direct
//! unit testing.

use crate::world::anchor::Path;
use crate::world::tree::{Child, NodeId, NodeKind, NodeLibrary};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActiveFrameKind {
    Cartesian,
    /// The render frame is rooted at a `NodeKind::WrappedPlane`
    /// node. The shader runs the X-wrap branch of `march_cartesian`
    /// at depth==0; the slab's `(dims, slab_depth)` are uploaded as
    /// `Uniforms.slab_dims`.
    WrappedPlane { dims: [u32; 3], slab_depth: u8 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveFrame {
    /// Path used by the linear ribbon / camera-local transforms.
    pub render_path: Path,
    /// Logical interaction/render layer path. Identical to
    /// `render_path` for the purely-Cartesian architecture.
    pub logical_path: Path,
    pub node_id: NodeId,
    pub kind: ActiveFrameKind,
    /// Rotation center for the shader's frame-entry R^T, in the
    /// frame root's [0,3)³ local coords. When the frame path passes
    /// through a TangentBlock and continues into Cartesian children
    /// below it, the shader applies R^T around this center so the
    /// DDA sees the same rotated view as rays entering from outside.
    /// `[1.5, 1.5, 1.5]` when the frame root IS the TB; shifts as
    /// the frame descends deeper. `[0; 3]` (unused) when no TB is
    /// on the frame path.
    pub tb_center: [f32; 3],
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

/// Resolve the active frame.
///
/// Descends from `world_root` along `camera_anchor` for at most
/// `desired_depth` slot steps. Stops early at a `WrappedPlane`
/// node — the wrap branch in the shader fires at depth==0 of the
/// marcher's local frame, so the render frame must be the slab
/// root, not a sub-cell of it.
///
/// When the descent hits a `TangentBlock`, it computes the camera's
/// R^T-rotated+scaled position in TB-local and continues descending
/// using slots derived from the rotated position. This lets the
/// frame go deeper within the TB (matching the shader's rotated
/// view) while keeping the render_path consistent with the camera's
/// actual position after rotation.
pub fn compute_render_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    camera_anchor: &Path,
    desired_depth: u8,
    cam_pos: Option<&crate::world::anchor::WorldPos>,
) -> ActiveFrame {
    let mut target = *camera_anchor;
    target.truncate(desired_depth);
    let mut node_id = world_root;
    let mut reached = Path::root();
    let mut kind = match library.get(world_root).map(|n| n.kind) {
        Some(NodeKind::WrappedPlane { dims, slab_depth }) => {
            ActiveFrameKind::WrappedPlane { dims, slab_depth }
        }
        _ => ActiveFrameKind::Cartesian,
    };
    let mut tb_center = [0.0f32; 3];
    let mut tb_rotation: Option<[[f32; 3]; 3]> = None;
    let mut tb_scale: f32 = 1.0;
    let mut tb_depth: usize = 0;

    for k in 0..target.depth() as usize {
        if matches!(kind, ActiveFrameKind::WrappedPlane { .. }) {
            break;
        }
        let Some(node) = library.get(node_id) else { break };
        let slot = if let Some(rot) = tb_rotation {
            // Past a TB: pick the slot from the ROTATED camera
            // position instead of the unrotated anchor. Compute
            // cam position in this node's [0,3)³, apply R^T + scale,
            // and use that to choose the cell.
            if let Some(pos) = cam_pos {
                let cam_in_node = pos.in_frame(&reached);
                let centered = [
                    cam_in_node[0] - 1.5,
                    cam_in_node[1] - 1.5,
                    cam_in_node[2] - 1.5,
                ];
                let s = tb_scale;
                let rotated = [
                    (rot[0][0]*centered[0] + rot[0][1]*centered[1] + rot[0][2]*centered[2]) / s,
                    (rot[1][0]*centered[0] + rot[1][1]*centered[1] + rot[1][2]*centered[2]) / s,
                    (rot[2][0]*centered[0] + rot[2][1]*centered[1] + rot[2][2]*centered[2]) / s,
                ];
                let rotated_pos = [
                    rotated[0] + 1.5,
                    rotated[1] + 1.5,
                    rotated[2] + 1.5,
                ];
                let cx = (rotated_pos[0].floor() as i32).clamp(0, 2) as usize;
                let cy = (rotated_pos[1].floor() as i32).clamp(0, 2) as usize;
                let cz = (rotated_pos[2].floor() as i32).clamp(0, 2) as usize;
                crate::world::tree::slot_index(cx, cy, cz)
            } else {
                target.slot(k) as usize
            }
        } else {
            target.slot(k) as usize
        };

        match node.children[slot] {
            Child::Node(child_id) => {
                reached.push(slot as u8);
                node_id = child_id;
                if let Some(child_node) = library.get(child_id) {
                    match child_node.kind {
                        NodeKind::WrappedPlane { dims, slab_depth } => {
                            kind = ActiveFrameKind::WrappedPlane { dims, slab_depth };
                            break;
                        }
                        NodeKind::TangentBlock { rotation } => {
                            tb_rotation = Some(rotation);
                            tb_scale = inscribed_cube_scale(&rotation);
                            tb_depth = k + 1;
                            tb_center = [1.5, 1.5, 1.5];
                        }
                        _ => {}
                    }
                }
                // Track TB center as we descend past the TB into
                // Cartesian children. The center shifts with each
                // level: center_child = (center_parent - slot_origin) * 3
                if tb_rotation.is_some() && k >= tb_depth {
                    let (sx, sy, sz) = crate::world::tree::slot_coords(slot);
                    tb_center = [
                        (tb_center[0] - sx as f32) * 3.0,
                        (tb_center[1] - sy as f32) * 3.0,
                        (tb_center[2] - sz as f32) * 3.0,
                    ];
                }
            }
            Child::Block(_) | Child::Empty | Child::EntityRef(_) => break,
        }
    }
    ActiveFrame {
        render_path: reached,
        logical_path: reached,
        node_id,
        kind,
        tb_center,
    }
}

fn inscribed_cube_scale(r: &[[f32; 3]; 3]) -> f32 {
    let mut max_extent = 0.0f32;
    for i in 0..3 {
        let extent = r[0][i].abs() + r[1][i].abs() + r[2][i].abs();
        max_extent = max_extent.max(extent);
    }
    if max_extent < 1e-6 { 1.0 } else { (1.0 / max_extent).min(1.0) }
}

pub fn with_render_margin(
    library: &NodeLibrary,
    world_root: NodeId,
    logical_path: &Path,
    render_margin: u8,
    cam_pos: Option<&crate::world::anchor::WorldPos>,
) -> ActiveFrame {
    let logical = compute_render_frame(library, world_root, logical_path, logical_path.depth(), cam_pos);
    let min_render_depth = logical.logical_path.depth();
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
    let render = compute_render_frame(library, world_root, &render_path, render_depth, cam_pos);
    ActiveFrame {
        render_path: render.render_path,
        logical_path: logical.logical_path,
        node_id: render.node_id,
        kind: render.kind,
        tb_center: render.tb_center,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, uniform_children, NodeLibrary};

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
        let frame = compute_render_frame(&lib, root, &anchor, 0, None);
        assert_eq!(frame.render_path.depth(), 0);
        assert_eq!(frame.node_id, root);
    }

    #[test]
    fn render_frame_descends_through_cartesian() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..4 { anchor.push(13); }
        let frame = compute_render_frame(&lib, root, &anchor, 3, None);
        assert_eq!(frame.render_path.depth(), 3);
    }

    #[test]
    fn render_frame_truncates_when_camera_anchor_shallow() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        anchor.push(13);
        let frame = compute_render_frame(&lib, root, &anchor, 5, None);
        assert!(frame.render_path.depth() <= 1);
    }

    #[test]
    fn render_frame_stops_when_path_misses_node() {
        let mut lib = NodeLibrary::default();
        let mut root_children = empty_children();
        root_children[5] = Child::Block(crate::world::palette::block::STONE);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        let mut anchor = Path::root();
        anchor.push(5);
        anchor.push(0);
        let frame = compute_render_frame(&lib, root, &anchor, 2, None);
        assert_eq!(frame.render_path.depth(), 0, "Block child terminates descent");
    }

    #[test]
    fn frame_from_slots_builds_exact_prefix() {
        let slots = [13u8, 16u8, 4u8];
        let p = frame_from_slots(&slots);
        assert_eq!(p.depth(), slots.len() as u8);
        assert_eq!(p.as_slice(), &slots);
    }

    #[test]
    fn render_frame_kind_is_wrapped_plane_when_descent_lands_on_one() {
        use crate::world::tree::{empty_children, slot_index, NodeKind};
        let mut lib = NodeLibrary::default();
        let mut wp_children = empty_children();
        wp_children[slot_index(0, 0, 0)] = Child::Block(crate::world::palette::block::GRASS);
        wp_children[slot_index(1, 0, 0)] = Child::Block(crate::world::palette::block::GRASS);
        wp_children[slot_index(2, 0, 0)] = Child::Block(crate::world::palette::block::GRASS);
        let wp = lib.insert_with_kind(
            wp_children,
            NodeKind::WrappedPlane { dims: [3, 1, 1], slab_depth: 1 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(wp);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(slot_index(1, 1, 1) as u8);
        anchor.push(slot_index(2, 0, 0) as u8);
        let frame = compute_render_frame(&lib, root, &anchor, 5, None);
        assert_eq!(frame.render_path.depth(), 1);
        match frame.kind {
            ActiveFrameKind::WrappedPlane { dims, slab_depth } => {
                assert_eq!(dims, [3, 1, 1]);
                assert_eq!(slab_depth, 1);
            }
            other => panic!("expected WrappedPlane kind, got {other:?}"),
        }
    }

    #[test]
    fn render_frame_kind_is_cartesian_when_descent_misses_wrapped_plane() {
        use crate::world::tree::{empty_children, slot_index, NodeKind};
        let mut lib = NodeLibrary::default();
        let wp_children = empty_children();
        let wp = lib.insert_with_kind(
            wp_children,
            NodeKind::WrappedPlane { dims: [3, 1, 1], slab_depth: 1 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(wp);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(0);
        let frame = compute_render_frame(&lib, root, &anchor, 3, None);
        assert!(matches!(frame.kind, ActiveFrameKind::Cartesian));
    }
}
