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

use crate::world::anchor::{Path, WorldPos};
use crate::world::tree::{slot_index, Child, NodeId, NodeKind, NodeLibrary};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActiveFrameKind {
    Cartesian,
    /// The render frame is rooted at a `NodeKind::WrappedPlane`
    /// node. The shader runs the X-wrap branch of `march_cartesian`
    /// at depth==0; the slab's `(dims, slab_depth)` are uploaded as
    /// `Uniforms.slab_dims`.
    WrappedPlane { dims: [u32; 3], slab_depth: u8 },
    /// The render frame is rooted at a `NodeKind::TangentBlock` node.
    /// shade_pixel pre-rotates camera.pos and ray_dir by Mᵀ before
    /// march_cartesian, so the DDA finds the rotated cells inside.
    /// The descent stops here (camera anchor can be deeper but the
    /// render frame doesn't follow into the rotated subtree's
    /// cartesian descendants — that would require tracking cumulative
    /// rotation through WorldPos.in_frame, a follow-up).
    TangentBlock,
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
    /// Path depth at which the descent crossed a `NodeKind::TangentBlock`
    /// (i.e. the path index of the TB node itself). `None` when the
    /// frame's path doesn't go through any rotated subtree. When set,
    /// camera-position math must apply Mᵀ (rotation) at this index
    /// during the slot walk so that frame-local coords inside the
    /// rotated subtree are interpreted as rotated-axes — see
    /// `WorldPos::in_frame_with_rotation`.
    pub tangent_crossing: Option<u8>,
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
/// Descends from `world_root` along the camera's *physical position*
/// for at most `desired_depth` slot steps. At each level, the slot
/// is picked by floor()-ing the camera's local-frame position, NOT
/// by reading the camera's stored anchor slot. This matters inside
/// `NodeKind::TangentBlock` subtrees: the anchor's slot path is
/// constructed by axis-aligned `zoom_in`, but cells inside a rotated
/// subtree subdivide rotated-axes, so the anchor's slot index past
/// a TB doesn't correspond to where the camera physically is. By
/// recomputing slots from the running camera-local position (and
/// applying Mᵀ at the TB crossing), the render frame chases the
/// camera correctly through rotated subtrees and `target_render_frame`
/// doesn't have to back off.
///
/// `tangent_rotation_cols` are the columns of M for any TB the
/// descent might cross (identity when no rotation is configured).
///
/// Stops early at `WrappedPlane` (its X-wrap fires at marcher
/// depth 0). Continues through TB into cartesian descendants —
/// anchor-descent into rotated subtrees relies on this.
pub fn compute_render_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    camera_pos: &WorldPos,
    tangent_rotation_cols: &[[f32; 3]; 3],
    desired_depth: u8,
) -> ActiveFrame {
    let mut node_id = world_root;
    let mut reached = Path::root();
    let mut kind = match library.get(world_root).map(|n| n.kind) {
        Some(NodeKind::WrappedPlane { dims, slab_depth }) => {
            ActiveFrameKind::WrappedPlane { dims, slab_depth }
        }
        Some(NodeKind::TangentBlock) => ActiveFrameKind::TangentBlock,
        _ => ActiveFrameKind::Cartesian,
    };
    let mut tangent_crossing: Option<u8> =
        if matches!(kind, ActiveFrameKind::TangentBlock) { Some(0) } else { None };
    let trace = std::env::var("DSG_FRAME_TRACE").map(|v| v != "0").unwrap_or(false);
    for iter_k in 0..desired_depth {
        // Both WrappedPlane and TangentBlock are HARD STOPS. The
        // walker that runs at the active frame (X-wrap shader for
        // WrappedPlane, march_in_tangent_cube for TangentBlock)
        // owns the [0, 3)³ box and traverses its full depth in
        // one shot. Descending past these would force rays exiting
        // the smaller frame to return sky (no ribbon support in
        // those walkers), losing far-away content visibility.
        if !matches!(kind, ActiveFrameKind::Cartesian) {
            if trace {
                eprintln!("frame_trace iter={} BREAK kind={:?} reached_depth={}", iter_k, kind, reached.depth());
            }
            break;
        }
        let Some(node) = library.get(node_id) else {
            if trace {
                eprintln!("frame_trace iter={} BREAK node_id={} not in library reached_depth={}", iter_k, node_id, reached.depth());
            }
            break;
        };
        // Recompute cam_local fresh from the camera's WorldPos at
        // each iteration. Iteratively scaling cam_local by 3× per
        // descent step compounds f32 error catastrophically — by
        // depth 15 a single ulp of starting error has grown to
        // ~1 cell. `in_frame` / `in_frame_with_rotation` use the
        // anchor's slot decomposition + offset directly, with f32
        // precision bounded by the frame's local extent (not by
        // the cumulative descent depth).
        let cam_local = match tangent_crossing {
            Some(crossing) => camera_pos.in_frame_with_rotation(
                &reached, tangent_rotation_cols, crossing,
            ),
            None => camera_pos.in_frame(&reached),
        };
        // If camera is outside the current frame's [0, 3)³, descent
        // can't continue — the camera isn't physically inside any
        // child of the current node.
        if cam_local[0] < 0.0 || cam_local[0] >= 3.0
            || cam_local[1] < 0.0 || cam_local[1] >= 3.0
            || cam_local[2] < 0.0 || cam_local[2] >= 3.0
        {
            if trace {
                eprintln!(
                    "frame_trace iter={} BREAK cam_oob cam=({:.3}, {:.3}, {:.3}) reached_depth={} kind={:?} crossing={:?}",
                    iter_k, cam_local[0], cam_local[1], cam_local[2], reached.depth(), kind, tangent_crossing,
                );
            }
            break;
        }
        // Pick slot from the camera's CURRENT local position, not
        // from the camera anchor's stored slots. Inside a rotated
        // subtree the anchor's axis-aligned `zoom_in` produces slots
        // that don't correspond to the camera's actual location;
        // recomputing here ensures the render frame chases the
        // camera through rotated descendants correctly.
        let sx = (cam_local[0]).floor().clamp(0.0, 2.0) as usize;
        let sy = (cam_local[1]).floor().clamp(0.0, 2.0) as usize;
        let sz = (cam_local[2]).floor().clamp(0.0, 2.0) as usize;
        let slot = slot_index(sx, sy, sz);
        match node.children[slot] {
            Child::Node(child_id) => {
                if trace {
                    eprintln!(
                        "frame_trace iter={} push slot={} cam=({:.3},{:.3},{:.3}) child_id={} kind={:?}",
                        iter_k, slot, cam_local[0], cam_local[1], cam_local[2], child_id, kind,
                    );
                }
                reached.push(slot as u8);
                node_id = child_id;
                if let Some(child_node) = library.get(child_id) {
                    match child_node.kind {
                        NodeKind::WrappedPlane { dims, slab_depth } => {
                            kind = ActiveFrameKind::WrappedPlane { dims, slab_depth };
                            break;
                        }
                        NodeKind::TangentBlock => {
                            // Cross into rotated subtree — record
                            // the crossing index so the next
                            // iteration's `in_frame_with_rotation`
                            // applies Mᵀ at the right path index.
                            kind = ActiveFrameKind::TangentBlock;
                            tangent_crossing = Some(reached.depth());
                            if trace {
                                eprintln!("frame_trace iter={} CROSSED_TB crossing={}", iter_k, reached.depth());
                            }
                        }
                        NodeKind::Cartesian => {}
                    }
                }
            }
            Child::Block(b) => {
                if trace {
                    eprintln!("frame_trace iter={} BREAK Block({}) at slot={} reached_depth={}", iter_k, b, slot, reached.depth());
                }
                break;
            }
            Child::Empty => {
                if trace {
                    eprintln!("frame_trace iter={} BREAK Empty at slot={} reached_depth={}", iter_k, slot, reached.depth());
                }
                break;
            }
            Child::EntityRef(_) => {
                if trace {
                    eprintln!("frame_trace iter={} BREAK EntityRef at slot={} reached_depth={}", iter_k, slot, reached.depth());
                }
                break;
            }
        }
    }
    if trace {
        eprintln!(
            "frame_trace EXIT reached_depth={} kind={:?} crossing={:?} desired_depth={}",
            reached.depth(), kind, tangent_crossing, desired_depth,
        );
    }
    ActiveFrame {
        render_path: reached,
        logical_path: reached,
        node_id,
        kind,
        tangent_crossing,
    }
}

/// Determine the `ActiveFrame` for an explicit path (the path is
/// authoritative; no slot-recomputation). Used at upload time when
/// the GPU pack returns the slots it actually reached — those
/// might differ from what camera-driven descent would pick. Walks
/// the path to figure out kind transitions and the TB crossing
/// index. Identical to a slot-walking compute_render_frame.
pub fn frame_for_path(
    library: &NodeLibrary,
    world_root: NodeId,
    path: &Path,
) -> ActiveFrame {
    let mut node_id = world_root;
    let mut reached = Path::root();
    let mut kind = match library.get(world_root).map(|n| n.kind) {
        Some(NodeKind::WrappedPlane { dims, slab_depth }) => {
            ActiveFrameKind::WrappedPlane { dims, slab_depth }
        }
        Some(NodeKind::TangentBlock) => ActiveFrameKind::TangentBlock,
        _ => ActiveFrameKind::Cartesian,
    };
    let mut tangent_crossing: Option<u8> =
        if matches!(kind, ActiveFrameKind::TangentBlock) { Some(0) } else { None };
    for k in 0..path.depth() as usize {
        if matches!(kind, ActiveFrameKind::WrappedPlane { .. }) {
            break;
        }
        let Some(node) = library.get(node_id) else { break };
        let slot = path.slot(k) as usize;
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
                        NodeKind::TangentBlock => {
                            kind = ActiveFrameKind::TangentBlock;
                            tangent_crossing = Some(reached.depth());
                        }
                        NodeKind::Cartesian => {}
                    }
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
        tangent_crossing,
    }
}

pub fn with_render_margin(
    library: &NodeLibrary,
    world_root: NodeId,
    camera_pos: &WorldPos,
    tangent_rotation_cols: &[[f32; 3]; 3],
    render_margin: u8,
) -> ActiveFrame {
    let logical = compute_render_frame(
        library, world_root, camera_pos, tangent_rotation_cols,
        camera_pos.anchor.depth(),
    );
    let min_render_depth = logical.logical_path.depth();
    let render_depth = logical
        .logical_path
        .depth()
        .saturating_sub(render_margin)
        .max(min_render_depth);
    if render_depth == logical.logical_path.depth() {
        return logical;
    }
    // Shallower descent uses the same camera position; the loop's
    // depth cap is what shortens it.
    let render = compute_render_frame(
        library, world_root, camera_pos, tangent_rotation_cols, render_depth,
    );
    ActiveFrame {
        render_path: render.render_path,
        logical_path: logical.logical_path,
        node_id: render.node_id,
        kind: render.kind,
        tangent_crossing: render.tangent_crossing,
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

    const ID: [[f32; 3]; 3] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];

    fn center_camera(depth: u8) -> WorldPos {
        // Center of every level along slot 13. Offset is in [0, 1)
        // per axis (sub-cell coords of the deepest frame).
        WorldPos::uniform_column(13, depth, [0.5, 0.5, 0.5])
    }

    #[test]
    fn render_frame_root_when_desired_depth_zero() {
        let (lib, root) = cartesian_chain(5);
        let cam = center_camera(3);
        let frame = compute_render_frame(&lib, root, &cam, &ID, 0);
        assert_eq!(frame.render_path.depth(), 0);
        assert_eq!(frame.node_id, root);
    }

    #[test]
    fn render_frame_descends_through_cartesian() {
        let (lib, root) = cartesian_chain(5);
        let cam = center_camera(4);
        let frame = compute_render_frame(&lib, root, &cam, &ID, 3);
        assert_eq!(frame.render_path.depth(), 3);
    }

    #[test]
    fn render_frame_truncates_when_camera_anchor_shallow() {
        // Camera at depth 1 (shallow). Even with desired_depth=5, the
        // descent runs until the loop hits Block/Empty children. With
        // cartesian_chain(5), descent hits Empty at depth 1 (the
        // chain's deepest node has empty_children).
        let (lib, root) = cartesian_chain(5);
        let cam = center_camera(1);
        let frame = compute_render_frame(&lib, root, &cam, &ID, 5);
        assert!(frame.render_path.depth() <= 5);
    }

    #[test]
    fn render_frame_stops_when_path_misses_node() {
        // Build root with a Block at slot 5 (not a Node). Camera in
        // slot 5 → descent picks slot 5, sees Block, stops at root.
        let mut lib = NodeLibrary::default();
        let mut root_children = empty_children();
        root_children[5] = Child::Block(crate::world::palette::block::STONE);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        // slot 5 = slot_index(2, 1, 0). Offset is sub-cell in [0, 1).
        let cam = WorldPos::uniform_column(5, 1, [0.5, 0.5, 0.5]);
        let frame = compute_render_frame(&lib, root, &cam, &ID, 2);
        assert_eq!(frame.render_path.depth(), 0, "Block child terminates descent");
    }

    #[test]
    fn frame_from_slots_builds_exact_prefix() {
        let slots = [13u8, 16u8, 4u8];
        let p = frame_from_slots(&slots);
        assert_eq!(p.depth(), slots.len() as u8);
        assert_eq!(p.as_slice(), &slots);
    }

    /// When the descent reaches a `WrappedPlane` node, the render
    /// frame must STOP there (kind = WrappedPlane) instead of
    /// descending into the slab subtree. The shader's wrap branch
    /// fires at marcher-local depth==0 — which means the render
    /// frame must be the slab root, not a sub-cell.
    #[test]
    fn render_frame_kind_is_wrapped_plane_when_descent_lands_on_one() {
        use crate::world::tree::{empty_children, slot_index, NodeKind};
        let mut lib = NodeLibrary::default();
        // Build a small WrappedPlane subtree (slab depth 1, dims
        // [3, 1, 1] — fills X axis only).
        let mut wp_children = empty_children();
        wp_children[slot_index(0, 0, 0)] = Child::Block(crate::world::palette::block::GRASS);
        wp_children[slot_index(1, 0, 0)] = Child::Block(crate::world::palette::block::GRASS);
        wp_children[slot_index(2, 0, 0)] = Child::Block(crate::world::palette::block::GRASS);
        let wp = lib.insert_with_kind(
            wp_children,
            NodeKind::WrappedPlane { dims: [3, 1, 1], slab_depth: 1 },
        );
        // Embed: root has WP at slot 13, everything else empty.
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(wp);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        // Camera in slot 13 of root (= WP cell) at center.
        let cam = WorldPos::uniform_column(13, 1, [0.5, 0.5, 0.5]);
        let frame = compute_render_frame(&lib, root, &cam, &ID, 5);
        // Render frame stops at the WP node (depth 1), not at the
        // sub-cell at depth 2.
        assert_eq!(frame.render_path.depth(), 1);
        match frame.kind {
            ActiveFrameKind::WrappedPlane { dims, slab_depth } => {
                assert_eq!(dims, [3, 1, 1]);
                assert_eq!(slab_depth, 1);
            }
            other => panic!("expected WrappedPlane kind, got {other:?}"),
        }
    }

    /// Camera anchor that doesn't enter the WrappedPlane subtree
    /// must produce a plain Cartesian render frame — wrap can't
    /// fire from outside the slab.
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

        // Camera in slot 0 of root — the WP is in slot 13 (centre).
        let cam = WorldPos::uniform_column(0, 1, [0.5, 0.5, 0.5]);
        let frame = compute_render_frame(&lib, root, &cam, &ID, 3);
        assert!(matches!(frame.kind, ActiveFrameKind::Cartesian));
    }
}
