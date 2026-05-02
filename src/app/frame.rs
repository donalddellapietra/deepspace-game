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
/// `desired_depth` slot steps. The descent walks any Cartesian
/// or `WrappedPlane` Node it finds; it does NOT stop early at
/// the `WrappedPlane` boundary the way it used to.
///
/// **Why descent must go past WrappedPlane.** Plain Cartesian's
/// precision discipline is "render frame follows the camera
/// anchor down with a small render margin". `WorldPos::in_frame`
/// then walks `[common_prefix..anchor_depth)` slots — short tail,
/// every contribution bounded by `WORLD_SIZE`, f32 precise. With
/// a WP-stop the render frame stays pinned at depth ≈ 2 while
/// the camera anchor goes to 18+, so the tail walk accumulates
/// ~16 levels of slot offsets in WP-local; deep slots
/// (`3 / 3^N`) sit below the f32 eps for the running sum and
/// the camera's WP-local position has ~2e-7 of fixed noise.
/// At deep anchor that noise is bigger than the cell the cursor
/// or pixel is supposed to resolve — visible as camera jitter
/// and "off by some" CPU raycast hits. Letting descent continue
/// past WP (through the slab subgrid, into a `TangentBlock`,
/// into the cube's Cartesian interior) keeps the tail walk short
/// and the camera local at full precision — same trick Cartesian
/// has always used.
///
/// The kind tracked here is the kind of the FINAL node landed on.
/// `WrappedPlane` only when the descent literally ends at the WP
/// node (camera anchor at WP depth, or shallower / pointed away).
/// Once descent goes through WP, kind reverts to whatever the
/// deeper node carries — typically Cartesian.
///
/// **X-wrap caveat.** `march_cartesian`'s X-wrap branch fires at
/// marcher-local `depth == 0` of a WP frame; with the render
/// frame past WP it can't fire from the player's POV. That's
/// acceptable — X-wrap matters when navigating across the slab
/// (camera at-or-above the surface, render frame stays at WP),
/// not when zoomed in on a single cube.
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
    for k in 0..target.depth() as usize {
        let Some(node) = library.get(node_id) else { break };
        let slot = target.slot(k) as usize;
        match node.children[slot] {
            Child::Node(child_id) => {
                reached.push(slot as u8);
                node_id = child_id;
            }
            Child::Block(_) | Child::Empty | Child::EntityRef(_) => break,
        }
    }
    let kind = match library.get(node_id).map(|n| n.kind) {
        Some(NodeKind::WrappedPlane { dims, slab_depth }) => {
            ActiveFrameKind::WrappedPlane { dims, slab_depth }
        }
        _ => ActiveFrameKind::Cartesian,
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
    logical_path: &Path,
    render_margin: u8,
) -> ActiveFrame {
    let logical = compute_render_frame(library, world_root, logical_path, logical_path.depth());
    // Shell architecture: the render frame IS the innermost
    // shell root. The shader pops outward via the ribbon for
    // coarser context. Each shell has a bounded depth budget.
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

        // Camera anchor sits at the WP node (depth 1) or deeper.
        let mut anchor = Path::root();
        anchor.push(slot_index(1, 1, 1) as u8); // depth 1 → WP node
        anchor.push(slot_index(2, 0, 0) as u8); // depth 2 → inside WP
        let frame = compute_render_frame(&lib, root, &anchor, 5);
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
        let mut anchor = Path::root();
        anchor.push(0);
        let frame = compute_render_frame(&lib, root, &anchor, 3);
        assert!(matches!(frame.kind, ActiveFrameKind::Cartesian));
    }
}
