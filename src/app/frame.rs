//! Render-frame helpers: walking the camera path to find the
//! active frame, transforming positions/AABBs into that frame.
//!
//! The "render frame" is the GPU's view of the world. The shader
//! starts ray marching at a frame root, with the camera expressed
//! in that frame's coordinates.
//!
//! All functions here are **pure** — no `App` state — for direct
//! unit testing.

use crate::world::anchor::Path;
use crate::world::tree::{Child, NodeId, NodeKind, NodeLibrary};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActiveFrameKind {
    Cartesian,
    /// Render frame is a `CubedSphereBody` cell. The shader's body
    /// arm (`march_face_subtree_curved`) handles everything below —
    /// the Cartesian DDA must NOT step into face subtrees or it
    /// treats curved UVR wedges as axis-aligned boxes. `logical_path`
    /// may descend deeper for editing; `render_path` stops here.
    Body { inner_r: f32, outer_r: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveFrame {
    /// Path used by the linear ribbon / camera-local transforms.
    pub render_path: Path,
    /// Logical interaction/render layer path. For Cartesian this is
    /// identical to `render_path`. For `Body` frames this may be
    /// deeper: the user's zoom path continues into the face subtree
    /// for editing purposes even though the renderer stops at the
    /// body cell.
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

/// Resolve the active frame by walking the camera path from `world_root`
/// down to `desired_depth`, terminating early at any non-Node child.
///
/// **Clamps at `CubedSphereBody` cells**: when the walk lands on a
/// node tagged `NodeKind::CubedSphereBody`, the body slot is included
/// in the returned path but descent stops immediately. Going deeper
/// would land on `NodeKind::CubedSphereFace` or plain `Cartesian`
/// nodes inside a face subtree — but those "Cartesian" nodes actually
/// partition UVR space, not XYZ, so the shader's Cartesian DDA would
/// treat curved wedges as axis-aligned boxes. The shader's body arm
/// takes over from the clamp point and does the right curved walk.
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
    let mut kind = ActiveFrameKind::Cartesian;
    for k in 0..target.depth() as usize {
        let Some(node) = library.get(node_id) else { break };
        let slot = target.slot(k) as usize;
        match node.children[slot] {
            Child::Node(child_id) => {
                reached.push(slot as u8);
                node_id = child_id;
                // Peek at the child's kind. If it's a CubedSphereBody,
                // we've hit the terminal render-frame root — stop
                // here and let the shader's body arm take over.
                if let Some(child_node) = library.get(child_id) {
                    if let NodeKind::CubedSphereBody { inner_r, outer_r } = child_node.kind {
                        kind = ActiveFrameKind::Body { inner_r, outer_r };
                        break;
                    }
                    // CubedSphereFace should never appear as a
                    // standalone render-frame terminus — it only
                    // exists beneath a CubedSphereBody, which we've
                    // already clamped on. Defensively, if somehow
                    // encountered, continue as Cartesian (don't
                    // break) so existing behavior holds.
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
    }
}

/// Like [`compute_render_frame`] but without the CubedSphereBody
/// clamp: the walker descends into face subtrees, stopping only at
/// non-Node children or the requested depth. Used to build the
/// `logical_path` that tracks the user's zoom target for editing;
/// `render_path` stays clamped at the body via [`compute_render_frame`].
pub fn compute_logical_frame(
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
    ActiveFrame {
        render_path: reached,
        logical_path: reached,
        node_id,
        kind: ActiveFrameKind::Cartesian,
    }
}

pub fn with_render_margin(
    library: &NodeLibrary,
    world_root: NodeId,
    logical_path: &Path,
    render_margin: u8,
) -> ActiveFrame {
    // Logical path must track the user's intended zoom depth — even
    // if that path passes through a CubedSphereBody cell into a face
    // subtree. Editing code relies on `logical_path` being deeper
    // than `render_path` in that case.
    let logical = compute_logical_frame(library, world_root, logical_path, logical_path.depth());
    // Render path stops at the body cell: compute_render_frame
    // returns the clamped version regardless of what the user asked
    // for below the body.
    let render = compute_render_frame(library, world_root, logical_path, logical_path.depth());

    // Shell architecture: the render frame IS the innermost shell
    // root. The shader pops outward via the ribbon for coarser
    // context. No render_margin needed — each shell has a bounded
    // depth budget.
    let render_depth = logical
        .logical_path
        .depth()
        .saturating_sub(render_margin)
        .max(logical.logical_path.depth());
    if render_depth == logical.logical_path.depth() {
        // Preserve the clamped render path + kind from `render`, but
        // use the deeper logical path from `logical`.
        return ActiveFrame {
            render_path: render.render_path,
            logical_path: logical.logical_path,
            node_id: render.node_id,
            kind: render.kind,
        };
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
    use crate::world::tree::{empty_children, uniform_children};

    fn cartesian_chain(depth: u8) -> (NodeLibrary, NodeId) {
        let mut lib = NodeLibrary::default();
        let mut node = lib.insert(empty_children());
        for _ in 1..depth {
            node = lib.insert(uniform_children(Child::Node(node)));
        }
        lib.ref_inc(node);
        (lib, node)
    }

    /// Build a world with a `CubedSphereBody` node at `body_slot`
    /// (depth 1 from root). Beneath the body, `face_depth` levels of
    /// uniform-empty Cartesian children sit in the face-center slot
    /// so we can test deep descents past the body cell.
    fn world_with_body_at_depth_1(
        body_slot: u8,
        face_depth: u8,
        inner_r: f32,
        outer_r: f32,
    ) -> (NodeLibrary, NodeId) {
        let mut lib = NodeLibrary::default();
        // Cartesian chain inside the body cell — acts as the face
        // subtree's deep content. Uses Cartesian kind; that's fine
        // for testing the clamp — the clamp triggers on the body's
        // own `NodeKind`, not on descendants.
        let mut inner = lib.insert(empty_children());
        for _ in 0..face_depth {
            inner = lib.insert(uniform_children(Child::Node(inner)));
        }
        // Body node with face subtree at slot 5 (arbitrary).
        let mut body_children = empty_children();
        body_children[5] = Child::Node(inner);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody { inner_r, outer_r },
        );
        // Root cell with body at `body_slot`.
        let mut root_children = empty_children();
        root_children[body_slot as usize] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        (lib, root)
    }

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

    // --- CubedSphereBody clamp tests --------------------------------

    #[test]
    fn render_frame_stops_at_cubed_sphere_body() {
        // Body at slot 13 (depth 1), then 5 levels of face-subtree
        // content. Anchor points deep past the body — render_path
        // must clamp at depth 1, logical_path must reach depth 6.
        let (lib, root) = world_with_body_at_depth_1(13, 5, 0.20, 0.45);
        let mut anchor = Path::root();
        anchor.push(13); // into body
        anchor.push(5);  // face subtree slot
        for _ in 0..5 {
            anchor.push(13); // deep into face content
        }
        let frame = compute_render_frame(&lib, root, &anchor, anchor.depth());
        assert_eq!(frame.render_path.depth(), 1, "render_path clamps at body cell");
        assert!(
            matches!(frame.kind, ActiveFrameKind::Body { .. }),
            "kind should be Body, got {:?}",
            frame.kind,
        );

        // with_render_margin produces the full logical path while
        // render_path stays clamped.
        let full = with_render_margin(&lib, root, &anchor, 4);
        assert_eq!(
            full.render_path.depth(),
            1,
            "render_path still clamps when routed through with_render_margin"
        );
        assert_eq!(
            full.logical_path.depth(),
            anchor.depth(),
            "logical_path preserves deeper camera anchor for editing",
        );
        assert!(matches!(full.kind, ActiveFrameKind::Body { .. }));
    }

    #[test]
    fn render_frame_body_radii_match_node_kind() {
        let (lib, root) = world_with_body_at_depth_1(13, 2, 0.10, 0.40);
        let mut anchor = Path::root();
        anchor.push(13);
        anchor.push(5);
        anchor.push(13);
        let frame = compute_render_frame(&lib, root, &anchor, anchor.depth());
        match frame.kind {
            ActiveFrameKind::Body { inner_r, outer_r } => {
                assert_eq!(inner_r, 0.10);
                assert_eq!(outer_r, 0.40);
            }
            _ => panic!("expected Body kind, got {:?}", frame.kind),
        }
    }

    #[test]
    fn render_frame_descends_into_cartesian_above_body() {
        // Root is Cartesian; slot 0 is a Cartesian child; inside
        // that, slot 1 is the body. Anchor = [0, 1, 5, 7] should
        // reach the body (render_path depth 2), logical_path depth 4.
        let mut lib = NodeLibrary::default();

        let mut inner = lib.insert(empty_children());
        for _ in 0..3 {
            inner = lib.insert(uniform_children(Child::Node(inner)));
        }
        let mut body_children = empty_children();
        body_children[5] = Child::Node(inner);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody { inner_r: 0.2, outer_r: 0.45 },
        );

        let mut mid_children = empty_children();
        mid_children[1] = Child::Node(body);
        let mid = lib.insert(mid_children);

        let mut root_children = empty_children();
        root_children[0] = Child::Node(mid);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(0); // into mid (Cartesian)
        anchor.push(1); // into body (CubedSphereBody)
        anchor.push(5); // into face subtree
        anchor.push(7); // deeper

        let frame = compute_render_frame(&lib, root, &anchor, anchor.depth());
        assert_eq!(
            frame.render_path.depth(),
            2,
            "render_path stops at body cell (depth 2 from root)",
        );
        assert_eq!(frame.render_path.as_slice(), &[0, 1]);
        assert!(matches!(frame.kind, ActiveFrameKind::Body { .. }));

        let full = with_render_margin(&lib, root, &anchor, 4);
        assert_eq!(full.render_path.depth(), 2);
        assert_eq!(full.logical_path.depth(), anchor.depth());
        assert_eq!(full.logical_path.as_slice(), anchor.as_slice());
    }

    #[test]
    fn render_frame_cartesian_only_world_unchanged() {
        // No body anywhere on the path: kind must stay Cartesian,
        // depths must match the pre-change behaviour.
        let (lib, root) = cartesian_chain(6);
        let mut anchor = Path::root();
        for _ in 0..5 { anchor.push(13); }
        let frame = compute_render_frame(&lib, root, &anchor, 5);
        assert_eq!(frame.render_path.depth(), 5);
        assert_eq!(frame.kind, ActiveFrameKind::Cartesian);

        let full = with_render_margin(&lib, root, &anchor, 4);
        assert_eq!(full.kind, ActiveFrameKind::Cartesian);
        assert_eq!(full.render_path, full.logical_path);
    }
}
