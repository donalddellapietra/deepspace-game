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
use crate::world::sphere_geom::{subframe_range, SphereSubFrameRange, DEFAULT_SPHERE_LAT_MAX};
use crate::world::tree::{Child, NodeId, NodeKind, NodeLibrary};

/// Standard body size for `WrappedPlane` nodes — they always
/// occupy their parent slot's full `[0, 3)³` local frame.
const WRAPPED_PLANE_BODY_SIZE: f32 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActiveFrameKind {
    Cartesian,
    /// The render frame is rooted at a `NodeKind::WrappedPlane`
    /// node. The shader runs the X-wrap branch of `march_cartesian`
    /// at depth==0; the slab's `(dims, slab_depth)` are uploaded as
    /// `Uniforms.slab_dims`.
    WrappedPlane { dims: [u32; 3], slab_depth: u8 },
    /// The render frame is rooted at a node INSIDE a `WrappedPlane`
    /// subtree (= deeper than the WP node itself). Sphere DDA
    /// operates with a ray reparameterized into this sub-frame's
    /// local rotated+translated coords; precision scales with the
    /// sub-frame extent rather than the camera-distance ULP that
    /// walls a body-rooted DDA at depth ~12.
    ///
    /// The sub-frame's (lat, lon, r) range is derived from the
    /// path under the WP via `sphere_geom::subframe_range`; the
    /// sub-frame's local axes are built from the range center.
    SphereSubFrame(SphereSubFrameRange),
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
/// `desired_depth` slot steps.
///
/// Frame-kind transitions during descent:
///   * Reaches a Cartesian Node mid-descent → `Cartesian` kind,
///     keep descending.
///   * Reaches a `WrappedPlane` node and stops there → `WrappedPlane`
///     kind. (Shader's wrap / sphere dispatch fires at marcher-
///     local depth==0, so the render frame IS the slab root.)
///   * Continues PAST a `WrappedPlane` node → `SphereSubFrame` kind
///     for every deeper level. The sub-frame's `(lat, lon, r)`
///     range is derived from the slot path under the WP.
///
/// The "stop at WP" gate of the previous architecture is now opt-in:
/// callers that don't want to descend past the WP can pass a
/// `desired_depth` ≤ the WP's depth.
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
    let mut wp_seen = false;
    let mut kind = match library.get(world_root).map(|n| n.kind) {
        Some(NodeKind::WrappedPlane { dims, slab_depth }) => {
            wp_seen = true;
            ActiveFrameKind::WrappedPlane { dims, slab_depth }
        }
        _ => ActiveFrameKind::Cartesian,
    };
    let mut last_k: usize = 0;
    let mut stopped_early = false;
    for k in 0..target.depth() as usize {
        let Some(node) = library.get(node_id) else { stopped_early = true; break };
        let slot = target.slot(k) as usize;
        match node.children[slot] {
            Child::Node(child_id) => {
                reached.push(slot as u8);
                node_id = child_id;
                last_k = k + 1;
                if let Some(child_node) = library.get(child_id) {
                    if let NodeKind::WrappedPlane { dims, slab_depth } = child_node.kind {
                        wp_seen = true;
                        kind = ActiveFrameKind::WrappedPlane { dims, slab_depth };
                        continue;
                    }
                }
                // Inside a WP subtree, every additional step turns
                // the kind into a SphereSubFrame at the refined
                // (lat, lon, r) range. We always recompute from the
                // full reached path because the WP was found at some
                // earlier step.
                if wp_seen {
                    if let Some(range) = subframe_range(
                        library,
                        world_root,
                        &reached,
                        WRAPPED_PLANE_BODY_SIZE,
                        DEFAULT_SPHERE_LAT_MAX,
                    ) {
                        kind = ActiveFrameKind::SphereSubFrame(range);
                    }
                }
                // Hybrid prototype: when the descent enters the proto
                // target cell's subtree (= [13,13,9,2,22, ...]),
                // upgrade kind to Cartesian. The renderer then
                // dispatches march_cartesian on the deep render
                // frame, and gpu_camera_for_frame computes
                // camera-in-frame with a SHORT walk (= bounded f32 =
                // no jitter). Sphere-mode camera ABOVE the planet
                // doesn't enter this branch — its anchor path goes
                // through Empty WP children, descent stops at WP.
                // But once the camera is inside the proto cell's
                // subtree, the precision-stable Cartesian frame
                // takes over.
                let proto_cell_path: [u8; 5] = [13, 13, 9, 2, 22];
                if reached.depth() as usize >= proto_cell_path.len()
                    && reached.as_slice()[..proto_cell_path.len()] == proto_cell_path
                {
                    kind = ActiveFrameKind::Cartesian;
                }
            }
            Child::Block(_) | Child::Empty | Child::EntityRef(_) => {
                stopped_early = true;
                break;
            }
        }
    }
    // Past-leaf SphereSubFrame refinement: when the library descent
    // stopped inside (or at) a WrappedPlane but the camera anchor
    // wanted to go deeper, KEEP refining the sub-frame range using
    // the remaining slots of the target path. The render frame's
    // `reached` path stays at the deepest reachable Node (= what the
    // GPU buffer can address), but the SphereSubFrame range tracks
    // the camera's logical (lat, lon, r) extent so deep-zoom rays
    // can run on bounded sub-frame coords. Without this, the kind
    // stays WrappedPlane whenever the camera anchor's f32-deepened
    // path hits an Empty slot above the slab — the exact case the
    // sub-frame architecture exists to fix.
    if wp_seen && stopped_early && last_k < target.depth() as usize {
        let mut virtual_path = reached;
        for k in last_k..target.depth() as usize {
            virtual_path.push(target.slot(k));
        }
        if virtual_path.depth() > reached.depth() {
            if let Some(range) = subframe_range(
                library,
                world_root,
                &virtual_path,
                WRAPPED_PLANE_BODY_SIZE,
                DEFAULT_SPHERE_LAT_MAX,
            ) {
                kind = ActiveFrameKind::SphereSubFrame(range);
            }
        }
    }
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

    /// When the library descent stops at the WP because the deeper
    /// slot is a Block leaf, `render_path` stays at the WP — the
    /// shader still runs its dispatch from the WP node. But the kind
    /// upgrades to `SphereSubFrame` whenever the camera anchor's
    /// target path extends past the WP, so the renderer projects the
    /// camera into sub-frame local coords (precision-stable) instead
    /// of body-rooted WP-local coords (which wall at depth ~22).
    #[test]
    fn render_frame_upgrades_to_sphere_subframe_past_wp_leaf() {
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

        // Camera anchor: WP (depth 1) → Block slot at depth 2 →
        // virtual deeper slots up to desired_depth=5.
        let mut anchor = Path::root();
        anchor.push(slot_index(1, 1, 1) as u8); // depth 1 → WP node
        anchor.push(slot_index(2, 0, 0) as u8); // depth 2 → Block (descent stops)
        anchor.push(slot_index(1, 1, 1) as u8); // virtual depth 3
        anchor.push(slot_index(1, 1, 1) as u8); // virtual depth 4
        anchor.push(slot_index(1, 1, 1) as u8); // virtual depth 5
        let frame = compute_render_frame(&lib, root, &anchor, 5);
        // Render path stops at the WP (deepest reachable Node).
        assert_eq!(frame.render_path.depth(), 1);
        // But the kind upgrades to SphereSubFrame because the target
        // path extends 4 slots past the WP.
        match frame.kind {
            ActiveFrameKind::SphereSubFrame(range) => {
                assert_eq!(range.wp_dims, [3, 1, 1]);
                assert_eq!(range.wp_slab_depth, 1);
                assert_eq!(range.wp_path_depth, 1);
                // 4 slots past WP → sub-frame extents shrink by 3^4.
                let expected_lon_extent = (2.0 * std::f32::consts::PI) / 3.0_f32.powi(4);
                assert!(
                    (range.lon_extent() - expected_lon_extent).abs() < 1e-5,
                    "lon extent should refine 4 levels past WP, got {}",
                    range.lon_extent(),
                );
            }
            other => panic!("expected SphereSubFrame kind, got {other:?}"),
        }
    }

    /// When `desired_depth` extends BEYOND the WP, descent now
    /// continues past the WP and reports `SphereSubFrame` for the
    /// deeper levels (sphere sub-frame architecture, step 2).
    #[test]
    fn render_frame_kind_is_sphere_subframe_when_descending_past_wrapped_plane() {
        use crate::world::tree::{empty_children, slot_index, NodeKind};
        let mut lib = NodeLibrary::default();
        let mut wp_children = empty_children();
        wp_children[slot_index(1, 1, 1)] = Child::Block(crate::world::palette::block::GRASS);
        let wp = lib.insert_with_kind(
            wp_children,
            NodeKind::WrappedPlane { dims: [3, 1, 1], slab_depth: 1 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(wp);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        // Anchor: descend to WP (1 slot), then attempt to go deeper.
        // The slab's slot (1,1,1) is a Block leaf, so descent still
        // stops there. But for a path that has a Node at the deeper
        // slot, the SphereSubFrame kind should appear. Construct
        // such a tree:
        let mut nested_inner = empty_children();
        nested_inner[slot_index(0, 0, 0)] = Child::Block(crate::world::palette::block::STONE);
        let nested = lib.insert(nested_inner);
        let mut wp2_children = empty_children();
        wp2_children[slot_index(1, 1, 1)] = Child::Node(nested);
        let wp2 = lib.insert_with_kind(
            wp2_children,
            NodeKind::WrappedPlane { dims: [3, 1, 1], slab_depth: 1 },
        );
        let mut root2_children = empty_children();
        root2_children[slot_index(1, 1, 1)] = Child::Node(wp2);
        let root2 = lib.insert(root2_children);
        lib.ref_inc(root2);

        let mut anchor = Path::root();
        anchor.push(slot_index(1, 1, 1) as u8);
        anchor.push(slot_index(1, 1, 1) as u8);
        let frame = compute_render_frame(&lib, root2, &anchor, 5);
        assert_eq!(frame.render_path.depth(), 2,
                   "should descend through WP into deeper sphere sub-frame");
        match frame.kind {
            ActiveFrameKind::SphereSubFrame(range) => {
                assert_eq!(range.wp_dims, [3, 1, 1]);
                assert_eq!(range.wp_slab_depth, 1);
                assert_eq!(range.wp_path_depth, 1);
            }
            other => panic!("expected SphereSubFrame, got {other:?}"),
        }
    }

    /// `desired_depth` ≤ the WP's depth still stops at the WP and
    /// reports `WrappedPlane` kind (= back-compat with the previous
    /// architecture; the shader's slab-cell DDA + ribbon still fires
    /// from the WP root in that regime).
    #[test]
    fn render_frame_kind_is_wrapped_plane_when_desired_depth_at_wp() {
        use crate::world::tree::{empty_children, slot_index, NodeKind};
        let mut lib = NodeLibrary::default();
        let wp = lib.insert_with_kind(
            empty_children(),
            NodeKind::WrappedPlane { dims: [3, 1, 1], slab_depth: 1 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(wp);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(slot_index(1, 1, 1) as u8);
        let frame = compute_render_frame(&lib, root, &anchor, 1);
        assert_eq!(frame.render_path.depth(), 1);
        assert!(matches!(frame.kind, ActiveFrameKind::WrappedPlane { .. }));
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
