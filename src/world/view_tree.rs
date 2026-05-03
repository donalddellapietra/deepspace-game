//! Camera-aware view tree: a per-frame pruning of `world.root`
//! that captures distance-based LOD.
//!
//! ## Why
//!
//! The current pack uses **content-aware** uniform-flatten: a Node
//! whose subtree is one block type collapses to a `Child::Block`
//! splat in its parent's slab. This decision is made once at pack
//! time, captures only the cell's content, and ignores the camera.
//! Result: adjacent cells aggregate differently if their content
//! differs (one uniform, one mixed) and the visible LOD pops as
//! the camera crosses cell boundaries.
//!
//! The view tree replaces this with a **camera-aware** prune. At
//! every internal Node we ask: is this child's cell sub-pixel from
//! the camera's POV? If yes, splat. If no, recurse. The decision
//! depends only on geometry, not on the cell's content
//! coincidentally being uniform.
//!
//! ## Behaviour
//!
//! For each `Child::Node(c)` in a Cartesian Node:
//! - Compute the child's world AABB from the parent's origin and
//!   slot offset.
//! - Distance from the camera to that AABB (clamped to box; 0 if
//!   inside).
//! - Pixel size: `cell_size / dist * focal`. Sub-pixel ⇒ splat.
//! - Splat as `Child::Block(representative)` (or `Child::Empty` if
//!   the representative is `REPRESENTATIVE_EMPTY`).
//! - Otherwise recurse and emit `Child::Node(view_subtree_id)`.
//!
//! `Child::Empty`, `Child::Block(_)`, `Child::EntityRef(_)` pass
//! through unchanged. Non-Cartesian Nodes (`WrappedPlane`,
//! `TangentBlock`) pass through too — their internal layout has
//! shader-side dispatch semantics this routine must not disturb.
//!
//! ## Dedup amortization
//!
//! Distant uniform regions of the world all collapse to the same
//! splat (same representative, same Empty/Block kind). Content-
//! addressed dedup in `NodeLibrary` makes those identical
//! prunings share NodeIds across frames and across siblings. The
//! per-frame cost is dominated by the small set of cells near the
//! camera that change; far regions reuse from previous frames.
//!
//! ## Edits
//!
//! Edits operate on `world.root`. The view tree is rebuilt every
//! upload from the latest world state; edits are visible the
//! following frame regardless of where the camera is, because the
//! cells near the edit get freshly walked and their splat color
//! (representative_block) reflects the new content.

use super::tree::{
    slot_coords, Child, NodeId, NodeKind, NodeLibrary, REPRESENTATIVE_EMPTY,
};

/// Build the view tree by pruning `world_root` against the camera's
/// world position. Returns a NodeId in `library` representing the
/// pruned tree. The world root is in the cubic frame `[0, 3)³`;
/// `camera_world` is the camera's position in that frame.
///
/// `focal_pixels` is `screen_height / (2 * tan(fov/2))` — the same
/// constant the shader uses in `at_lod = lod_pixels <
/// LOD_PIXEL_THRESHOLD`. Caller passes it once.
///
/// `pixel_threshold` matches `LOD_PIXEL_THRESHOLD` in the shader
/// (default 1.0 = sub-pixel).
pub fn build_view_tree(
    library: &mut NodeLibrary,
    world_root: NodeId,
    camera_world: [f32; 3],
    focal_pixels: f32,
    pixel_threshold: f32,
) -> NodeId {
    build_recursive(
        library,
        world_root,
        [0.0, 0.0, 0.0],
        3.0,
        camera_world,
        focal_pixels,
        pixel_threshold,
    )
}

fn build_recursive(
    library: &mut NodeLibrary,
    node_id: NodeId,
    cell_origin: [f32; 3],
    cell_size: f32,
    camera: [f32; 3],
    focal: f32,
    threshold: f32,
) -> NodeId {
    let (children_in, kind) = match library.get(node_id) {
        Some(n) => (n.children, n.kind),
        None => return node_id,
    };

    // Non-Cartesian nodes pass through — shader-side dispatch needs
    // their layout intact.
    if !matches!(kind, NodeKind::Cartesian) {
        return node_id;
    }

    let mut new_children = children_in;
    let child_size = cell_size / 3.0;

    for slot in 0..27 {
        let Child::Node(child_id) = children_in[slot] else { continue };

        let (sx, sy, sz) = slot_coords(slot);
        let child_origin = [
            cell_origin[0] + sx as f32 * child_size,
            cell_origin[1] + sy as f32 * child_size,
            cell_origin[2] + sz as f32 * child_size,
        ];
        let dist = aabb_distance(camera, child_origin, child_size);
        let pixels = if dist <= 1e-6 {
            f32::INFINITY
        } else {
            (child_size / dist) * focal
        };

        if pixels < threshold {
            // Sub-pixel: splat with the subtree's representative.
            let rep = library
                .get(child_id)
                .map(|n| n.representative_block)
                .unwrap_or(REPRESENTATIVE_EMPTY);
            new_children[slot] = if rep == REPRESENTATIVE_EMPTY {
                Child::Empty
            } else {
                Child::Block(rep)
            };
        } else {
            // Visible: recurse.
            let pruned = build_recursive(
                library, child_id, child_origin, child_size, camera, focal, threshold,
            );
            new_children[slot] = Child::Node(pruned);
        }
    }

    library.insert(new_children)
}

/// Euclidean distance from `point` to the AABB `[origin, origin + size]³`.
/// Zero if the point is inside the box.
fn aabb_distance(point: [f32; 3], origin: [f32; 3], size: f32) -> f32 {
    let mut sq = 0.0_f32;
    for i in 0..3 {
        let lo = origin[i];
        let hi = origin[i] + size;
        let p = point[i];
        let d = if p < lo { lo - p } else if p > hi { p - hi } else { 0.0 };
        sq += d * d;
    }
    sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, slot_index, uniform_children};

    fn solid(library: &mut NodeLibrary, depth: u32, block: u16) -> NodeId {
        let mut child = Child::Block(block);
        for _ in 0..depth {
            let id = library.insert(uniform_children(child));
            child = Child::Node(id);
        }
        match child {
            Child::Node(id) => id,
            _ => unreachable!(),
        }
    }

    #[test]
    fn near_camera_keeps_full_detail() {
        // World tree: depth 3 of uniform stone. Camera right inside.
        // Should keep all internal Nodes (recurse).
        let mut lib = NodeLibrary::default();
        let root = solid(&mut lib, 3, crate::world::palette::block::STONE);
        let camera = [1.5_f32, 1.5, 1.5];
        let focal = 720.0 / (2.0 * (1.2_f32 * 0.5).tan());
        let view = build_view_tree(&mut lib, root, camera, focal, 1.0);

        // Walk the view tree — should still have Nodes at depth 1 and 2.
        let n0 = lib.get(view).unwrap();
        match n0.children[slot_index(1, 1, 1)] {
            Child::Node(_) => {} // good
            other => panic!("expected Node at depth 1, got {other:?}"),
        }
    }

    #[test]
    fn far_camera_collapses_to_splat() {
        // Camera far away → near-root cells should be sub-pixel and
        // emitted as Block(representative).
        let mut lib = NodeLibrary::default();
        let root = solid(&mut lib, 6, crate::world::palette::block::STONE);
        // Camera position outside [0, 3)^3 by a large margin.
        let camera = [1e6_f32, 1.5, 1.5];
        let focal = 720.0 / (2.0 * (1.2_f32 * 0.5).tan());
        let view = build_view_tree(&mut lib, root, camera, focal, 1.0);

        let n0 = lib.get(view).unwrap();
        // At root, the entire world is a single sub-pixel splat.
        // Each Child::Node(_) at depth 0 should have collapsed to Block.
        let solid_count = n0
            .children
            .iter()
            .filter(|c| matches!(c, Child::Block(_)))
            .count();
        assert!(
            solid_count > 0,
            "expected at least some Block splats at far distance"
        );
    }

    #[test]
    fn empty_subtrees_collapse_to_empty() {
        // A tree where the root has both an Empty child and a Block child.
        // The Block is far → collapses to Block (representative).
        // The Empty stays empty.
        let mut lib = NodeLibrary::default();
        let stone_subtree = solid(&mut lib, 3, crate::world::palette::block::STONE);
        let mut children = empty_children();
        children[slot_index(2, 2, 2)] = Child::Node(stone_subtree);
        let root = lib.insert(children);

        let camera = [1e6_f32, 1.5, 1.5];
        let focal = 720.0 / (2.0 * (1.2_f32 * 0.5).tan());
        let view = build_view_tree(&mut lib, root, camera, focal, 1.0);

        let n = lib.get(view).unwrap();
        // The stone subtree should collapse to Block(stone).
        match n.children[slot_index(2, 2, 2)] {
            Child::Block(_) => {}
            other => panic!("expected Block splat for far stone subtree, got {other:?}"),
        }
        // Other slots stay Empty.
        for s in 0..27 {
            if s == slot_index(2, 2, 2) {
                continue;
            }
            assert!(
                matches!(n.children[s], Child::Empty),
                "slot {s} should be Empty"
            );
        }
    }

    #[test]
    fn idempotent_when_camera_unchanged() {
        let mut lib = NodeLibrary::default();
        let root = solid(&mut lib, 4, crate::world::palette::block::STONE);
        let camera = [1.5_f32, 1.5, 1.5];
        let focal = 720.0 / (2.0 * (1.2_f32 * 0.5).tan());
        let v1 = build_view_tree(&mut lib, root, camera, focal, 1.0);
        let v2 = build_view_tree(&mut lib, root, camera, focal, 1.0);
        assert_eq!(v1, v2, "same camera → same view tree NodeId via dedup");
    }

    #[test]
    fn does_not_descend_through_non_cartesian() {
        use crate::world::tree::NodeKind;
        let mut lib = NodeLibrary::default();
        let leaf = solid(&mut lib, 2, crate::world::palette::block::STONE);
        let mut wp_children = empty_children();
        wp_children[slot_index(1, 1, 1)] = Child::Node(leaf);
        let wp = lib.insert_with_kind(
            wp_children,
            NodeKind::WrappedPlane { dims: [3, 1, 1], slab_depth: 1 },
        );
        let camera = [1.5_f32, 1.5, 1.5];
        let focal = 720.0 / (2.0 * (1.2_f32 * 0.5).tan());
        let view = build_view_tree(&mut lib, wp, camera, focal, 1.0);
        // WrappedPlane returned as-is.
        assert_eq!(view, wp);
    }

    #[test]
    fn distance_zero_inside_box() {
        let d = aabb_distance([1.5, 1.5, 1.5], [0.0, 0.0, 0.0], 3.0);
        assert_eq!(d, 0.0);
    }

    #[test]
    fn distance_outside_box() {
        let d = aabb_distance([5.0, 1.5, 1.5], [0.0, 0.0, 0.0], 3.0);
        assert_eq!(d, 2.0);
    }
}
