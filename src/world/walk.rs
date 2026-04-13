//! Tree walk: DFS traversal of the content-addressed tree, collecting
//! visible nodes at the renderer's emit layer.
//!
//! The walk is layer-aware: it descends from the root to `emit_layer`,
//! culling nodes whose AABB is outside a radius sphere around the
//! camera. The output is a list of `Visit`s — one per visible node —
//! which the renderer reconciles into Bevy entities.

use bevy::prelude::*;

use super::tree::{
    slot_coords, NodeId, BRANCH_FACTOR, CHILDREN_PER_NODE, EMPTY_NODE,
    MAX_LAYER,
};
use super::view::{extent_for_layer, scale_for_layer, WorldAnchor};

// ----------------------------------------------------------- SmallPath

/// A compact identifier for a node's position in the tree during a
/// single-frame walk: `depth` significant slot indices from the root.
/// Used as the `RenderState.entities` key, so reuse survives across
/// frames as long as the camera keeps looking at the same spot.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
pub struct SmallPath {
    pub depth: u8,
    pub slots: [u8; MAX_LAYER as usize],
}

impl SmallPath {
    pub fn empty() -> Self {
        Self {
            depth: 0,
            slots: [0; MAX_LAYER as usize],
        }
    }

    pub fn push(&self, slot: u8) -> Self {
        let mut out = *self;
        out.slots[out.depth as usize] = slot;
        out.depth += 1;
        out
    }
}

// --------------------------------------------------------------- Visit

/// One "visit" the tree walk wants the reconciler to spawn/update.
/// `origin` is anchor-relative — already the final Bevy `Transform`
/// translation the renderer will give the entity.
pub struct Visit {
    pub path: SmallPath,
    pub node_id: NodeId,
    pub origin: Vec3,
    pub scale: f32,
}

/// One frame on the `walk()` DFS stack.
pub struct WalkFrame {
    pub node_id: NodeId,
    pub path: SmallPath,
    pub origin_leaves: [i64; 3],
    pub depth: u8,
}

// ---------------------------------------------------------------- walk

/// Walk the tree from root to `emit_layer`, collecting visible nodes.
///
/// Accumulates each node's absolute leaf-space origin as the walker
/// descends, then converts to a Bevy `Vec3` relative to the camera
/// anchor only when the node passes the cull / emit test. Tracking
/// the origin in `i64` keeps `f32` precision small even when the
/// player is billions of leaves deep inside the root.
pub fn walk(
    world: &super::state::WorldState,
    emit_layer: u8,
    target_layer: u8,
    camera_pos: Vec3,
    radius_bevy: f32,
    anchor: &WorldAnchor,
    stack: &mut Vec<WalkFrame>,
    out: &mut Vec<Visit>,
) {
    stack.clear();
    out.clear();
    if world.root == EMPTY_NODE {
        return;
    }

    let mut child_extent_leaves: [i64; MAX_LAYER as usize + 1] =
        [0; MAX_LAYER as usize + 1];
    {
        let mut ext: i64 = super::state::world_extent_voxels();
        child_extent_leaves[0] = ext;
        for layer in 1..=(MAX_LAYER as usize) {
            ext /= 5;
            child_extent_leaves[layer] = ext;
        }
    }

    stack.push(WalkFrame {
        node_id: world.root,
        path: SmallPath::empty(),
        origin_leaves: [0; 3],
        depth: 0,
    });

    let radius_sq = radius_bevy * radius_bevy;

    while let Some(frame) = stack.pop() {
        let WalkFrame { node_id, path, origin_leaves, depth } = frame;

        let origin_bevy = Vec3::new(
            (origin_leaves[0] - anchor.leaf_coord[0]) as f32,
            (origin_leaves[1] - anchor.leaf_coord[1]) as f32,
            (origin_leaves[2] - anchor.leaf_coord[2]) as f32,
        );
        let extent = extent_for_layer(depth);
        let aabb_min = origin_bevy;
        let aabb_max = origin_bevy + Vec3::splat(extent);

        let dx = (aabb_min.x - camera_pos.x)
            .max(0.0)
            .max(camera_pos.x - aabb_max.x);
        let dy = (aabb_min.y - camera_pos.y)
            .max(0.0)
            .max(camera_pos.y - aabb_max.y);
        let dz = (aabb_min.z - camera_pos.z)
            .max(0.0)
            .max(camera_pos.z - aabb_max.z);
        let min_dist_sq = dx * dx + dy * dy + dz * dz;
        if min_dist_sq > radius_sq {
            continue;
        }

        if depth == emit_layer {
            out.push(Visit {
                path,
                node_id,
                origin: origin_bevy,
                scale: scale_for_layer(target_layer),
            });
            continue;
        }

        let Some(node) = world.library.get(node_id) else { continue };
        let Some(children) = node.children.as_ref() else {
            out.push(Visit {
                path,
                node_id,
                origin: origin_bevy,
                scale: scale_for_layer(depth),
            });
            continue;
        };

        let child_extent_i64 = child_extent_leaves[(depth + 1) as usize];
        for slot in 0..CHILDREN_PER_NODE {
            let child_id = children[slot];
            if child_id == EMPTY_NODE {
                continue;
            }
            let (sx, sy, sz) = slot_coords(slot);
            let child_origin_leaves = [
                origin_leaves[0] + (sx as i64) * child_extent_i64,
                origin_leaves[1] + (sy as i64) * child_extent_i64,
                origin_leaves[2] + (sz as i64) * child_extent_i64,
            ];
            let child_path = path.push(slot as u8);
            stack.push(WalkFrame {
                node_id: child_id,
                path: child_path,
                origin_leaves: child_origin_leaves,
                depth: depth + 1,
            });
        }
    }
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::render::{CameraZoom, RADIUS_VIEW_CELLS, MIN_ZOOM, MAX_ZOOM};
    use crate::world::state::WorldState;
    use crate::world::tree::MAX_LAYER;
    use crate::world::view::{cell_size_at_layer, target_layer_for};

    fn anchor_origin() -> WorldAnchor {
        WorldAnchor { leaf_coord: [0, 0, 0] }
    }

    #[test]
    fn walk_grassland_at_leaves_emits_at_least_one_visit() {
        let world = WorldState::new_grassland();
        let mut stack = Vec::new();
        let mut visits = Vec::new();
        let target = target_layer_for(MAX_LAYER);
        let emit = target.saturating_sub(1);
        walk(
            &world, emit, target, Vec3::ZERO,
            RADIUS_VIEW_CELLS * cell_size_at_layer(MAX_LAYER),
            &anchor_origin(), &mut stack, &mut visits,
        );
        assert!(!visits.is_empty());
    }

    #[test]
    fn walk_radius_limits_emit_count() {
        let world = WorldState::new_grassland();
        let mut stack = Vec::new();
        let mut visits = Vec::new();
        let target = target_layer_for(MAX_LAYER);
        let emit = target.saturating_sub(1);
        walk(
            &world, emit, target, Vec3::ZERO,
            RADIUS_VIEW_CELLS * cell_size_at_layer(MAX_LAYER),
            &anchor_origin(), &mut stack, &mut visits,
        );
        assert!(visits.len() < 200_000, "walk emitted {} visits", visits.len());
    }

    #[test]
    fn walk_radius_scales_with_view_layer() {
        let world = WorldState::new_grassland();
        let anchor = anchor_origin();
        let mut stack = Vec::new();
        for view_layer in (MIN_ZOOM..=MAX_ZOOM).rev() {
            let target = target_layer_for(view_layer);
            let emit = target.saturating_sub(1);
            let radius = RADIUS_VIEW_CELLS * cell_size_at_layer(view_layer);
            let mut visits = Vec::new();
            walk(
                &world, emit, target, Vec3::ZERO, radius,
                &anchor, &mut stack, &mut visits,
            );
            assert!(visits.len() > 0, "view layer {view_layer}: 0 visits");
        }
    }

    #[test]
    fn small_path_push() {
        let p = SmallPath::empty().push(7).push(12);
        assert_eq!(p.depth, 2);
        assert_eq!(p.slots[0], 7);
        assert_eq!(p.slots[1], 12);
    }
}
