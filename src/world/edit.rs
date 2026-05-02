//! Block editing: break / place / install subtree.
//!
//! Edits flow HitInfo → `propagate_edit`, which rebuilds ancestors
//! clone-on-write from the hit depth back to a fresh root.

use super::anchor::Path;
use super::raycast::HitInfo;
use super::state::WorldState;
use super::tree::{
    empty_children, slot_coords, slot_index, Child, NodeId, NodeKind, NodeLibrary, EMPTY_NODE,
};

/// Break (remove) the block at the hit location.
pub fn break_block(world: &mut WorldState, hit: &HitInfo) -> bool {
    propagate_edit(world, hit, Child::Empty)
}

/// Place any child (block or subtree) adjacent to the hit face.
/// For blocks, use `Child::Block(idx)`. For saved meshes, use
/// `Child::Node(saved_node_id)`.
pub fn place_child(world: &mut WorldState, hit: &HitInfo, new_child: Child) -> bool {
    // Sphere hits carry an explicit placement path (the last empty
    // cell the ray traversed before hitting the block). Place there
    // directly — the face→xyz-delta derivation below is nonsense for
    // face-subtree (u,v,r) slots.
    if let Some(ref place_path) = hit.place_path {
        let place_hit = HitInfo {
            path: place_path.clone(),
            face: hit.face,
            t: hit.t,
            place_path: None,
        };
        return propagate_edit(world, &place_hit, new_child);
    }
    let (_parent_id, slot) = *hit.path.last().unwrap();
    let (x, y, z) = slot_coords(slot);
    let (dx, dy, dz): (i32, i32, i32) = match hit.face {
        0 => (1, 0, 0),
        1 => (-1, 0, 0),
        2 => (0, 1, 0),
        3 => (0, -1, 0),
        4 => (0, 0, 1),
        5 => (0, 0, -1),
        _ => return false,
    };

    let nx = x as i32 + dx;
    let ny = y as i32 + dy;
    let nz = z as i32 + dz;

    // In-node: neighbor is within the same 2x2x2 node.
    if (0..=1).contains(&nx) && (0..=1).contains(&ny) && (0..=1).contains(&nz) {
        let adj_slot = slot_index(nx as usize, ny as usize, nz as usize);
        let parent_id = hit.path.last().unwrap().0;

        let node = match world.library.get(parent_id) {
            Some(n) => n,
            None => return false,
        };

        if !is_placeable(&world.library, node.children[adj_slot]) {
            return false;
        }

        let mut place_hit = hit.clone();
        place_hit.path.last_mut().unwrap().1 = adj_slot;
        return propagate_edit(world, &place_hit, new_child);
    }

    // Cross-node: use pure slot/path arithmetic — no f32 coordinates.
    // Build a Path from the hit's slot indices, step it in the face
    // normal direction (with carry/borrow propagation), then walk the
    // tree along the new path.
    let (axis, direction) = match hit.face {
        0 => (0usize, 1i32),
        1 => (0, -1),
        2 => (1, 1),
        3 => (1, -1),
        4 => (2, 1),
        5 => (2, -1),
        _ => return false,
    };
    let mut target_path = Path::root();
    for &(_, slot) in &hit.path {
        target_path.push(slot as u8);
    }
    target_path.step_neighbor_cartesian(axis, direction);

    place_child_at_path(world, &target_path, new_child)
}

/// Place a block adjacent to the hit face. Builds a uniform subtree
/// that matches the depth of siblings at the placement site, so the
/// placed block has full recursive structure like the terrain around it.
pub fn place_block(world: &mut WorldState, hit: &HitInfo, block_type: u16) -> bool {
    // Figure out how deep siblings are at the placement site.
    let sibling_depth = if let Some(&(parent_id, _)) = hit.path.last() {
        if let Some(parent) = world.library.get(parent_id) {
            let mut max_d = 0u32;
            for child in &parent.children {
                if let Child::Node(nid) = child {
                    let d = depth_of_node(&world.library, *nid);
                    if d > max_d { max_d = d; }
                }
            }
            max_d
        } else {
            0
        }
    } else {
        0
    };

    let child = world.library.build_uniform_subtree(block_type, sibling_depth);
    place_child(world, hit, child)
}

/// Install a subtree (NodeId) at a given path from root. Used for
/// placing saved meshes / imported models.
pub fn install_subtree(world: &mut WorldState, ancestor_slots: &[usize], new_node_id: NodeId) {
    if ancestor_slots.is_empty() { return; }

    // Phase 1: Descent — record (parent_id, slot) pairs.
    let mut descent: Vec<(NodeId, usize)> = Vec::with_capacity(ancestor_slots.len());
    let mut current_id = world.root;

    for &slot in ancestor_slots {
        descent.push((current_id, slot));
        let Some(node) = world.library.get(current_id) else { return };
        match node.children[slot] {
            Child::Node(child_id) => current_id = child_id,
            _ => break,
        }
    }

    // Phase 2: Ascent — walk back up, cloning children arrays.
    // Preserve each ancestor's NodeKind (see `propagate_edit`).
    let mut child = Child::Node(new_node_id);
    for &(parent_id, slot) in descent.iter().rev() {
        let Some(node) = world.library.get(parent_id) else { return };
        let original_kind = node.kind;
        let mut new_children = node.children;
        new_children[slot] = child;
        child = Child::Node(world.library.insert_with_kind(new_children, original_kind));
    }

    if let Child::Node(new_root) = child {
        world.swap_root(new_root);
    }
}

/// Compute depth of a single node (non-memoized, but uniform nodes are O(1)).
fn depth_of_node(library: &NodeLibrary, id: NodeId) -> u32 {
    let Some(node) = library.get(id) else { return 0 };
    let first_child = node.children[0];
    match first_child {
        Child::Node(child_id) => 1 + depth_of_node(library, child_id),
        _ => 1,
    }
}

/// Place a block at the position identified by a slot path. Walks the
/// tree along the path, materializing empty intermediate nodes as
/// needed. Uses pure integer slot arithmetic — no f32 coordinates —
/// so it works at all 63 layers without precision loss.
fn place_child_at_path(
    world: &mut WorldState,
    target_path: &Path,
    new_child: Child,
) -> bool {
    let target_depth = target_path.depth() as usize;
    if target_depth == 0 {
        return false;
    }
    let slots = target_path.as_slice();

    let mut path: Vec<(NodeId, usize)> = Vec::with_capacity(target_depth);
    let mut current_id = world.root;

    for level in 0..target_depth {
        let slot = slots[level] as usize;
        path.push((current_id, slot));

        let node = match world.library.get(current_id) {
            Some(n) => n,
            None => return false,
        };

        let is_last = level == target_depth - 1;

        match node.children[slot] {
            Child::Node(child_id) if !is_last => {
                current_id = child_id;
            }
            child if is_last && is_placeable(&world.library, child) => {
                let place_hit = HitInfo { path, face: 0, t: 0.0, place_path: None };
                return propagate_edit(world, &place_hit, new_child);
            }
            child if !is_last && is_placeable(&world.library, child) => {
                let remaining_slots = &slots[level + 1..target_depth];
                let chain_id = build_placement_chain_from_slots(
                    world, remaining_slots, new_child,
                );
                let place_hit = HitInfo { path, face: 0, t: 0.0, place_path: None };
                return propagate_edit(world, &place_hit, Child::Node(chain_id));
            }
            _ => {
                return false;
            }
        }
    }

    false
}

/// Build a chain of empty nodes along `slots`, placing `leaf_child`
/// at the deepest level. Pure slot arithmetic, no f32.
fn build_placement_chain_from_slots(
    world: &mut WorldState,
    slots: &[u8],
    leaf_child: Child,
) -> NodeId {
    let mut child = leaf_child;
    for &slot in slots.iter().rev() {
        let mut children = empty_children();
        children[slot as usize] = child;
        let id = world.library.insert(children);
        child = Child::Node(id);
    }

    match child {
        Child::Node(id) => id,
        _ => unreachable!("remaining > 0 guarantees at least one wrapping node"),
    }
}

/// Apply an edit and propagate clone-on-write up to root.
fn propagate_edit(world: &mut WorldState, hit: &HitInfo, new_child: Child) -> bool {
    if hit.path.is_empty() {
        return false;
    }

    let mut replacement: Option<NodeId> = None;

    for i in (0..hit.path.len()).rev() {
        let (node_id, slot) = hit.path[i];
        let (children_template, original_kind) = if node_id == EMPTY_NODE {
            (empty_children(), NodeKind::Cartesian)
        } else {
            let node = match world.library.get(node_id) {
                Some(n) => n,
                None => return false,
            };
            (node.children, node.kind)
        };

        let mut new_children = children_template;
        if let Some(nid) = replacement {
            new_children[slot] = Child::Node(nid);
        } else {
            new_children[slot] = new_child;
        }

        replacement = Some(world.library.insert_with_kind(new_children, original_kind));
    }

    if let Some(new_root) = replacement {
        world.swap_root(new_root);
        true
    } else {
        false
    }
}

/// A cell is placeable if it's Empty or an all-empty Node subtree
/// (`representative_block == REPRESENTATIVE_EMPTY`). At coarser zoom
/// levels, air regions are represented as Node subtrees rather than
/// Child::Empty.
fn is_placeable(library: &NodeLibrary, child: Child) -> bool {
    use crate::world::tree::REPRESENTATIVE_EMPTY;
    match child {
        Child::Empty => true,
        Child::Node(id) => library
            .get(id)
            .map_or(false, |n| n.representative_block == REPRESENTATIVE_EMPTY),
        Child::Block(_) => false,
        // Entity cells aren't placeable — something is already there
        // (the entity). Players edit entities via a different path.
        Child::EntityRef(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::aabb::hit_aabb;
    use crate::world::bootstrap::plain_test_world;
    use crate::world::palette::block;
    use crate::world::raycast::{cpu_raycast, is_solid_at};

    #[test]
    fn break_block_modifies_world() {
        let mut world = plain_test_world();
        let old_root = world.root;
        let hit = cpu_raycast(
            &world.library, world.root,
            [1.5, 2.5, 1.5], [0.0, -1.0, 0.0], 8,
        ).unwrap();
        assert!(break_block(&mut world, &hit));
        assert_ne!(world.root, old_root, "Root should change after edit");
    }

    #[test]
    fn place_block_on_ground() {
        let mut world = plain_test_world();
        let hit = cpu_raycast(
            &world.library, world.root,
            [1.5, 2.5, 1.5], [0.0, -1.0, 0.0], 8,
        ).unwrap();
        assert!(break_block(&mut world, &hit));

        let hit2 = cpu_raycast(
            &world.library, world.root,
            [1.5, 2.5, 1.5], [0.0, -1.0, 0.0], 8,
        ).unwrap();
        let old_root = world.root;
        assert!(place_block(&mut world, &hit2, block::BRICK));
        assert_ne!(world.root, old_root);
    }

    #[test]
    fn cross_node_placement_upward() {
        let mut world = plain_test_world();
        let hit = cpu_raycast(
            &world.library, world.root,
            [1.5, 2.5, 1.5], [0.0, -1.0, 0.0], 2,
        );
        assert!(hit.is_some());
        let hit = hit.unwrap();
        let (_, slot) = *hit.path.last().unwrap();
        let (_x, y, _z) = slot_coords(slot);
        if y == 2 {
            let old_root = world.root;
            assert!(place_block(&mut world, &hit, block::BRICK));
            assert_ne!(world.root, old_root);
        }
    }

    #[test]
    fn cross_node_placement_into_empty_subtree() {
        let mut world = plain_test_world();
        let hit = cpu_raycast(
            &world.library, world.root,
            [0.5, 2.5, 0.5], [0.0, -1.0, 0.0], 3,
        );
        assert!(hit.is_some(), "Should hit ground");
        let hit = hit.unwrap();
        let (aabb_min, aabb_max) = hit_aabb(&world.library, &hit);
        let cell_size = aabb_max[0] - aabb_min[0];

        let target_center_y = (aabb_min[1] + aabb_max[1]) * 0.5 + cell_size;
        if target_center_y < 2.0 {
            let old_root = world.root;
            let placed = place_block(&mut world, &hit, block::BRICK);
            assert!(placed, "Should place into empty subtree");
            assert_ne!(world.root, old_root);

            let target = [
                (aabb_min[0] + aabb_max[0]) * 0.5,
                target_center_y,
                (aabb_min[2] + aabb_max[2]) * 0.5,
            ];
            assert!(is_solid_at(&world.library, world.root, target, 8),
                "Placed block should be solid at target");
        }
    }

    #[test]
    fn placement_outside_world_returns_false() {
        let mut world = plain_test_world();
        let hit = HitInfo {
            path: vec![(world.root, slot_index(1, 1, 1))],
            face: 2, t: 1.0, place_path: None,
        };
        assert!(!place_block(&mut world, &hit, block::BRICK));
    }

    /// Cross-node placement must work identically at shallow and deep
    /// layers. The old f32-based `place_child_at_point` lost precision
    /// past depth ~23; the new path-based `place_child_at_path` uses
    /// pure slot arithmetic and works at all 63 layers.
    #[test]
    fn cross_node_placement_works_at_depth_40() {
        let depth: usize = 40;
        let mut lib = NodeLibrary::default();
        let solid = lib.build_uniform_subtree(block::BRICK, depth as u32);

        let mut root_children = empty_children();
        root_children[slot_index(0, 1, 1)] = solid;
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        let mut world = WorldState { root, library: lib };

        let mut path = Vec::new();
        let mut current = root;
        for level in 0..depth {
            let slot = if level == 0 {
                slot_index(0, 1, 1)
            } else {
                slot_index(1, 1, 1)
            };
            path.push((current, slot));
            if let Some(node) = world.library.get(current) {
                if let Child::Node(child_id) = node.children[slot] {
                    current = child_id;
                }
            }
        }

        let hit = HitInfo {
            path: path.clone(),
            face: 0,
            t: 1.0,
            place_path: None,
        };

        let result = place_block(&mut world, &hit, block::STONE);
        assert!(result, "Cross-node placement at depth {} must succeed", depth);

        let mut verify_path = Path::root();
        for &(_, slot) in &path {
            verify_path.push(slot as u8);
        }
        verify_path.step_neighbor_cartesian(0, 1);

        assert_eq!(
            verify_path.as_slice()[0],
            slot_index(1, 1, 1) as u8,
            "root slot must cross from x=0 to x=1",
        );
        for d in 1..depth {
            assert_eq!(
                verify_path.as_slice()[d],
                slot_index(0, 1, 1) as u8,
                "depth {} must wrap x=1 to x=0", d + 1,
            );
        }

        let placed_slots = verify_path.as_slice();
        let mut node_id = world.root;
        for (i, &slot) in placed_slots.iter().enumerate() {
            let node = world.library.get(node_id).expect("node must exist");
            let child = node.children[slot as usize];
            if i == placed_slots.len() - 1 {
                assert!(!matches!(child, Child::Empty),
                    "Placed block at depth {} must not be empty (slot {})",
                    i + 1, slot);
            } else if let Child::Node(next) = child {
                node_id = next;
            } else {
                panic!("Expected Node at depth {} slot {}, got {:?}", i + 1, slot, child);
            }
        }
    }
}
