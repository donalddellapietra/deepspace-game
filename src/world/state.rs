//! Runtime world state: the content-addressed voxel tree wrapped as
//! a Bevy resource.
//!
//! `WorldState` owns the root `NodeId` and the full `NodeLibrary`.
//! Every gameplay write goes through the edit walks in `edit.rs`,
//! which keep voxels and meshes consistent.
//!
//! The v1 world is an infinite grassland with a solid ground layer
//! that is [`GROUND_Y_VOXELS`] leaf voxels deep and air above it.
//! The top face of the grass sits at root-local leaf y =
//! [`GROUND_Y_VOXELS`], and the floating [`WorldAnchor`](super::view::WorldAnchor)
//! maps that to Bevy `y = 0` whenever the player is resting on it.

use bevy::prelude::*;

use super::generator::{generate_air_leaf, generate_grass_leaf};
use super::tree::{
    downsample_from_library, slot_coords, uniform_children, Children, NodeId,
    NodeLibrary, BRANCH_FACTOR, CHILDREN_PER_NODE, EMPTY_NODE, MAX_LAYER,
    NODE_VOXELS_PER_AXIS,
};

/// Root-local y-offset of the ground surface, in leaf voxels. Every
/// leaf whose y-range in root-local coords is ≤ this value is solid
/// grass; every leaf whose y-range is ≥ this is empty air.
///
/// At `125` = one layer-11 extent, the ground is `125` leaf voxels
/// (5 leaves) deep — enough for the player to dig down noticeably
/// before hitting the void at the bottom of the root.
pub const GROUND_Y_VOXELS: i64 = 125;

/// Full world extent along one axis, in leaf voxels.
/// `25 × 5^MAX_LAYER ≈ 6.1 billion` — overflows `i32`, lives in `i64`.
pub const fn world_extent_voxels() -> i64 {
    let mut n: i64 = 1;
    let mut i = 0;
    while i < MAX_LAYER as usize {
        n *= BRANCH_FACTOR as i64;
        i += 1;
    }
    n * (NODE_VOXELS_PER_AXIS as i64)
}

#[derive(Resource)]
pub struct WorldState {
    pub root: NodeId,
    pub library: NodeLibrary,
}

impl Default for WorldState {
    fn default() -> Self {
        Self::new_grassland()
    }
}

impl WorldState {
    /// Build a fresh grassland world with a ground layer. Content
    /// collapses to a small number of library entries thanks to
    /// dedup: two leaf patterns (grass and air), plus two non-leaf
    /// patterns per layer (a "bottom-row" pattern and an "all air"
    /// pattern) until the loop reaches the root.
    pub fn new_grassland() -> Self {
        let mut state = Self {
            root: EMPTY_NODE,
            library: NodeLibrary::default(),
        };
        state.build_grassland_root();
        state
    }

    /// (Re)build the root. Safe to call on an already-built world —
    /// dedup makes every insertion a library hit, so the world id
    /// is preserved.
    pub fn build_grassland_root(&mut self) -> NodeId {
        // Insert the two leaf patterns.
        let grass_leaf = self.library.insert_leaf(generate_grass_leaf());
        let air_leaf = self.library.insert_leaf(generate_air_leaf());

        // `cur_bottom` is the NodeId of the "pattern at root-local
        // y_min = 0" for the layer BELOW the one we're currently
        // constructing. `cur_air` is the "all air" pattern at the
        // same layer. We iterate from layer MAX_LAYER-1 up to layer 0,
        // and at each step we build new layer-K versions of both
        // patterns from the previous (layer-K+1) ones.
        let mut cur_bottom = grass_leaf;
        let mut cur_air = air_leaf;

        let extent_at_root: i64 = world_extent_voxels();

        for k in (0..MAX_LAYER).rev() {
            // Axis size of a layer-K node, in leaf voxels.
            // extent_at_root / 5^K
            let k_extent = layer_extent_voxels(k, extent_at_root);

            // Build the "bottom-row" pattern at layer K. If the
            // entire layer-K y-range fits inside the grass region,
            // the pattern is uniform (all children are `cur_bottom`).
            // Otherwise the layer-K straddles the ground and its
            // children are split by y-slot: `sy == 0` children use
            // `cur_bottom`, `sy >= 1` children use `cur_air`.
            let bot_children: Children = if k_extent <= GROUND_Y_VOXELS {
                uniform_children(cur_bottom)
            } else {
                mixed_bottom_children(cur_bottom, cur_air)
            };
            let bot_voxels = downsample_from_library(&self.library, bot_children.as_ref());
            let new_bottom = self.library.insert_non_leaf(bot_voxels, bot_children);

            // Build the "all air" pattern at layer K. Skip at layer 0
            // because nothing references it (the root is always a
            // "bottom-row" pattern).
            let new_air = if k > 0 {
                let air_children = uniform_children(cur_air);
                let air_voxels =
                    downsample_from_library(&self.library, air_children.as_ref());
                self.library.insert_non_leaf(air_voxels, air_children)
            } else {
                cur_air
            };

            cur_bottom = new_bottom;
            cur_air = new_air;
        }

        // `cur_bottom` is now the root. Transfer the external ref
        // from the previous root. Order matters: ref_inc new first,
        // then ref_dec old, so that a round-trip rebuild with the
        // same content doesn't evict-and-remint the root.
        self.library.ref_inc(cur_bottom);
        if self.root != EMPTY_NODE {
            self.library.ref_dec(self.root);
        }
        self.root = cur_bottom;
        self.root
    }
}

// -------------------------------------------------------------- helpers

/// Axis size of a layer-K node in leaf voxels. Layer 0 is the root
/// (full world extent), layer `MAX_LAYER` is a leaf (25).
fn layer_extent_voxels(layer: u8, root_extent: i64) -> i64 {
    let mut n = root_extent;
    for _ in 0..layer {
        n /= BRANCH_FACTOR as i64;
    }
    n
}

/// Build the children array for a "bottom-row" pattern at a layer
/// whose extent straddles the ground: children at `sy == 0` use the
/// supplied bottom id, children at `sy >= 1` use the air id.
fn mixed_bottom_children(bottom: NodeId, air: NodeId) -> Children {
    let v: Vec<NodeId> = (0..CHILDREN_PER_NODE)
        .map(|slot| {
            let (_sx, sy, _sz) = slot_coords(slot);
            if sy == 0 {
                bottom
            } else {
                air
            }
        })
        .collect();
    v.into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| unreachable!("size constant"))
}

// ----------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{voxel_from_block, voxel_idx, EMPTY_VOXEL, NODE_VOXELS_PER_AXIS};
    use crate::block::BlockType;

    fn grass() -> u8 {
        voxel_from_block(Some(BlockType::Grass))
    }

    #[test]
    fn new_grassland_builds_root() {
        let world = WorldState::new_grassland();
        assert_ne!(world.root, EMPTY_NODE);
        assert!(world.library.get(world.root).is_some());
    }

    /// Grass leaf + air leaf + 2 non-leaf patterns at every layer
    /// 1..=11 + 1 root = 25 entries.
    #[test]
    fn grassland_library_has_expected_entry_count() {
        let world = WorldState::new_grassland();
        let expected = 2 /* leaves */
            + 2 * ((MAX_LAYER as usize) - 1) /* layers 1..=MAX_LAYER-1 */
            + 1 /* root (layer 0 bottom-row only) */;
        assert_eq!(world.library.len(), expected);
    }

    #[test]
    fn rebuilding_is_idempotent() {
        let mut world = WorldState::new_grassland();
        let r0 = world.root;
        let l0 = world.library.len();
        world.build_grassland_root();
        assert_eq!(world.root, r0);
        assert_eq!(world.library.len(), l0);
    }

    #[test]
    fn root_has_external_ref() {
        let world = WorldState::new_grassland();
        assert!(world.library.get(world.root).unwrap().ref_count >= 1);
    }

    /// The "all-zero path" leaf sits at root-local `y in (0, 25)`
    /// — entirely below ground — so it should be all-grass.
    #[test]
    fn all_zero_leaf_is_all_grass() {
        let world = WorldState::new_grassland();
        let grass = grass();
        // Walk down the zero path and inspect the leaf.
        let mut id = world.root;
        for _ in 0..MAX_LAYER {
            let node = world.library.get(id).expect("node");
            let children = node.children.as_ref().expect("non-leaf");
            id = children[0];
        }
        let leaf = world.library.get(id).expect("leaf");
        for x in 0..NODE_VOXELS_PER_AXIS {
            for y in 0..NODE_VOXELS_PER_AXIS {
                for z in 0..NODE_VOXELS_PER_AXIS {
                    assert_eq!(leaf.voxels[voxel_idx(x, y, z)], grass);
                }
            }
        }
    }

    /// A path whose layer-10 y-slot is 1 lands in the "air above
    /// ground" region — the leaf should be all-empty.
    #[test]
    fn air_region_leaf_is_all_empty() {
        let world = WorldState::new_grassland();
        let mut id = world.root;
        let mut path = [0u8; MAX_LAYER as usize];
        // slot_index(0, 1, 0) = 1*5 = 5 at depth 10 pushes us into
        // the air layer above ground.
        path[10] = 5;
        for depth in 0..MAX_LAYER as usize {
            let node = world.library.get(id).expect("node");
            let children = node.children.as_ref().expect("non-leaf");
            id = children[path[depth] as usize];
        }
        let leaf = world.library.get(id).expect("leaf");
        for x in 0..NODE_VOXELS_PER_AXIS {
            for y in 0..NODE_VOXELS_PER_AXIS {
                for z in 0..NODE_VOXELS_PER_AXIS {
                    assert_eq!(leaf.voxels[voxel_idx(x, y, z)], EMPTY_VOXEL);
                }
            }
        }
    }
}
