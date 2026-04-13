//! Runtime world state: the content-addressed voxel tree wrapped as
//! a Bevy resource.
//!
//! `WorldState` owns the root `NodeId` and the full `NodeLibrary`.
//! Every gameplay write goes through the edit walks in `edit.rs`,
//! which keep voxels and meshes consistent.
//!
//! The v1 world is an infinite grassland with a solid ground layer
//! that is [`GROUND_Y_VOXELS`] leaf voxels deep (`5^(MAX_LAYER - 1)`,
//! one layer-1 child extent) and air above it. The top face of the
//! grass sits at root-local leaf y = [`GROUND_Y_VOXELS`], and the
//! floating [`WorldAnchor`](super::view::WorldAnchor) maps that to
//! Bevy `y = 0` whenever the player is resting on it.

use bevy::prelude::*;

use super::generator::{
    generate_air_leaf, generate_grass_leaf,
    generate_sphere_leaf, aabb_inside_sphere, aabb_outside_sphere, SphereParams,
};
use super::tree::{
    downsample_from_library, filled_voxel_grid, slot_coords, uniform_children,
    voxel_from_block, Children, NodeId, NodeLibrary, BRANCH_FACTOR,
    CHILDREN_PER_NODE, EMPTY_NODE, MAX_LAYER, NODE_VOXELS_PER_AXIS,
};
use crate::block::BlockType;

/// Root-local y-offset of the ground surface, in leaf voxels. Every
/// leaf whose y-range in root-local coords is ≤ this value is solid
/// grass; every leaf whose y-range is ≥ this is empty air.
///
/// Set to `5^(MAX_LAYER - 1)` — one layer-1 child extent — so the
/// ground spans at least 5 view cells at every zoom level, including
/// [`MIN_ZOOM`](super::render::MIN_ZOOM). This guarantees the
/// layer-uniform UX principle: the outline, placement, and collision
/// feel identical at every view layer.
///
/// Thanks to content dedup, the deeper ground adds zero extra library
/// entries — the grassland bootstrap produces exactly 25 nodes
/// regardless of this value.
pub const GROUND_Y_VOXELS: i64 = {
    // 5^(MAX_LAYER - 1). Computed as a const because f64::powi isn't
    // available in const context.
    let mut n: i64 = 1;
    let mut i = 0;
    while i < MAX_LAYER - 1 {
        n *= BRANCH_FACTOR as i64;
        i += 1;
    }
    n
};

/// The depth in a [`Position`]'s path at which the grass/air
/// transition first appears. At this depth, the child extent equals
/// [`GROUND_Y_VOXELS`], so `sy = 1` is the first air slot.
///
/// For `GROUND_Y_VOXELS = 5^(MAX_LAYER - 1)` the transition depth
/// is 2 — depths 0 and 1 are still inside the "bottom" mixed
/// pattern, and depth 2 is where `sy = 0` is all-grass and
/// `sy >= 1` is all-air.
pub const GROUND_TRANSITION_DEPTH: usize = {
    // Find the shallowest depth d where
    // child_extent(d) = world_extent / 5^(d+1) <= GROUND_Y_VOXELS.
    // child_extent(d) = NODE_VOXELS_PER_AXIS * 5^(MAX_LAYER - d - 1).
    let mut d: usize = 0;
    loop {
        let mut extent: i64 = NODE_VOXELS_PER_AXIS as i64;
        let exp = MAX_LAYER as usize - d - 1;
        let mut j = 0;
        while j < exp {
            extent *= BRANCH_FACTOR as i64;
            j += 1;
        }
        if extent <= GROUND_Y_VOXELS {
            break d;
        }
        d += 1;
    }
};

// ---------------------------------------------------------- sphere parameters

/// Radius of the test sphere in leaf voxels.
pub const SPHERE_RADIUS: i64 = 500;

/// Depth from the sphere surface at which all voxels are Stone.
const STONE_DEPTH: i64 = 10;

/// Leaf-coordinate centre of the sphere (world centre on all axes).
pub fn sphere_center() -> [i64; 3] {
    let half = world_extent_voxels() / 2;
    [half, half, half]
}

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
        Self::new_sphere()
    }
}

impl WorldState {
    /// Build a sphere world from the default sphere parameters.
    pub fn new_sphere() -> Self {
        let mut state = Self {
            root: EMPTY_NODE,
            library: NodeLibrary::default(),
        };
        let params = SphereParams {
            center: sphere_center(),
            radius: SPHERE_RADIUS,
        };
        state.build_sphere_root(&params);
        state
    }

    /// Build the sphere tree. Pre-builds uniform "all air" and "all
    /// stone" subtree towers so the recursive builder can skip
    /// entirely empty (exterior) or deep-interior regions without
    /// recursing to leaves.
    pub fn build_sphere_root(&mut self, params: &SphereParams) -> NodeId {
        let air_leaf = self.library.insert_leaf(generate_air_leaf());
        let stone_leaf = self.library.insert_leaf(
            filled_voxel_grid(voxel_from_block(Some(BlockType::Stone))),
        );

        let layer_count = MAX_LAYER as usize + 1;
        let mut air_tower = vec![EMPTY_NODE; layer_count];
        let mut solid_tower = vec![EMPTY_NODE; layer_count];
        air_tower[MAX_LAYER as usize] = air_leaf;
        solid_tower[MAX_LAYER as usize] = stone_leaf;

        for k in (0..MAX_LAYER as usize).rev() {
            let air_ch = uniform_children(air_tower[k + 1]);
            let air_vox = downsample_from_library(&self.library, air_ch.as_ref());
            air_tower[k] = self.library.insert_non_leaf(air_vox, air_ch);

            let solid_ch = uniform_children(solid_tower[k + 1]);
            let solid_vox = downsample_from_library(&self.library, solid_ch.as_ref());
            solid_tower[k] = self.library.insert_non_leaf(solid_vox, solid_ch);
        }

        let extent = world_extent_voxels();
        let root_id = self.build_sphere_node(
            [0, 0, 0], extent, 0, params, &air_tower, &solid_tower,
        );
        self.swap_root(root_id);
        root_id
    }

    /// Recursive sphere builder. At each node, checks whether the
    /// AABB is entirely outside (air tower), entirely deep inside
    /// (solid tower), or intersects the surface (recurse / eval).
    fn build_sphere_node(
        &mut self,
        origin: [i64; 3],
        extent: i64,
        layer: u8,
        params: &SphereParams,
        air_tower: &[NodeId],
        solid_tower: &[NodeId],
    ) -> NodeId {
        if aabb_outside_sphere(origin, extent, params) {
            return air_tower[layer as usize];
        }
        if aabb_inside_sphere(
            origin, extent, params.center, params.radius - STONE_DEPTH,
        ) {
            return solid_tower[layer as usize];
        }
        if layer == MAX_LAYER {
            let grid = generate_sphere_leaf(origin, params);
            return self.library.insert_leaf(grid);
        }
        let child_extent = extent / BRANCH_FACTOR as i64;
        let mut child_ids = Vec::with_capacity(CHILDREN_PER_NODE);
        for slot in 0..CHILDREN_PER_NODE {
            let (sx, sy, sz) = slot_coords(slot);
            let child_origin = [
                origin[0] + sx as i64 * child_extent,
                origin[1] + sy as i64 * child_extent,
                origin[2] + sz as i64 * child_extent,
            ];
            child_ids.push(self.build_sphere_node(
                child_origin, child_extent, layer + 1, params,
                air_tower, solid_tower,
            ));
        }
        let children_arr: Children = child_ids
            .into_boxed_slice()
            .try_into()
            .unwrap_or_else(|_| unreachable!("size constant"));
        let voxels = downsample_from_library(&self.library, children_arr.as_ref());
        self.library.insert_non_leaf(voxels, children_arr)
    }

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

    /// Replace the world root with `new_root_id`, transferring the
    /// external ref in the order that keeps the library consistent
    /// even when the new root and the old root share descendants.
    ///
    /// The order matters: if the new root transiently held no external
    /// refs and we dec'd the old root first, a cascading eviction
    /// could free a node the new root was about to reference. Always
    /// `ref_inc` the new root first, then `ref_dec` the old root,
    /// then commit the pointer swap. Captures `self.root` before
    /// `ref_dec` so the pointer swap can actually occur before the
    /// decrement fires — this is functionally equivalent to the order
    /// described above (the value decremented is the old root either
    /// way) but keeps `self.root` pointing at a live node at every
    /// observable moment.
    ///
    /// No-op when `new_root_id == self.root` so callers can be lazy
    /// about checking. This also keeps round-trip edits (edit then
    /// undo) from uselessly ref-cycling the same id.
    pub fn swap_root(&mut self, new_root_id: NodeId) {
        if new_root_id == self.root {
            return;
        }
        self.library.ref_inc(new_root_id);
        let old_root = self.root;
        self.root = new_root_id;
        self.library.ref_dec(old_root);
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

        // `cur_bottom` is now the root. Hand off the external ref via
        // `swap_root`, which does the ref_inc-then-ref_dec dance in
        // the order that keeps the library consistent. On an idempotent
        // rebuild (dedup makes `cur_bottom` equal to the existing
        // root), `swap_root`'s no-op guard short-circuits and avoids
        // a pointless ref-cycle. On the very first build, `self.root`
        // is `EMPTY_NODE`, which `ref_dec` treats as a no-op.
        self.swap_root(cur_bottom);
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
    use crate::world::tree::{slot_index, voxel_from_block, voxel_idx, EMPTY_VOXEL, NODE_VOXELS_PER_AXIS};
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

    /// A path whose root y-slot is 1 lands in the "air above
    /// ground" region — the leaf should be all-empty.
    #[test]
    fn air_region_leaf_is_all_empty() {
        let world = WorldState::new_grassland();
        let mut id = world.root;
        let mut path = [0u8; MAX_LAYER as usize];
        // slot_index(0, 1, 0) = 5 at depth 0 (the root) pushes us
        // into the air region above the ground.
        path[0] = slot_index(0, 1, 0) as u8;
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

    // --------------------------------------------------------- sphere tests

    #[test]
    fn sphere_builds_root() {
        let world = WorldState::new_sphere();
        assert_ne!(world.root, EMPTY_NODE);
        assert!(world.library.get(world.root).is_some());
    }

    #[test]
    fn sphere_center_is_solid() {
        let world = WorldState::new_sphere();
        let center = sphere_center();
        let pos = crate::world::view::position_from_leaf_coord(center)
            .expect("center inside world");
        assert_ne!(
            crate::world::edit::get_voxel(&world, &pos),
            EMPTY_VOXEL,
            "sphere center should be solid"
        );
    }

    #[test]
    fn sphere_exterior_is_empty() {
        let world = WorldState::new_sphere();
        let center = sphere_center();
        let coord = [center[0], center[1] + SPHERE_RADIUS + 100, center[2]];
        let pos = crate::world::view::position_from_leaf_coord(coord)
            .expect("outside point inside world");
        assert_eq!(
            crate::world::edit::get_voxel(&world, &pos),
            EMPTY_VOXEL,
            "point outside sphere should be empty"
        );
    }

    #[test]
    fn sphere_library_is_compact() {
        let world = WorldState::new_sphere();
        assert!(
            world.library.len() < 20_000,
            "library has {} entries — dedup not working?",
            world.library.len()
        );
    }

    #[test]
    fn sphere_surface_visible_at_root() {
        let world = WorldState::new_sphere();
        let root = world.library.get(world.root).expect("root");
        let has_solid = root.voxels.iter().any(|&v| v != EMPTY_VOXEL);
        assert!(has_solid, "root voxel grid is all-empty — sphere not visible");
    }
}
