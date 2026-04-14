//! Runtime world state: the content-addressed tree.

use super::palette::block;
use super::sdf::{self, Planet, Vec3};
use super::tree::*;

pub struct WorldState {
    pub root: NodeId,
    pub library: NodeLibrary,
    /// Gravity sources placed in world-space [0, 3). Collision and
    /// camera orientation query `dominant_planet(pos)` to decide
    /// which (if any) planet's gravity applies at a given position.
    pub planets: Vec<Planet>,
}

impl WorldState {
    /// The planet whose influence sphere contains `pos` and whose
    /// center is closest; `None` if `pos` is in empty space.
    pub fn dominant_planet(&self, pos: Vec3) -> Option<&Planet> {
        let mut best: Option<(&Planet, f32)> = None;
        for p in &self.planets {
            let d = sdf::length(sdf::sub(pos, p.center));
            if d <= p.influence_radius {
                match best {
                    None => best = Some((p, d)),
                    Some((_, bd)) if d < bd => best = Some((p, d)),
                    _ => {}
                }
            }
        }
        best.map(|(p, _)| p)
    }

    /// Blended "up" direction at `pos`: contributions from each
    /// planet whose influence reaches `pos`, plus a residual toward
    /// world +Y. Weights taper smoothly from 1 at the planet surface
    /// to 0 at the influence boundary, so flying into / out of
    /// gravity rotates the horizon gradually instead of snapping.
    pub fn blended_up(&self, pos: Vec3) -> Vec3 {
        let mut accum = [0.0f32, 0.0, 0.0];
        let mut total_weight = 0.0f32;
        for p in &self.planets {
            let to = sdf::sub(pos, p.center);
            let r = sdf::length(to);
            if r >= p.influence_radius || r < 1e-8 { continue; }
            // t = 0 at surface, 1 at influence boundary.
            let t = if r <= p.radius {
                0.0
            } else {
                ((r - p.radius) / (p.influence_radius - p.radius).max(1e-6))
                    .clamp(0.0, 1.0)
            };
            // Smoothstep falloff: full pull inside, zero at boundary.
            let s = 1.0 - t;
            let w = s * s * (3.0 - 2.0 * s);
            accum = sdf::add(accum, sdf::scale(p.up_at(pos), w));
            total_weight += w;
        }
        let world_up = [0.0f32, 1.0, 0.0];
        let residual = (1.0 - total_weight).max(0.0);
        accum = sdf::add(accum, sdf::scale(world_up, residual));
        sdf::normalize(accum)
    }
}

impl WorldState {
    /// Build a 3-level test world with obvious visual features.
    ///
    /// Level 0 (finest): individual blocks.
    /// Level 1: 3x3x3 groups of blocks (27 blocks each).
    /// Level 2: 3x3x3 groups of level-1 nodes (9x9x9 blocks total per axis).
    /// Level 3 (root): 3x3x3 groups of level-2 nodes (27x27x27 blocks total).
    ///
    /// The root spans [0, 27) in each axis at block resolution.
    /// Ground at y<9 (bottom third), grass at y=9..18, air above.
    /// With a checkerboard pattern and some pillars for visual variety.
    pub fn test_world() -> Self {
        let mut library = NodeLibrary::default();

        // --- Level 1 nodes: 3x3x3 uniform block clusters ---
        let stone_l1 = library.insert(uniform_children(Child::Block(block::STONE)));
        let dirt_l1 = library.insert(uniform_children(Child::Block(block::DIRT)));
        let grass_l1 = library.insert(uniform_children(Child::Block(block::GRASS)));
        let wood_l1 = library.insert(uniform_children(Child::Block(block::WOOD)));
        let leaf_l1 = library.insert(uniform_children(Child::Block(block::LEAF)));
        let sand_l1 = library.insert(uniform_children(Child::Block(block::SAND)));
        let brick_l1 = library.insert(uniform_children(Child::Block(block::BRICK)));
        let air_l1 = library.insert(empty_children());

        // A mixed level-1 node: checkerboard of stone and dirt.
        let mut checker_children = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    let slot = slot_index(x, y, z);
                    if (x + y + z) % 2 == 0 {
                        checker_children[slot] = Child::Block(block::STONE);
                    } else {
                        checker_children[slot] = Child::Block(block::DIRT);
                    }
                }
            }
        }
        let checker_l1 = library.insert(checker_children);

        // --- Level 2 nodes: 3x3x3 of level-1 nodes ---

        // Solid ground layer (level 2): all stone.
        let stone_l2 = library.insert(uniform_children(Child::Node(stone_l1)));

        // Checkerboard ground (level 2): alternating stone and dirt groups.
        let mut checker_ground_children = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    let slot = slot_index(x, y, z);
                    if (x + z) % 2 == 0 {
                        checker_ground_children[slot] = Child::Node(stone_l1);
                    } else {
                        checker_ground_children[slot] = Child::Node(checker_l1);
                    }
                }
            }
        }
        let checker_ground_l2 = library.insert(checker_ground_children);

        // Grass surface layer (level 2): grass on top, dirt below.
        let mut grass_surface_children = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                let slot_bottom = slot_index(x, 0, z);
                let slot_mid = slot_index(x, 1, z);
                let slot_top = slot_index(x, 2, z);
                grass_surface_children[slot_bottom] = Child::Node(dirt_l1);
                grass_surface_children[slot_mid] = Child::Node(grass_l1);
                grass_surface_children[slot_top] = Child::Node(air_l1);
            }
        }
        let grass_surface_l2 = library.insert(grass_surface_children);

        // Air layer (level 2).
        let air_l2 = library.insert(uniform_children(Child::Node(air_l1)));

        // Feature layer (level 2): mostly air with some structures.
        let mut features_children = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                for y in 0..BRANCH {
                    features_children[slot_index(x, y, z)] = Child::Node(air_l1);
                }
            }
        }
        // A wood pillar at (1, *, 1).
        features_children[slot_index(1, 0, 1)] = Child::Node(wood_l1);
        features_children[slot_index(1, 1, 1)] = Child::Node(wood_l1);
        features_children[slot_index(1, 2, 1)] = Child::Node(leaf_l1);
        // A brick wall at x=0.
        features_children[slot_index(0, 0, 0)] = Child::Node(brick_l1);
        features_children[slot_index(0, 0, 1)] = Child::Node(brick_l1);
        features_children[slot_index(0, 0, 2)] = Child::Node(brick_l1);
        features_children[slot_index(0, 1, 0)] = Child::Node(brick_l1);
        features_children[slot_index(0, 1, 1)] = Child::Node(brick_l1);
        features_children[slot_index(0, 1, 2)] = Child::Node(brick_l1);
        // Sand pile at (2, 0, 0).
        features_children[slot_index(2, 0, 0)] = Child::Node(sand_l1);
        let features_l2 = library.insert(features_children);

        // --- Level 3 (root): 3x3x3 of level-2 nodes ---
        // World spans [0, 27) in block coords.
        // y=0: solid stone ground
        // y=1: grass surface (dirt + grass + air)
        // y=2: air with features in some spots
        let mut root_children = empty_children();
        for z in 0..BRANCH {
            for x in 0..BRANCH {
                // Bottom: stone or checkerboard.
                if (x + z) % 2 == 0 {
                    root_children[slot_index(x, 0, z)] = Child::Node(stone_l2);
                } else {
                    root_children[slot_index(x, 0, z)] = Child::Node(checker_ground_l2);
                }
                // Middle: grass surface.
                root_children[slot_index(x, 1, z)] = Child::Node(grass_surface_l2);
                // Top: air, with features at (1,2,1).
                if x == 1 && z == 1 {
                    root_children[slot_index(x, 2, z)] = Child::Node(features_l2);
                } else {
                    root_children[slot_index(x, 2, z)] = Child::Node(air_l2);
                }
            }
        }

        let root = library.insert(root_children);
        library.ref_inc(root);

        eprintln!(
            "Test world: {} library entries, root spans [0, 27) per axis",
            library.len(),
        );

        Self { root, library, planets: Vec::new() }
    }

    /// Compute the maximum depth of the tree from the root.
    /// Depth = number of Node→Node edges from root to the deepest
    /// terminal (Block/Empty) children.
    /// Memoized — each unique NodeId is visited at most once.
    pub fn tree_depth(&self) -> u32 {
        let mut cache = std::collections::HashMap::new();
        self.depth_of(self.root, &mut cache)
    }

    fn depth_of(&self, id: NodeId, cache: &mut std::collections::HashMap<NodeId, u32>) -> u32 {
        if let Some(&d) = cache.get(&id) { return d; }
        let Some(node) = self.library.get(id) else { return 0 };
        let mut max_child_depth = 0u32;
        for child in &node.children {
            if let Child::Node(child_id) = child {
                max_child_depth = max_child_depth.max(self.depth_of(*child_id, cache));
            }
        }
        let d = 1 + max_child_depth;
        cache.insert(id, d);
        d
    }

    pub fn swap_root(&mut self, new_root: NodeId) {
        if new_root == self.root { return; }
        self.library.ref_inc(new_root);
        let old = self.root;
        self.root = new_root;
        self.library.ref_dec(old);
    }
}
