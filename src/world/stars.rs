//! Stars-world preset: a planet cube you stand on + distant star
//! cubes scattered across the sky.
//!
//! # Geometry
//!
//! The camera sits at world position `(1.5, 1.5, 1.5)` — the exact
//! center of the root cell. `1.5` has the infinite ternary expansion
//! `0.1̄` (repeating 1), so `deepened_to(20)` gives the camera
//! anchor path `(1,1,1)²⁰` with a stable sub-cell offset at every
//! depth.
//!
//! The planet is a `Child::Block(GRASS)` at path
//! `[(1,1,1)¹⁹, (1,0,1)]` — the sibling cell immediately below the
//! camera's depth-20 anchor cell. Its top face sits at
//! `y = 1.5 − 0.5 · 3⁻¹⁹`, microscopically below the camera.
//! Rays going `-Y` pop one level to the depth-19 parent frame, then
//! hit the planet block at local sub-slot `(1,0,1)`.
//!
//! Stars live in the 26 non-center root slots. Each star is a small
//! cube at some depth 3..6 within its root slot, placed so it sits
//! at a distinct angular position in the camera's field. Sizes
//! range `1/9 .. 1/729` — roughly `0.07 rad .. 0.001 rad` angular,
//! so some stars read as small bright cubes and others as pinpoint
//! dots. The sky gradient shows through every gap between stars.
//!
//! # Precision
//!
//! At `anchor_depth = 20`, ribbon pops to root take 19 hops. Every
//! star at a root slot requires the full 19 pops to reach, so this
//! world exercises the same deep-ribbon precision path the d21
//! descent test covers — with explicit star occupants instead of an
//! empty-sky aperture.

use crate::world::anchor::WorldPos;
#[cfg(test)]
use crate::world::anchor::Path;
use crate::world::bootstrap::WorldBootstrap;
use crate::world::palette::{block, ColorRegistry};
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, Child, NodeId, NodeLibrary, MAX_DEPTH,
};
use std::collections::HashMap;

/// Star color — warm white, distinctly different from the sky
/// gradient (shades of blue) and from grass (green-dominant).
const STAR_RGB: (u8, u8, u8) = (255, 240, 200);

/// A single star specification.
///
/// The star Block sits at the leaf of the path
/// `[root_slot, sub_slot, (1,1,1)^(depth - 2)]`, producing a cell
/// of size `3^(1 - depth)` at a deterministic angular position
/// within the root slot.
///
/// - `root_slot`: one of the 26 non-`(1,1,1)` root children. Picks
///   which direction the star is in from the camera.
/// - `sub_slot`: one of the 27 children at depth 2. Picks the
///   specific angular position within the root slot — different
///   `sub_slot` values put the star at non-overlapping dots.
/// - `depth`: total tree depth of the Block. Larger = smaller star.
struct StarSpec {
    root_slot: (u8, u8, u8),
    sub_slot: (u8, u8, u8),
    depth: u8,
}

const STARS: &[StarSpec] = &[
    // +Y hemisphere — visible when looking up. Mix of sizes at
    // different sub-positions within the (1,2,1) and adjacent
    // "sky" root slots.
    StarSpec { root_slot: (1, 2, 1), sub_slot: (1, 1, 1), depth: 3 }, // bright central, 1/9
    StarSpec { root_slot: (1, 2, 1), sub_slot: (0, 0, 2), depth: 4 }, // medium, 1/27
    StarSpec { root_slot: (1, 2, 1), sub_slot: (2, 2, 0), depth: 4 },
    StarSpec { root_slot: (1, 2, 1), sub_slot: (0, 2, 0), depth: 5 }, // small, 1/81
    StarSpec { root_slot: (1, 2, 1), sub_slot: (2, 0, 2), depth: 5 },
    StarSpec { root_slot: (2, 2, 1), sub_slot: (0, 1, 1), depth: 4 }, // +X sky
    StarSpec { root_slot: (2, 2, 1), sub_slot: (2, 0, 0), depth: 5 },
    StarSpec { root_slot: (0, 2, 1), sub_slot: (2, 1, 1), depth: 4 }, // -X sky
    StarSpec { root_slot: (0, 2, 1), sub_slot: (0, 2, 2), depth: 6 }, // tiny, 1/243
    StarSpec { root_slot: (1, 2, 2), sub_slot: (1, 0, 0), depth: 4 }, // +Z sky
    StarSpec { root_slot: (1, 2, 2), sub_slot: (1, 2, 2), depth: 5 },
    StarSpec { root_slot: (1, 2, 0), sub_slot: (1, 0, 2), depth: 4 }, // -Z sky
    StarSpec { root_slot: (1, 2, 0), sub_slot: (1, 2, 0), depth: 5 },
    StarSpec { root_slot: (2, 2, 2), sub_slot: (0, 0, 0), depth: 5 }, // corner stars
    StarSpec { root_slot: (0, 2, 0), sub_slot: (2, 0, 2), depth: 5 },
    StarSpec { root_slot: (2, 2, 0), sub_slot: (0, 0, 2), depth: 5 },
    StarSpec { root_slot: (0, 2, 2), sub_slot: (2, 0, 0), depth: 5 },
    // Horizon stars — cardinal root slots at y=1.
    StarSpec { root_slot: (2, 1, 1), sub_slot: (0, 2, 1), depth: 4 }, // east
    StarSpec { root_slot: (2, 1, 1), sub_slot: (2, 0, 2), depth: 5 },
    StarSpec { root_slot: (0, 1, 1), sub_slot: (2, 2, 1), depth: 4 }, // west
    StarSpec { root_slot: (0, 1, 1), sub_slot: (0, 0, 0), depth: 5 },
    StarSpec { root_slot: (1, 1, 2), sub_slot: (1, 2, 0), depth: 4 }, // front
    StarSpec { root_slot: (1, 1, 2), sub_slot: (2, 0, 2), depth: 5 },
    StarSpec { root_slot: (1, 1, 0), sub_slot: (1, 2, 2), depth: 4 }, // back
    StarSpec { root_slot: (1, 1, 0), sub_slot: (0, 0, 0), depth: 5 },
];

/// Default spawn depth — camera anchor sits at `(1,1,1)²⁰`.
pub const DEFAULT_STARS_SPAWN_DEPTH: u8 = 20;

/// Build a "single Block at (1,1,1)^n" subtree. The returned node
/// is a tree of depth `n` whose only non-empty cell is a
/// `Child::Block(block_id)` at the deepest center. All levels
/// above are Node chains with only the `(1,1,1)` slot populated.
fn single_block_subtree(lib: &mut NodeLibrary, n: u8, block_id: u16) -> NodeId {
    debug_assert!(n >= 1);
    let mut leaf = empty_children();
    leaf[slot_index(1, 1, 1)] = Child::Block(block_id);
    let mut current = lib.insert(leaf);
    for _ in 1..n {
        let mut c = empty_children();
        c[slot_index(1, 1, 1)] = Child::Node(current);
        current = lib.insert(c);
    }
    current
}

/// Build the `stars-world` preset. `total_depth` sets tree depth;
/// default 40. The camera anchor lives at
/// `(1,1,1)^DEFAULT_STARS_SPAWN_DEPTH`, regardless of `total_depth`,
/// so spawn-depth ≥ 20 scenarios render the camera inside a tiny
/// depth-20 cell with the planet one cell below.
pub fn bootstrap_stars_world(total_depth: u8) -> WorldBootstrap {
    assert!(total_depth >= 20, "stars world needs total_depth ≥ 20 for deep ribbons");
    assert!(
        (total_depth as usize) <= MAX_DEPTH,
        "total_depth {} exceeds MAX_DEPTH {}",
        total_depth, MAX_DEPTH,
    );

    let mut lib = NodeLibrary::default();
    let mut registry = ColorRegistry::new();
    let star_block = registry
        .register(STAR_RGB.0, STAR_RGB.1, STAR_RGB.2, 255)
        .expect("palette has room for star color");
    let planet_block = block::GRASS;

    // Camera's (1,1,1)^k Node chain from depth `total_depth - 1`
    // up to depth 2 (root's immediate child of (1,1,1)). Each
    // intermediate level has (1,1,1) → deeper Node, all other
    // slots empty. At depth 2 (the root-level (1,1,1) child), the
    // (1,0,1) slot carries the planet Block.
    //
    // Bottom-up: start with a deepest-level all-empty leaf, wrap
    // in Node layers.
    let mut camera_column = lib.insert(empty_children());
    for k in (2..total_depth).rev() {
        let mut children = empty_children();
        children[slot_index(1, 1, 1)] = Child::Node(camera_column);
        // Planet: immediately below the camera's depth-20
        // ancestor. At k = 2 we're building the root's direct
        // (1,1,1) child; its (1,0,1) sub-slot is the planet.
        //
        // This puts the planet at a depth-2 cell, size 1/3, top
        // face at y = 4/3. Big enough to see as a clear ground
        // surface. The camera at y = 1.5 stands 1/6 above it
        // (within the (1,1,1) sibling cell of (1,0,1) at depth 2).
        if k == 2 {
            children[slot_index(1, 0, 1)] = Child::Block(planet_block);
        }
        camera_column = lib.insert(children);
    }

    // Root: (1,1,1) = camera_column, stars scatter across the
    // other 26 slots. Multiple stars in the same root slot would
    // need to merge into one subtree; keep it simple — one star
    // per root slot per StarSpec. Overlap on `root_slot` means
    // only the last spec for that slot wins.
    let mut root_children = empty_children();
    root_children[slot_index(1, 1, 1)] = Child::Node(camera_column);
    // Group specs by root_slot. If multiple stars share a root
    // slot, merge their sub_slot placements into one subtree.
    let mut by_slot: HashMap<(u8, u8, u8), Vec<&StarSpec>> = HashMap::new();
    for s in STARS {
        by_slot.entry(s.root_slot).or_default().push(s);
    }
    for (root_slot, specs) in by_slot {
        let node = build_root_slot_with_multiple_stars(&mut lib, &specs, star_block);
        root_children[slot_index(
            root_slot.0 as usize,
            root_slot.1 as usize,
            root_slot.2 as usize,
        )] = Child::Node(node);
    }
    let root = lib.insert(root_children);
    lib.ref_inc(root);
    let world = WorldState { root, library: lib };

    // Camera at world center (1.5, 1.5, 1.5). Anchor depth = 20
    // by default; tree depth is 40 so there's headroom for
    // further zoom-in. `uniform_column` constructs the
    // `(center_slot)^depth` anchor directly — see its docs for
    // the f32-drift trap it sidesteps.
    let spawn_pos = WorldPos::uniform_column(
        slot_index(1, 1, 1) as u8,
        DEFAULT_STARS_SPAWN_DEPTH,
        [0.5, 0.5, 0.5],
    );

    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: 0.0,
        default_spawn_pitch: 0.0, // horizon
        // 0 = "not a plain world" — skips `plain_surface_spawn`
        // respawning and `carve_air_pocket` tree mutation when
        // the caller passes `--spawn-depth`.
        plain_layers: 0,
        color_registry: registry,
    }
}

/// Build a root-slot node with possibly multiple stars at different
/// `sub_slot` positions. Each sub_slot chain is built independently,
/// then merged into a single depth-2 Children array.
///
/// Mixing sub-slot chains of different `depth` values is safe:
/// each one is a `single_block_subtree` at its own remaining depth,
/// inserted at its `sub_slot` index of the root-slot node. Child
/// entries at un-used sub_slots stay `Child::Empty`.
fn build_root_slot_with_multiple_stars(
    lib: &mut NodeLibrary,
    specs: &[&StarSpec],
    star_block: u16,
) -> NodeId {
    let mut sub_children = empty_children();
    for spec in specs {
        debug_assert!(spec.depth >= 3, "star depth must be ≥ 3");
        let remaining = spec.depth - 2; // levels below the sub_slot Child
        let idx = slot_index(
            spec.sub_slot.0 as usize,
            spec.sub_slot.1 as usize,
            spec.sub_slot.2 as usize,
        );
        sub_children[idx] = if remaining == 0 {
            Child::Block(star_block)
        } else {
            Child::Node(single_block_subtree(lib, remaining, star_block))
        };
    }
    lib.insert(sub_children)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stars_world_has_expected_depth() {
        let b = bootstrap_stars_world(40);
        assert_eq!(b.world.tree_depth(), 40);
    }

    #[test]
    fn stars_world_spawn_is_world_center() {
        let b = bootstrap_stars_world(40);
        let xyz = b.default_spawn_pos.in_frame(&Path::root());
        assert!((xyz[0] - 1.5).abs() < 1e-6);
        assert!((xyz[1] - 1.5).abs() < 1e-6);
        assert!((xyz[2] - 1.5).abs() < 1e-6);
        assert_eq!(b.default_spawn_pos.anchor.depth(), DEFAULT_STARS_SPAWN_DEPTH);
    }

    #[test]
    fn stars_world_spawn_anchor_is_all_center() {
        // (1.5, 1.5, 1.5) deepens down the (1,1,1)^k column —
        // verify every slot is the center (13).
        let b = bootstrap_stars_world(40);
        let slots = b.default_spawn_pos.anchor.as_slice();
        for (i, &s) in slots.iter().enumerate() {
            assert_eq!(s, 13, "slot {i} should be center (13), got {s}");
        }
    }
}
