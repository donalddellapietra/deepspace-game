//! Stars-world preset: a planet cube you stand on + distant star
//! cubes at varying ribbon depths.
//!
//! Exists to validate that the renderer preserves precision across
//! arbitrarily many ribbon pops — i.e. that distant occupants (stars)
//! remain visible regardless of how deep the camera's anchor sits.
//!
//! # Layout
//!
//! The camera's anchor path is the all-center column `(1,1,1)^N`.
//! Along this column every intermediate node is a `Child::Node` so
//! the render frame can descend all the way to `anchor_depth = N`.
//! The planet sits in the sibling cell directly below the camera
//! (slot `(1,0,1)` of the depth-2 parent), with its top face at
//! `y = 4/3`. Camera spawns just above that face.
//!
//! Each star is a single `Child::Block(STAR_COLOR)` at a non-center
//! slot of some ancestor level `d`. A star placed at level `d`
//! shares `d-1` slots of prefix with the camera, so a ray from the
//! camera to that star pops `N - (d-1)` ribbon levels before
//! descending into the star's sibling-chain. Placing stars at a
//! range of `d` values exercises every ribbon depth from 0 to N.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::palette::{block, ColorRegistry};
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, Child, NodeLibrary, MAX_DEPTH,
};

/// Star color — bright warm yellow, chosen so `b < r` and `b < g`
/// (the sky-dominance heuristic picks blue pixels; stars must not
/// be confused with sky).
const STAR_RGB: (u8, u8, u8) = (255, 220, 80);

/// Star placement spec: `(divergence_depth, (slot_x, slot_y, slot_z))`.
/// The star is a single `Child::Block` at this slot of the
/// depth-`divergence_depth` node on the camera's column.
///
/// Slots must not equal `(1, 1, 1)` — that's the camera's onward
/// path. Every other slot is fair game.
///
/// We spread the stars across `±X`, `±Z`, and `+Y` hemispheres so
/// screenshots from multiple look-directions each catch at least
/// one. Depths are spaced across `1..=N-1` so every ribbon count
/// from "just one pop" to "full depth" is exercised.
const STAR_SPECS: &[(u8, (u8, u8, u8))] = &[
    // Root-level giants: slots in the +Y, -X, +X, ±Z hemispheres.
    // Huge, visible across any look direction.
    (1,  (1, 2, 1)), // directly above (+Y)
    (1,  (0, 1, 1)), // straight left (-X)
    (1,  (2, 1, 1)), // straight right (+X)
    (1,  (1, 1, 0)), // straight back (-Z)
    (1,  (1, 1, 2)), // straight front (+Z)
    // Mid-depth stars — exercise moderate ribbon counts.
    (5,  (0, 2, 1)),
    (10, (2, 2, 0)),
    (15, (1, 2, 2)),
    // Deep stars — the precision fix's original target range.
    (20, (0, 2, 2)),
    (24, (2, 2, 0)),
];

/// Build the stars world. `total_depth` sets the tree depth and the
/// camera's anchor depth (= `total_depth - 1`). Keep it ≥ 25 to
/// force deep ribbons that would have failed without the
/// `ray_dir`-unscaled pop transform.
pub fn bootstrap_stars_world(total_depth: u8) -> WorldBootstrap {
    assert!(total_depth >= 3, "stars world needs at least 3 levels");
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

    // Bottom-up build. `current` starts as the deepest leaf and
    // each iteration wraps it in one more `Node` layer.
    let mut current = lib.insert(empty_children());

    // k is the depth-from-root of the node we're constructing, so
    // its children sit at depth `k+1`. We walk from the deepest
    // non-leaf down to the root (k = 1 .. total_depth - 1
    // inclusive). At k = total_depth we've already built the leaf.
    for k in (1..total_depth).rev() {
        let mut children = empty_children();
        // Camera's onward path: (1,1,1) always points to the
        // deeper Node so compute_render_frame can descend all the
        // way to the camera's anchor depth.
        children[slot_index(1, 1, 1)] = Child::Node(current);

        // Planet: one cube at path [(1,1,1), (1,0,1)] — slot
        // (1,0,1) of the depth-1 node on the camera's column.
        // The node being constructed at iteration k = 2 IS that
        // depth-1 node, so this is where the planet block goes.
        // Cell: [4/3, 5/3]×[1, 4/3]×[4/3, 5/3]. Top face at y=4/3.
        // Camera spawns just above that face.
        if k == 2 {
            children[slot_index(1, 0, 1)] = Child::Block(planet_block);
        }

        // Stars at this depth.
        for &(d, (sx, sy, sz)) in STAR_SPECS {
            if d as u32 == k as u32 {
                debug_assert!(
                    (sx, sy, sz) != (1, 1, 1),
                    "star at (1,1,1) would occlude camera's descent column",
                );
                children[slot_index(sx as usize, sy as usize, sz as usize)] =
                    Child::Block(star_block);
            }
        }

        current = lib.insert(children);
    }
    lib.ref_inc(current);
    let world = WorldState { root: current, library: lib };

    // Camera just above the planet's top face. The planet sits in
    // slot (1,0,1) of (1,1,1) of root → cell y = [1, 4/3]. Spawn
    // at y = 1.35 so the camera stands on the top face with a
    // 0.017 clearance. X and Z are at 1.5 (center of root's
    // (1,1,1) slot), so the camera's path at every depth is the
    // all-center column.
    //
    // `from_frame_local` at shallow depth (3) then `deepened_to`
    // walks the ternary expansion precisely — no f32 drift as N
    // grows. Camera anchor is `total_depth - 1` so the render
    // frame has full Node-chain coverage (the leaf at total_depth
    // has all-empty children).
    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [1.5, 1.35, 1.5],
        3,
    )
    .deepened_to(total_depth - 1);

    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: 0.0,
        // Pitch slightly up so the default view catches the +Y
        // hemisphere stars. Screenshot tests override this anyway.
        default_spawn_pitch: 0.6,
        plain_layers: total_depth,
        color_registry: registry,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stars_world_has_expected_depth() {
        let b = bootstrap_stars_world(25);
        assert_eq!(b.world.tree_depth(), 25);
    }

    #[test]
    fn stars_world_spawn_is_above_planet_top() {
        let b = bootstrap_stars_world(25);
        let xyz = b.default_spawn_pos.in_frame(&Path::root());
        // Planet top face is at y = 4/3.
        assert!(xyz[1] > 4.0 / 3.0, "camera y {} must be above planet top 4/3", xyz[1]);
        assert!(xyz[1] < 5.0 / 3.0, "camera y {} must be inside (1,1,1) sub-cell of (1,1,1) of root", xyz[1]);
    }
}
