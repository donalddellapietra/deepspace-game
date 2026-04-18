//! Menger sponge — canonical ternary fractal, native to our scale-3 tree.
//!
//! # Structure
//!
//! Of the 27 sub-cells in a 3×3×3 node, the 7 where ≥2 coords equal the
//! center (= 6 face centers + 1 body center) are removed; the remaining
//! 20 recurse into sub-sponges. This is the exact Menger definition.
//!
//! The 20 kept cells split into two **structural roles**:
//!
//! - **8 cube corners** (all coords ∈ {0, 2}) — the vertex scaffold.
//! - **12 edge midpoints** (exactly one coord = 1) — the spanning ribs.
//!
//! # Coloring
//!
//! PySpace's `menger` scene uses `color=(.2, .5, 1.0)` — a cool blue —
//! and `mausoleum` (a Menger embellishment) uses orbit-trap ochre. We
//! blend the two: corners get the PySpace menger blue, edges get a
//! warmer structural bronze, so the scaffold-vs-rib split is visible
//! at every zoom level. This is how we reproduce the orbit-trap
//! "different folds, different hues" look in a discrete voxel tree.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::fractals::{self_similar_fractal, Slot};
use crate::world::palette::{block, ColorRegistry};
use crate::world::state::WorldState;
use crate::world::tree::{empty_children, slot_index, Child, NodeLibrary, BRANCH, MAX_DEPTH};

/// Kept-cell predicate for the Menger sponge. Excludes cells where
/// ≥2 coords are the center (1).
#[inline]
fn is_menger_kept(x: usize, y: usize, z: usize) -> bool {
    let count = (x == 1) as u8 + (y == 1) as u8 + (z == 1) as u8;
    count < 2
}

/// Uniform-block Menger sponge (backward-compat helper). All 20
/// kept cells become `Child::Block(STONE)` at the deepest level.
///
/// Existing callers (e.g. `gpu/pack.rs` baseline tests) rely on
/// this function's exact output; don't change its block choice.
pub fn menger_world(depth: u8) -> WorldState {
    let mut lib = NodeLibrary::default();

    let mut children = empty_children();
    for z in 0..BRANCH {
        for y in 0..BRANCH {
            for x in 0..BRANCH {
                if !is_menger_kept(x, y, z) { continue; }
                children[slot_index(x, y, z)] = Child::Block(block::STONE);
            }
        }
    }
    let mut current = lib.insert(children);

    for _ in 1..depth {
        let mut children = empty_children();
        for z in 0..BRANCH {
            for y in 0..BRANCH {
                for x in 0..BRANCH {
                    if !is_menger_kept(x, y, z) { continue; }
                    children[slot_index(x, y, z)] = Child::Node(current);
                }
            }
        }
        current = lib.insert(children);
    }

    lib.ref_inc(current);
    WorldState { root: current, library: lib }
}

/// Two-colour Menger: 8 corner cells painted with `corner_block`, 12
/// edge-midpoint cells with `edge_block`. See module docs for why.
fn menger_world_two_tone(depth: u8, corner_block: u8, edge_block: u8) -> WorldState {
    let mut slots: Vec<Slot> = Vec::with_capacity(20);
    for z in 0u8..3 {
        for y in 0u8..3 {
            for x in 0u8..3 {
                if !is_menger_kept(x as usize, y as usize, z as usize) { continue; }
                let center_axes = (x == 1) as u8 + (y == 1) as u8 + (z == 1) as u8;
                let block_id = if center_axes == 0 { corner_block } else { edge_block };
                slots.push((x, y, z, block_id));
            }
        }
    }
    let mut lib = NodeLibrary::default();
    let root = self_similar_fractal(&mut lib, depth, &slots);
    lib.ref_inc(root);
    WorldState { root, library: lib }
}

pub(crate) fn bootstrap_menger_world(depth: u8) -> WorldBootstrap {
    let depth = depth.min(MAX_DEPTH as u8);

    // Palette inspired by PySpace's `menger` (cool blue, .2 .5 1.0) +
    // `mausoleum` orbit-trap bronze (.70 .55 .25): corners get the
    // structural bronze, the connecting edges get the blue. At every
    // zoom level you see the same corner-rib weave.
    let mut registry = ColorRegistry::new();
    let corner = registry.register(178, 140, 64, 255).unwrap();   // bronze (.70 .55 .25)
    let edge = registry.register(51, 128, 255, 255).unwrap();     // PySpace menger blue (.20 .50 1.0)

    let world = menger_world_two_tone(depth, corner, edge);

    // Far-diagonal pose (see `scripts/test-fractals.sh`): (+X,+Y,+Z)
    // corner of the root cell looking back along the body diagonal.
    // Reveals the sponge's 3D structure with both corner and edge
    // tones clearly readable.
    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [2.8, 2.8, 2.8],
        2,
    )
    .deepened_to(8);
    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        default_spawn_yaw: std::f32::consts::FRAC_PI_4,
        default_spawn_pitch: -0.615,
        plain_layers: depth,
        color_registry: registry,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn menger_world_has_expected_depth() {
        let w = menger_world(5);
        assert_eq!(w.tree_depth(), 5);
    }

    #[test]
    fn menger_world_dedups_to_one_node_per_level() {
        // All 20 filled cells at each level share one child → O(depth).
        let w = menger_world(6);
        assert_eq!(w.library.len(), 6);
    }

    #[test]
    fn two_tone_has_same_topology() {
        // Corner/edge split uses different blocks but SAME kept-cell
        // set, so tree_depth and library size match uniform Menger.
        let a = menger_world(6);
        let b = menger_world_two_tone(6, 11, 12);
        assert_eq!(a.tree_depth(), b.tree_depth());
        assert_eq!(a.library.len(), b.library.len());
    }
}
