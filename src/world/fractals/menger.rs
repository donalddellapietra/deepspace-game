//! Menger sponge analogue — 7/8 "hollow cube" fractal for the base-2 octree.
//!
//! # Structure
//!
//! The classic Menger sponge is inherently ternary (20/27 cells kept).
//! In a base-2 octree there is no center row to remove. The closest
//! analogue is a "hollow cube" pattern: fill 7 of 8 slots, leaving
//! one corner empty. We omit (0,0,0) and keep the remaining 7 slots.
//! 7/8 = 87.5% occupancy per level.
//!
//! The 7 kept cells split into two **structural roles**:
//!
//! - **3 face-adjacent** slots (exactly 1 coord differs from the
//!   removed corner): (1,0,0), (0,1,0), (0,0,1).
//! - **4 distant** slots (2+ coords differ from the removed corner):
//!   (1,1,0), (1,0,1), (0,1,1), (1,1,1).
//!
//! # Coloring
//!
//! PySpace's `menger` scene uses `color=(.2, .5, 1.0)` — a cool blue —
//! and `mausoleum` (a Menger embellishment) uses orbit-trap ochre. We
//! blend the two: face-adjacent slots get the PySpace menger blue,
//! distant slots get a warmer structural bronze, so the role split is
//! visible at every zoom level.

use crate::world::anchor::{Path, WorldPos};
use crate::world::bootstrap::WorldBootstrap;
use crate::world::fractals::{self_similar_fractal, Slot};
use crate::world::palette::{block, ColorRegistry};
use crate::world::state::WorldState;
use crate::world::tree::{empty_children, slot_index, Child, NodeLibrary, BRANCH, MAX_DEPTH};

/// Kept-cell predicate for the base-2 Menger analogue. Keeps all
/// slots except (0,0,0).
#[inline]
fn is_menger_kept(x: usize, y: usize, z: usize) -> bool {
    !(x == 0 && y == 0 && z == 0)
}

/// Uniform-block Menger sponge (backward-compat helper). All 7
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

/// Two-colour Menger: 3 face-adjacent cells painted with `edge_block`,
/// 4 distant cells with `corner_block`. See module docs for why.
fn menger_world_two_tone(depth: u8, corner_block: u16, edge_block: u16) -> WorldState {
    let mut slots: Vec<Slot> = Vec::with_capacity(7);
    for z in 0u8..2 {
        for y in 0u8..2 {
            for x in 0u8..2 {
                if !is_menger_kept(x as usize, y as usize, z as usize) { continue; }
                // Face-adjacent to (0,0,0): exactly one coord is 1
                let nonzero = (x as u8) + (y as u8) + (z as u8);
                let block_id = if nonzero == 1 { edge_block } else { corner_block };
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
    // `mausoleum` orbit-trap bronze (.70 .55 .25): distant slots get the
    // structural bronze, the face-adjacent slots get the blue.
    let mut registry = ColorRegistry::new();
    let corner = registry.register(178, 140, 64, 255).unwrap();   // bronze (.70 .55 .25)
    let edge = registry.register(51, 128, 255, 255).unwrap();     // PySpace menger blue (.20 .50 1.0)

    let world = menger_world_two_tone(depth, corner, edge);

    // Far-diagonal pose: (+X,+Y,+Z) corner of the root cell looking
    // back along the body diagonal.
    let spawn_pos = WorldPos::from_frame_local(
        &Path::root(),
        [1.8, 1.8, 1.8],
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
        // All 7 filled cells at each level share one child -> O(depth).
        let w = menger_world(6);
        assert_eq!(w.library.len(), 6);
    }

    #[test]
    fn two_tone_has_same_topology() {
        // Role split uses different blocks but SAME kept-cell
        // set, so tree_depth and library size match uniform Menger.
        let a = menger_world(6);
        let b = menger_world_two_tone(6, 11, 12);
        assert_eq!(a.tree_depth(), b.tree_depth());
        assert_eq!(a.library.len(), b.library.len());
    }
}
