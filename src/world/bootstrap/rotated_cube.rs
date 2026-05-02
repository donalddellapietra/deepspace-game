//! Side-by-side cubes — minimal world for the TangentBlock-from-
//! Cartesian dispatch prototype.
//!
//! Pure Cartesian world tree. Two cubes side by side at the deepest
//! container cell:
//!   - LEFT (`slot 12`): `NodeKind::TangentBlock` containing uniform
//!     grass — the rotated cube. The shader's `march_cartesian`
//!     recognises the kind on descent and transforms the ray into the
//!     cube's local frame using a per-cell rotation.
//!   - RIGHT (`slot 14`): a regular Cartesian Node containing uniform
//!     stone — the axis-aligned reference. Renders through the
//!     standard Cartesian DDA path with no rotation.
//!
//! Side-by-side comparison is mandatory: a yaw-only rotation keeps
//! vertical edges vertical, so a single rotated cube viewed head-on
//! can look identical to an axis-aligned cube without a reference.
//!
//! Precision discipline (the user's hard constraint):
//! - The cube container is buried at `total_depth - 2` levels via a
//!   centered descent chain. Total tree depth defaults to 30 to
//!   exercise the deep-anchor render path.
//! - The CPU spawn uses `Path::push` integer-arithmetic only; no
//!   absolute world XYZ coordinates anywhere. Camera anchor lives in
//!   the container's slot 22 (front-centre) at depth `total_depth - 1`.
//! - The shader's TangentBlock dispatch operates in the parent
//!   render-frame's `[0, 3)` local coords (`cur_node_origin`,
//!   `cur_cell_size`). Rotation is computed from those bounded
//!   inputs and never propagates through deeper descent.

use super::WorldBootstrap;
use crate::world::anchor::{Path, WorldPos};
use crate::world::palette::{block, ColorRegistry};
use crate::world::state::WorldState;
use crate::world::tree::{
    empty_children, slot_index, uniform_children, Child, NodeKind, NodeLibrary, MAX_DEPTH,
};

/// Default total tree depth for the rotated-cube preset. Deep enough
/// to exercise the precision-stable render-frame path the user
/// requires — the cubes live at this depth with a 28-level centered
/// descent chain above them.
pub const DEFAULT_ROTATED_CUBE_DEPTH: u8 = 30;

pub fn rotated_cube_world() -> WorldState {
    rotated_cube_world_at_depth(DEFAULT_ROTATED_CUBE_DEPTH)
}

pub fn rotated_cube_world_at_depth(total_depth: u8) -> WorldState {
    assert!(total_depth >= 2, "total_depth must be >= 2");
    assert!(
        (total_depth as usize) <= MAX_DEPTH,
        "total_depth {} exceeds MAX_DEPTH {}",
        total_depth,
        MAX_DEPTH,
    );

    let mut library = NodeLibrary::default();

    // The rotated cube: TangentBlock whose 27 children are uniform
    // grass. Cannot uniform-flatten, so the node stays explicit; the
    // shader keys on `NodeKind::TangentBlock` to dispatch the
    // per-cell rotation transform.
    let rotated_cube = library.insert_with_kind(
        uniform_children(Child::Block(block::GRASS)),
        NodeKind::TangentBlock,
    );

    // The axis-aligned reference: regular Cartesian node, uniform
    // stone, same shape. Renders through the unmodified Cartesian
    // DDA path. A side-by-side comparison is mandatory — a yaw
    // rotation keeps vertical edges vertical, so a lone rotated cube
    // can look identical to an axis-aligned one head-on.
    let reference_cube = library.insert(uniform_children(Child::Block(block::STONE)));

    // Container holds the two cubes in adjacent slots:
    //   slot 12 (cell 0,1,1) = rotated grass cube (LEFT)
    //   slot 14 (cell 2,1,1) = reference stone cube (RIGHT)
    //   slot 22 (cell 1,1,2) = empty — camera anchor lives here
    // Other slots are empty, giving each cube an isolated silhouette.
    let mut container_children = empty_children();
    container_children[slot_index(0, 1, 1)] = Child::Node(rotated_cube);
    container_children[slot_index(2, 1, 1)] = Child::Node(reference_cube);
    let mut current = library.insert(container_children);

    // Centered descent chain above the container. Each wrap is a
    // Cartesian node whose only non-empty slot is slot 13 (centre).
    // After (total_depth - 2) wraps, the container sits at tree depth
    // (total_depth - 2); cubes at (total_depth - 1); block leaves at
    // total_depth. Pure integer arithmetic, no f32 anywhere.
    for _ in 0..(total_depth - 2) {
        let mut wrapper = empty_children();
        wrapper[slot_index(1, 1, 1)] = Child::Node(current);
        current = library.insert(wrapper);
    }

    library.ref_inc(current);
    let world = WorldState { root: current, library };
    eprintln!(
        "Rotated cube world: depth={}, library_entries={} \
         (rotated grass @ slot12, reference stone @ slot14, container @ depth {})",
        world.tree_depth(),
        world.library.len(),
        total_depth - 2,
    );
    world
}

pub(crate) fn bootstrap_rotated_cube_world() -> WorldBootstrap {
    bootstrap_rotated_cube_world_at_depth(DEFAULT_ROTATED_CUBE_DEPTH)
}

pub(crate) fn bootstrap_rotated_cube_world_at_depth(total_depth: u8) -> WorldBootstrap {
    let world = rotated_cube_world_at_depth(total_depth);

    // Camera anchor: precision-stable construction via integer Path
    // pushes — no absolute world XYZ at any step. The chain descends
    // (total_depth - 3) levels of slot 13, then a final slot 22 push
    // (cell 1,1,2 = +Z neighbour of the next chain step). That places
    // the camera as a SIBLING of the container at depth `total_depth -
    // 2` rather than inside it — far enough back that both cubes fit
    // in the frame instead of filling it.
    //
    // Geometry in the wrapper's [0, 3) local frame at depth
    // `total_depth - 3`:
    //   slot 13 = container at local [1, 2)³ (cubes are inside)
    //   slot 22 = camera cell at local [1, 2)×[1, 2)×[2, 3)
    //   camera offset (0.5, 0.5, 0.5) → wrapper-local (1.5, 1.5, 2.5)
    //   cubes inside container project to wrapper-local
    //     [1, 4/3) × [4/3, 5/3) × [4/3, 5/3)   (rotated grass)
    //     [5/3, 2) × [4/3, 5/3) × [4/3, 5/3)   (reference stone)
    //   ≈ 18° angular offset from forward — comfortable framing.
    let mut anchor = Path::root();
    let chain_len = total_depth - 3;
    for _ in 0..chain_len {
        anchor.push(slot_index(1, 1, 1) as u8); // centre slot 13
    }
    anchor.push(slot_index(1, 1, 2) as u8); // slot 22 = +Z neighbour
    let spawn_pos = WorldPos::new(anchor, [0.5, 0.5, 0.5]);

    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        // yaw=0 is -Z (looking into the screen) per camera.rs basis.
        default_spawn_yaw: 0.0,
        default_spawn_pitch: 0.0,
        // `plain_layers = 0` opts out of `--plain-world` surface
        // tracking that would override our spawn anchor.
        plain_layers: 0,
        color_registry: ColorRegistry::new(),
    }
}
