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

/// Default total tree depth for the rotated-cube preset.
pub const DEFAULT_ROTATED_CUBE_DEPTH: u8 = 30;
/// Default depth of uniform recursive subtree INSIDE each cube. This
/// is what makes the world meaningfully "N layers" — without this,
/// the cubes are 1-cell uniform leaves and zooming in reveals
/// nothing. Mirrors `wrapped_planet`'s `cell_subtree_depth = 20`.
pub const DEFAULT_ROTATED_CUBE_SUBTREE_DEPTH: u8 = 20;

pub fn rotated_cube_world() -> WorldState {
    rotated_cube_world_at_depth(
        DEFAULT_ROTATED_CUBE_DEPTH,
        DEFAULT_ROTATED_CUBE_SUBTREE_DEPTH,
    )
}

pub fn rotated_cube_world_at_depth(
    total_depth: u8,
    cube_subtree_depth: u8,
) -> WorldState {
    assert!(total_depth >= 4, "total_depth must be >= 4");
    assert!(cube_subtree_depth >= 1, "cube_subtree_depth must be >= 1");
    assert!(
        cube_subtree_depth + 2 <= total_depth,
        "cube_subtree_depth ({}) + 2 must fit in total_depth ({})",
        cube_subtree_depth, total_depth,
    );
    assert!(
        (total_depth as usize) <= MAX_DEPTH,
        "total_depth {} exceeds MAX_DEPTH {}",
        total_depth,
        MAX_DEPTH,
    );

    let mut library = NodeLibrary::default();

    // Helper: build a uniform recursive subtree of `depth` Cartesian
    // levels, all 27 children = the inner (eventually `Block`).
    // Content-addressed dedup keeps the chain to O(depth) library
    // entries no matter how many cells it covers. Mirrors
    // `wrapped_planet::build_uniform_anchor`.
    fn uniform_anchor(library: &mut NodeLibrary, block_id: u16, depth: u8) -> Child {
        if depth == 0 {
            return Child::Block(block_id);
        }
        let inner = uniform_anchor(library, block_id, depth - 1);
        Child::Node(library.insert(uniform_children(inner)))
    }

    // Rotated cube: TangentBlock whose 27 children are deep uniform
    // grass anchors. The TangentBlock node itself is at the cube
    // root level; INSIDE it, `cube_subtree_depth - 1` more levels of
    // uniform grass form the recursive content. Total levels at and
    // below the TangentBlock = `cube_subtree_depth`.
    let grass_inner = uniform_anchor(&mut library, block::GRASS, cube_subtree_depth - 1);
    let rotated_cube = library.insert_with_kind(
        uniform_children(grass_inner),
        NodeKind::TangentBlock,
    );

    // Reference cube: regular Cartesian, same depth + structure but
    // uniform stone. Renders through the unmodified Cartesian DDA so
    // any visual difference between the two cubes attributes solely
    // to the rotation transform.
    let stone_inner = uniform_anchor(&mut library, block::STONE, cube_subtree_depth - 1);
    let reference_cube = library.insert(uniform_children(stone_inner));

    // Container holds the two cubes in adjacent slots:
    //   slot 12 (cell 0,1,1) = rotated grass cube (LEFT)
    //   slot 14 (cell 2,1,1) = reference stone cube (RIGHT)
    //   slot 22 (cell 1,1,2) = empty — camera anchor lives here
    let mut container_children = empty_children();
    container_children[slot_index(0, 1, 1)] = Child::Node(rotated_cube);
    container_children[slot_index(2, 1, 1)] = Child::Node(reference_cube);
    let mut current = library.insert(container_children);

    // Centered descent chain above the container. Chain length is
    // `total_depth - 1 - cube_subtree_depth` so that:
    //   block leaves          @ depth `total_depth`
    //   cube node             @ depth `total_depth - cube_subtree_depth`
    //   container             @ depth `chain_len`
    //   camera anchor cell    @ depth `chain_len + 1` (slot 22 of last
    //                          chain wrapper — sibling of container)
    let chain_len = total_depth - 1 - cube_subtree_depth;
    for _ in 0..chain_len {
        let mut wrapper = empty_children();
        wrapper[slot_index(1, 1, 1)] = Child::Node(current);
        current = library.insert(wrapper);
    }

    library.ref_inc(current);
    let world = WorldState { root: current, library };
    eprintln!(
        "Rotated cube world: depth={}, library_entries={}, \
         cube_subtree_depth={}, container @ depth {}",
        world.tree_depth(),
        world.library.len(),
        cube_subtree_depth,
        chain_len,
    );
    world
}

pub(crate) fn bootstrap_rotated_cube_world() -> WorldBootstrap {
    bootstrap_rotated_cube_world_at_depth(
        DEFAULT_ROTATED_CUBE_DEPTH,
        DEFAULT_ROTATED_CUBE_SUBTREE_DEPTH,
    )
}

pub(crate) fn bootstrap_rotated_cube_world_at_depth(
    total_depth: u8,
    cube_subtree_depth: u8,
) -> WorldBootstrap {
    let world = rotated_cube_world_at_depth(total_depth, cube_subtree_depth);

    // Camera anchor: pure integer Path::push arithmetic; no absolute
    // world XYZ. The centered chain descends (chain_len - 1) levels
    // of slot 13, then a final slot 22 push placing the camera as a
    // SIBLING of the container at depth `chain_len` (cell 1,1,2 of
    // the same parent that holds the container at slot 13).
    //
    // Geometry in that parent's [0, 3) local frame:
    //   slot 13 = container at local [1, 2)³  (rotated + reference)
    //   slot 22 = camera cell at local [1, 2)×[1, 2)×[2, 3)
    //   camera offset (0.5, 0.5, 0.5) → frame-local (1.5, 1.5, 2.5)
    //   cubes project to ≈ 18° angular offset from forward (-Z).
    let chain_len = total_depth - 1 - cube_subtree_depth;
    let mut anchor = Path::root();
    for _ in 0..(chain_len - 1) {
        anchor.push(slot_index(1, 1, 1) as u8);
    }
    anchor.push(slot_index(1, 1, 2) as u8); // slot 22 = +Z neighbour
    // Diagonal spawn — three distinct off-centre offsets so the camera
    // position lies on no slot or face boundary in any axis. Combined
    // with the yaw / pitch below this guarantees the viewing ray
    // never coincides with a frame axis, so frame transitions don't
    // look like the camera rotating into the new frame.
    let spawn_pos = WorldPos::new(anchor, [0.42, 0.58, 0.71]);

    WorldBootstrap {
        world,
        planet_path: None,
        default_spawn_pos: spawn_pos,
        // Diagonal yaw + pitch — off-cardinal so the camera's viewing
        // line is never aligned with any axis of any render frame it
        // descends into. Frames in this engine are nested and
        // axis-aligned; a diagonal viewer means there's no frame
        // transition at which the camera's apparent orientation
        // discontinuously snaps to a frame axis. Magnitudes kept
        // small enough that both cubes (at ≈ ±18° from -Z) stay in
        // frame.
        default_spawn_yaw: 0.2617994,    // π/12 = 15°
        default_spawn_pitch: -0.2617994, // -π/12 = -15°
        // `plain_layers = 0` opts out of `--plain-world` surface
        // tracking that would override our spawn anchor.
        plain_layers: 0,
        color_registry: ColorRegistry::new(),
    }
}
