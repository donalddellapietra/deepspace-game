//! Phase 2 unit tests for the X-wrap CPU primitive.
//!
//! These exercise `WorldPos::add_local` against a `WrappedPlane`-rooted
//! world, verifying that walking east past the slab footprint comes
//! back from the west — the defining feature of the wrapped-Cartesian
//! planet. Runs entirely on the CPU (no harness, no GPU); the GPU
//! parity is checked by `wrapped_planet_visual::ray_east_hits_west_marker`.

#![cfg(not(target_arch = "wasm32"))]

use deepspace_game::world::anchor::{Path, Transition, WorldPos};
use deepspace_game::world::bootstrap::{
    wrapped_planet_world, DEFAULT_WRAPPED_PLANET_CELL_SUBTREE_DEPTH,
    DEFAULT_WRAPPED_PLANET_EMBEDDING_DEPTH, DEFAULT_WRAPPED_PLANET_SLAB_DEPTH,
    DEFAULT_WRAPPED_PLANET_SLAB_DIMS,
};

/// Build the canonical wrapped-planet world for tests. Returns
/// `(world, slab_root_path)` where `slab_root_path` is the path
/// from the world root down to the WrappedPlane node — the centre-
/// slot column repeated `embedding_depth` times.
fn canonical_world() -> (
    deepspace_game::world::state::WorldState,
    Path,
    [u32; 3],
    u8,
) {
    let embedding_depth = DEFAULT_WRAPPED_PLANET_EMBEDDING_DEPTH;
    let dims = DEFAULT_WRAPPED_PLANET_SLAB_DIMS;
    let slab_depth = DEFAULT_WRAPPED_PLANET_SLAB_DEPTH;
    let world = wrapped_planet_world(
        embedding_depth, dims, slab_depth,
        DEFAULT_WRAPPED_PLANET_CELL_SUBTREE_DEPTH,
    );
    let mut slab_root = Path::root();
    for _ in 0..embedding_depth {
        slab_root.push(deepspace_game::world::tree::slot_index(1, 1, 1) as u8);
    }
    (world, slab_root, dims, slab_depth)
}

/// Build a `WorldPos` whose anchor is the slab's leaf cell at
/// `(slab_x, slab_y, slab_z)` (subgrid coordinates), with the offset
/// at the cell centre. Walks the path through the WrappedPlane subtree
/// using ternary digit decomposition.
fn pos_at_slab_cell(
    slab_root: &Path,
    slab_x: u32,
    slab_y: u32,
    slab_z: u32,
    slab_depth: u8,
) -> WorldPos {
    use deepspace_game::world::tree::slot_index;
    let mut anchor = *slab_root;
    // Walk from depth 0 to slab_depth-1 inside the WrappedPlane.
    // At each level k (0-indexed from the slab root), the slot's
    // (x, y, z) coordinate is the ternary digit of (slab_x, slab_y,
    // slab_z) at position (slab_depth - 1 - k).
    for k in 0..slab_depth {
        let pow = 3u32.pow((slab_depth - 1 - k) as u32);
        let cx = (slab_x / pow) % 3;
        let cy = (slab_y / pow) % 3;
        let cz = (slab_z / pow) % 3;
        anchor.push(slot_index(cx as usize, cy as usize, cz as usize) as u8);
    }
    WorldPos::new(anchor, [0.5, 0.5, 0.5])
}

#[test]
fn step_neighbor_in_world_wraps_east_to_west() {
    let (world, slab_root, dims, slab_depth) = canonical_world();
    let mut pos = pos_at_slab_cell(&slab_root, dims[0] - 1, 1, 1, slab_depth);
    let anchor_before = pos.anchor.as_slice().to_vec();
    let dir = 1; // east
    let wrapped = pos
        .anchor
        .step_neighbor_in_world(&world.library, world.root, 0, dir);
    assert!(wrapped, "stepping east off slab east edge should wrap");
    let after = pos_at_slab_cell(&slab_root, 0, 1, 1, slab_depth);
    assert_eq!(
        pos.anchor.as_slice(),
        after.anchor.as_slice(),
        "after wrap, anchor should match slab cell (0, 1, 1); before={:?}",
        anchor_before,
    );
}

#[test]
fn step_neighbor_in_world_wraps_west_to_east() {
    let (world, slab_root, dims, slab_depth) = canonical_world();
    let mut pos = pos_at_slab_cell(&slab_root, 0, 1, 1, slab_depth);
    let dir = -1; // west
    let wrapped = pos
        .anchor
        .step_neighbor_in_world(&world.library, world.root, 0, dir);
    assert!(wrapped, "stepping west off slab west edge should wrap");
    let after = pos_at_slab_cell(&slab_root, dims[0] - 1, 1, 1, slab_depth);
    assert_eq!(
        pos.anchor.as_slice(),
        after.anchor.as_slice(),
        "after wrap, anchor should match slab cell (dims_x-1, 1, 1)",
    );
}

#[test]
fn add_local_east_returns_west_after_full_circumference() {
    let (world, slab_root, dims, slab_depth) = canonical_world();
    let mut pos = pos_at_slab_cell(&slab_root, 0, 1, 1, slab_depth);
    let original_anchor = pos.anchor.as_slice().to_vec();
    let original_offset = pos.offset;

    // Cell-local delta is one full leaf cell along +X. The leaf
    // anchor is at slab_depth levels below the WrappedPlane node;
    // each level scales the cell width by 1/3, so a step of
    // exactly 1.0 in the leaf frame steps one cell. Repeat
    // `dims[0]` times — that's a full circumference.
    let mut transitions_seen = 0usize;
    for _ in 0..dims[0] {
        let t = pos.add_local([1.0, 0.0, 0.0], &world.library, world.root);
        if matches!(t, Transition::WrappedPlaneWrap { .. }) {
            transitions_seen += 1;
        }
    }
    assert_eq!(
        transitions_seen, 1,
        "exactly one wrap should fire per full circumference; got {transitions_seen}",
    );
    assert_eq!(
        pos.anchor.as_slice(),
        original_anchor,
        "after a full circumference the anchor should return to the original",
    );
    for i in 0..3 {
        assert!(
            (pos.offset[i] - original_offset[i]).abs() < 1e-3,
            "offset[{i}] drifted: {} -> {}",
            original_offset[i],
            pos.offset[i],
        );
    }
}

#[test]
fn add_local_east_one_cell_at_x_max_wraps_to_x0() {
    let (world, slab_root, dims, slab_depth) = canonical_world();
    let mut pos = pos_at_slab_cell(&slab_root, dims[0] - 1, 1, 1, slab_depth);
    let t = pos.add_local([1.0, 0.0, 0.0], &world.library, world.root);
    assert!(
        matches!(t, Transition::WrappedPlaneWrap { axis: 0 }),
        "expected an X-wrap transition; got {t:?}",
    );
    let expected = pos_at_slab_cell(&slab_root, 0, 1, 1, slab_depth);
    assert_eq!(
        pos.anchor.as_slice(),
        expected.anchor.as_slice(),
        "east wrap should land on slab cell (0, 1, 1)",
    );
}

#[test]
fn y_axis_does_not_wrap() {
    // Y is not a wrap axis. Stepping past slab_y == dims[1] - 1
    // bubbles out of the slab subtree (anchor walks UP through the
    // WrappedPlane to a sibling of the WP node, since the embedding
    // is Cartesian). The exact anchor we land on depends on the
    // embedding chain; what we must verify is that NO wrap fires.
    let (world, slab_root, _dims, slab_depth) = canonical_world();
    let mut pos = pos_at_slab_cell(&slab_root, 0, 1, 1, slab_depth);
    // Many cells north — far enough to definitely exit the slab.
    let t = pos.add_local([0.0, 100.0, 0.0], &world.library, world.root);
    assert!(
        !matches!(t, Transition::WrappedPlaneWrap { .. }),
        "Y-axis motion must NOT trigger an X-wrap transition; got {t:?}",
    );
}

#[test]
fn z_axis_does_not_wrap() {
    let (world, slab_root, _dims, slab_depth) = canonical_world();
    let mut pos = pos_at_slab_cell(&slab_root, 0, 1, 1, slab_depth);
    let t = pos.add_local([0.0, 0.0, 100.0], &world.library, world.root);
    assert!(
        !matches!(t, Transition::WrappedPlaneWrap { .. }),
        "Z-axis motion must NOT trigger an X-wrap transition; got {t:?}",
    );
}
