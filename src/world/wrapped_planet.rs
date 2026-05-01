//! Wrapped Cartesian planet construction. Phase 1.2: hardcoded
//! 18x9x3 slab. No wrap or polar handling yet -- those are Phases
//! 2 and 3.
//!
//! Geometry:
//!
//! `active_subdepth=2` means there are two subdivisions of the
//! planet's `[0, 3)`-cubed local frame below the planet root, with
//! the planet root counted as subdepth 0:
//!
//!   - subdepth 0 = planet root, 27 children at the next level.
//!   - subdepth 1 = 9 cells per axis from the planet root.
//!   - subdepth 2 = 27 cells per axis from the planet root. Active
//!     cells live here; each spans 1/9 of the planet frame.
//!
//! For an active cell at coord `(x, y, z)` with `x in [0, width)`,
//! `y in [0, height)`, `z in [0, depth)`:
//!
//! | level | slot indices                                |
//! |-------|---------------------------------------------|
//! | 1     | `(x / 9, y / 9, z / 9)`                     |
//! | 2     | `((x % 9) / 3, (y % 9) / 3, (z % 9) / 3)`   |
//! | 3     | `(x % 3, y % 3, z % 3)`                     |
//!
//! Banned slots outside the active region are filled with
//! `Child::Empty`, so the existing dedup machinery collapses the
//! all-empty subtrees to a single shared empty node.

use crate::world::palette::block;
use crate::world::tree::{
    empty_children, slot_index, Child, NodeId, NodeKind, NodeLibrary, BRANCH,
};

/// Phase 1 hardcoded width along X (longitude). Wraps modularly. 18
/// cells = 2x the latitude extent (9), so longitude:latitude = 2:1
/// (Mercator-correct — square cells at the equator). 18 divides
/// `3^active_subdepth = 9` cleanly so the active edge falls on a
/// cell boundary at every march-DDA depth.
pub const PHASE1_WIDTH: u16 = 18;
/// Phase 1 hardcoded height along Y (vertical / shell thickness).
/// 3 cells = a thin gravity-aligned crust: y=0 dirt (bedrock band),
/// y=1 stone, y=2 grass (top band). At `active_subdepth=2` the active
/// Y region spans planet-frame `[0, 3/9) = [0, 1/3)`.
pub const PHASE1_HEIGHT: u16 = 3;
/// Phase 1 hardcoded depth along Z (latitude). Bounded; cells outside
/// `[0, depth)` are polar bands. 9 cells covers the full planet-frame
/// Z extent at `active_subdepth=2` (9/9 = 1.0 planet-frame unit).
pub const PHASE1_DEPTH: u16 = 9;
/// Phase 1 active subdepth: 2 levels of subdivision below the planet
/// root *as the wrap-math denominator* (`pow3_u(2) = 9` active cells
/// per planet-root cell along each axis). The build itself nests 3
/// levels (L1/L2/L3) so leaves sit at planet-root-depth + 3, each
/// spanning 1/9 of the planet frame.
pub const PHASE1_ACTIVE_SUBDEPTH: u8 = 2;

/// Build the active-region voxel content for an 18x9x3 slab and wrap
/// it in a `NodeKind::WrappedPlanet` node. See module docs for the
/// geometry.
///
/// Active-cell block content:
///   - `y == 0`           -> dirt (bedrock band)
///   - `y == height - 1`  -> grass (top band)
///   - else               -> stone
pub fn build_phase1_planet(lib: &mut NodeLibrary) -> NodeId {
    build_planet(
        lib,
        PHASE1_WIDTH,
        PHASE1_HEIGHT,
        PHASE1_DEPTH,
        PHASE1_ACTIVE_SUBDEPTH,
    )
}

/// General builder for a wrapped Cartesian planet. Currently only
/// `active_subdepth == 2` is supported -- other values panic. Phase 5
/// may relax this.
///
/// `width`, `height`, `depth` are active-cell counts along X, Y, Z.
/// Each axis is capped at `3^active_subdepth` (27 at the default
/// subdepth) so the active region fits inside the planet root's
/// `[0, 3)^3` frame.
pub fn build_planet(
    lib: &mut NodeLibrary,
    width: u16,
    height: u16,
    depth: u16,
    active_subdepth: u8,
) -> NodeId {
    assert_eq!(
        active_subdepth, 2,
        "only active_subdepth=2 supported in Phase 1",
    );
    assert!(
        width <= 27 && height <= 27 && depth <= 27,
        "width/height/depth must each be <= 27 at active_subdepth=2 (got {}x{}x{})",
        width, height, depth,
    );

    let w = width as usize;
    let h = height as usize;
    let d = depth as usize;

    // Build level 1 of the planet root: 3x3x3 children. Each child
    // spans 1.0 planet-frame units (one "huge cell"). For each huge
    // cell we either return an empty subtree (if the active region
    // doesn't intersect it at all) or descend to level 2.
    let mut root_children = empty_children();
    for l1z in 0..BRANCH {
        for l1y in 0..BRANCH {
            for l1x in 0..BRANCH {
                let big = build_level1_child(lib, w, h, d, l1x, l1y, l1z);
                root_children[slot_index(l1x, l1y, l1z)] = big;
            }
        }
    }

    lib.insert_with_kind(
        root_children,
        NodeKind::WrappedPlanet {
            width,
            height,
            depth,
            active_subdepth,
        },
    )
}

/// Build a level-1 ("huge") child of the planet root. Returns
/// `Child::Empty` if no active cell lands inside this huge cell,
/// otherwise a `Child::Node` holding the level-2 subtree.
fn build_level1_child(
    lib: &mut NodeLibrary,
    w: usize,
    h: usize,
    d: usize,
    l1x: usize,
    l1y: usize,
    l1z: usize,
) -> Child {
    // Active cell coord ranges that fall in this huge cell are
    //   x in [l1x*9, l1x*9 + 9), y in [l1y*9, l1y*9 + 9),
    //   z in [l1z*9, l1z*9 + 9).
    let x_lo = l1x * 9;
    let y_lo = l1y * 9;
    let z_lo = l1z * 9;
    if x_lo >= w || y_lo >= h || z_lo >= d {
        return Child::Empty;
    }

    let mut children = empty_children();
    for l2z in 0..BRANCH {
        for l2y in 0..BRANCH {
            for l2x in 0..BRANCH {
                let big = build_level2_child(
                    lib,
                    w,
                    h,
                    d,
                    l1x * 9 + l2x * 3,
                    l1y * 9 + l2y * 3,
                    l1z * 9 + l2z * 3,
                );
                children[slot_index(l2x, l2y, l2z)] = big;
            }
        }
    }
    Child::Node(lib.insert(children))
}

/// Build a level-2 ("big") child. `(bx_lo, by_lo, bz_lo)` is the
/// lowest active-cell coord that falls in this big cell. Returns
/// `Child::Empty` if entirely outside the active region.
fn build_level2_child(
    lib: &mut NodeLibrary,
    w: usize,
    h: usize,
    d: usize,
    bx_lo: usize,
    by_lo: usize,
    bz_lo: usize,
) -> Child {
    if bx_lo >= w || by_lo >= h || bz_lo >= d {
        return Child::Empty;
    }
    let mut children = empty_children();
    for l3z in 0..BRANCH {
        for l3y in 0..BRANCH {
            for l3x in 0..BRANCH {
                let cx = bx_lo + l3x;
                let cy = by_lo + l3y;
                let cz = bz_lo + l3z;
                let cell = if cx < w && cy < h && cz < d {
                    Child::Block(block_for_active_cell(cy, h))
                } else {
                    Child::Empty
                };
                children[slot_index(l3x, l3y, l3z)] = cell;
            }
        }
    }
    Child::Node(lib.insert(children))
}

/// Block content for an active cell. `cy` is the active-cell Y
/// coordinate (`0..height`).
#[inline]
fn block_for_active_cell(cy: usize, height: usize) -> u16 {
    if cy == 0 {
        block::DIRT
    } else if cy == height - 1 {
        block::GRASS
    } else {
        block::STONE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::raycast::cpu_raycast;
    use crate::world::tree::REPRESENTATIVE_EMPTY;

    #[test]
    fn phase1_planet_inserts_with_correct_kind() {
        let mut lib = NodeLibrary::default();
        let id = build_phase1_planet(&mut lib);
        let node = lib.get(id).expect("planet node missing");
        assert!(
            matches!(
                node.kind,
                NodeKind::WrappedPlanet {
                    width: PHASE1_WIDTH,
                    height: PHASE1_HEIGHT,
                    depth: PHASE1_DEPTH,
                    active_subdepth: PHASE1_ACTIVE_SUBDEPTH,
                }
            ),
            "unexpected planet NodeKind: {:?}",
            node.kind,
        );
    }

    #[test]
    fn phase1_planet_has_non_empty_representative() {
        let mut lib = NodeLibrary::default();
        let id = build_phase1_planet(&mut lib);
        let node = lib.get(id).unwrap();
        assert_ne!(
            node.representative_block,
            REPRESENTATIVE_EMPTY,
            "Phase 1 planet should contain stone/grass/dirt",
        );
    }

    #[test]
    fn phase1_planet_dedups_when_built_twice() {
        let mut lib = NodeLibrary::default();
        let a = build_phase1_planet(&mut lib);
        let b = build_phase1_planet(&mut lib);
        assert_eq!(a, b, "identical planets must dedup via content addressing");
    }

    #[test]
    fn phase1_planet_library_dedups_to_small_node_count() {
        // The 18x3x9 planet at active_subdepth=2 dedups aggressively
        // because the height=3 vertical structure (dirt/stone/grass)
        // fits inside ONE leaf node along Y, making every active leaf
        // identical regardless of its (x, z) position.
        //
        // Build trace (the build nests TWO levels of node, not three:
        // build_level1_child wraps a 3^3 array of build_level2_child
        // results, each of which IS the leaf node):
        //   - Active region: x in [0, 18), y in [0, 3), z in [0, 9).
        //   - Leaf node (build_level2_child): 27 children at
        //     (l3x, l3y, l3z) in [0,3)^3. For active leaves (those
        //     with by_lo=0, in-bounds bx_lo and bz_lo), every child
        //     has cy = l3y in [0, 3), so the dirt/stone/grass column
        //     is fully captured inside one leaf node. All 27 children
        //     are blocks. -> 1 unique leaf NodeId.
        //   - L1 node (build_level1_child): 27 children. Active L1
        //     (l1y=l1z=0, l1x in {0,1}) holds 9 leaf refs (l2y=0,
        //     all l2x, all l2z) and 18 Empty (l2y=1,2 rows). The two
        //     active L1 instances both reference the same single
        //     leaf NodeId in the same 9 slot positions, so they
        //     dedup. -> 1 unique L1 NodeId.
        //   - Planet root: 27 children, 2 active (slots (0,0,0) and
        //     (1,0,0)) sharing the L1 NodeId, 25 Empty. The root
        //     itself has NodeKind::WrappedPlanet which makes it
        //     distinct from any Cartesian node with the same
        //     children. -> 1 planet root NodeId.
        //
        // Expected library size: planet root + L1 + leaf = 3.
        let mut lib = NodeLibrary::default();
        let _ = build_phase1_planet(&mut lib);
        assert_eq!(lib.len(), 3, "planet library size = {}", lib.len());
    }

    // Phase 1.3 banned-cell acceptance: rays that traverse only
    // banned (out-of-active-region) coords must miss. With 18x3x9 at
    // active_subdepth=2 the active region in planet-frame coords is
    // [0, 2.0) x [0, 0.333) x [0, 1.0). Banned regions are filled
    // with `Child::Empty`, so the existing empty-cell DDA advance
    // already produces a no-hit. These tests pin that contract so a
    // future "optimize the planet by partially flattening banned
    // regions into the planet root's NodeKind" change cannot
    // accidentally revive content there.

    #[test]
    fn ray_through_x_banned_column_misses() {
        // Planet-frame X active region is [0, 2.0); cell.x=2 of the
        // planet root is entirely banned. Ray sits at x=2.5 going +Z
        // through banned region.
        let mut lib = NodeLibrary::default();
        let planet = build_phase1_planet(&mut lib);
        let hit = cpu_raycast(&lib, planet, [2.5, 0.1, 0.1], [0.0, 0.0, 1.0], 4);
        assert!(hit.is_none(), "ray entirely within X-banned column should miss");
    }

    #[test]
    fn ray_through_y_banned_row_misses() {
        // Planet-frame Y active region is [0, 0.333); y=0.5 sits in
        // the banned row at planet-root cell.y=0 layer above the
        // active sub-cell band. Ray sweeps +X staying at y=0.5.
        let mut lib = NodeLibrary::default();
        let planet = build_phase1_planet(&mut lib);
        let hit = cpu_raycast(&lib, planet, [0.0, 0.5, 0.5], [1.0, 0.0, 0.0], 4);
        assert!(hit.is_none(), "ray entirely within Y-banned row should miss");
    }

    #[test]
    fn ray_through_z_banned_region_misses() {
        // Planet-frame Z active region is [0, 1.0); z=1.5 sits in
        // cell.z=1 of the planet root, entirely banned (active
        // region spans only cell.z=0). Ray sweeps +X staying at z=1.5.
        let mut lib = NodeLibrary::default();
        let planet = build_phase1_planet(&mut lib);
        let hit = cpu_raycast(&lib, planet, [0.0, 0.1, 1.5], [1.0, 0.0, 0.0], 4);
        assert!(hit.is_none(), "ray entirely within Z-banned region should miss");
    }

    #[test]
    fn ray_into_active_region_hits() {
        // Sanity: a downward ray from above the active region hits
        // the grass top. With height=3 the active y top is at
        // planet-frame y = 3/9 = 0.333 (active y=2 occupies
        // [2/9, 3/9)). Camera at y=1.5 looking straight down crosses
        // banned y first then lands on grass at y=0.333. Using
        // (1.0, 1.5, 0.1) keeps z in the active region [0, 1.0).
        let mut lib = NodeLibrary::default();
        let planet = build_phase1_planet(&mut lib);
        let hit = cpu_raycast(&lib, planet, [1.0, 1.5, 0.1], [0.0, -1.0, 0.0], 4);
        assert!(hit.is_some(), "downward ray over active region should hit grass");
    }

    // ---- Phase 2 wrap-aware path arithmetic ----
    //
    // The CPU `Path::step_neighbor_in_world` mirrors the shader's
    // X-axis modular wrap. Tests below exercise:
    //   - Walking east `width` cells returns to the same path.
    //   - West-step at the west edge wraps to the east edge (with the
    //     correct L1/L2/L3 slot decomposition).
    //   - Stepping into a banned Y row no-ops (the move is rejected
    //     instead of entering the polar band).

    #[test]
    fn walking_east_planet_width_returns_to_origin() {
        use crate::world::anchor::Path;

        let mut lib = NodeLibrary::default();
        let planet = build_phase1_planet(&mut lib);

        // Wrap planet in a single Cartesian ancestor at slot 13
        // (center). Path indexing in step_neighbor_in_world uses
        // `planet_idx = 0` for "the slot at index 0 led INTO the
        // planet", so descendants live at indices 1..=3.
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(planet);
        let world_root = lib.insert(root_children);

        // Path pointing at active cell (cx, cy, cz) = (5, 1, 4):
        //   L1 = (5/9, 1/9, 4/9) = (0, 0, 0)
        //   L2 = ((5%9)/3, (1%9)/3, (4%9)/3) = (1, 0, 1)
        //   L3 = (5%3, 1%3, 4%3) = (2, 1, 1)
        let mut path = Path::root();
        path.push(slot_index(1, 1, 1) as u8); // planet at slot 13 of root
        path.push(slot_index(0, 0, 0) as u8); // L1
        path.push(slot_index(1, 0, 1) as u8); // L2
        path.push(slot_index(2, 1, 1) as u8); // L3

        let start_path = path;

        for _ in 0..PHASE1_WIDTH {
            path.step_neighbor_in_world(&lib, world_root, 0, 1);
        }

        assert_eq!(
            path, start_path,
            "walking east `width` cells must return to the same path",
        );
    }

    #[test]
    fn walking_west_one_cell_at_origin_wraps_to_east_edge() {
        use crate::world::anchor::Path;

        let mut lib = NodeLibrary::default();
        let planet = build_phase1_planet(&mut lib);
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(planet);
        let world_root = lib.insert(root_children);

        // Path pointing at active cell (cx, cy, cz) = (0, 1, 4):
        //   L1 = (0, 0, 0)
        //   L2 = (0, 0, 1)
        //   L3 = (0, 1, 1)
        let mut path = Path::root();
        path.push(slot_index(1, 1, 1) as u8);
        path.push(slot_index(0, 0, 0) as u8);
        path.push(slot_index(0, 0, 1) as u8);
        path.push(slot_index(0, 1, 1) as u8);

        // Step west once. cx=0 → -1 → wraps to cx=17 (= width - 1).
        // For (cx, cy, cz) = (17, 1, 4):
        //   L1 = (17/9, 1/9, 4/9) = (1, 0, 0)
        //   L2 = ((17%9)/3, 0, 1) = (8/3, 0, 1) = (2, 0, 1)
        //   L3 = (17%3, 1%3, 4%3) = (2, 1, 1)
        path.step_neighbor_in_world(&lib, world_root, 0, -1);

        assert_eq!(path.depth(), 4);
        assert_eq!(path.slot(0), slot_index(1, 1, 1) as u8);
        assert_eq!(path.slot(1), slot_index(1, 0, 0) as u8);
        assert_eq!(path.slot(2), slot_index(2, 0, 1) as u8);
        assert_eq!(path.slot(3), slot_index(2, 1, 1) as u8);
    }

    #[test]
    fn walking_y_into_polar_band_does_not_step() {
        use crate::world::anchor::Path;

        let mut lib = NodeLibrary::default();
        let planet = build_phase1_planet(&mut lib);
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(planet);
        let world_root = lib.insert(root_children);

        // Path at (cx, cy, cz) = (5, 2, 4) — top of the active height
        // (height=3, so cy ∈ [0, 3); cy=2 is the grass row).
        //   L1 = (0, 0, 0)
        //   L2 = (1, 0, 1)
        //   L3 = (2, 2, 1)
        let mut path = Path::root();
        path.push(slot_index(1, 1, 1) as u8);
        path.push(slot_index(0, 0, 0) as u8);
        path.push(slot_index(1, 0, 1) as u8);
        path.push(slot_index(2, 2, 1) as u8);
        let before = path;

        // Try to step up (axis=1, direction=+1): would go to cy=3,
        // banned. Should no-op.
        path.step_neighbor_in_world(&lib, world_root, 1, 1);
        assert_eq!(
            path, before,
            "stepping into the polar band must no-op",
        );
    }
}
