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

/// Phase 1 hardcoded width along X (longitude). 18 cells = 2x the
/// height; chosen so the active region tiles cleanly at
/// `active_subdepth=2` (18 divides 9 evenly so the active edge falls
/// on a cell boundary at every march-DDA depth).
pub const PHASE1_WIDTH: u16 = 18;
/// Phase 1 hardcoded height along Y (latitude). 9 cells covers the
/// full planet-frame Y extent at `active_subdepth=2` (9/9 = 1.0).
pub const PHASE1_HEIGHT: u16 = 9;
/// Phase 1 hardcoded depth along Z. 3 cells is the simplest 3^N-aligned
/// crust that lets later phases bump it up without rework.
pub const PHASE1_DEPTH: u16 = 3;
/// Phase 1 active subdepth: 2 levels of subdivision below the planet
/// root, so each active cell spans 1/9 of the planet frame.
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
        // The 18x9x3 planet at active_subdepth=2 is heavily dedup'd:
        // along Y (latitude) only the per-row block content varies,
        // and along X (longitude) every L1 sibling sees the same Y
        // structure. Expected library size: 1 planet root + 1 L1
        // child + 3 L2 children (one per Y row) = 5 nodes.
        let mut lib = NodeLibrary::default();
        let _ = build_phase1_planet(&mut lib);
        assert_eq!(lib.len(), 5, "planet library size = {}", lib.len());
    }

    // Phase 1.3 banned-cell acceptance: rays that traverse only
    // banned (out-of-active-region) coords must miss. With 18x9x3 at
    // active_subdepth=2 the active region in planet-frame coords is
    // [0, 2.0) x [0, 1.0) x [0, 0.333). Banned regions are filled
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
        let hit = cpu_raycast(&lib, planet, [2.5, 0.5, 0.1], [0.0, 0.0, 1.0], 4);
        assert!(hit.is_none(), "ray entirely within X-banned column should miss");
    }

    #[test]
    fn ray_through_y_banned_row_misses() {
        // Planet-frame Y active region is [0, 1.0); cells at
        // planet-frame y>=1 are banned (the polar bands).
        let mut lib = NodeLibrary::default();
        let planet = build_phase1_planet(&mut lib);
        let hit = cpu_raycast(&lib, planet, [1.0, 1.5, 0.1], [0.001, 0.0, 1.0], 4);
        assert!(hit.is_none(), "ray entirely within Y-banned row should miss");
    }

    #[test]
    fn ray_through_z_banned_region_misses() {
        // Planet-frame Z active region is [0, 0.333); z=1.5 is well
        // inside cell.z=1, banned. Ray sweeps +X across the planet
        // staying in z=1.5.
        let mut lib = NodeLibrary::default();
        let planet = build_phase1_planet(&mut lib);
        let hit = cpu_raycast(&lib, planet, [0.0, 0.5, 1.5], [1.0, 0.0, 0.0], 4);
        assert!(hit.is_none(), "ray entirely within Z-banned region should miss");
    }

    #[test]
    fn ray_into_active_region_hits() {
        // Sanity: a downward ray from above the active region hits
        // the grass top. Active y top is at planet-frame y=1.0
        // (active y=8 occupies [8/9, 9/9)). Camera at y=1.5 looking
        // straight down lands on grass at y=1.0.
        let mut lib = NodeLibrary::default();
        let planet = build_phase1_planet(&mut lib);
        let hit = cpu_raycast(&lib, planet, [1.0, 1.5, 0.1], [0.0, -1.0, 0.0], 4);
        assert!(hit.is_some(), "downward ray over active region should hit grass");
    }
}
