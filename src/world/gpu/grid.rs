//! Base-3 acceleration grid for the root Cartesian frame.
//!
//! A dense 3^GRID_DEPTH grid over `[0, 3)³` that acts as a pre-filter
//! for the ray-march shader. Each cell stores:
//!
//! - bit 7: occupied (any tree content lies in this cell's sub-region)
//! - bits 0-6: Chebyshev distance to the nearest occupied cell, in
//!   grid cells, clamped to 127
//!
//! The shader walks this grid first, using the distance field to
//! skip empty runs in a single `t +=` jump instead of cell-by-cell
//! DDA. Only when it enters an occupied cell does it hand off to the
//! tree walker (`march_cartesian`) — which still does the real
//! recursive descent to find actual geometry.
//!
//! Why base-3 and not base-2: the grid at depth G is exactly the tree
//! truncated to depth G. Bake reduces to a single tree walk with a
//! per-node depth check — no ray-box overlap math, no scale factors,
//! no coordinate conversions. Preserves the project's
//! every-layer-identical invariant.
//!
//! Size at GRID_DEPTH = 4: 3^4 = 81 per axis, 81³ = 531 441 cells.
//! Packed at 1 byte per cell into u32s, that's 132 861 u32s = 0.51 MB
//! — a fixed allocation independent of `plain_layers` or tree depth.
//!
//! Not rebuilt on every edit yet. Rebuilt on initial pack and on
//! `update_root` via the edit path — the edit path appends new nodes
//! for the edit-path ancestors, so a full grid re-bake is
//! O(3^(3*GRID_DEPTH)) ≈ half a million operations, which takes under
//! a millisecond on any modern CPU. Fine for now; can amortize later
//! if edit rate becomes a bottleneck.

use crate::world::tree::{Child, NodeId, NodeLibrary};

/// Grid depth in the base-3 tree. Grid is 3^GRID_DEPTH cells per axis.
pub const GRID_DEPTH: u32 = 4;

/// Cells per axis (3^GRID_DEPTH).
pub const GRID_DIM: u32 = 81; // 3^4

/// Total grid cells.
pub const GRID_SIZE: usize = (GRID_DIM as usize).pow(3);

/// Packed storage size in u32s (4 cells per u32, rounded up).
/// 81³ = 531441 isn't divisible by 4 so the last u32 carries a partial
/// load with zero-valued trailing cells.
pub const GRID_U32_COUNT: usize = (GRID_SIZE + 3) / 4;

/// Maximum representable distance-field value (7-bit field).
const DF_MAX: u8 = 127;

/// Bake the acceleration grid from the tree.
///
/// Returns a `Vec<u32>` of length `GRID_U32_COUNT` suitable for
/// upload to a GPU storage buffer. Each u32 packs 4 cells (8 bits
/// each), little-endian within the u32: cell `idx` lives at
/// `buffer[idx / 4]` bits `(idx % 4) * 8 .. (idx % 4) * 8 + 8`.
pub fn bake_grid(library: &NodeLibrary, root: NodeId) -> Vec<u32> {
    let mut occ = vec![false; GRID_SIZE];
    mark_occupancy(library, root, 0, 0, 0, 0, &mut occ);
    let df = chebyshev_distance_transform(&occ);
    pack(&occ, &df)
}

/// Recursively mark grid cells covered by tree content. A cell is
/// marked occupied if any non-empty content lies in its sub-region.
fn mark_occupancy(
    library: &NodeLibrary,
    nid: NodeId,
    cur_depth: u32,
    x: u32,
    y: u32,
    z: u32,
    occ: &mut [bool],
) {
    let Some(node) = library.get(nid) else { return; };
    // Early-out: uniform-empty subtrees can't contribute anything.
    if node.representative_block == 255 {
        return;
    }
    // Size of this node's footprint in grid cells.
    let span = 3u32.pow(GRID_DEPTH.saturating_sub(cur_depth));
    if cur_depth == GRID_DEPTH {
        // This tree node is exactly one grid cell. Mark it.
        let idx = (z * GRID_DIM * GRID_DIM + y * GRID_DIM + x) as usize;
        occ[idx] = true;
        return;
    }
    // Recurse into 27 children. Each child covers span/3 cells per axis.
    let child_span = span / 3;
    for slot in 0..27usize {
        let sx = (slot % 3) as u32;
        let sy = ((slot / 3) % 3) as u32;
        let sz = (slot / 9) as u32;
        let cx = x + sx * child_span;
        let cy = y + sy * child_span;
        let cz = z + sz * child_span;
        match node.children[slot] {
            Child::Empty => {}
            Child::Block(_) => {
                // Fill this child's grid footprint solid.
                fill_region(occ, cx, cy, cz, child_span);
            }
            Child::Node(child_nid) => {
                mark_occupancy(
                    library, child_nid, cur_depth + 1,
                    cx, cy, cz, occ,
                );
            }
        }
    }
}

/// Fill a cubic region of the grid with `true`.
fn fill_region(occ: &mut [bool], x0: u32, y0: u32, z0: u32, span: u32) {
    for dz in 0..span {
        for dy in 0..span {
            let row_base = ((z0 + dz) * GRID_DIM * GRID_DIM + (y0 + dy) * GRID_DIM + x0) as usize;
            for dx in 0..span {
                occ[row_base + dx as usize] = true;
            }
        }
    }
}

/// 3-pass separable Chebyshev (L∞) distance transform.
///
/// Each pass sweeps one axis forward-then-backward, propagating the
/// per-axis "distance to nearest occupied cell" minus 1 along that
/// axis. For Chebyshev, the final value is `min` across axes of the
/// per-axis results — which is exactly what three independent 1D
/// passes produce when combined with a per-axis `min` carry.
///
/// Occupied cells get DF = 0. Empty cells get the count of cells one
/// must step (in the worst axis direction) before hitting an
/// occupied cell.
fn chebyshev_distance_transform(occ: &[bool]) -> Vec<u8> {
    let n = GRID_DIM as usize;
    let mut df = vec![DF_MAX; GRID_SIZE];
    for (i, &o) in occ.iter().enumerate() {
        if o {
            df[i] = 0;
        }
    }
    // X-axis passes.
    for z in 0..n {
        for y in 0..n {
            let row = z * n * n + y * n;
            // Forward.
            let mut running = DF_MAX;
            for x in 0..n {
                let v = df[row + x];
                running = running.saturating_add(1).min(v);
                df[row + x] = running;
            }
            // Backward.
            running = DF_MAX;
            for x in (0..n).rev() {
                let v = df[row + x];
                running = running.saturating_add(1).min(v);
                df[row + x] = running;
            }
        }
    }
    // Y-axis passes.
    for z in 0..n {
        for x in 0..n {
            let col_stride = n;
            let plane = z * n * n;
            // Forward.
            let mut running = DF_MAX;
            for y in 0..n {
                let idx = plane + y * col_stride + x;
                let v = df[idx];
                running = running.saturating_add(1).min(v);
                df[idx] = running;
            }
            // Backward.
            running = DF_MAX;
            for y in (0..n).rev() {
                let idx = plane + y * col_stride + x;
                let v = df[idx];
                running = running.saturating_add(1).min(v);
                df[idx] = running;
            }
        }
    }
    // Z-axis passes.
    for y in 0..n {
        for x in 0..n {
            let plane_stride = n * n;
            // Forward.
            let mut running = DF_MAX;
            for z in 0..n {
                let idx = z * plane_stride + y * n + x;
                let v = df[idx];
                running = running.saturating_add(1).min(v);
                df[idx] = running;
            }
            // Backward.
            running = DF_MAX;
            for z in (0..n).rev() {
                let idx = z * plane_stride + y * n + x;
                let v = df[idx];
                running = running.saturating_add(1).min(v);
                df[idx] = running;
            }
        }
    }
    df
}

/// Pack per-cell (occupied, df) into u32s, 4 cells per u32.
fn pack(occ: &[bool], df: &[u8]) -> Vec<u32> {
    let mut out = vec![0u32; GRID_U32_COUNT];
    for i in 0..GRID_SIZE {
        let byte = (if occ[i] { 0x80 } else { 0 }) | (df[i] & 0x7F);
        let word = i >> 2;
        let shift = ((i & 3) * 8) as u32;
        out[word] |= (byte as u32) << shift;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::bootstrap::{bootstrap_world, WorldPreset};

    /// Unpack one grid cell from the packed representation.
    fn cell(buf: &[u32], x: u32, y: u32, z: u32) -> (bool, u8) {
        let idx = (z * GRID_DIM * GRID_DIM + y * GRID_DIM + x) as usize;
        let word = buf[idx >> 2];
        let byte = ((word >> ((idx & 3) * 8)) & 0xFF) as u8;
        ((byte & 0x80) != 0, byte & 0x7F)
    }

    #[test]
    fn grid_sizes() {
        assert_eq!(GRID_DIM, 81);
        assert_eq!(GRID_SIZE, 81 * 81 * 81);
        assert_eq!(GRID_U32_COUNT, (GRID_SIZE + 3) / 4);
        // 531441 odd → 132861 u32s with one unused slot.
        assert_eq!(GRID_U32_COUNT, 132861);
    }

    #[test]
    fn empty_world_has_no_occupied_cells() {
        let mut lib = NodeLibrary::default();
        let root = lib.insert(crate::world::tree::empty_children());
        lib.ref_inc(root);
        let grid = bake_grid(&lib, root);
        for z in 0..GRID_DIM {
            for y in 0..GRID_DIM {
                for x in 0..GRID_DIM {
                    let (occ, df) = cell(&grid, x, y, z);
                    assert!(!occ, "empty world should have no occupied cells");
                    assert_eq!(df, DF_MAX, "empty world should have saturated df");
                }
            }
        }
    }

    #[test]
    fn jerusalem_cross_has_sparse_occupancy() {
        let boot = bootstrap_world(WorldPreset::JerusalemCross, Some(20));
        let grid = bake_grid(&boot.world.library, boot.world.root);
        let mut occupied = 0usize;
        let mut total = 0usize;
        for z in 0..GRID_DIM {
            for y in 0..GRID_DIM {
                for x in 0..GRID_DIM {
                    let (occ, _) = cell(&grid, x, y, z);
                    if occ { occupied += 1; }
                    total += 1;
                }
            }
        }
        // Jerusalem at depth 4 should mark 7^4 = 2401 cells of 81^3.
        let ratio = occupied as f64 / total as f64;
        assert!(
            ratio < 0.02,
            "Jerusalem should be very sparse at grid depth 4; got {}/{} = {:.4}",
            occupied, total, ratio,
        );
        assert!(
            occupied >= 2000,
            "Jerusalem depth-4 should mark ~2401 cells; got {}",
            occupied,
        );
    }

    #[test]
    fn df_zero_at_occupied_cells() {
        let boot = bootstrap_world(WorldPreset::JerusalemCross, Some(20));
        let grid = bake_grid(&boot.world.library, boot.world.root);
        let mut checked = 0usize;
        for z in 0..GRID_DIM {
            for y in 0..GRID_DIM {
                for x in 0..GRID_DIM {
                    let (occ, df) = cell(&grid, x, y, z);
                    if occ {
                        assert_eq!(df, 0, "occupied cell must have df=0 ({x},{y},{z})");
                        checked += 1;
                    }
                }
            }
        }
        assert!(checked > 0, "no occupied cells found in Jerusalem grid");
    }

    #[test]
    fn df_propagates_monotonically() {
        // Single occupied cell at origin. DF at (k, 0, 0) should be k
        // for k up to DF_MAX.
        let mut lib = NodeLibrary::default();
        // Build a tree with a single solid cell at (0,0,0) at depth GRID_DEPTH.
        // Easiest: depth-GRID_DEPTH subtree with slot 0 = Block.
        let mut children = crate::world::tree::empty_children();
        children[0] = Child::Block(crate::world::palette::block::STONE);
        let leaf = lib.insert(children);
        // Build ancestor chain: each depth's slot 0 = the previous node.
        let mut cur = leaf;
        for _ in 0..GRID_DEPTH - 1 {
            let mut parents = crate::world::tree::empty_children();
            parents[0] = Child::Node(cur);
            cur = lib.insert(parents);
        }
        // Root (depth 0).
        let mut root_children = crate::world::tree::empty_children();
        root_children[0] = Child::Node(cur);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let grid = bake_grid(&lib, root);
        let (occ_origin, df_origin) = cell(&grid, 0, 0, 0);
        assert!(occ_origin, "origin cell should be occupied");
        assert_eq!(df_origin, 0);
        for k in 1..10u32 {
            let (_, df) = cell(&grid, k, 0, 0);
            assert_eq!(
                df as u32, k,
                "df at ({k}, 0, 0) should propagate monotonically",
            );
        }
    }
}
