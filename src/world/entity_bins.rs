//! Uniform hash grid over the current render frame's [0, WORLD_SIZE)³
//! for entity spatial indexing.
//!
//! Built CPU-side every frame from the entity GPU list; uploaded as
//! two storage buffers the shader can walk via a cheap 3D DDA.
//!
//! ## Layout
//!
//! Two parallel arrays, prefix-sum indexed:
//!
//! ```text
//! offsets: [res³ + 1]        offset into `entries` where bin i starts
//! entries: [sum_of_counts]   flat list of entity indices, grouped by bin
//!
//! count(bin i) = offsets[i + 1] - offsets[i]
//! entities in bin i = entries[offsets[i] .. offsets[i + 1]]
//! ```
//!
//! Prefix sums save a u32 per bin over a naive `(offset, count)`
//! layout and avoid having to special-case the last bin.
//!
//! ## Overlap insertion
//!
//! An entity whose bbox straddles multiple bins is registered in
//! EVERY bin its AABB touches. The shader may test the same entity
//! twice if the ray crosses two of its bins, but `min(t)` composition
//! makes duplicate tests a perf cost, not a correctness issue.
//!
//! ## Scope
//!
//! Only bins entities whose bbox overlaps [0, WORLD_SIZE)³. Entities
//! entirely outside the frame cell are dropped (they're off-screen
//! under the current Cartesian frame; ribbon-pop rays can't reach
//! them without reconstructing the bin grid in the ancestor frame —
//! a follow-up).

use crate::world::anchor::WORLD_SIZE;
use crate::world::gpu::GpuEntity;

/// Per-axis bin count. 32³ = 32768 bins. At sqrt(10_000) ≈ 100
/// entities/axis, 32 gives ~0.3 entities/bin — each ray tests
/// roughly that many entities per bin it crosses, vs. O(N) in the
/// brute-force path.
///
/// Keep this in sync with the shader-side `BIN_GRID_RES` constant
/// in `entities.wgsl`.
pub const BIN_GRID_RES: u32 = 32;

/// Built bin data, ready to upload to the GPU.
///
/// `offsets.len() == (BIN_GRID_RES.pow(3) + 1) as usize`.
/// `entries.len() == offsets.last().unwrap()` (the last prefix sum).
pub struct EntityBins {
    pub offsets: Vec<u32>,
    pub entries: Vec<u32>,
}

impl EntityBins {
    /// Empty scaffold: one offset of 0, no entries. Used when no
    /// entities are live (entity_count == 0 in the uniform gates
    /// the shader's bin-walk entirely).
    pub fn empty() -> Self {
        let cells = BIN_GRID_RES.pow(3) as usize;
        Self {
            offsets: vec![0u32; cells + 1],
            entries: Vec::new(),
        }
    }

    /// Build a bin grid from the entity list. Each entity is
    /// registered in every bin its bbox overlaps.
    ///
    /// O(N * avg_bins_per_entity) construction. With ~2-3 bins/axis
    /// per entity at our current scale (~5-20 bins each), 10k
    /// entities → ~100k insertions — on the order of 0.5 ms on CPU.
    pub fn build(entities: &[GpuEntity]) -> Self {
        let cells = BIN_GRID_RES.pow(3) as usize;
        let mut offsets = vec![0u32; cells + 1];

        if entities.is_empty() {
            return Self {
                offsets,
                entries: Vec::new(),
            };
        }

        let bin_size = WORLD_SIZE / BIN_GRID_RES as f32;
        let res = BIN_GRID_RES as i32;

        // Pass 1: count insertions per bin.
        let mut counts = vec![0u32; cells];
        for e in entities {
            let (raw_min, raw_max) = bin_range(e, bin_size, res);
            if !bin_range_in_grid(raw_min, raw_max, res) {
                continue;
            }
            let (b_min, b_max) = clamp_bin_range(raw_min, raw_max, res);
            for bz in b_min[2]..=b_max[2] {
                for by in b_min[1]..=b_max[1] {
                    for bx in b_min[0]..=b_max[0] {
                        let id = bin_id(bx, by, bz);
                        counts[id] += 1;
                    }
                }
            }
        }

        // Pass 2: prefix sum → offsets.
        let mut running = 0u32;
        for i in 0..cells {
            offsets[i] = running;
            running += counts[i];
        }
        offsets[cells] = running;

        // Pass 3: fill entries. Use `counts` as a write cursor per
        // bin, decrementing from the post-sum offset.
        let mut entries = vec![0u32; running as usize];
        let mut cursor = offsets.clone();
        for (e_idx, e) in entities.iter().enumerate() {
            let (raw_min, raw_max) = bin_range(e, bin_size, res);
            if !bin_range_in_grid(raw_min, raw_max, res) {
                continue;
            }
            let (b_min, b_max) = clamp_bin_range(raw_min, raw_max, res);
            for bz in b_min[2]..=b_max[2] {
                for by in b_min[1]..=b_max[1] {
                    for bx in b_min[0]..=b_max[0] {
                        let id = bin_id(bx, by, bz);
                        let slot = cursor[id] as usize;
                        entries[slot] = e_idx as u32;
                        cursor[id] += 1;
                    }
                }
            }
        }

        Self { offsets, entries }
    }
}

/// Compute the inclusive unclamped bin range `[min, max]` that
/// covers the entity's AABB. Callers must check overlap with the
/// grid via `bin_range_in_grid` BEFORE clamping — clamping first
/// would incorrectly pull entirely-outside entities into edge bins.
#[inline]
fn bin_range(e: &GpuEntity, bin_size: f32, _res: i32) -> ([i32; 3], [i32; 3]) {
    let min = [
        (e.bbox_min[0] / bin_size).floor() as i32,
        (e.bbox_min[1] / bin_size).floor() as i32,
        (e.bbox_min[2] / bin_size).floor() as i32,
    ];
    // Exclusive max from floor((max - eps) / bin_size); adding a
    // small epsilon avoids spurious +1 bin when bbox_max falls
    // exactly on a bin boundary.
    let max = [
        ((e.bbox_max[0] - 1e-6) / bin_size).floor() as i32,
        ((e.bbox_max[1] - 1e-6) / bin_size).floor() as i32,
        ((e.bbox_max[2] - 1e-6) / bin_size).floor() as i32,
    ];
    (min, max)
}

/// True when the entity's bin range overlaps the grid `[0, res)³`.
/// Uses UNCLAMPED bin indices so entirely-outside entities are
/// correctly rejected rather than snapping to an edge bin.
#[inline]
fn bin_range_in_grid(b_min: [i32; 3], b_max: [i32; 3], res: i32) -> bool {
    b_min[0] < res && b_max[0] >= 0 &&
        b_min[1] < res && b_max[1] >= 0 &&
        b_min[2] < res && b_max[2] >= 0
}

/// Clamp a bin range to the grid after `bin_range_in_grid` has
/// confirmed overlap. Overlapping range never produces an empty
/// post-clamp range.
#[inline]
fn clamp_bin_range(b_min: [i32; 3], b_max: [i32; 3], res: i32) -> ([i32; 3], [i32; 3]) {
    (
        [
            b_min[0].max(0),
            b_min[1].max(0),
            b_min[2].max(0),
        ],
        [
            b_max[0].min(res - 1),
            b_max[1].min(res - 1),
            b_max[2].min(res - 1),
        ],
    )
}

#[inline]
fn bin_id(x: i32, y: i32, z: i32) -> usize {
    let r = BIN_GRID_RES as i32;
    (x + y * r + z * r * r) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entity(min: [f32; 3], size: f32) -> GpuEntity {
        GpuEntity {
            bbox_min: min,
            representative_block: 0,
            bbox_max: [min[0] + size, min[1] + size, min[2] + size],
            subtree_bfs: 0,
        }
    }

    #[test]
    fn empty_list_produces_valid_scaffold() {
        let bins = EntityBins::build(&[]);
        assert_eq!(bins.offsets.len(), (BIN_GRID_RES.pow(3) + 1) as usize);
        assert!(bins.entries.is_empty());
        assert!(bins.offsets.iter().all(|&o| o == 0));
    }

    #[test]
    fn single_entity_in_one_bin() {
        let bin_size = WORLD_SIZE / BIN_GRID_RES as f32;
        let center = bin_size * 0.5;
        let e = entity([center - 0.01, center - 0.01, center - 0.01], 0.02);
        let bins = EntityBins::build(&[e]);
        let bin0 = bin_id(0, 0, 0);
        assert_eq!(bins.offsets[bin0 + 1] - bins.offsets[bin0], 1);
        assert_eq!(bins.entries[bins.offsets[bin0] as usize], 0);
    }

    #[test]
    fn spanning_entity_registered_in_multiple_bins() {
        let bin_size = WORLD_SIZE / BIN_GRID_RES as f32;
        // Entity straddling bin (0,0,0) and (1,0,0): bbox crosses x=bin_size.
        let e = entity([bin_size - 0.01, 0.01, 0.01], bin_size * 0.5);
        let bins = EntityBins::build(&[e]);
        let count_00 = bins.offsets[bin_id(0, 0, 0) + 1] - bins.offsets[bin_id(0, 0, 0)];
        let count_10 = bins.offsets[bin_id(1, 0, 0) + 1] - bins.offsets[bin_id(1, 0, 0)];
        assert_eq!(count_00, 1);
        assert_eq!(count_10, 1);
    }

    #[test]
    fn entity_outside_grid_dropped() {
        // bbox entirely outside [0, WORLD_SIZE)³ on the +x side.
        let e = entity([WORLD_SIZE + 1.0, 0.5, 0.5], 0.1);
        let bins = EntityBins::build(&[e]);
        assert!(bins.entries.is_empty());
    }

    #[test]
    fn prefix_sum_totals_match_entries_len() {
        // 10 entities scattered around the grid.
        let entities: Vec<GpuEntity> = (0..10)
            .map(|i| {
                let p = (i as f32) * 0.25;
                entity([p, p, p], 0.1)
            })
            .collect();
        let bins = EntityBins::build(&entities);
        let last = *bins.offsets.last().unwrap();
        assert_eq!(last as usize, bins.entries.len());
    }
}
