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
use crate::world::gpu::{GpuBinEntry, GpuEntity};

/// Per-axis bin count. 128³ = 2 097 152 bins. With ~10k entities in
/// a dense cluster (the motion_10000 test), 32³ left ~1000
/// entities/bin and the shader DDA paid O(~1000) AABB tests per
/// ray. 128³ puts typical dense clusters at ~10 entities/bin and
/// pushes shader cost roughly linearly lower; the extra bin visits
/// along the ray (128 vs 32) are cheap array lookups compared to
/// the AABB tests they save.
///
/// Keep this in sync with the shader-side `BIN_GRID_RES` constant
/// in `entities.wgsl`.
pub const BIN_GRID_RES: u32 = 64;

/// Cached bin data, rebuilt in place each frame. The owning
/// allocation is reused across frames to avoid zeroing the
/// `res³ + 1`-entry offsets buffer every frame (8 MB at res=128).
///
/// Call [`EntityBins::new`] once at startup, then
/// [`EntityBins::rebuild`] per frame to overwrite contents. The
/// public `offsets` / `entries` slices always reflect the most
/// recent `rebuild` (or the empty scaffold immediately after `new`).
pub struct EntityBins {
    pub offsets: Vec<u32>,
    /// Bin entries carry the entity's bbox INLINE alongside its
    /// index so the shader's per-bin AABB cull reads sequentially.
    /// See `GpuBinEntry` doc in `gpu::types` for the layout
    /// rationale.
    pub entries: Vec<GpuBinEntry>,
    /// Scratch buffer reused across rebuilds for the "write cursor"
    /// pass. Kept alongside `offsets` so we don't clone offsets on
    /// every rebuild.
    cursor: Vec<u32>,
}

impl EntityBins {
    /// Allocate the `res³ + 1` offsets array and empty entries list.
    /// Zero-initialized so it's a valid empty scaffold immediately.
    pub fn new() -> Self {
        let cells = BIN_GRID_RES.pow(3) as usize;
        Self {
            offsets: vec![0u32; cells + 1],
            entries: Vec::new(),
            cursor: vec![0u32; cells + 1],
        }
    }

    /// Convenience ctor for empty state without forcing the large
    /// allocation. Used by the no-entity fast path so we still have
    /// a valid `offsets`/`entries` pair to upload without paying
    /// 8 MB of allocation on an empty frame.
    pub fn empty() -> Self {
        let cells = BIN_GRID_RES.pow(3) as usize;
        Self {
            offsets: vec![0u32; cells + 1],
            entries: Vec::new(),
            cursor: Vec::new(),
        }
    }

    /// One-shot build used by tests. Allocates fresh vectors; hot
    /// paths call [`EntityBins::new`] once and [`rebuild`] every
    /// frame instead.
    pub fn build(entities: &[GpuEntity]) -> Self {
        let mut bins = Self::new();
        bins.rebuild(entities);
        bins
    }

    /// Rewrite `offsets` and `entries` from `entities`, reusing the
    /// existing allocations. Correctness is identical to a freshly
    /// built `EntityBins`; the only difference is no per-frame
    /// `res³`-entry zeroing allocation.
    pub fn rebuild(&mut self, entities: &[GpuEntity]) {
        let cells = BIN_GRID_RES.pow(3) as usize;
        debug_assert_eq!(self.offsets.len(), cells + 1);
        debug_assert_eq!(self.cursor.len(), cells + 1);

        if entities.is_empty() {
            // Zero the offsets (they may hold stale values from a
            // previous non-empty frame) and clear entries. Zeroing
            // 8 MB is still cheaper than allocating it.
            for o in self.offsets.iter_mut() { *o = 0; }
            self.entries.clear();
            return;
        }

        let bin_size = WORLD_SIZE / BIN_GRID_RES as f32;
        let res = BIN_GRID_RES as i32;

        // Pass 1: count insertions per bin. We'd like to write
        // directly into offsets[i] (the prefix-sum pass then shifts
        // those counts into starts), but that forces a two-vec
        // split for the cursor; easier to keep the semantics clear
        // by zeroing offsets first and treating it as the count
        // buffer through pass 2.
        for o in self.offsets.iter_mut() { *o = 0; }
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
                        self.offsets[id] += 1;
                    }
                }
            }
        }

        // Pass 2: in-place prefix sum. `offsets[i]` becomes the
        // start offset; `offsets[cells]` is the total.
        let mut running = 0u32;
        for i in 0..cells {
            let c = self.offsets[i];
            self.offsets[i] = running;
            self.cursor[i] = running;
            running += c;
        }
        self.offsets[cells] = running;
        self.cursor[cells] = running;

        // Pass 3: fill entries. Resize in place so the Vec's capacity
        // grows but never shrinks across frames. Each entry carries
        // the entity's bbox inline so the shader's AABB cull reads
        // sequentially.
        let total = running as usize;
        if self.entries.len() < total {
            self.entries.resize(total, GpuBinEntry::default());
        } else {
            self.entries.truncate(total);
        }
        for (e_idx, e) in entities.iter().enumerate() {
            let (raw_min, raw_max) = bin_range(e, bin_size, res);
            if !bin_range_in_grid(raw_min, raw_max, res) {
                continue;
            }
            let (b_min, b_max) = clamp_bin_range(raw_min, raw_max, res);
            let entry = GpuBinEntry {
                bbox_min: e.bbox_min,
                e_idx: e_idx as u32,
                bbox_max: e.bbox_max,
                _pad: 0,
            };
            for bz in b_min[2]..=b_max[2] {
                for by in b_min[1]..=b_max[1] {
                    for bx in b_min[0]..=b_max[0] {
                        let id = bin_id(bx, by, bz);
                        let slot = self.cursor[id] as usize;
                        self.entries[slot] = entry;
                        self.cursor[id] += 1;
                    }
                }
            }
        }
    }
}

impl Default for EntityBins {
    fn default() -> Self {
        Self::empty()
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
        let e = entity([center - 0.01 * bin_size, center - 0.01 * bin_size, center - 0.01 * bin_size], 0.02 * bin_size);
        let bins = EntityBins::build(&[e]);
        let bin0 = bin_id(0, 0, 0);
        assert_eq!(bins.offsets[bin0 + 1] - bins.offsets[bin0], 1);
        assert_eq!(bins.entries[bins.offsets[bin0] as usize].e_idx, 0);
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

    #[test]
    fn rebuild_reuses_allocation_and_matches_build() {
        // Build once, then rebuild with a different set and confirm
        // the resulting offsets/entries match a fresh build.
        let a: Vec<GpuEntity> = (0..50)
            .map(|i| entity([(i as f32) * 0.03, 0.4, 0.4], 0.08))
            .collect();
        let b: Vec<GpuEntity> = (0..100)
            .map(|i| entity([0.4, (i as f32) * 0.02, 0.4], 0.06))
            .collect();

        let mut pool = EntityBins::new();
        pool.rebuild(&a);
        pool.rebuild(&b);

        let fresh = EntityBins::build(&b);
        assert_eq!(pool.offsets, fresh.offsets);
        assert_eq!(pool.entries, fresh.entries);
    }

    #[test]
    fn rebuild_then_empty_clears_entries() {
        let a: Vec<GpuEntity> = (0..10)
            .map(|i| entity([(i as f32) * 0.1, 0.4, 0.4], 0.05))
            .collect();
        let mut pool = EntityBins::new();
        pool.rebuild(&a);
        assert!(!pool.entries.is_empty());
        pool.rebuild(&[]);
        assert!(pool.entries.is_empty());
        assert!(pool.offsets.iter().all(|&o| o == 0));
    }
}
