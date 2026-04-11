//! Core voxel data: one `Chunk` = 5³ blocks, one `FlatWorld` = the single
//! integer-block grid every depth shares. `FlatWorld` also owns a spatial
//! hash (`ChunkIndex`) of its chunk keys so the render system can do range
//! queries in `O(|visible|)` instead of scanning every chunk each frame.

use std::collections::{HashMap, HashSet};

use bevy::prelude::*;

use crate::block::{BlockType, MODEL_SIZE};

/// Chunk side length in integer blocks.
pub const S: i32 = MODEL_SIZE as i32;
/// Super-chunk side length in integer blocks (5 chunks × 5 blocks = 25).
pub const SUPER: i32 = S * S;

/// Spatial bucket side length for [`ChunkIndex`], in chunks per axis.
/// Arbitrary — not tied to `MODEL_SIZE` or any zoom level — it's only a
/// tuning knob balancing bucket-count vs chunks-per-bucket during queries.
const CHUNK_INDEX_BUCKET: i32 = 8;

#[derive(Clone)]
pub struct Chunk {
    pub blocks: [[[Option<BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
    /// `true` if a player edit has touched this chunk. The streaming
    /// generator must never overwrite a user_modified chunk, and removed
    /// cells are kept as empty `user_modified=true` chunks (tombstones).
    pub user_modified: bool,
    /// Flipped to `true` when the chunk's content has changed since the last
    /// render pass. Signals `ensure_level1_id` to re-resolve the library.
    pub mesh_dirty: bool,
    /// Cached `MeshLibrary` level-1 id. `None` = not yet resolved.
    pub level1_id: Option<u64>,
}

impl Chunk {
    pub fn new_empty() -> Self {
        Self {
            blocks: [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
            user_modified: false,
            mesh_dirty: true,
            level1_id: None,
        }
    }

    pub fn new_filled(bt: BlockType) -> Self {
        Self {
            blocks: [[[Some(bt); MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
            user_modified: false,
            mesh_dirty: true,
            level1_id: None,
        }
    }

    /// An empty chunk the streaming generator must leave alone.
    pub fn tombstone() -> Self {
        let mut c = Self::new_empty();
        c.user_modified = true;
        c
    }
}

/// Generic spatial hash over chunk keys. Buckets are `CHUNK_INDEX_BUCKET`
/// chunks wide on each axis and hold a set of chunk keys. Range queries walk
/// only the populated buckets (pure spatial optimization — the bucket size
/// is arbitrary and not tied to any zoom level).
#[derive(Clone, Default)]
pub struct ChunkIndex {
    buckets: HashMap<IVec3, HashSet<IVec3>>,
}

impl ChunkIndex {
    pub fn add(&mut self, chunk_key: IVec3) {
        self.buckets
            .entry(Self::bucket(chunk_key))
            .or_default()
            .insert(chunk_key);
    }

    pub fn remove(&mut self, chunk_key: IVec3) {
        let bk = Self::bucket(chunk_key);
        if let Some(bucket) = self.buckets.get_mut(&bk) {
            bucket.remove(&chunk_key);
            if bucket.is_empty() {
                self.buckets.remove(&bk);
            }
        }
    }

    /// Return every chunk key within squared euclidean `radius` chunks of
    /// `center`. Iterates populated buckets only, with a loose-radius bucket
    /// cull before descending into each bucket's chunk list. Squared-distance
    /// math is done in `i64` so wildly out-of-range queries (e.g. a player
    /// that fell a long way) can't overflow.
    pub fn chunks_in_sphere(&self, center: IVec3, radius: i32) -> Vec<IVec3> {
        let radius_sq: i64 = (radius as i64) * (radius as i64);
        let bucket_diag = ((CHUNK_INDEX_BUCKET as f32) * 1.7321) as i32 + 1;
        let loose = (radius as i64) + (bucket_diag as i64);
        let loose_sq: i64 = loose * loose;
        let bucket_half = CHUNK_INDEX_BUCKET / 2;

        let mut result = Vec::new();
        for (&bk, chunk_keys) in &self.buckets {
            let bucket_center = bk * CHUNK_INDEX_BUCKET + IVec3::splat(bucket_half);
            let dbx = (bucket_center.x - center.x) as i64;
            let dby = (bucket_center.y - center.y) as i64;
            let dbz = (bucket_center.z - center.z) as i64;
            if dbx * dbx + dby * dby + dbz * dbz > loose_sq {
                continue;
            }
            for &chunk_key in chunk_keys {
                let dx = (chunk_key.x - center.x) as i64;
                let dy = (chunk_key.y - center.y) as i64;
                let dz = (chunk_key.z - center.z) as i64;
                if dx * dx + dy * dy + dz * dz <= radius_sq {
                    result.push(chunk_key);
                }
            }
        }
        result
    }

    /// Return every chunk key that falls inside the inclusive cube
    /// `[min, max_incl]`. Walks the bucket range overlapping the cube and
    /// filters each bucket's chunk list by exact containment. Faster than
    /// `chunks_in_sphere` for axis-aligned queries and a good fit for the
    /// `compute_levelN_key` path, which needs all existing chunks in one
    /// super-chunk or super-super-chunk-sized region.
    pub fn chunks_in_cube(&self, min: IVec3, max_incl: IVec3) -> Vec<IVec3> {
        let bsize = CHUNK_INDEX_BUCKET;
        let b_min = IVec3::new(
            min.x.div_euclid(bsize),
            min.y.div_euclid(bsize),
            min.z.div_euclid(bsize),
        );
        let b_max = IVec3::new(
            max_incl.x.div_euclid(bsize),
            max_incl.y.div_euclid(bsize),
            max_incl.z.div_euclid(bsize),
        );

        let mut result = Vec::new();
        for bz in b_min.z..=b_max.z {
            for by in b_min.y..=b_max.y {
                for bx in b_min.x..=b_max.x {
                    let bk = IVec3::new(bx, by, bz);
                    if let Some(bucket) = self.buckets.get(&bk) {
                        for &ck in bucket {
                            if ck.x >= min.x
                                && ck.x <= max_incl.x
                                && ck.y >= min.y
                                && ck.y <= max_incl.y
                                && ck.z >= min.z
                                && ck.z <= max_incl.z
                            {
                                result.push(ck);
                            }
                        }
                    }
                }
            }
        }
        result
    }

    /// Look up a single bucket's contents (used by FlatWorld for cube-range
    /// queries like super_chunk_solid / super_super_chunk_solid).
    pub fn bucket_at(&self, bucket_key: IVec3) -> Option<&HashSet<IVec3>> {
        self.buckets.get(&bucket_key)
    }

    /// Convert a chunk key to the bucket coord it lives in.
    pub fn bucket_of(chunk_key: IVec3) -> IVec3 {
        Self::bucket(chunk_key)
    }

    /// The bucket side length. Exposed so FlatWorld can walk a bucket range.
    pub const fn bucket_size() -> i32 {
        CHUNK_INDEX_BUCKET
    }

    fn bucket(chunk_key: IVec3) -> IVec3 {
        IVec3::new(
            chunk_key.x.div_euclid(CHUNK_INDEX_BUCKET),
            chunk_key.y.div_euclid(CHUNK_INDEX_BUCKET),
            chunk_key.z.div_euclid(CHUNK_INDEX_BUCKET),
        )
    }
}

/// The single integer-block grid. Every depth is a view over this.
#[derive(Clone, Default)]
pub struct FlatWorld {
    pub chunks: HashMap<IVec3, Chunk>,
    /// Spatial hash of chunk keys, kept in sync with `chunks` by
    /// `insert_chunk` / `remove_chunk` / `set`.
    pub index: ChunkIndex,
}

impl FlatWorld {
    pub fn get(&self, coord: IVec3) -> Option<BlockType> {
        let (key, local) = Self::decompose(coord);
        self.chunks.get(&key)?.blocks[local.y as usize][local.z as usize][local.x as usize]
    }

    /// Raw block write. **Tests only.** Game code must go through
    /// [`super::state::WorldState::edit_block`] so the mesh library's
    /// refcount stays correct.
    pub fn set(&mut self, coord: IVec3, block: Option<BlockType>) {
        let (key, local) = Self::decompose(coord);
        let was_new = !self.chunks.contains_key(&key);
        {
            let chunk = self.chunks.entry(key).or_insert_with(Chunk::new_empty);
            chunk.blocks[local.y as usize][local.z as usize][local.x as usize] = block;
            chunk.mesh_dirty = true;
            chunk.level1_id = None;
        }
        if was_new {
            self.index.add(key);
        }
    }

    /// Insert (or replace) a chunk, maintaining the spatial index. Returns
    /// the displaced chunk if one was present. All gameplay code should
    /// prefer this over `chunks.insert` directly.
    pub fn insert_chunk(&mut self, key: IVec3, chunk: Chunk) -> Option<Chunk> {
        let old = self.chunks.insert(key, chunk);
        if old.is_none() {
            self.index.add(key);
        }
        old
    }

    /// Remove a chunk, maintaining the spatial index.
    pub fn remove_chunk(&mut self, key: IVec3) -> Option<Chunk> {
        let old = self.chunks.remove(&key);
        if old.is_some() {
            self.index.remove(key);
        }
        old
    }

    pub fn is_solid(&self, coord: IVec3) -> bool {
        self.get(coord).is_some()
    }

    /// `true` iff chunk `key` exists and contains at least one block.
    pub fn chunk_solid(&self, key: IVec3) -> bool {
        self.chunks
            .get(&key)
            .is_some_and(|c| c.blocks.iter().flatten().flatten().any(|b| b.is_some()))
    }

    /// `true` iff any of the 5×5×5 chunks inside super-chunk `super_key` has
    /// at least one block.
    pub fn super_chunk_solid(&self, super_key: IVec3) -> bool {
        for cz in 0..S {
            for cy in 0..S {
                for cx in 0..S {
                    let chunk_key = super_key * S + IVec3::new(cx, cy, cz);
                    if self.chunk_solid(chunk_key) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// `true` iff any chunk inside super-super-chunk `sss_key` (25×25×25
    /// chunks) has at least one block. Goes through the spatial index so
    /// empty SSSes are a handful of bucket misses instead of 15k HashMap
    /// lookups.
    pub fn super_super_chunk_solid(&self, sss_key: IVec3) -> bool {
        let min = sss_key * SUPER;
        let max = min + IVec3::splat(SUPER - 1);
        let b_size = ChunkIndex::bucket_size();
        let b_min = IVec3::new(
            min.x.div_euclid(b_size),
            min.y.div_euclid(b_size),
            min.z.div_euclid(b_size),
        );
        let b_max = IVec3::new(
            max.x.div_euclid(b_size),
            max.y.div_euclid(b_size),
            max.z.div_euclid(b_size),
        );
        for bz in b_min.z..=b_max.z {
            for by in b_min.y..=b_max.y {
                for bx in b_min.x..=b_max.x {
                    let bk = IVec3::new(bx, by, bz);
                    if let Some(bucket) = self.index.bucket_at(bk) {
                        for &ck in bucket {
                            if ck.x >= min.x
                                && ck.x <= max.x
                                && ck.y >= min.y
                                && ck.y <= max.y
                                && ck.z >= min.z
                                && ck.z <= max.z
                                && self.chunk_solid(ck)
                            {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        false
    }

    fn decompose(coord: IVec3) -> (IVec3, IVec3) {
        let key = IVec3::new(
            coord.x.div_euclid(S),
            coord.y.div_euclid(S),
            coord.z.div_euclid(S),
        );
        let local = IVec3::new(
            coord.x.rem_euclid(S),
            coord.y.rem_euclid(S),
            coord.z.rem_euclid(S),
        );
        (key, local)
    }
}
