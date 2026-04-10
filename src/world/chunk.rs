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
    /// cull before descending into each bucket's chunk list.
    pub fn chunks_in_sphere(&self, center: IVec3, radius: i32) -> Vec<IVec3> {
        let radius_sq = radius * radius;
        // A bucket can contribute only if its center is within
        // `radius + bucket diagonal` of the query center.
        let bucket_diag = ((CHUNK_INDEX_BUCKET as f32) * 1.7321) as i32 + 1;
        let loose = radius + bucket_diag;
        let loose_sq = loose * loose;
        let bucket_half = CHUNK_INDEX_BUCKET / 2;

        let mut result = Vec::new();
        for (&bk, chunk_keys) in &self.buckets {
            let bucket_center = bk * CHUNK_INDEX_BUCKET + IVec3::splat(bucket_half);
            let db = bucket_center - center;
            if db.x * db.x + db.y * db.y + db.z * db.z > loose_sq {
                continue;
            }
            for &chunk_key in chunk_keys {
                let d = chunk_key - center;
                if d.x * d.x + d.y * d.y + d.z * d.z <= radius_sq {
                    result.push(chunk_key);
                }
            }
        }
        result
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
