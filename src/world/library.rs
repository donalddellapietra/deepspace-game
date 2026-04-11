//! Content-addressed mesh cache with refcount + eviction.
//!
//! Every *unique* 5×5×5 chunk pattern produces one `Level1Entry`. Every
//! unique 5×5×5 layout of level-1 ids (a super-chunk) produces one
//! `Level2Entry`. Entries are refcounted: as soon as the last chunk or
//! rendered super-chunk stops pointing at an entry, it's evicted and its
//! mesh handles are dropped.
//!
//! For procedural terrain the library shrinks to a handful of entries
//! regardless of world size — that's the "infinite scaling" win.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::block::{BlockType, MODEL_SIZE};
use crate::model::mesher::{bake_model, bake_volume};
use crate::model::BakedSubMesh;

use super::chunk::{Chunk, FlatWorld, S, SUPER};

pub type Level1Key = [[[Option<BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
pub type Level2Key = [[[u64; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
/// Level-3 key = 5×5×5 arrangement of level-2 ids (child super-chunks of
/// a super-super-chunk). Same shape as Level2Key — the ids at level 3
/// address level-2 entries instead of level-1 entries.
pub type Level3Key = [[[u64; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];

/// Reserved id for "nothing to render here".
pub const EMPTY_ID: u64 = 0;

pub struct Level1Entry {
    pub content: Level1Key,
    pub baked: Vec<BakedSubMesh>,
    pub ref_count: u32,
}

pub struct Level2Entry {
    pub key: Level2Key,
    pub baked: Vec<BakedSubMesh>,
    pub ref_count: u32,
}

pub struct Level3Entry {
    pub key: Level3Key,
    pub baked: Vec<BakedSubMesh>,
    pub ref_count: u32,
}

#[derive(Resource)]
pub struct MeshLibrary {
    level1: HashMap<u64, Level1Entry>,
    level1_by_content: HashMap<Level1Key, u64>,
    level2: HashMap<u64, Level2Entry>,
    level2_by_key: HashMap<Level2Key, u64>,
    level3: HashMap<u64, Level3Entry>,
    level3_by_key: HashMap<Level3Key, u64>,
    next_id: u64,
}

impl Default for MeshLibrary {
    fn default() -> Self {
        Self {
            level1: HashMap::new(),
            level1_by_content: HashMap::new(),
            level2: HashMap::new(),
            level2_by_key: HashMap::new(),
            level3: HashMap::new(),
            level3_by_key: HashMap::new(),
            // 0 is reserved for "empty".
            next_id: 1,
        }
    }
}

impl MeshLibrary {
    // ------------------------------------------------------------------ level 1

    pub fn level1_lookup_or_bake(
        &mut self,
        content: &Level1Key,
        meshes: &mut Assets<Mesh>,
    ) -> u64 {
        if let Some(&id) = self.level1_by_content.get(content) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        let baked = bake_model(content, meshes);
        self.level1.insert(
            id,
            Level1Entry {
                content: *content,
                baked,
                ref_count: 0,
            },
        );
        self.level1_by_content.insert(*content, id);
        id
    }

    pub fn level1_get(&self, id: u64) -> Option<&Level1Entry> {
        self.level1.get(&id)
    }

    pub fn level1_increment(&mut self, id: u64) {
        if id == EMPTY_ID {
            return;
        }
        if let Some(entry) = self.level1.get_mut(&id) {
            entry.ref_count = entry.ref_count.saturating_add(1);
        }
    }

    pub fn level1_decrement(&mut self, id: u64) {
        if id == EMPTY_ID {
            return;
        }
        let should_evict = {
            let Some(entry) = self.level1.get_mut(&id) else {
                return;
            };
            entry.ref_count = entry.ref_count.saturating_sub(1);
            entry.ref_count == 0
        };
        if should_evict {
            if let Some(evicted) = self.level1.remove(&id) {
                self.level1_by_content.remove(&evicted.content);
            }
        }
    }

    // ------------------------------------------------------------------ level 2

    pub fn level2_lookup_or_bake(
        &mut self,
        key: Level2Key,
        world: &FlatWorld,
        super_key: IVec3,
        meshes: &mut Assets<Mesh>,
    ) -> u64 {
        if let Some(&id) = self.level2_by_key.get(&key) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        let baked = bake_super_chunk(world, super_key, meshes);
        self.level2.insert(
            id,
            Level2Entry {
                key,
                baked,
                ref_count: 0,
            },
        );
        self.level2_by_key.insert(key, id);
        id
    }

    pub fn level2_get(&self, id: u64) -> Option<&Level2Entry> {
        self.level2.get(&id)
    }

    pub fn level2_increment(&mut self, id: u64) {
        if id == EMPTY_ID {
            return;
        }
        if let Some(entry) = self.level2.get_mut(&id) {
            entry.ref_count = entry.ref_count.saturating_add(1);
        }
    }

    pub fn level2_decrement(&mut self, id: u64) {
        if id == EMPTY_ID {
            return;
        }
        let should_evict = {
            let Some(entry) = self.level2.get_mut(&id) else {
                return;
            };
            entry.ref_count = entry.ref_count.saturating_sub(1);
            entry.ref_count == 0
        };
        if should_evict {
            if let Some(evicted) = self.level2.remove(&id) {
                self.level2_by_key.remove(&evicted.key);
            }
        }
    }

    // ------------------------------------------------------------------ level 3

    pub fn level3_lookup_or_bake(
        &mut self,
        key: Level3Key,
        world: &FlatWorld,
        sss_key: IVec3,
        meshes: &mut Assets<Mesh>,
    ) -> u64 {
        if let Some(&id) = self.level3_by_key.get(&key) {
            return id;
        }
        let id = self.next_id;
        self.next_id += 1;
        let baked = bake_super_super_chunk(world, sss_key, meshes);
        self.level3.insert(
            id,
            Level3Entry {
                key,
                baked,
                ref_count: 0,
            },
        );
        self.level3_by_key.insert(key, id);
        id
    }

    pub fn level3_get(&self, id: u64) -> Option<&Level3Entry> {
        self.level3.get(&id)
    }

    pub fn level3_increment(&mut self, id: u64) {
        if id == EMPTY_ID {
            return;
        }
        if let Some(entry) = self.level3.get_mut(&id) {
            entry.ref_count = entry.ref_count.saturating_add(1);
        }
    }

    pub fn level3_decrement(&mut self, id: u64) {
        if id == EMPTY_ID {
            return;
        }
        let should_evict = {
            let Some(entry) = self.level3.get_mut(&id) else {
                return;
            };
            entry.ref_count = entry.ref_count.saturating_sub(1);
            entry.ref_count == 0
        };
        if should_evict {
            if let Some(evicted) = self.level3.remove(&id) {
                self.level3_by_key.remove(&evicted.key);
            }
        }
    }
}

/// Resolve a chunk's level-1 id, baking its mesh on miss and keeping the
/// library's refcount accurate through the transition.
pub fn ensure_level1_id(
    chunk: &mut Chunk,
    library: &mut MeshLibrary,
    meshes: &mut Assets<Mesh>,
) -> u64 {
    if let Some(id) = chunk.level1_id {
        if !chunk.mesh_dirty {
            return id;
        }
    }

    let cached_id = chunk.level1_id;

    let new_id = if chunk.blocks.iter().flatten().flatten().all(|b| b.is_none()) {
        EMPTY_ID
    } else {
        library.level1_lookup_or_bake(&chunk.blocks, meshes)
    };

    if cached_id != Some(new_id) {
        if let Some(old) = cached_id {
            library.level1_decrement(old);
        }
        library.level1_increment(new_id);
    }

    chunk.level1_id = Some(new_id);
    chunk.mesh_dirty = false;
    new_id
}

/// Walk the 125 children of a super-chunk, ensuring each has a fresh level-1
/// id, and return the resulting level-2 key.
pub fn compute_level2_key(
    world: &mut FlatWorld,
    super_key: IVec3,
    library: &mut MeshLibrary,
    meshes: &mut Assets<Mesh>,
) -> Level2Key {
    let origin = super_key * S;
    let mut child_ids: Level2Key = [[[EMPTY_ID; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
    for cz in 0..MODEL_SIZE {
        for cy in 0..MODEL_SIZE {
            for cx in 0..MODEL_SIZE {
                let chunk_key = origin + IVec3::new(cx as i32, cy as i32, cz as i32);
                if let Some(chunk) = world.chunks.get_mut(&chunk_key) {
                    child_ids[cz][cy][cx] = ensure_level1_id(chunk, library, meshes);
                }
            }
        }
    }
    child_ids
}

/// Build a super-super-chunk's level-3 key by a single spatial-index sweep
/// over its 25×25×25 chunk region.
///
/// The naive implementation ran `compute_level2_key` 125 times, each one
/// touching 125 chunk positions whether they existed or not — ~15,625
/// `HashMap::get_mut` calls per new SSS, ~1 ms on procedural ground. This
/// version asks the `ChunkIndex` for the existing chunks in the SSS cube
/// once, and fills all 125 level-2 keys in one pass. Empty regions above
/// the ground are skipped entirely.
pub fn compute_level3_key(
    world: &mut FlatWorld,
    sss_key: IVec3,
    library: &mut MeshLibrary,
    meshes: &mut Assets<Mesh>,
) -> Level3Key {
    // The SSS covers chunk keys [sss_min, sss_max] inclusive (SUPER = 25
    // chunks per axis).
    let sss_min = sss_key * SUPER;
    let sss_max = sss_min + IVec3::splat(SUPER - 1);

    // Stack-local 125-slot table of level-2 keys, indexed as
    // `level2_keys[sz][sy][sx]`. Each slot is itself a `[[[u64; 5]; 5]; 5]`
    // indexed `[lz][ly][lx]`. Total ~122 KB, comfortably on stack.
    let mut level2_keys: [[[Level2Key; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE] =
        [[[[[[EMPTY_ID; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
            MODEL_SIZE];

    // Single spatial query → only the chunks that actually exist, no empty
    // positions iterated.
    let existing = world.index.chunks_in_cube(sss_min, sss_max);

    for chunk_key in existing {
        let rel = chunk_key - sss_min; // 0..SUPER on each axis
        let sx = rel.x.div_euclid(S) as usize;
        let sy = rel.y.div_euclid(S) as usize;
        let sz = rel.z.div_euclid(S) as usize;
        let lx = rel.x.rem_euclid(S) as usize;
        let ly = rel.y.rem_euclid(S) as usize;
        let lz = rel.z.rem_euclid(S) as usize;

        if let Some(chunk) = world.chunks.get_mut(&chunk_key) {
            let id = ensure_level1_id(chunk, library, meshes);
            level2_keys[sz][sy][sx][lz][ly][lx] = id;
        }
    }

    // Resolve each of the 125 super-chunks to a level-2 id and assemble
    // the level-3 key.
    let mut level3_key: Level3Key = [[[EMPTY_ID; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
    for sz in 0..MODEL_SIZE {
        for sy in 0..MODEL_SIZE {
            for sx in 0..MODEL_SIZE {
                let l2_key = level2_keys[sz][sy][sx];
                let all_empty = l2_key
                    .iter()
                    .flatten()
                    .flatten()
                    .all(|&id| id == EMPTY_ID);
                let l2_id = if all_empty {
                    EMPTY_ID
                } else {
                    let super_key = IVec3::new(
                        sss_key.x * S + sx as i32,
                        sss_key.y * S + sy as i32,
                        sss_key.z * S + sz as i32,
                    );
                    library.level2_lookup_or_bake(l2_key, world, super_key, meshes)
                };
                level3_key[sz][sy][sx] = l2_id;
            }
        }
    }

    level3_key
}

/// Bake a super-super-chunk's 125³ integer grid into per-block-type
/// sub-meshes. Pre-fetches the 15,625 chunks once into a boxed flat array
/// so the bake closure is array indexing.
pub fn bake_super_super_chunk(
    world: &FlatWorld,
    sss_key: IVec3,
    meshes: &mut Assets<Mesh>,
) -> Vec<BakedSubMesh> {
    // 125 chunks per axis == SUPER per axis when measured in chunks, times
    // S blocks per chunk gives SUPER * S integer blocks per axis. SUPER = 25,
    // so total = 125.
    let chunk_span: i32 = SUPER; // 25 chunks per axis
    let total: i32 = SUPER * S; // 125 integer blocks per axis
    let chunk_origin = sss_key * chunk_span;

    // Pre-fetch 15,625 chunk refs into a boxed Vec so the closure is a
    // cheap array lookup. Heap-allocated to keep the stack safe.
    let span = chunk_span as usize;
    let mut chunk_refs: Vec<Option<&Chunk>> = vec![None; span * span * span];
    for cz in 0..chunk_span {
        for cy in 0..chunk_span {
            for cx in 0..chunk_span {
                let key = chunk_origin + IVec3::new(cx, cy, cz);
                let idx = (cz as usize * span + cy as usize) * span + cx as usize;
                chunk_refs[idx] = world.chunks.get(&key);
            }
        }
    }

    bake_volume(
        total,
        |x, y, z| {
            if x < 0 || y < 0 || z < 0 || x >= total || y >= total || z >= total {
                return None;
            }
            let cx = x / S;
            let cy = y / S;
            let cz = z / S;
            let lx = (x % S) as usize;
            let ly = (y % S) as usize;
            let lz = (z % S) as usize;
            let idx = (cz as usize * span + cy as usize) * span + cx as usize;
            chunk_refs[idx].and_then(|c| c.blocks[ly][lz][lx])
        },
        meshes,
    )
}

/// Bake a super-chunk's 25³ integer grid into per-block-type sub-meshes.
/// Pre-fetches the 125 chunks into a local array so the bake closure is
/// array indexing rather than a HashMap lookup per cell.
pub fn bake_super_chunk(
    world: &FlatWorld,
    super_key: IVec3,
    meshes: &mut Assets<Mesh>,
) -> Vec<BakedSubMesh> {
    let origin = super_key * S;
    let mut chunks: [[[Option<&Chunk>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE] =
        [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
    for cz in 0..MODEL_SIZE {
        for cy in 0..MODEL_SIZE {
            for cx in 0..MODEL_SIZE {
                let key = origin + IVec3::new(cx as i32, cy as i32, cz as i32);
                chunks[cz][cy][cx] = world.chunks.get(&key);
            }
        }
    }

    bake_volume(
        SUPER,
        |x, y, z| {
            if x < 0 || y < 0 || z < 0 || x >= SUPER || y >= SUPER || z >= SUPER {
                return None;
            }
            let cx = (x / S) as usize;
            let cy = (y / S) as usize;
            let cz = (z / S) as usize;
            let lx = (x % S) as usize;
            let ly = (y % S) as usize;
            let lz = (z % S) as usize;
            chunks[cz][cy][cx].and_then(|c| c.blocks[ly][lz][lx])
        },
        meshes,
    )
}
