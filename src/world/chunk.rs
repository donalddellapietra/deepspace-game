//! Core voxel data: one `Chunk` = 5³ blocks, one `FlatWorld` = the single
//! integer-block grid every depth shares.

use std::collections::HashMap;

use bevy::prelude::*;

use crate::block::{BlockType, MODEL_SIZE};

/// Chunk side length in integer blocks.
pub const S: i32 = MODEL_SIZE as i32;
/// Super-chunk side length in integer blocks (5 chunks × 5 blocks = 25).
pub const SUPER: i32 = S * S;

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

/// The single integer-block grid. Every depth is a view over this.
#[derive(Clone, Default)]
pub struct FlatWorld {
    pub chunks: HashMap<IVec3, Chunk>,
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
        let chunk = self.chunks.entry(key).or_insert_with(Chunk::new_empty);
        chunk.blocks[local.y as usize][local.z as usize][local.x as usize] = block;
        chunk.mesh_dirty = true;
        chunk.level1_id = None;
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
