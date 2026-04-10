//! `WorldState` — the single source of truth at runtime. Owns the one
//! `FlatWorld`, the current zoom `depth`, and the set of super-chunks whose
//! cached level-2 bake is stale.
//!
//! All gameplay writes into the world should go through [`WorldState::edit_block`]
//! or [`WorldState::replace_chunk`] so the `MeshLibrary` refcounts stay correct.

use std::collections::HashSet;

use bevy::prelude::*;

use crate::block::BlockType;

use super::chunk::{Chunk, FlatWorld, S, SUPER};
use super::collision;
use super::library::MeshLibrary;

/// Maximum drill depth. 0 = most zoomed out, `MAX_DEPTH` = most zoomed in.
pub const MAX_DEPTH: usize = 2;

#[derive(Resource, Default)]
pub struct WorldState {
    pub world: FlatWorld,
    pub depth: usize,
    /// Super-chunk keys whose cached level-2 mesh may be stale and need a
    /// re-lookup on the next render pass.
    pub dirty_supers: HashSet<IVec3>,
}

impl collision::SolidQuery for WorldState {
    fn is_solid(&self, coord: IVec3) -> bool {
        match self.depth {
            0 => self.world.super_chunk_solid(coord),
            1 => self.world.chunk_solid(coord),
            _ => self.world.is_solid(coord),
        }
    }
}

impl WorldState {
    pub fn is_top_layer(&self) -> bool {
        self.depth == 0
    }

    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Zoom in by one level. The world is untouched — only the player's
    /// bevy-units scale so they stay over the same integer-block location.
    pub fn drill_in(&mut self, player_pos: Vec3) -> Option<Vec3> {
        if self.depth >= MAX_DEPTH {
            return None;
        }
        self.depth += 1;
        Some(player_pos * S as f32)
    }

    /// Zoom out by one level.
    pub fn drill_out(&mut self, player_pos: Vec3) -> Option<Vec3> {
        if self.depth == 0 {
            return None;
        }
        self.depth -= 1;
        Some(player_pos / S as f32)
    }

    pub fn dirty_super_for_block(&mut self, block_coord: IVec3) {
        self.dirty_supers.insert(IVec3::new(
            block_coord.x.div_euclid(SUPER),
            block_coord.y.div_euclid(SUPER),
            block_coord.z.div_euclid(SUPER),
        ));
    }

    pub fn dirty_super_for_chunk(&mut self, chunk_key: IVec3) {
        self.dirty_supers.insert(IVec3::new(
            chunk_key.x.div_euclid(S),
            chunk_key.y.div_euclid(S),
            chunk_key.z.div_euclid(S),
        ));
    }

    /// Edit a single block, decrementing the containing chunk's old library
    /// entry so refcounts stay correct. Marks the owning super-chunk dirty.
    pub fn edit_block(
        &mut self,
        coord: IVec3,
        block: Option<BlockType>,
        library: &mut MeshLibrary,
    ) {
        let key = IVec3::new(
            coord.x.div_euclid(S),
            coord.y.div_euclid(S),
            coord.z.div_euclid(S),
        );
        if let Some(chunk) = self.world.chunks.get_mut(&key) {
            if let Some(id) = chunk.level1_id.take() {
                library.level1_decrement(id);
            }
        }
        self.world.set(coord, block);
        self.dirty_super_for_block(coord);
    }

    /// Insert (or replace) a whole chunk. If a chunk already lived at `key`
    /// its level-1 refcount is decremented before we drop it.
    pub fn replace_chunk(
        &mut self,
        key: IVec3,
        chunk: Chunk,
        library: &mut MeshLibrary,
    ) {
        if let Some(old) = self.world.insert_chunk(key, chunk) {
            if let Some(id) = old.level1_id {
                library.level1_decrement(id);
            }
        }
        self.dirty_super_for_chunk(key);
    }
}
