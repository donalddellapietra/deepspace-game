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

/// Maximum drill depth. 0 = most zoomed out (super-super-chunks visible),
/// `MAX_DEPTH` = most zoomed in (individual blocks). Each step between
/// depths is a factor of `S` in linear scale.
pub const MAX_DEPTH: usize = 3;

#[derive(Resource, Default)]
pub struct WorldState {
    pub world: FlatWorld,
    pub depth: usize,
    /// Chunk keys whose content may have changed since the last render.
    /// Each render pass derives its own per-level dirty set from this
    /// (e.g. `render_super_chunks` divides by `S`, `render_super_super_chunks`
    /// divides by `SUPER`) and uses it to skip re-computing keys for
    /// entities that are already rendered and haven't been touched. Cleared
    /// at the end of `render_world`.
    pub dirty_chunks: HashSet<IVec3>,
}

impl collision::SolidQuery for WorldState {
    fn is_solid(&self, coord: IVec3) -> bool {
        match self.depth {
            0 => self.world.super_super_chunk_solid(coord),
            1 => self.world.super_chunk_solid(coord),
            2 => self.world.chunk_solid(coord),
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
    ///
    /// After the scale change, the player's new `y` may land inside a
    /// solid bevy-block (e.g. depth-1 `y=2` divided by `S=5` becomes `y=0.4`
    /// which is inside the depth-0 ground cube spanning `0..1`). The swept-
    /// AABB collision can't resolve "already overlapping," so we push the
    /// player up to sit on top of whatever cell their feet landed in.
    pub fn drill_out(&mut self, player_pos: Vec3) -> Option<Vec3> {
        if self.depth == 0 {
            return None;
        }
        self.depth -= 1;
        use collision::SolidQuery;
        let mut new_pos = player_pos / S as f32;
        // Bound the push-up to a few cells so a pathological world can't
        // loop forever.
        for _ in 0..8 {
            let cell = IVec3::new(
                new_pos.x.floor() as i32,
                new_pos.y.floor() as i32,
                new_pos.z.floor() as i32,
            );
            if !SolidQuery::is_solid(self, cell) {
                break;
            }
            new_pos.y = (cell.y + 1) as f32 + 0.001;
        }
        Some(new_pos)
    }

    /// Mark a chunk as dirty so the next render re-derives whatever level
    /// entity contains it.
    pub fn dirty_chunk(&mut self, chunk_key: IVec3) {
        self.dirty_chunks.insert(chunk_key);
    }

    /// Convenience: mark the chunk that owns `block_coord` as dirty.
    pub fn dirty_chunk_for_block(&mut self, block_coord: IVec3) {
        self.dirty_chunks.insert(IVec3::new(
            block_coord.x.div_euclid(S),
            block_coord.y.div_euclid(S),
            block_coord.z.div_euclid(S),
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
        self.dirty_chunk_for_block(coord);
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
        self.dirty_chunk(key);
    }
}
