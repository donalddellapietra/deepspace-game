//! Procedural generation + streaming. Pure `generate_chunk` is a deterministic
//! function of the chunk coord; `generate_terrain` is the Bevy system that
//! keeps the world populated around the player.

use bevy::prelude::*;

use crate::block::{BlockType, MODEL_SIZE};
use crate::player::Player;

use super::chunk::{Chunk, S, SUPER};
use super::state::WorldState;

const WORLD_SEED: u64 = 0xDEAD_BEEF_F00D_CAFE;

/// Stream radius in chunks. Must cover the largest per-depth view radius.
/// Depth 0 at rd=12 wants 12 SSS × 25 chunks/SSS = 300 chunks, plus margin.
const STREAM_RADIUS_CHUNKS: i32 = 12 * SUPER + S;

#[derive(Resource, Default)]
pub struct StreamState {
    /// (cx, cz) of the chunk column the player was over on the last stream
    /// pass. We only re-stream when this changes.
    pub last_column: Option<IVec2>,
}

/// Deterministic per-chunk content. Pure function of the chunk coord.
pub fn generate_chunk(coord: IVec3) -> Option<Chunk> {
    use BlockType::*;
    match coord.y {
        0 | 1 => Some(Chunk::new_filled(Stone)),
        2 | 3 => Some(Chunk::new_filled(Dirt)),
        4 => Some(Chunk::new_filled(Grass)),
        5 => {
            // Sparse layer above the grass: one rock per 5×5 column, laid out
            // by the chunk's LOCAL position within its super-chunk so every
            // rock super-chunk worldwide collapses to one level-2 library
            // entry.
            let mut chunk = Chunk::new_empty();
            let (rx, rz) = rock_position(coord.x, coord.z);
            chunk.blocks[0][rz][rx] = Some(Stone);
            Some(chunk)
        }
        _ => None,
    }
}

/// Rock position is hashed from the chunk's LOCAL coords within its
/// super-chunk (0..5), not its absolute world coords. This is what lets the
/// mesh library collapse every rock super-chunk worldwide into a single
/// level-2 entry: the 5×5 rock layout inside a super-chunk is identical
/// everywhere, so every level-2 key matches.
fn rock_position(cx: i32, cz: i32) -> (usize, usize) {
    let local_x = cx.rem_euclid(S) as u64;
    let local_z = cz.rem_euclid(S) as u64;
    let h = WORLD_SEED
        ^ local_x.wrapping_mul(0x9E3779B97F4A7C15)
        ^ local_z.wrapping_mul(0xBF58476D1CE4E5B9);
    let rx = (h % MODEL_SIZE as u64) as usize;
    let rz = ((h / MODEL_SIZE as u64) % MODEL_SIZE as u64) as usize;
    (rx, rz)
}

/// Delta-streamed terrain generator. Only generates the new edge when the
/// player's chunk column changes — skips all cells that were in the previous
/// window.
pub fn generate_terrain(
    mut stream: ResMut<StreamState>,
    mut state: ResMut<WorldState>,
    player_q: Query<&Transform, With<Player>>,
) {
    let Ok(tf) = player_q.single() else { return };

    // Convert bevy-space position to integer-block coords, depth-aware.
    // At depth N, 1 bevy unit = S^(MAX_DEPTH - N) integer blocks.
    let scale = match state.depth {
        0 => (S * S * S) as f32, // super-super-chunk: 125 blocks
        1 => (S * S) as f32,     // super-chunk:       25 blocks
        2 => S as f32,           // chunk:              5 blocks
        _ => 1.0,                // block:              1 block
    };
    let int_x = (tf.translation.x * scale).floor() as i32;
    let int_z = (tf.translation.z * scale).floor() as i32;
    let cx_player = int_x.div_euclid(S);
    let cz_player = int_z.div_euclid(S);
    let new_column = IVec2::new(cx_player, cz_player);

    let prev = stream.last_column;
    if prev == Some(new_column) {
        return;
    }
    stream.last_column = Some(new_column);

    let r = STREAM_RADIUS_CHUNKS;

    for dz in -r..=r {
        for dx in -r..=r {
            let cx = new_column.x + dx;
            let cz = new_column.y + dz;

            // Delta: skip anything that was already in the previous window.
            if let Some(p) = prev {
                if (cx - p.x).abs() <= r && (cz - p.y).abs() <= r {
                    continue;
                }
            }

            for cy in 0..=5 {
                let key = IVec3::new(cx, cy, cz);
                if state.world.chunks.contains_key(&key) {
                    continue; // user_modified / tombstoned / already generated
                }
                let Some(chunk) = generate_chunk(key) else { continue };
                state.world.insert_chunk(key, chunk);
                state.dirty_chunk(key);
            }
        }
    }
}
