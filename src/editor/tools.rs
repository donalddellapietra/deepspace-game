use bevy::prelude::*;

use crate::block::{BlockType, MODEL_SIZE};
use crate::camera::{CursorLocked, FpsCam};
use crate::interaction::TargetedBlock;
use crate::inventory::InventoryState;
use crate::model::mesher::bake_model;
use crate::model::{ModelRegistry, VoxelModel};
use crate::player::{Player, Velocity};
use crate::world::{Chunk, RenderState, WorldState, MAX_DEPTH, SUPER};

const S: i32 = MODEL_SIZE as i32;

// ============================================================
// Drill down / up — change zoom only, never modify world data
// ============================================================

pub fn drill_down(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    mut state: ResMut<WorldState>,
    mut rs: ResMut<RenderState>,
    mut player_q: Query<(&mut Transform, &mut Velocity), With<Player>>,
    mut cam_q: Query<&mut FpsCam>,
) {
    if inv.open {
        return;
    }
    if !keyboard.just_pressed(KeyCode::KeyF) {
        return;
    }
    if state.depth >= MAX_DEPTH {
        return;
    }
    let Ok((mut tf, mut vel)) = player_q.single_mut() else {
        return;
    };

    let Some(new_pos) = state.drill_in(tf.translation) else {
        return;
    };
    tf.translation = new_pos;
    vel.0 = Vec3::ZERO;
    if let Ok(mut cam) = cam_q.single_mut() {
        cam.yaw = 0.4;
        cam.pitch = 0.8;
    }
    rs.needs_full_refresh = true;
}

pub fn drill_up(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    mut state: ResMut<WorldState>,
    mut rs: ResMut<RenderState>,
    mut player_q: Query<(&mut Transform, &mut Velocity), With<Player>>,
) {
    if inv.open {
        return;
    }
    if !keyboard.just_pressed(KeyCode::KeyQ) {
        return;
    }
    if state.depth == 0 {
        return;
    }
    let Ok((mut tf, mut vel)) = player_q.single_mut() else {
        return;
    };

    let Some(new_pos) = state.drill_out(tf.translation) else {
        return;
    };
    tf.translation = new_pos;
    vel.0 = Vec3::ZERO;
    rs.needs_full_refresh = true;
}

// ============================================================
// Block editing — granularity follows the current depth
// ============================================================

pub fn remove_block(
    mouse: Res<ButtonInput<MouseButton>>,
    locked: Res<CursorLocked>,
    inv: Res<InventoryState>,
    targeted: Res<TargetedBlock>,
    mut state: ResMut<WorldState>,
) {
    if inv.open || !locked.0 {
        return;
    }
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let Some(hit) = targeted.hit else { return };

    match state.depth {
        0 => {
            // hit is a super-chunk key. Tombstone all 5×5×5 chunks beneath it
            // so the streaming generator never refills them.
            for cz in 0..S {
                for cy in 0..S {
                    for cx in 0..S {
                        let chunk_key = hit * S + IVec3::new(cx, cy, cz);
                        state.world.chunks.insert(chunk_key, Chunk::tombstone());
                    }
                }
            }
            state.dirty_supers.insert(hit);
        }
        1 => {
            // hit is a chunk key.
            state.world.chunks.insert(hit, Chunk::tombstone());
            state.dirty_super_for_chunk(hit);
        }
        _ => {
            // hit is an integer block coord.
            state.world.set(hit, None);
            // Mark the containing chunk user_modified so the generator will
            // never overwrite the user's hole.
            let key = IVec3::new(
                hit.x.div_euclid(S),
                hit.y.div_euclid(S),
                hit.z.div_euclid(S),
            );
            if let Some(chunk) = state.world.chunks.get_mut(&key) {
                chunk.user_modified = true;
            }
            state.dirty_super_for_block(hit);
        }
    }
}

pub fn place_block(
    mouse: Res<ButtonInput<MouseButton>>,
    locked: Res<CursorLocked>,
    inv: Res<InventoryState>,
    targeted: Res<TargetedBlock>,
    hotbar: Res<super::Hotbar>,
    registry: Res<ModelRegistry>,
    mut state: ResMut<WorldState>,
) {
    if inv.open || !locked.0 {
        return;
    }
    if !mouse.just_pressed(MouseButton::Right) {
        return;
    }
    let Some(hit) = targeted.hit else { return };
    let Some(normal) = targeted.normal else { return };

    let place = hit + normal;

    match state.depth {
        0 => {
            // place is a super-chunk key. Refuse if already populated.
            if state.world.super_chunk_solid(place) {
                return;
            }
            let bt = match hotbar.active_item() {
                super::HotbarItem::Block(bt) => *bt,
                super::HotbarItem::SavedModel(_) => BlockType::Stone, // models are too small at this granularity
            };
            let mut filled = Chunk::new_filled(bt);
            filled.user_modified = true;
            for cz in 0..S {
                for cy in 0..S {
                    for cx in 0..S {
                        let chunk_key = place * S + IVec3::new(cx, cy, cz);
                        state.world.chunks.insert(chunk_key, filled.clone());
                    }
                }
            }
            state.dirty_supers.insert(place);
        }
        1 => {
            if state.world.chunk_solid(place) {
                return;
            }
            let chunk = match hotbar.active_item() {
                super::HotbarItem::Block(bt) => {
                    let mut c = Chunk::new_filled(*bt);
                    c.user_modified = true;
                    c
                }
                super::HotbarItem::SavedModel(idx) => {
                    let Some(model) = registry.models.get(*idx) else {
                        return;
                    };
                    Chunk {
                        blocks: model.blocks,
                        mesh_dirty: true,
                        baked: vec![],
                        user_modified: true,
                    }
                }
            };
            state.world.chunks.insert(place, chunk);
            state.dirty_super_for_chunk(place);
        }
        _ => {
            if state.world.is_solid(place) {
                return;
            }
            match hotbar.active_item() {
                super::HotbarItem::Block(bt) => {
                    state.world.set(place, Some(*bt));
                    mark_block_user_modified(&mut state, place);
                }
                super::HotbarItem::SavedModel(idx) => {
                    let Some(model) = registry.models.get(*idx) else {
                        return;
                    };
                    let blocks = model.blocks;
                    for y in 0..MODEL_SIZE {
                        for z in 0..MODEL_SIZE {
                            for x in 0..MODEL_SIZE {
                                if let Some(bt) = blocks[y][z][x] {
                                    let coord =
                                        place + IVec3::new(x as i32, y as i32, z as i32);
                                    state.world.set(coord, Some(bt));
                                    mark_block_user_modified(&mut state, coord);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Silence unused warning when SUPER isn't directly referenced.
    let _ = SUPER;
}

fn mark_block_user_modified(state: &mut WorldState, coord: IVec3) {
    let key = IVec3::new(
        coord.x.div_euclid(S),
        coord.y.div_euclid(S),
        coord.z.div_euclid(S),
    );
    if let Some(chunk) = state.world.chunks.get_mut(&key) {
        chunk.user_modified = true;
    }
    state.dirty_super_for_block(coord);
}

// ============================================================
// Hotbar + save
// ============================================================

pub fn cycle_hotbar_slot(keyboard: Res<ButtonInput<KeyCode>>, mut hotbar: ResMut<super::Hotbar>) {
    for (key, idx) in [
        (KeyCode::Digit1, 0),
        (KeyCode::Digit2, 1),
        (KeyCode::Digit3, 2),
        (KeyCode::Digit4, 3),
        (KeyCode::Digit5, 4),
        (KeyCode::Digit6, 5),
        (KeyCode::Digit7, 6),
        (KeyCode::Digit8, 7),
        (KeyCode::Digit9, 8),
        (KeyCode::Digit0, 9),
    ] {
        if keyboard.just_pressed(key) {
            hotbar.active = idx;
        }
    }
}

pub fn save_as_template(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    state: Res<WorldState>,
    player_q: Query<&Transform, With<Player>>,
    mut registry: ResMut<ModelRegistry>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if inv.open {
        return;
    }
    if !keyboard.just_pressed(KeyCode::KeyP) {
        return;
    }
    let Ok(tf) = player_q.single() else { return };

    // Save the chunk currently under the player. The player's bevy units
    // depend on depth, so convert via the per-depth granularity to find the
    // integer chunk key.
    let chunk_key = match state.depth {
        0 => IVec3::new(
            (tf.translation.x * S as f32) as i32,
            (tf.translation.y * S as f32) as i32,
            (tf.translation.z * S as f32) as i32,
        ),
        1 => IVec3::new(
            tf.translation.x as i32,
            tf.translation.y as i32,
            tf.translation.z as i32,
        ),
        _ => IVec3::new(
            (tf.translation.x as i32).div_euclid(S),
            (tf.translation.y as i32).div_euclid(S),
            (tf.translation.z as i32).div_euclid(S),
        ),
    };
    let Some(chunk) = state.world.chunks.get(&chunk_key) else {
        return;
    };
    let baked = bake_model(&chunk.blocks, &mut meshes);
    let name = format!("Custom {}", registry.models.len());
    registry.register(VoxelModel {
        name,
        blocks: chunk.blocks,
        baked,
    });
}
