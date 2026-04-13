//! Debug tools for visual inspection. Not part of normal gameplay.

use bevy::prelude::*;

use crate::block::BlockType;
use crate::player::spawn_position;
use crate::world::edit::edit_at_layer_pos;
use crate::world::tree::{voxel_from_block, EMPTY_VOXEL, MAX_LAYER};
use crate::world::WorldState;

/// F9 → carve a pit near spawn and place one of each block type so
/// textures can be visually inspected. No-ops after first press.
pub fn debug_texture_test(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut world: ResMut<WorldState>,
    mut done: Local<bool>,
) {
    if *done || !keyboard.just_pressed(KeyCode::F9) {
        return;
    }
    *done = true;

    let spawn = spawn_position();
    let base_path = spawn.path;

    let mut edit = |dx: i8, dy: i8, dz: i8, voxel: u8| {
        let cx = (spawn.voxel[0] as i8 + dx) as u8;
        let cy = (spawn.voxel[1] as i8 + dy) as u8;
        let cz = (spawn.voxel[2] as i8 + dz) as u8;
        if cx < 25 && cy < 25 && cz < 25 {
            let lp = crate::world::position::LayerPos::from_path_and_cell(
                base_path,
                [cx, cy, cz],
                MAX_LAYER,
            );
            edit_at_layer_pos(&mut world, &lp, voxel);
        }
    };

    // Carve a 5x3x5 pit in front of the player (negative Z)
    for dx in -2..=2i8 {
        for dz in -5..=-1i8 {
            for dy in -3..=-1i8 {
                edit(dx, dy, dz, EMPTY_VOXEL);
            }
        }
    }

    // Place a row of different block types along the pit floor
    let blocks = BlockType::ALL;
    for (i, bt) in blocks.iter().enumerate() {
        let dx = (i as i8) - 2;
        if (-2..=2).contains(&dx) {
            edit(dx, -3, -3, voxel_from_block(Some(*bt)));
        }
    }
    // Place remaining blocks in second row
    for (i, bt) in blocks.iter().enumerate().skip(5) {
        let dx = (i as i8) - 7;
        if (-2..=2).contains(&dx) {
            edit(dx, -3, -4, voxel_from_block(Some(*bt)));
        }
    }

    info!("Debug texture test: carved pit and placed block samples near spawn. Look down!");
}
