use bevy::prelude::*;

use crate::block::materials::BlockMaterials;
use crate::block::{BlockType, MODEL_SIZE};
use crate::camera::FpsCam;
use crate::interaction::TargetedBlock;
use crate::layer::{EditingContext, GameLayer};
use crate::model::ModelRegistry;
use crate::player::Player;
use crate::world::Layer1World;

use super::{EditorState, SharedCubeMesh};
use super::grid::{EditBlock, EditEntity};

/// In World mode, press E on a targeted cell to enter editing.
pub fn enter_edit_mode(
    keyboard: Res<ButtonInput<KeyCode>>,
    world: Res<Layer1World>,
    player_q: Query<&Transform, With<Player>>,
    mut cam_q: Query<&mut FpsCam>,
    mut commands: Commands,
    mut next_state: ResMut<NextState<GameLayer>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyE) { return }

    let Ok(player_tf) = player_q.single() else { return };
    let pos = player_tf.translation;

    // Cell at player's feet
    let coord = IVec3::new(
        (pos.x / MODEL_SIZE as f32).floor() as i32,
        0,
        (pos.z / MODEL_SIZE as f32).floor() as i32,
    );

    let Some(cell_data) = world.cells.get(&coord) else { return };

    commands.insert_resource(EditingContext {
        cell_coord: coord,
        model_id: cell_data.model_id,
        return_position: pos,
    });

    // Reset camera to look toward -Z (into the cell)
    if let Ok(mut cam) = cam_q.single_mut() {
        cam.yaw = 0.0;
        cam.pitch = 0.3; // slight downward angle
    }

    next_state.set(GameLayer::Editing);
}

/// In Editing mode, press Q to exit back to world.
pub fn exit_edit_shortcut(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<GameLayer>>,
) {
    if keyboard.just_pressed(KeyCode::KeyQ) {
        next_state.set(GameLayer::World);
    }
}

/// Right click: place a block adjacent to the targeted face (Minecraft convention).
pub fn place_block(
    mouse: Res<ButtonInput<MouseButton>>,
    targeted: Res<TargetedBlock>,
    editor: Res<EditorState>,
    context: Res<EditingContext>,
    mut registry: ResMut<ModelRegistry>,
    mut commands: Commands,
    materials: Res<BlockMaterials>,
    cube_mesh: Option<Res<SharedCubeMesh>>,
) {
    if !mouse.just_pressed(MouseButton::Right) { return }
    let Some(hit) = targeted.hit else { return };
    let Some(normal) = targeted.normal else { return };
    let Some(cube) = cube_mesh else { return };

    let place_pos = hit + normal;
    let s = MODEL_SIZE as i32;
    if place_pos.x < 0 || place_pos.x >= s
        || place_pos.y < 0 || place_pos.y >= s
        || place_pos.z < 0 || place_pos.z >= s
    {
        return;
    }

    let Some(model) = registry.get_mut(context.model_id) else { return };
    let p = place_pos;

    if model.blocks[p.y as usize][p.z as usize][p.x as usize].is_some() {
        return;
    }

    model.blocks[p.y as usize][p.z as usize][p.x as usize] = Some(editor.selected_block);

    let cell_origin = Vec3::new(
        context.cell_coord.x as f32 * MODEL_SIZE as f32,
        context.cell_coord.y as f32 * MODEL_SIZE as f32,
        context.cell_coord.z as f32 * MODEL_SIZE as f32,
    );
    let world_pos = cell_origin + Vec3::new(p.x as f32 + 0.5, p.y as f32 + 0.5, p.z as f32 + 0.5);

    commands.spawn((
        EditEntity,
        EditBlock { local_pos: place_pos },
        Mesh3d(cube.0.clone()),
        MeshMaterial3d(materials.get(editor.selected_block)),
        Transform::from_translation(world_pos),
    ));
}

/// Left click: remove the targeted block (Minecraft convention).
pub fn remove_block(
    mouse: Res<ButtonInput<MouseButton>>,
    targeted: Res<TargetedBlock>,
    context: Res<EditingContext>,
    mut registry: ResMut<ModelRegistry>,
    mut commands: Commands,
    blocks_q: Query<(Entity, &EditBlock), With<EditEntity>>,
) {
    if !mouse.just_pressed(MouseButton::Left) { return }
    let Some(hit) = targeted.hit else { return };

    let Some(model) = registry.get_mut(context.model_id) else { return };
    model.blocks[hit.y as usize][hit.z as usize][hit.x as usize] = None;

    for (entity, edit_block) in &blocks_q {
        if edit_block.local_pos == hit {
            commands.entity(entity).despawn();
            break;
        }
    }
}

/// Number keys to change selected block type.
pub fn cycle_block_type(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut editor: ResMut<EditorState>,
) {
    let key_map = [
        (KeyCode::Digit1, 0), (KeyCode::Digit2, 1), (KeyCode::Digit3, 2),
        (KeyCode::Digit4, 3), (KeyCode::Digit5, 4), (KeyCode::Digit6, 5),
        (KeyCode::Digit7, 6), (KeyCode::Digit8, 7), (KeyCode::Digit9, 8),
        (KeyCode::Digit0, 9),
    ];

    for (key, idx) in key_map {
        if keyboard.just_pressed(key) {
            if let Some(bt) = BlockType::from_index(idx) {
                editor.selected_block = bt;
            }
        }
    }
}
