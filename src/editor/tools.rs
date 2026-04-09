use bevy::prelude::*;

use crate::block::materials::BlockMaterials;
use crate::block::{BlockType, MODEL_SIZE};
use crate::camera::FpsCam;
use crate::interaction::TargetedBlock;
use crate::layer::{EditingContext, GameLayer};
use crate::model::mesher::bake_model;
use crate::model::{ModelRegistry, VoxelModel};
use crate::player::Player;
use crate::world::Layer1World;

use super::{EditorState, SharedCubeMesh};
use super::grid::{EditBlock, EditEntity};

/// Press E in world mode to enter editing on the cell at your feet.
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

    let coord = IVec3::new(
        (pos.x / MODEL_SIZE as f32).floor() as i32,
        0,
        (pos.z / MODEL_SIZE as f32).floor() as i32,
    );

    if !world.cells.contains_key(&coord) { return }

    commands.insert_resource(EditingContext {
        cell_coord: coord,
        return_position: pos,
    });

    if let Ok(mut cam) = cam_q.single_mut() {
        cam.yaw = 0.0;
        cam.pitch = 0.3;
    }

    next_state.set(GameLayer::Editing);
}

/// Press Q to exit editing.
pub fn exit_edit_shortcut(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut next_state: ResMut<NextState<GameLayer>>,
) {
    if keyboard.just_pressed(KeyCode::KeyQ) {
        next_state.set(GameLayer::World);
    }
}

/// Right click: place block on the face adjacent to targeted block.
pub fn place_block(
    mouse: Res<ButtonInput<MouseButton>>,
    targeted: Res<TargetedBlock>,
    editor: Res<EditorState>,
    context: Res<EditingContext>,
    mut world: ResMut<Layer1World>,
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
    { return; }

    let Some(cell_data) = world.cells.get_mut(&context.cell_coord) else { return };
    let p = place_pos;

    if cell_data.blocks[p.y as usize][p.z as usize][p.x as usize].is_some() { return; }

    cell_data.blocks[p.y as usize][p.z as usize][p.x as usize] = Some(editor.selected_block);

    let cell_origin = context.cell_coord.as_vec3() * MODEL_SIZE as f32;
    let world_pos = cell_origin + Vec3::new(p.x as f32 + 0.5, p.y as f32 + 0.5, p.z as f32 + 0.5);

    commands.spawn((
        EditEntity,
        EditBlock { local_pos: place_pos },
        Mesh3d(cube.0.clone()),
        MeshMaterial3d(materials.get(editor.selected_block)),
        Transform::from_translation(world_pos),
    ));
}

/// Left click: remove targeted block.
pub fn remove_block(
    mouse: Res<ButtonInput<MouseButton>>,
    targeted: Res<TargetedBlock>,
    context: Res<EditingContext>,
    mut world: ResMut<Layer1World>,
    mut commands: Commands,
    blocks_q: Query<(Entity, &EditBlock), With<EditEntity>>,
) {
    if !mouse.just_pressed(MouseButton::Left) { return }
    let Some(hit) = targeted.hit else { return };

    let Some(cell_data) = world.cells.get_mut(&context.cell_coord) else { return };
    cell_data.blocks[hit.y as usize][hit.z as usize][hit.x as usize] = None;

    for (entity, edit_block) in &blocks_q {
        if edit_block.local_pos == hit {
            commands.entity(entity).despawn();
            break;
        }
    }
}

/// Number keys to select block type (editing mode).
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

/// Press P to save the current cell as a new model template.
pub fn save_as_template(
    keyboard: Res<ButtonInput<KeyCode>>,
    context: Res<EditingContext>,
    world: Res<Layer1World>,
    mut registry: ResMut<ModelRegistry>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyP) { return }

    let Some(cell_data) = world.cells.get(&context.cell_coord) else { return };

    let name = format!("Custom {}", registry.models.len());
    let baked = bake_model(&cell_data.blocks, &mut meshes);

    registry.register(VoxelModel {
        name,
        blocks: cell_data.blocks,
        baked,
    });
}

/// Number keys in world mode to select model template.
pub fn cycle_model_slot(
    keyboard: Res<ButtonInput<KeyCode>>,
    registry: Res<ModelRegistry>,
    mut hotbar: ResMut<crate::world::WorldHotbar>,
) {
    let key_map = [
        (KeyCode::Digit1, 0), (KeyCode::Digit2, 1), (KeyCode::Digit3, 2),
        (KeyCode::Digit4, 3), (KeyCode::Digit5, 4), (KeyCode::Digit6, 5),
        (KeyCode::Digit7, 6), (KeyCode::Digit8, 7), (KeyCode::Digit9, 8),
        (KeyCode::Digit0, 9),
    ];
    for (key, idx) in key_map {
        if keyboard.just_pressed(key) {
            if (idx as usize) < registry.models.len() {
                hotbar.selected_slot = idx as usize;
            }
        }
    }
}

/// Right click in world mode to place a model cell at the targeted position.
pub fn place_cell(
    mouse: Res<ButtonInput<MouseButton>>,
    hotbar: Res<crate::world::WorldHotbar>,
    registry: Res<ModelRegistry>,
    player_q: Query<&Transform, With<Player>>,
    camera_q: Query<&FpsCam>,
    mut world: ResMut<Layer1World>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if !mouse.just_pressed(MouseButton::Right) { return }
    let Ok(player_tf) = player_q.single() else { return };
    let Ok(cam) = camera_q.single() else { return };

    let Some(template) = registry.models.get(hotbar.selected_slot) else { return };

    // Simple: place at the cell the player is looking at + one above ground
    let forward = Vec3::new(-cam.yaw.sin(), 0.0, -cam.yaw.cos());
    let target_pos = player_tf.translation + forward * 5.0;
    let coord = IVec3::new(
        (target_pos.x / MODEL_SIZE as f32).floor() as i32,
        1, // one layer above ground
        (target_pos.z / MODEL_SIZE as f32).floor() as i32,
    );

    if world.cells.contains_key(&coord) { return; }

    let baked = bake_model(&template.blocks, &mut meshes);
    world.cells.insert(coord, crate::world::CellData {
        template_id: crate::model::ModelId(hotbar.selected_slot),
        blocks: template.blocks,
        baked,
    });
}

/// Left click in world mode to remove a cell (not ground level).
pub fn remove_cell(
    mouse: Res<ButtonInput<MouseButton>>,
    player_q: Query<&Transform, With<Player>>,
    camera_q: Query<&FpsCam>,
    mut world: ResMut<Layer1World>,
    mut loaded: ResMut<crate::world::LoadedCells>,
    mut commands: Commands,
) {
    if !mouse.just_pressed(MouseButton::Left) { return }
    let Ok(player_tf) = player_q.single() else { return };
    let Ok(cam) = camera_q.single() else { return };

    let forward = Vec3::new(-cam.yaw.sin(), 0.0, -cam.yaw.cos());
    let target_pos = player_tf.translation + forward * 5.0;
    let coord = IVec3::new(
        (target_pos.x / MODEL_SIZE as f32).floor() as i32,
        1,
        (target_pos.z / MODEL_SIZE as f32).floor() as i32,
    );

    // Don't allow removing ground (y=0)
    if coord.y <= 0 { return; }
    if world.cells.remove(&coord).is_some() {
        if let Some(entity) = loaded.entities.remove(&coord) {
            commands.entity(entity).despawn();
        }
    }
}
