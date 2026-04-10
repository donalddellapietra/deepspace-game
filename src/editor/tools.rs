use bevy::prelude::*;

use crate::block::materials::BlockMaterials;
use crate::block::{BlockType, MODEL_SIZE};
use crate::camera::{CursorLocked, FpsCam};
use crate::interaction::TargetedBlock;
use crate::inventory::InventoryState;
use crate::layer::{ActiveLayer, NavEntry};
use crate::model::mesher::bake_model;
use crate::model::{ModelRegistry, VoxelModel};
use crate::player::{Player, Velocity};
use crate::world::{self, CellSlot, RenderState, SharedCubeMesh, VoxelGrid, VoxelWorld};

// ============================================================
// Drill down / up
// ============================================================

/// Press F: drill into the cell/block the crosshair is pointing at.
pub fn drill_down(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    targeted: Res<TargetedBlock>,
    mut active: ResMut<ActiveLayer>,
    mut world: ResMut<VoxelWorld>,
    mut rs: ResMut<RenderState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut player_q: Query<(&mut Transform, &mut Velocity), With<Player>>,
    mut cam_q: Query<&mut FpsCam>,
) {
    if inv.open { return }
    if !keyboard.just_pressed(KeyCode::KeyF) { return }

    // Must be targeting something
    let Some(hit) = targeted.hit else { return };
    let Ok((mut tf, mut vel)) = player_q.single_mut() else { return };

    if active.is_top_layer() {
        // Drilling into a top-layer cell
        if !world.cells.contains_key(&hit) { return }
    } else {
        // Drilling deeper: the targeted block must be a Child or convertible Block
        let Some(grid) = world.get_grid_mut(&active.nav_stack) else { return };
        let h = hit;
        if h.x < 0 || h.x >= MODEL_SIZE as i32 || h.y < 0 || h.y >= MODEL_SIZE as i32
            || h.z < 0 || h.z >= MODEL_SIZE as i32 { return }

        match &grid.slots[h.y as usize][h.z as usize][h.x as usize] {
            CellSlot::Empty => return,
            CellSlot::Child(_) => {} // already a child, can drill in
            CellSlot::Block(bt) => {
                // Auto-create a child grid from this block
                let bt = *bt;
                let mut inner = VoxelGrid::new_empty();
                // A block is solid — fill ALL 5 layers
                for y in 0..MODEL_SIZE {
                    for z in 0..MODEL_SIZE {
                        for x in 0..MODEL_SIZE {
                            inner.slots[y][z][x] = CellSlot::Block(bt);
                        }
                    }
                }
                inner.rebake(&mut meshes);
                grid.slots[h.y as usize][h.z as usize][h.x as usize] =
                    CellSlot::Child(Box::new(inner));
            }
        }
    }

    // Push navigation
    active.nav_stack.push(NavEntry {
        cell_coord: hit,
        return_position: tf.translation,
    });

    // Find floor inside the target grid
    let floor_y = if let Some(grid) = world.get_grid(&active.nav_stack) {
        grid.column_top(MODEL_SIZE / 2, MODEL_SIZE / 2)
    } else { 0.0 };

    // Teleport player to stand on the floor inside the cell
    let s = MODEL_SIZE as f32;
    tf.translation = Vec3::new(s / 2.0, floor_y, s / 2.0);
    vel.0 = Vec3::ZERO;

    if let Ok(mut cam) = cam_q.single_mut() {
        cam.yaw = 0.4;
        cam.pitch = 0.8; // look down at the surface
    }

    rs.needs_refresh = true;
}

/// Press Q: drill back up to parent layer.
pub fn drill_up(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    mut active: ResMut<ActiveLayer>,
    mut world: ResMut<VoxelWorld>,
    mut rs: ResMut<RenderState>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut player_q: Query<(&mut Transform, &mut Velocity), With<Player>>,
) {
    if inv.open { return }
    if !keyboard.just_pressed(KeyCode::KeyQ) { return }
    if active.nav_stack.is_empty() { return }

    // Re-bake the grid we're leaving
    if let Some(grid) = world.get_grid_mut(&active.nav_stack) {
        grid.rebake(&mut meshes);
    }

    let entry = active.nav_stack.pop().unwrap();

    if let Ok((mut tf, mut vel)) = player_q.single_mut() {
        tf.translation = entry.return_position;
        vel.0 = Vec3::ZERO;
    }

    rs.needs_refresh = true;
}

// ============================================================
// Block interaction — works at EVERY layer
// ============================================================

/// Left click: remove the targeted block/cell.
pub fn remove_block(
    mouse: Res<ButtonInput<MouseButton>>,
    locked: Res<CursorLocked>,
    inv: Res<InventoryState>,
    targeted: Res<TargetedBlock>,
    active: Res<ActiveLayer>,
    mut world: ResMut<VoxelWorld>,
    mut commands: Commands,
    mut rs: ResMut<RenderState>,
) {
    if inv.open { return }
    if !locked.0 { return } // cursor not grabbed yet
    if !mouse.just_pressed(MouseButton::Left) { return }
    let Some(hit) = targeted.hit else { return };

    if active.is_top_layer() {
        // Remove a top-layer cell
        world.cells.remove(&hit);
        if let Some(e) = rs.entities.remove(&hit) { commands.entity(e).despawn(); }
    } else {
        // Remove a block inside the current grid
        let Some(grid) = world.get_grid_mut(&active.nav_stack) else { return };
        let h = hit;
        if h.x < 0 || h.x >= MODEL_SIZE as i32 || h.y < 0 || h.y >= MODEL_SIZE as i32
            || h.z < 0 || h.z >= MODEL_SIZE as i32 { return }
        grid.slots[h.y as usize][h.z as usize][h.x as usize] = CellSlot::Empty;
        if let Some(e) = rs.entities.remove(&hit) { commands.entity(e).despawn(); }
    }
}

/// Right click: place using the active hotbar item.
/// - If it's a Block: place a single block (inside grid) or solid cell (top layer).
/// - If it's a SavedModel: place the model's blocks as a new cell (top layer) or
///   as a Child grid (inside grid).
pub fn place_block(
    mouse: Res<ButtonInput<MouseButton>>,
    locked: Res<CursorLocked>,
    inv: Res<InventoryState>,
    targeted: Res<TargetedBlock>,
    hotbar: Res<super::Hotbar>,
    registry: Res<ModelRegistry>,
    active: Res<ActiveLayer>,
    mut world: ResMut<VoxelWorld>,
    mut commands: Commands,
    materials: Res<BlockMaterials>,
    cube: Option<Res<SharedCubeMesh>>,
    mut rs: ResMut<RenderState>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if inv.open { return }
    if !locked.0 { return }
    if !mouse.just_pressed(MouseButton::Right) { return }
    let Some(hit) = targeted.hit else { return };
    let Some(normal) = targeted.normal else { return };

    let place = hit + normal;
    let item = hotbar.active_item();

    // Build the block array to place
    let place_blocks: [[[Option<BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE] = match item {
        super::HotbarItem::Block(bt) => {
            let mut blocks = [[[None; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE];
            // At top layer: solid cell. Inside grid: single block.
            if active.is_top_layer() {
                for y in 0..MODEL_SIZE { for z in 0..MODEL_SIZE { for x in 0..MODEL_SIZE {
                    blocks[y][z][x] = Some(*bt);
                }}}
            } else {
                // Just place one block
                let s = MODEL_SIZE as i32;
                if place.x < 0 || place.x >= s || place.y < 0 || place.y >= s
                    || place.z < 0 || place.z >= s { return }
                let Some(grid) = world.get_grid_mut(&active.nav_stack) else { return };
                if grid.slots[place.y as usize][place.z as usize][place.x as usize].is_solid() { return }
                grid.slots[place.y as usize][place.z as usize][place.x as usize] = CellSlot::Block(*bt);

                if let Some(cube) = cube {
                    let e = commands.spawn((
                        world::LayerEntity, Mesh3d(cube.0.clone()),
                        MeshMaterial3d(materials.get(*bt)),
                        Transform::from_translation(place.as_vec3() + Vec3::splat(0.5)),
                    )).id();
                    rs.entities.insert(place, e);
                }
                return;
            }
            blocks
        }
        super::HotbarItem::SavedModel(idx) => {
            let Some(model) = registry.models.get(*idx) else { return };
            model.blocks
        }
    };

    if active.is_top_layer() {
        if world.cells.contains_key(&place) { return }
        world.cells.insert(place, VoxelGrid::from_blocks(&place_blocks, &mut meshes));
        rs.needs_refresh = true;
    } else {
        // Inside a grid: place as a Child grid
        let s = MODEL_SIZE as i32;
        if place.x < 0 || place.x >= s || place.y < 0 || place.y >= s
            || place.z < 0 || place.z >= s { return }
        let Some(grid) = world.get_grid_mut(&active.nav_stack) else { return };
        if grid.slots[place.y as usize][place.z as usize][place.x as usize].is_solid() { return }

        let child = VoxelGrid::from_blocks(&place_blocks, &mut meshes);
        grid.slots[place.y as usize][place.z as usize][place.x as usize] =
            CellSlot::Child(Box::new(child));
        rs.needs_refresh = true;
    }
}

// ============================================================
// Hotbar slot selection + save
// ============================================================

pub fn cycle_hotbar_slot(keyboard: Res<ButtonInput<KeyCode>>, mut hotbar: ResMut<super::Hotbar>) {
    for (key, idx) in [
        (KeyCode::Digit1, 0), (KeyCode::Digit2, 1), (KeyCode::Digit3, 2),
        (KeyCode::Digit4, 3), (KeyCode::Digit5, 4), (KeyCode::Digit6, 5),
        (KeyCode::Digit7, 6), (KeyCode::Digit8, 7), (KeyCode::Digit9, 8),
        (KeyCode::Digit0, 9),
    ] {
        if keyboard.just_pressed(key) {
            hotbar.active = idx;
        }
    }
}

/// Press P: save the current grid as a reusable model template.
pub fn save_as_template(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    active: Res<ActiveLayer>,
    world: Res<VoxelWorld>,
    mut registry: ResMut<ModelRegistry>,
    mut meshes: ResMut<Assets<Mesh>>,
) {
    if inv.open { return }
    if !keyboard.just_pressed(KeyCode::KeyP) { return }
    let Some(grid) = world.get_grid(&active.nav_stack) else { return };
    let blocks = grid.to_block_array();
    let baked = bake_model(&blocks, &mut meshes);
    let name = format!("Custom {}", registry.models.len());
    registry.register(VoxelModel { name, blocks, baked });
}
