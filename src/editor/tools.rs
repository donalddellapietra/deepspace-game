//! Editor input handlers: zoom, hotbar cycling, and place/remove
//! block hooks driven by the raycast in `crate::interaction`.

use bevy::prelude::*;

use crate::camera::CursorLocked;
use crate::interaction::TargetedBlock;
use crate::inventory::InventoryState;
use crate::world::collision::position_from_bevy;
use crate::world::edit::edit_leaf;
use crate::world::tree::{voxel_from_block, EMPTY_VOXEL};
use crate::world::{CameraZoom, WorldState};

pub fn zoom_in(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    mut zoom: ResMut<CameraZoom>,
) {
    if inv.open {
        return;
    }
    if keyboard.just_pressed(KeyCode::KeyF) {
        zoom.zoom_in();
    }
}

pub fn zoom_out(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    mut zoom: ResMut<CameraZoom>,
) {
    if inv.open {
        return;
    }
    if keyboard.just_pressed(KeyCode::KeyQ) {
        zoom.zoom_out();
    }
}

pub fn cycle_hotbar_slot(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut hotbar: ResMut<super::Hotbar>,
) {
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

/// Left-click → delete the targeted voxel (set it to empty).
///
/// Skips the frame when `CursorLocked` just transitioned from false
/// to true — that click was consumed by the camera grab in
/// `manage_cursor`, it should NOT also delete a block.
pub fn remove_block(
    mouse: Res<ButtonInput<MouseButton>>,
    locked: Res<CursorLocked>,
    inv: Res<InventoryState>,
    targeted: Res<TargetedBlock>,
    mut world: ResMut<WorldState>,
) {
    if inv.open || !locked.0 || locked.is_changed() {
        return;
    }
    if !mouse.just_pressed(MouseButton::Left) {
        return;
    }
    let Some(pos) = targeted.hit_position else {
        return;
    };
    edit_leaf(&mut world, &pos, EMPTY_VOXEL);
}

/// Right-click → place the active hotbar block on the face of the
/// targeted voxel pointed at by `normal`.
pub fn place_block(
    mouse: Res<ButtonInput<MouseButton>>,
    locked: Res<CursorLocked>,
    inv: Res<InventoryState>,
    targeted: Res<TargetedBlock>,
    hotbar: Res<super::Hotbar>,
    mut world: ResMut<WorldState>,
) {
    if inv.open || !locked.0 || locked.is_changed() {
        return;
    }
    if !mouse.just_pressed(MouseButton::Right) {
        return;
    }
    let (Some(hit), Some(normal)) = (targeted.hit, targeted.normal) else {
        return;
    };
    let place_coord = hit + normal;
    let center = Vec3::new(
        place_coord.x as f32 + 0.5,
        place_coord.y as f32 + 0.5,
        place_coord.z as f32 + 0.5,
    );
    let Some(place_pos) = position_from_bevy(center) else {
        return;
    };
    let voxel = voxel_from_block(Some(hotbar.active_block()));
    edit_leaf(&mut world, &place_pos, voxel);
}
