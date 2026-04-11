//! Editor input handlers: zoom, player reset, hotbar cycling, and
//! place/remove block hooks driven by the raycast in
//! `crate::interaction`.

use bevy::prelude::*;

use crate::camera::CursorLocked;
use crate::interaction::TargetedBlock;
use crate::inventory::InventoryState;
use crate::player::{Player, Velocity};
use crate::world::collision::{
    bevy_center_of_layer_pos, layer_pos_from_bevy,
};
use crate::world::edit::edit_at_layer_pos;
use crate::world::render::cell_size_at_layer;
use crate::world::tree::{voxel_from_block, EMPTY_VOXEL};
use crate::world::{CameraZoom, WorldState};

/// F → zoom in (show finer detail). Increments `CameraZoom.layer`
/// toward the leaf layer. No-op at `MAX_ZOOM` (the leaves).
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

/// Q → zoom out (show larger area at coarser detail). Decrements
/// `CameraZoom.layer` toward the root. No-op at `MIN_ZOOM`.
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

/// R → teleport the player back to the spawn point and zero their
/// velocity. Handy when you fall into the void off the edge of the
/// ground layer, or to reset after a physics glitch.
pub fn reset_player(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    mut player_q: Query<(&mut Transform, &mut Velocity), With<Player>>,
) {
    if inv.open {
        return;
    }
    if !keyboard.just_pressed(KeyCode::KeyR) {
        return;
    }
    let Ok((mut tf, mut vel)) = player_q.single_mut() else {
        return;
    };
    tf.translation = Vec3::new(0.0, 5.0, 0.0);
    vel.0 = Vec3::ZERO;
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

/// Left-click → delete the view-layer cell under the crosshair.
///
/// The raycast returns a [`LayerPos`] at the current view layer, so
/// we just hand it straight to [`edit_at_layer_pos`], which
/// dispatches on the layer:
///
/// - at the leaf layer, it removes a single voxel,
/// - one layer up, it removes a `5³` region inside a leaf,
/// - any higher layer, it replaces a whole layer-`(L + 2)` subtree.
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
    let Some(lp) = targeted.hit_layer_pos.as_ref() else {
        return;
    };
    edit_at_layer_pos(&mut world, lp, EMPTY_VOXEL);
}

/// Right-click → place the active hotbar block on the face of the
/// targeted cell pointed at by `normal`. The hit cell is a
/// [`LayerPos`] at the current view layer; we step one cell in the
/// normal direction by computing the placement cell's centre in
/// Bevy space and routing it back through `layer_pos_from_bevy`.
pub fn place_block(
    mouse: Res<ButtonInput<MouseButton>>,
    locked: Res<CursorLocked>,
    inv: Res<InventoryState>,
    targeted: Res<TargetedBlock>,
    hotbar: Res<super::Hotbar>,
    zoom: Res<CameraZoom>,
    mut world: ResMut<WorldState>,
) {
    if inv.open || !locked.0 || locked.is_changed() {
        return;
    }
    if !mouse.just_pressed(MouseButton::Right) {
        return;
    }
    let (Some(hit), Some(normal)) =
        (targeted.hit_layer_pos.as_ref(), targeted.normal)
    else {
        return;
    };

    let cell_size = cell_size_at_layer(zoom.layer);
    let hit_center = bevy_center_of_layer_pos(hit);
    let place_center = hit_center + normal.as_vec3() * cell_size;
    let Some(place_lp) = layer_pos_from_bevy(place_center, zoom.layer) else {
        return;
    };
    let voxel = voxel_from_block(Some(hotbar.active_block()));
    edit_at_layer_pos(&mut world, &place_lp, voxel);
}
