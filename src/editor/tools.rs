//! Editor input handlers: zoom, player reset, hotbar cycling, and
//! place/remove block hooks driven by the raycast in
//! `crate::interaction`.

use bevy::prelude::*;

use crate::camera::CursorLocked;
use crate::interaction::TargetedBlock;
use crate::inventory::InventoryState;
use crate::player::{Player, Velocity};
use crate::world::collision::position_from_bevy;
use crate::world::edit::edit_at_layer_pos;
use crate::world::position::LayerPos;
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
/// The raycast returns a leaf `Position`; we project it down to the
/// camera's current view layer and call [`edit_at_layer_pos`], which
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
    zoom: Res<CameraZoom>,
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
    let lp = LayerPos::from_leaf(&pos, zoom.layer);
    edit_at_layer_pos(&mut world, &lp, EMPTY_VOXEL);
}

/// Right-click → place the active hotbar block on the face of the
/// targeted cell pointed at by `normal`. Like `remove_block`, the
/// targeted cell and the placement cell are view-layer cells, so the
/// edit dispatches on `CameraZoom.layer`.
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
    let (Some(hit), Some(normal)) = (targeted.hit, targeted.normal) else {
        return;
    };

    // Compute the placement cell by stepping one view-layer cell in
    // the normal direction from the targeted cell. At view layer L,
    // one cell is `5^(MAX_LAYER - L)` Bevy units.
    let cell_size = crate::world::render::cell_size_at_layer(zoom.layer);
    let place_center = Vec3::new(
        hit.x as f32 + 0.5 + normal.x as f32 * cell_size,
        hit.y as f32 + 0.5 + normal.y as f32 * cell_size,
        hit.z as f32 + 0.5 + normal.z as f32 * cell_size,
    );
    let Some(place_leaf_pos) = position_from_bevy(place_center) else {
        return;
    };
    let lp = LayerPos::from_leaf(&place_leaf_pos, zoom.layer);
    let voxel = voxel_from_block(Some(hotbar.active_block()));
    edit_at_layer_pos(&mut world, &lp, voxel);
}
