//! Editor input handlers: zoom, player reset, hotbar cycling, and
//! place/remove block hooks driven by the raycast in
//! `crate::interaction`.

use bevy::prelude::*;

use crate::camera::CursorLocked;
use crate::interaction::TargetedBlock;
use crate::inventory::InventoryState;
use crate::player::{spawn_position, Player, Velocity};
use crate::world::collision;
use crate::world::edit::edit_at_layer_pos;
use crate::world::tree::{voxel_from_block, EMPTY_VOXEL};
use crate::world::view::{
    bevy_center_of_layer_pos, cell_size_at_layer, layer_pos_from_bevy, WorldAnchor,
};
use crate::world::{CameraZoom, WorldPosition, WorldState};

/// F → zoom in (show finer detail). Increments `CameraZoom.layer`
/// toward the leaf layer. No-op at `MAX_ZOOM` (the leaves).
///
/// After a successful zoom, snap the player onto the apparent ground
/// at the new view layer. The collision grid changes size when the
/// target layer changes, and thin features (like the 125-leaf-deep
/// grassland ground) inflate upward at coarse layers. Without this
/// snap, zooming in while standing on an inflated block would leave
/// the player hanging in mid-air; zooming out would put them inside
/// a solid block and gravity would pull them straight through. See
/// [`collision::snap_to_ground`] for the full story.
pub fn zoom_in(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    world: Res<WorldState>,
    mut zoom: ResMut<CameraZoom>,
    mut player_q: Query<(&mut WorldPosition, &mut Velocity), With<Player>>,
) {
    if inv.open {
        return;
    }
    if keyboard.just_pressed(KeyCode::KeyF) && zoom.zoom_in() {
        if let Ok((mut pos, mut vel)) = player_q.single_mut() {
            collision::snap_to_ground(&mut pos.0, &world, zoom.layer);
            // Zero vertical velocity so the frame after the snap
            // doesn't inherit whatever fall/jump we were in the
            // middle of.
            vel.0.y = 0.0;
        }
    }
}

/// Q → zoom out (show larger area at coarser detail). Decrements
/// `CameraZoom.layer` toward the root. No-op at `MIN_ZOOM`. Same
/// post-zoom snap as [`zoom_in`] for the mirror case.
pub fn zoom_out(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    world: Res<WorldState>,
    mut zoom: ResMut<CameraZoom>,
    mut player_q: Query<(&mut WorldPosition, &mut Velocity), With<Player>>,
) {
    if inv.open {
        return;
    }
    if keyboard.just_pressed(KeyCode::KeyQ) && zoom.zoom_out() {
        if let Ok((mut pos, mut vel)) = player_q.single_mut() {
            collision::snap_to_ground(&mut pos.0, &world, zoom.layer);
            vel.0.y = 0.0;
        }
    }
}

/// R → teleport the player back to the spawn point and zero their
/// velocity. Handy when you fall into the void off the edge of the
/// ground layer, or to reset after a physics glitch. The teleport
/// goes to [`spawn_translation`] and is then snapped to the apparent
/// ground at the current view layer — that way reset-at-any-zoom
/// lands the player standing on the visible ground instead of
/// inside an inflated collision block.
pub fn reset_player(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    world: Res<WorldState>,
    zoom: Res<CameraZoom>,
    mut player_q: Query<(&mut WorldPosition, &mut Velocity), With<Player>>,
) {
    if inv.open {
        return;
    }
    if !keyboard.just_pressed(KeyCode::KeyR) {
        return;
    }
    let Ok((mut pos, mut vel)) = player_q.single_mut() else {
        return;
    };
    pos.0 = spawn_position();
    vel.0 = Vec3::ZERO;
    collision::snap_to_ground(&mut pos.0, &world, zoom.layer);
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
    anchor: Res<WorldAnchor>,
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
    let hit_center = bevy_center_of_layer_pos(hit, &anchor);
    let place_center = hit_center + normal.as_vec3() * cell_size;
    let Some(place_lp) = layer_pos_from_bevy(place_center, zoom.layer, &anchor)
    else {
        return;
    };
    let voxel = voxel_from_block(Some(hotbar.active_block()));
    edit_at_layer_pos(&mut world, &place_lp, voxel);
}
