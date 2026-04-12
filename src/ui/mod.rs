//! UI module — cursor management and state resources.
//! The actual HUD rendering is handled by the React overlay (see `overlay` module).

pub mod color_picker;

use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::camera::CursorLocked;
use crate::inventory::InventoryState;
use crate::overlay::UiFocused;

// ── Plugin ─────────────────────────────────────────────────────────

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(color_picker::ColorPickerPlugin)
            .add_systems(Update, sync_cursor);
    }
}

// ── Cursor sync ──────────────────────────────────────────────────

/// Single owner of cursor lock state. Panels set their `open` bool;
/// click-to-grab and Escape-to-release are handled here too.
///
/// `UiFocused` is set by the React overlay when the mouse enters/leaves
/// an interactive UI element, replacing the old `egui_wants_pointer` flag.
pub fn sync_cursor(
    inv: Res<InventoryState>,
    picker: Res<color_picker::ColorPickerState>,
    ui_focused: Res<UiFocused>,
    mouse: Res<ButtonInput<MouseButton>>,
    key: Res<ButtonInput<KeyCode>>,
    mut cursor_options: Single<&mut CursorOptions>,
    mut cursor_locked: ResMut<CursorLocked>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
) {
    let any_panel_open = inv.open || picker.open;

    if any_panel_open {
        let was_locked = cursor_locked.0;
        cursor_options.visible = true;
        cursor_options.grab_mode = CursorGrabMode::None;
        cursor_locked.0 = false;

        // When transitioning from locked → unlocked, warp the cursor
        // to screen center so the browser/OS has a valid position.
        if was_locked {
            let center = Vec2::new(window.width() / 2.0, window.height() / 2.0);
            window.set_cursor_position(Some(center));
        }
        return;
    }

    // Don't grab if the React UI has pointer focus
    if ui_focused.0 {
        return;
    }

    // Click to grab
    if mouse.just_pressed(MouseButton::Left) && !cursor_locked.0 {
        cursor_options.visible = false;
        cursor_options.grab_mode = CursorGrabMode::Locked;
        cursor_locked.0 = true;
        return;
    }

    // Escape to release
    if key.just_pressed(KeyCode::Escape) {
        cursor_options.visible = true;
        cursor_options.grab_mode = CursorGrabMode::None;
        cursor_locked.0 = false;
    }
}
