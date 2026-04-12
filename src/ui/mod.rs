//! UI module — cursor management and state resources.
//! The actual HUD rendering is handled by the React overlay (see `overlay` module).

pub mod color_picker;

use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::camera::CursorLocked;
use crate::inventory::InventoryState;
use crate::overlay::{PointerLockLost, UiFocused};

// ── Plugin ─────────────────────────────────────────────────────────

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(color_picker::ColorPickerPlugin)
            .insert_resource(WasPanel(false))
            .add_systems(Update, sync_cursor);
    }
}

/// Tracks whether a panel was open last frame so we can auto-relock
/// on the frame the panel closes.
#[derive(Resource)]
pub struct WasPanel(bool);

// ── Cursor sync ──────────────────────────────────────────────────

/// Single owner of cursor lock state. Panels set their `open` bool;
/// click-to-grab and Escape-to-release are handled here too.
///
/// `UiFocused` is set by the React overlay when the mouse enters/leaves
/// an interactive UI element, replacing the old `egui_wants_pointer` flag.
pub fn sync_cursor(
    mut inv: ResMut<InventoryState>,
    mut picker: ResMut<color_picker::ColorPickerState>,
    ui_focused: Res<UiFocused>,
    mouse: Res<ButtonInput<MouseButton>>,
    key: Res<ButtonInput<KeyCode>>,
    mut cursor_options: Single<&mut CursorOptions>,
    mut cursor_locked: ResMut<CursorLocked>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
    mut was_panel: ResMut<WasPanel>,
    mut lock_lost: ResMut<PointerLockLost>,
) {
    // ── Escape (native) ──
    if key.just_pressed(KeyCode::Escape) {
        cursor_options.visible = true;
        cursor_options.grab_mode = CursorGrabMode::None;
        cursor_locked.0 = false;
        was_panel.0 = false;
        inv.open = false;
        picker.open = false;
        lock_lost.0 = false;
        return;
    }

    // ── Browser pointer lock loss (WASM) ──
    // The browser exits Pointer Lock on Escape and consumes the keypress.
    // JS detects `pointerlockchange` and sets this flag.
    // Only act on it if we thought we were locked — ignore if Bevy
    // intentionally unlocked for a panel (cursor_locked is already false).
    if lock_lost.0 {
        lock_lost.0 = false;
        if cursor_locked.0 {
            cursor_options.visible = true;
            cursor_options.grab_mode = CursorGrabMode::None;
            cursor_locked.0 = false;
            was_panel.0 = false;
            inv.open = false;
            picker.open = false;
            return;
        }
    }

    let any_panel_open = inv.open || picker.open;

    if any_panel_open {
        let was_locked = cursor_locked.0;
        cursor_options.visible = true;
        cursor_options.grab_mode = CursorGrabMode::None;
        cursor_locked.0 = false;
        was_panel.0 = true;

        if was_locked {
            let center = Vec2::new(window.width() / 2.0, window.height() / 2.0);
            window.set_cursor_position(Some(center));
        }
        return;
    }

    // ── Auto-relock when panel just closed ──
    if was_panel.0 {
        was_panel.0 = false;
        cursor_options.visible = false;
        cursor_options.grab_mode = CursorGrabMode::Locked;
        cursor_locked.0 = true;
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
    }
}
