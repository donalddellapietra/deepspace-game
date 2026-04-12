//! UI module — cursor management and state resources.
//! The actual HUD rendering is handled by the React overlay (see `overlay` module).
//!
//! ## Cursor state machine
//!
//! There are exactly three states:
//!
//! ```text
//!   ┌──────────┐  E / C   ┌──────────┐  Esc    ┌──────────┐
//!   │ Gameplay  │────────►│  Panel   │───────►│Unfocused │
//!   │  locked   │◄────────│ unlocked │        │ unlocked │
//!   └──────────┘ E / C    └──────────┘        └──────────┘
//!        ▲                                         │
//!        │              click anywhere             │
//!        └─────────────────────────────────────────┘
//! ```
//!
//! - **Gameplay**: cursor locked, game has input.
//! - **Panel**: a panel is open, cursor free.
//! - **Unfocused**: no panel, cursor free, waiting for click-to-grab.
//!
//! `sync_cursor` is the single owner of transitions.

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
            .add_systems(Update, sync_cursor);
    }
}

// ── Cursor sync ──────────────────────────────────────────────────

pub fn sync_cursor(
    mut inv: ResMut<InventoryState>,
    mut picker: ResMut<color_picker::ColorPickerState>,
    mut ui_focused: ResMut<UiFocused>,
    mouse: Res<ButtonInput<MouseButton>>,
    mut keyboard: ResMut<ButtonInput<KeyCode>>,
    mut cursor_options: Single<&mut CursorOptions>,
    mut cursor_locked: ResMut<CursorLocked>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
    mut lock_lost: ResMut<PointerLockLost>,
    #[cfg(not(target_arch = "wasm32"))]
    primary: Query<Entity, With<PrimaryWindow>>,
) {
    // ── 1. Escape — close everything, go to Unfocused ──
    if keyboard.just_pressed(KeyCode::Escape) {
        inv.open = false;
        picker.open = false;
        go_unlocked(&mut cursor_options, &mut cursor_locked, &mut ui_focused, &mut lock_lost);
        return;
    }

    // ── 2. Browser pointer-lock loss (WASM) ──
    if lock_lost.0 && cursor_locked.0 {
        inv.open = false;
        picker.open = false;
        go_unlocked(&mut cursor_options, &mut cursor_locked, &mut ui_focused, &mut lock_lost);
        return;
    }
    lock_lost.0 = false;

    let any_panel = inv.open || picker.open;

    // ── 3. A panel is open → stay unlocked ──
    if any_panel {
        if cursor_locked.0 {
            unlock_cursor(&mut cursor_options, &mut cursor_locked);
            let center = Vec2::new(window.width() / 2.0, window.height() / 2.0);
            window.set_cursor_position(Some(center));
        }
        return;
    }

    // ── 4. No panel open. If one JUST closed this frame → re-lock ──
    // inv/picker are change-detected; if either flipped to false this
    // frame the panel was open last frame and closed now.
    if (inv.is_changed() || picker.is_changed()) && !cursor_locked.0 {
        go_locked(
            &mut cursor_options, &mut cursor_locked,
            &mut ui_focused, &mut keyboard,
            #[cfg(not(target_arch = "wasm32"))]
            &primary,
        );
        return;
    }

    // ── 5. Already locked → nothing to do ──
    if cursor_locked.0 {
        return;
    }

    // ── 6. Unlocked, no panel — click ANYWHERE to re-grab ──
    if mouse.just_pressed(MouseButton::Left) {
        go_locked(
            &mut cursor_options, &mut cursor_locked,
            &mut ui_focused, &mut keyboard,
            #[cfg(not(target_arch = "wasm32"))]
            &primary,
        );
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn unlock_cursor(opts: &mut CursorOptions, locked: &mut CursorLocked) {
    opts.visible = true;
    opts.grab_mode = CursorGrabMode::None;
    locked.0 = false;
}

/// Transition to the Unfocused state (no panel, cursor free).
fn go_unlocked(
    opts: &mut CursorOptions,
    locked: &mut CursorLocked,
    ui_focused: &mut UiFocused,
    lock_lost: &mut PointerLockLost,
) {
    unlock_cursor(opts, locked);
    ui_focused.0 = false;
    lock_lost.0 = false;
}

/// Transition to the Gameplay state (cursor locked, game has input).
fn go_locked(
    opts: &mut CursorOptions,
    locked: &mut CursorLocked,
    ui_focused: &mut UiFocused,
    keyboard: &mut ButtonInput<KeyCode>,
    #[cfg(not(target_arch = "wasm32"))]
    primary: &Query<Entity, With<PrimaryWindow>>,
) {
    opts.visible = false;
    opts.grab_mode = CursorGrabMode::Locked;
    locked.0 = true;
    // Clear stale flags: the panel's React element may have unmounted
    // before onMouseLeave could fire, leaving UiFocused stuck true.
    ui_focused.0 = false;
    // The webview may have consumed key-up events while it was first
    // responder.  Release everything so Bevy doesn't think keys are
    // still held ("stuck WASD" bug).
    keyboard.release_all();
    // Hand keyboard first-responder back to the Bevy content view.
    #[cfg(not(target_arch = "wasm32"))]
    if let Ok(entity) = primary.single() {
        crate::overlay::webview::refocus_content_view(entity);
    }
}
