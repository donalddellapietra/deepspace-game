//! Color picker state — the actual UI is rendered by the React overlay.
//! This module keeps the Bevy resource that other systems read.

use bevy::prelude::*;

use crate::inventory::InventoryState;

// ── Plugin ─────────────────────────────────────────────────────────

pub struct ColorPickerPlugin;

impl Plugin for ColorPickerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ColorPickerState>()
            .add_systems(Update, toggle_color_picker);
    }
}

// ── State ──────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct ColorPickerState {
    pub open: bool,
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

impl Default for ColorPickerState {
    fn default() -> Self {
        Self {
            open: false,
            r: 0.5,
            g: 0.5,
            b: 0.5,
        }
    }
}

impl ColorPickerState {
    pub fn current_color(&self) -> Color {
        Color::srgb(self.r, self.g, self.b)
    }
}

// ── Toggle ─────────────────────────────────────────────────────────

fn toggle_color_picker(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut inv: ResMut<InventoryState>,
    mut state: ResMut<ColorPickerState>,
) {
    if keyboard.just_pressed(KeyCode::KeyC) {
        state.open = !state.open;
        if state.open {
            inv.open = false;
        }
    }
}
