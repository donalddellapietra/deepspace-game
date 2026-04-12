//! Inventory state — press E to open/close.
//! The actual UI is rendered by the React overlay (see `overlay` module).

use bevy::prelude::*;

// ── Plugin ─────────────────────────────────────────────────────────

pub struct InventoryPlugin;

impl Plugin for InventoryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InventoryState>()
            .add_systems(Update, toggle_inventory);
    }
}

// ── State ──────────────────────────────────────────────────────────

#[derive(Resource, Default)]
pub struct InventoryState {
    pub open: bool,
}

// ── Toggle ─────────────────────────────────────────────────────────

fn toggle_inventory(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<InventoryState>,
) {
    if keyboard.just_pressed(KeyCode::KeyE) {
        state.open = !state.open;
    }
}
