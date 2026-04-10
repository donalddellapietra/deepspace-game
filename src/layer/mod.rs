// Layer management is now handled by WorldState in world/mod.rs.
// This module exists only for backwards compatibility with plugins that reference it.

use bevy::prelude::*;

pub struct LayerPlugin;

impl Plugin for LayerPlugin {
    fn build(&self, _app: &mut App) {
        // WorldState is registered by WorldPlugin
    }
}
