pub mod save_mode;
pub mod toast;
pub mod tools;

use std::collections::HashMap;

use bevy::prelude::*;

use crate::block::BlockType;
use crate::world::render::{MAX_ZOOM, MIN_ZOOM};

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Hotbar>()
            .init_resource::<save_mode::SaveMode>()
            .init_resource::<save_mode::SavedMeshes>()
            .init_resource::<save_mode::SaveTintState>()
            .add_systems(Startup, save_mode::init_save_tint_material)
            .add_systems(
                Update,
                (
                    tools::zoom_in.before(crate::player::move_player),
                    tools::zoom_out.before(crate::player::move_player),
                    tools::reset_player.before(crate::player::move_player),
                    tools::cycle_hotbar_slot,
                    tools::remove_block.after(crate::player::sync_anchor_to_player),
                    tools::place_block.after(crate::player::sync_anchor_to_player),
                    save_mode::toggle_save_mode,
                    save_mode::save_on_click
                        .after(crate::player::sync_anchor_to_player)
                        .after(crate::interaction::update_target),
                    // Tinting reads the raycast target and has to
                    // observe the entities the renderer spawned
                    // this frame, so it runs strictly after both.
                    save_mode::update_save_tint
                        .after(crate::interaction::update_target)
                        .after(crate::world::render::render_world),
                    toast::tick_toasts,
                ),
            );
    }
}

/// What a hotbar slot contains. A `Model` entry points into
/// `SavedMeshes` by index — we only store the index (not the
/// `NodeId`) so the hotbar stays stable when the saved list grows.
#[derive(Clone, Debug)]
pub enum HotbarItem {
    Block(BlockType),
    Model(usize),
}

/// Per-layer hotbar. Zooming swaps which `[HotbarItem; 10]` is
/// active, so assignments made at zoom L don't leak into zoom L+1.
/// The default for every layer is the ten block types in order;
/// only saved meshes make per-layer segregation interesting, but
/// we apply it uniformly so the rule is easy to reason about.
#[derive(Resource)]
pub struct Hotbar {
    per_layer: HashMap<u8, [HotbarItem; 10]>,
    pub active: usize,
}

fn default_slots() -> [HotbarItem; 10] {
    BlockType::ALL.map(HotbarItem::Block)
}

impl Default for Hotbar {
    fn default() -> Self {
        let mut per_layer = HashMap::new();
        for layer in MIN_ZOOM..=MAX_ZOOM {
            per_layer.insert(layer, default_slots());
        }
        Self {
            per_layer,
            active: 0,
        }
    }
}

impl Hotbar {
    pub fn slots(&self, layer: u8) -> &[HotbarItem; 10] {
        self.per_layer
            .get(&layer)
            .expect("hotbar pre-populated for every legal zoom layer")
    }

    pub fn slots_mut(&mut self, layer: u8) -> &mut [HotbarItem; 10] {
        self.per_layer
            .get_mut(&layer)
            .expect("hotbar pre-populated for every legal zoom layer")
    }

    pub fn active_item(&self, layer: u8) -> &HotbarItem {
        &self.slots(layer)[self.active]
    }
}
