pub mod tools;

use bevy::prelude::*;

use crate::block::BlockType;

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Hotbar>().add_systems(
            Update,
            (
                tools::zoom_in,
                tools::zoom_out,
                tools::reset_player,
                tools::cycle_hotbar_slot,
                tools::remove_block,
                tools::place_block,
            ),
        );
    }
}

/// What a hotbar slot contains. v1 only supports block types;
/// saved models are deferred.
#[derive(Clone, Debug)]
pub enum HotbarItem {
    Block(BlockType),
}

#[derive(Resource)]
pub struct Hotbar {
    pub slots: [HotbarItem; 10],
    pub active: usize,
}

impl Default for Hotbar {
    fn default() -> Self {
        Self {
            slots: BlockType::ALL.map(HotbarItem::Block),
            active: 0,
        }
    }
}

impl Hotbar {
    pub fn active_item(&self) -> &HotbarItem {
        &self.slots[self.active]
    }

    pub fn active_block(&self) -> BlockType {
        match self.active_item() {
            HotbarItem::Block(bt) => *bt,
        }
    }
}
