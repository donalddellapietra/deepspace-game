pub mod tools;

use bevy::prelude::*;

use crate::block::BlockType;

pub struct EditorPlugin;

impl Plugin for EditorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Hotbar>()
            .add_systems(Update, (
                tools::drill_down,
                tools::drill_up,
                tools::place_block,
                tools::remove_block,
                tools::cycle_hotbar_slot,
                tools::save_as_template,
            ));
    }
}

/// What a hotbar slot contains.
#[derive(Clone, Debug)]
pub enum HotbarItem {
    Block(BlockType),
    /// Index into ModelRegistry.models
    SavedModel(usize),
}

/// The 10 hotbar slots + which one is active.
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

    pub fn active_block(&self) -> Option<BlockType> {
        match self.active_item() {
            HotbarItem::Block(bt) => Some(*bt),
            _ => None,
        }
    }
}
