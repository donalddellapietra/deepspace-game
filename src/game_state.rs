//! UI-facing game state: hotbar, inventory, color picker, panels.
//!
//! This module owns all the state that the React overlay cares about
//! and handles UiCommands from the overlay.  It does NOT own the
//! world tree or camera — those stay in `App`.

use crate::bridge::*;
use crate::world::tree::BlockType;

// ── Hotbar slot ──────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum HotbarItem {
    Block(u8),
    Mesh(usize),
}

// ── Custom block palette entry ───────────────────────────────────

#[derive(Clone, Debug)]
pub struct CustomBlock {
    pub name: String,
    pub color: [f32; 4],
}

// ── Main state ───────────────────────────────────────────────────

pub struct GameUiState {
    pub active_slot: usize,
    pub slots: [HotbarItem; 10],
    pub inventory_open: bool,
    pub color_picker_open: bool,
    pub picker_r: f32,
    pub picker_g: f32,
    pub picker_b: f32,
    pub custom_blocks: Vec<CustomBlock>,
    pub ui_focused: bool,
    pub zoom_level: i32,
}

impl GameUiState {
    pub fn new() -> Self {
        let slots = std::array::from_fn(|i| {
            let voxel = match i {
                0 => BlockType::Stone as u8,
                1 => BlockType::Dirt as u8,
                2 => BlockType::Grass as u8,
                3 => BlockType::Wood as u8,
                4 => BlockType::Leaf as u8,
                5 => BlockType::Sand as u8,
                6 => BlockType::Brick as u8,
                7 => BlockType::Metal as u8,
                8 => BlockType::Glass as u8,
                _ => BlockType::Stone as u8,
            };
            HotbarItem::Block(voxel)
        });

        Self {
            active_slot: 0,
            slots,
            inventory_open: false,
            color_picker_open: false,
            picker_r: 0.5,
            picker_g: 0.5,
            picker_b: 0.5,
            custom_blocks: Vec::new(),
            ui_focused: false,
            zoom_level: 0,
        }
    }

    pub fn any_panel_open(&self) -> bool {
        self.inventory_open || self.color_picker_open
    }

    pub fn active_block_type(&self) -> Option<BlockType> {
        match &self.slots[self.active_slot] {
            HotbarItem::Block(voxel) => BlockType::from_index(*voxel),
            HotbarItem::Mesh(_) => None,
        }
    }

    /// Handle a UiCommand. Returns true if panel open/close state changed.
    pub fn handle_command(&mut self, cmd: UiCommand) -> bool {
        let panels_were_open = self.any_panel_open();

        match cmd {
            UiCommand::SelectHotbarSlot { slot } => {
                if slot < 10 { self.active_slot = slot; }
            }
            UiCommand::AssignBlockToSlot { voxel } => {
                self.slots[self.active_slot] = HotbarItem::Block(voxel);
            }
            UiCommand::AssignMeshToSlot { mesh_index } => {
                self.slots[self.active_slot] = HotbarItem::Mesh(mesh_index);
            }
            UiCommand::SetColorPickerRgb { r, g, b } => {
                self.picker_r = r;
                self.picker_g = g;
                self.picker_b = b;
            }
            UiCommand::CreateBlock => {
                let idx = BlockType::ALL.len() as u8 + self.custom_blocks.len() as u8;
                let name = format!("Custom #{}", self.custom_blocks.len() + 1);
                let color = [self.picker_r, self.picker_g, self.picker_b, 1.0];
                self.custom_blocks.push(CustomBlock { name, color });
                self.slots[self.active_slot] = HotbarItem::Block(idx);
                self.color_picker_open = false;
                log::info!("Created custom block voxel={idx}");
            }
            UiCommand::ToggleInventory => {
                self.inventory_open = !self.inventory_open;
                if self.inventory_open { self.color_picker_open = false; }
            }
            UiCommand::ToggleColorPicker => {
                self.color_picker_open = !self.color_picker_open;
                if self.color_picker_open { self.inventory_open = false; }
            }
            UiCommand::CloseAllPanels => {
                self.inventory_open = false;
                self.color_picker_open = false;
            }
            UiCommand::UiFocused { focused } => {
                self.ui_focused = focused;
            }
            _ => {}
        }

        panels_were_open != self.any_panel_open()
    }

    /// Handle a key press. Returns true if panel state changed.
    pub fn handle_key(&mut self, code: winit::keyboard::KeyCode, pressed: bool) -> bool {
        use winit::keyboard::KeyCode;
        if !pressed { return false; }

        let panels_were_open = self.any_panel_open();

        match code {
            KeyCode::KeyE => {
                self.inventory_open = !self.inventory_open;
                if self.inventory_open { self.color_picker_open = false; }
            }
            KeyCode::KeyC => {
                self.color_picker_open = !self.color_picker_open;
                if self.color_picker_open { self.inventory_open = false; }
            }
            KeyCode::Digit1 => { self.active_slot = 0; }
            KeyCode::Digit2 => { self.active_slot = 1; }
            KeyCode::Digit3 => { self.active_slot = 2; }
            KeyCode::Digit4 => { self.active_slot = 3; }
            KeyCode::Digit5 => { self.active_slot = 4; }
            KeyCode::Digit6 => { self.active_slot = 5; }
            KeyCode::Digit7 => { self.active_slot = 6; }
            KeyCode::Digit8 => { self.active_slot = 7; }
            KeyCode::Digit9 => { self.active_slot = 8; }
            KeyCode::Digit0 => { self.active_slot = 9; }
            _ => {}
        }

        panels_were_open != self.any_panel_open()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn push_to_overlay(&mut self) {
        use crate::overlay;

        let layer = self.zoom_level.max(0) as u8;

        // Hotbar
        let slots: Vec<SlotInfo> = self.slots.iter().map(|item| {
            match item {
                HotbarItem::Block(voxel) => {
                    let (name, color) = self.block_info(*voxel);
                    SlotInfo { kind: "block", index: *voxel as u32, name, color }
                }
                HotbarItem::Mesh(idx) => SlotInfo {
                    kind: "model", index: *idx as u32,
                    name: format!("Mesh #{}", idx),
                    color: [0.20, 0.85, 0.75, 0.7],
                },
            }
        }).collect();

        overlay::push_state(&GameStateUpdate::Hotbar(HotbarState {
            active: self.active_slot, slots, layer,
        }));

        // Inventory
        let builtin_blocks: Vec<BlockInfo> = BlockType::ALL.iter().map(|bt| {
            BlockInfo { voxel: *bt as u8, name: format!("{:?}", bt), color: builtin_block_color(*bt) }
        }).collect();

        let custom_blocks: Vec<BlockInfo> = self.custom_blocks.iter().enumerate().map(|(i, cb)| {
            BlockInfo { voxel: BlockType::ALL.len() as u8 + i as u8, name: cb.name.clone(), color: cb.color }
        }).collect();

        overlay::push_state(&GameStateUpdate::Inventory(InventoryStateJs {
            open: self.inventory_open, builtin_blocks, custom_blocks,
            saved_meshes: Vec::new(), layer,
        }));

        // Color picker
        overlay::push_state(&GameStateUpdate::ColorPicker(ColorPickerStateJs {
            open: self.color_picker_open, r: self.picker_r, g: self.picker_g, b: self.picker_b,
        }));

        // Mode indicator
        overlay::push_state(&GameStateUpdate::ModeIndicator(ModeIndicatorStateJs {
            layer, save_mode: false, save_eligible: false, entity_edit_mode: false,
        }));
    }

    fn block_info(&self, voxel: u8) -> (String, [f32; 4]) {
        let builtin_count = BlockType::ALL.len() as u8;
        if voxel < builtin_count {
            if let Some(bt) = BlockType::from_index(voxel) {
                (format!("{:?}", bt), builtin_block_color(bt))
            } else {
                (format!("Block {}", voxel), [0.3, 0.3, 0.3, 1.0])
            }
        } else {
            let custom_idx = (voxel - builtin_count) as usize;
            if let Some(cb) = self.custom_blocks.get(custom_idx) {
                (cb.name.clone(), cb.color)
            } else {
                (format!("Custom {}", voxel), [0.3, 0.3, 0.3, 1.0])
            }
        }
    }
}

fn builtin_block_color(bt: BlockType) -> [f32; 4] {
    match bt {
        BlockType::Stone => [0.5, 0.5, 0.5, 1.0],
        BlockType::Dirt  => [0.6, 0.4, 0.2, 1.0],
        BlockType::Grass => [0.3, 0.7, 0.2, 1.0],
        BlockType::Wood  => [0.6, 0.4, 0.15, 1.0],
        BlockType::Leaf  => [0.2, 0.6, 0.1, 1.0],
        BlockType::Sand  => [0.9, 0.85, 0.6, 1.0],
        BlockType::Water => [0.2, 0.4, 0.9, 0.7],
        BlockType::Brick => [0.7, 0.3, 0.2, 1.0],
        BlockType::Metal => [0.8, 0.8, 0.85, 1.0],
        BlockType::Glass => [0.7, 0.85, 0.9, 0.4],
    }
}
