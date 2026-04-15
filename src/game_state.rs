//! UI-facing game state: hotbar, inventory, color picker, panels.
//!
//! This module owns all the state that the React overlay cares about
//! and handles UiCommands from the overlay.  It does NOT own the
//! world tree or camera — those stay in `App`.

use crate::bridge::*;
use crate::world::palette::{self, block, ColorRegistry};
use crate::world::tree::NodeId;

// ── Hotbar slot ──────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum HotbarItem {
    Block(u8),
    Mesh(usize),
}

// ── Saved meshes (subtrees captured with V key) ─────────────────

#[derive(Clone, Debug)]
pub struct SavedMesh {
    pub node_id: NodeId,
}

#[derive(Default)]
pub struct SavedMeshes {
    pub items: Vec<SavedMesh>,
}

impl SavedMeshes {
    /// Save a subtree. Caller must `ref_inc` the node_id first to pin it.
    /// Returns the index, or the existing index if already saved.
    pub fn save(&mut self, node_id: NodeId) -> usize {
        if let Some(idx) = self.items.iter().position(|s| s.node_id == node_id) {
            return idx;
        }
        let idx = self.items.len();
        self.items.push(SavedMesh { node_id });
        idx
    }
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
    pub last_hotbar_sent: Option<HotbarState>,
    pub last_inventory_sent: Option<InventoryStateJs>,
    pub last_color_picker_sent: Option<ColorPickerStateJs>,
    pub last_mode_indicator_sent: Option<ModeIndicatorStateJs>,
}

impl GameUiState {
    pub fn new() -> Self {
        // Default hotbar: first 9 builtin blocks + stone in slot 10
        let slots = std::array::from_fn(|i| {
            let voxel = if i < palette::BUILTINS.len() {
                palette::BUILTINS[i].0
            } else {
                block::STONE
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
            last_hotbar_sent: None,
            last_inventory_sent: None,
            last_color_picker_sent: None,
            last_mode_indicator_sent: None,
        }
    }

    pub fn any_panel_open(&self) -> bool {
        self.inventory_open || self.color_picker_open
    }

    /// The palette index of the active hotbar slot, or None for mesh slots.
    pub fn active_block_type(&self) -> Option<u8> {
        match &self.slots[self.active_slot] {
            HotbarItem::Block(idx) => Some(*idx),
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
                let idx = block::BUILTIN_COUNT + self.custom_blocks.len() as u8;
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
    pub fn push_to_overlay(&mut self, registry: &ColorRegistry) {
        use crate::overlay;

        let layer = self.zoom_level.max(0) as u8;

        // Hotbar
        let slots: Vec<SlotInfo> = self.slots.iter().map(|item| {
            match item {
                HotbarItem::Block(voxel) => {
                    SlotInfo {
                        kind: "block",
                        index: *voxel as u32,
                        name: registry.name(*voxel).to_string(),
                        color: registry.color(*voxel),
                    }
                }
                HotbarItem::Mesh(idx) => SlotInfo {
                    kind: "model", index: *idx as u32,
                    name: format!("Mesh #{}", idx),
                    color: [0.20, 0.85, 0.75, 0.7],
                },
            }
        }).collect();

        let hotbar = HotbarState {
            active: self.active_slot, slots, layer,
        };
        if self.last_hotbar_sent.as_ref() != Some(&hotbar) {
            overlay::push_state(&GameStateUpdate::Hotbar(hotbar.clone()));
            self.last_hotbar_sent = Some(hotbar);
        }

        // Inventory: builtin blocks from the palette
        let builtin_blocks: Vec<BlockInfo> = palette::BUILTINS.iter().map(|&(idx, name, color)| {
            BlockInfo { voxel: idx, name: name.to_string(), color }
        }).collect();

        let custom_blocks: Vec<BlockInfo> = self.custom_blocks.iter().enumerate().map(|(i, cb)| {
            BlockInfo { voxel: block::BUILTIN_COUNT + i as u8, name: cb.name.clone(), color: cb.color }
        }).collect();

        let inventory = InventoryStateJs {
            open: self.inventory_open, builtin_blocks, custom_blocks,
            saved_meshes: Vec::new(), layer,
        };
        if self.last_inventory_sent.as_ref() != Some(&inventory) {
            overlay::push_state(&GameStateUpdate::Inventory(inventory.clone()));
            self.last_inventory_sent = Some(inventory);
        }

        // Color picker
        let color_picker = ColorPickerStateJs {
            open: self.color_picker_open, r: self.picker_r, g: self.picker_g, b: self.picker_b,
        };
        if self.last_color_picker_sent.as_ref() != Some(&color_picker) {
            overlay::push_state(&GameStateUpdate::ColorPicker(color_picker.clone()));
            self.last_color_picker_sent = Some(color_picker);
        }

        // Mode indicator
        let mode_indicator = ModeIndicatorStateJs {
            layer, save_mode: false, save_eligible: false, entity_edit_mode: false,
        };
        if self.last_mode_indicator_sent.as_ref() != Some(&mode_indicator) {
            overlay::push_state(&GameStateUpdate::ModeIndicator(mode_indicator.clone()));
            self.last_mode_indicator_sent = Some(mode_indicator);
        }
    }
}
