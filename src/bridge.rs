//! Serializable types for the Rust ↔ JS UI bridge.
//!
//! On WASM these are passed via `wasm_bindgen` / `js_sys`.
//! On native these go over wry IPC.

use serde::{Deserialize, Serialize};

// ── Rust → JS state pushes ────────────────────────────────────────

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct SlotInfo {
    pub kind: &'static str, // "block" or "model"
    pub index: u32,
    pub name: String,
    pub color: [f32; 4],
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct HotbarState {
    pub active: usize,
    pub slots: Vec<SlotInfo>,
    pub layer: u8,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct BlockInfo {
    pub voxel: u8,
    pub name: String,
    pub color: [f32; 4],
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct MeshInfo {
    pub index: usize,
    pub layer: u8,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct InventoryStateJs {
    pub open: bool,
    pub builtin_blocks: Vec<BlockInfo>,
    pub custom_blocks: Vec<BlockInfo>,
    pub saved_meshes: Vec<MeshInfo>,
    pub layer: u8,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct ColorPickerStateJs {
    pub open: bool,
    pub r: f32,
    pub g: f32,
    pub b: f32,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ModeIndicatorStateJs {
    pub layer: u8,
    pub save_mode: bool,
    pub save_eligible: bool,
    pub entity_edit_mode: bool,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct ToastMessageJs {
    pub text: String,
    pub id: u64,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct PauseMenuStateJs {
    pub open: bool,
    pub save_status: Option<String>,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct DebugOverlayStateJs {
    pub visible: bool,
    pub fps: f64,
    pub frame_time_ms: f64,
    pub zoom_level: i32,
    pub tree_depth: u32,
    pub edit_depth: u32,
    pub visual_depth: u32,
    pub camera_anchor_depth: u32,
    pub camera_local: [f32; 3],
    pub fov: f32,
    pub node_count: usize,
}

#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(tag = "type", content = "data")]
#[serde(rename_all = "camelCase")]
pub enum GameStateUpdate {
    Hotbar(HotbarState),
    Inventory(InventoryStateJs),
    ColorPicker(ColorPickerStateJs),
    ModeIndicator(ModeIndicatorStateJs),
    Toast(ToastMessageJs),
    PauseMenu(PauseMenuStateJs),
    DebugOverlay(DebugOverlayStateJs),
}

// ── JS → Rust commands ────────────────────────────────────────────

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "cmd", rename_all = "camelCase")]
pub enum UiCommand {
    SelectHotbarSlot { slot: usize },
    AssignBlockToSlot { voxel: u8 },
    #[serde(rename_all = "camelCase")]
    AssignMeshToSlot { mesh_index: usize },
    SetColorPickerRgb { r: f32, g: f32, b: f32 },
    CreateBlock,
    ToggleInventory,
    ToggleColorPicker,
    CloseAllPanels,
    UiFocused { focused: bool },
    PointerLockLost,
    SaveGame,
    LoadGame,
    ClosePauseMenu,
}
