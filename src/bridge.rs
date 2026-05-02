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
    pub voxel: u16,
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

/// Crosshair reticle state. The crosshair is an HTML overlay element
/// — it's always rendered at physical (native) resolution regardless
/// of the 3D path's internal scaling or temporal resolves, so voxel
/// aliasing / TAA blur never touches it. The renderer pushes this
/// each time the "aimed-at a target" bit flips; CSS toggles the
/// hit-state color.
#[derive(Serialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct CrosshairStateJs {
    /// True when the center-pixel ray hit any voxel this frame.
    /// Derived from the same CPU raycast that feeds `update_highlight`,
    /// so the crosshair stays in sync with cursor-aim feedback without
    /// a separate ray cast.
    pub on_target: bool,
    /// True when the crosshair should be drawn at all. False during
    /// menus / unlocked cursor / harness runs; the CSS hides the
    /// element. Bundled here so the UI doesn't need a second state
    /// channel for "gameplay mode vs. menu".
    pub visible: bool,
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
    /// Camera position in root-frame world coords. Lets the reader
    /// read "I'm at (x, y, z) in the root cell" directly, without
    /// knowing the render path.
    pub camera_root_xyz: [f32; 3],
    /// `WORLD_SIZE / 3^anchor_depth`. One anchor cell's width in
    /// root-frame units. `camera_root_xyz[i] / anchor_cell_size_root`
    /// gives the camera's position in anchor cells from origin.
    pub anchor_cell_size_root: f32,
    /// `[s0, s1, ...]` — the camera's anchor slot path. Rendered
    /// as a short string in the overlay.
    pub anchor_slots_csv: String,
    /// `Cartesian` or `WrappedPlane(...)` or `TangentBlock`.
    pub active_frame_kind: String,
    /// Render path (usually a prefix of the anchor). Empty when
    /// rendering from root.
    pub render_path_csv: String,
    /// `true` if the camera's anchor path crosses any
    /// `NodeKind::TangentBlock`. Useful for debugging rotation
    /// chain handling.
    pub tb_on_anchor_path: bool,
    /// Cumulative Y rotation along the anchor path, in degrees.
    /// (Approximation: only Y-axis rotations show meaningfully.)
    pub anchor_cumulative_yaw_deg: f32,
    /// Monotonic counter incremented each time the user presses `[`
    /// while the debug overlay is visible. The UI watches for changes
    /// and copies the formatted overlay text to the clipboard.
    pub copy_seq: u64,
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
    Crosshair(CrosshairStateJs),
}

// ── JS → Rust commands ────────────────────────────────────────────

#[derive(Deserialize, Clone, Debug)]
#[serde(tag = "cmd", rename_all = "camelCase")]
pub enum UiCommand {
    SelectHotbarSlot { slot: usize },
    AssignBlockToSlot { voxel: u16 },
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
