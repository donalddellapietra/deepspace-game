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
    /// Camera position in root-frame world coords. `camera_local −
    /// render_path_origin` transformed back to root. Lets a reader
    /// read "I'm at (x, y, z) in the root cell" directly, without
    /// knowing the render path.
    pub camera_root_xyz: [f32; 3],
    /// `3^−anchor_depth`. One anchor cell's width in root-frame
    /// units. `camera_root_xyz[i] / anchor_cell_size_root` gives
    /// the camera's position in anchor cells from the root origin.
    pub anchor_cell_size_root: f32,
    /// `[s0, s1, ...]` — the camera's anchor slot path. Rendered
    /// as a short string in the overlay.
    pub anchor_slots_csv: String,
    /// `Cartesian` or `Body(ir, or)`.
    pub active_frame_kind: String,
    /// Render path (usually a prefix of the anchor). Empty when
    /// rendering from root.
    pub render_path_csv: String,
    /// Sphere state summary. Empty when the camera is NOT sphere-
    /// aware (`sphere=None`). Otherwise
    /// `"face=N body_d=D uvr_d=D uvr_off=[..]"`.
    pub sphere_state: String,
    /// Signed distance from the sphere center to the camera, in
    /// body-local units. Populated when the anchor path crosses a
    /// `CubedSphereBody` node. `NaN` when no body on path.
    pub sphere_dist_center: f32,
    /// Signed distance from camera to the outer shell, body-local.
    /// Positive = outside shell (above surface), negative = inside
    /// crust. `NaN` when no body on path.
    pub sphere_dist_outer: f32,
    /// Signed distance from camera to the inner shell, body-local.
    /// Positive = outside inner shell (inside crust or beyond),
    /// negative = inside the hollow core. `NaN` otherwise.
    pub sphere_dist_inner: f32,
    /// Body radii `[inner_r, outer_r]` in body-local (`[0, 1)`-frame)
    /// units. `[NaN, NaN]` when no body on path.
    pub sphere_radii: [f32; 2],
    /// Current sphere debug paint mode (0..=6). Mirrors the renderer
    /// uniform; cycled by F6.
    pub sphere_debug_mode: u32,
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
