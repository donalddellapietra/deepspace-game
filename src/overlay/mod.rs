//! React UI overlay — Bevy plugin that bridges game state to/from a
//! React frontend via wasm-bindgen (WASM) or WebSocket (native, later).
//!
//! The plugin:
//! * Pushes game state to JS whenever Bevy resources change (change detection)
//! * Polls a JS command queue each frame and applies mutations to Bevy resources

pub mod bridge;
#[cfg(not(target_arch = "wasm32"))]
pub mod webview;
#[cfg(not(target_arch = "wasm32"))]
pub mod ws_server;

use bevy::prelude::*;
use serde_json;

use bridge::*;

use crate::block::{BlockType, BslMaterial, Palette, PaletteEntry};
use crate::editor::save_mode::{save_mode_eligible, SaveMode, SavedMeshes};
use crate::editor::{Hotbar, HotbarItem};
use crate::inventory::InventoryState;
use crate::ui::color_picker::ColorPickerState;
use crate::world::view::target_layer_for;
use crate::world::CameraZoom;

// ── Plugin ────────────────────────────────────────────────────────

pub struct OverlayPlugin;

impl Plugin for OverlayPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(UiFocused(false))
            .insert_resource(PointerLockLost(false))
            .insert_resource(ToastIdCounter(0))
            .add_message::<ToastEvent>()
            .add_systems(
                Update,
                (
                    push_hotbar,
                    push_mode_indicator,
                    push_inventory,
                    push_color_picker,
                    push_toasts,
                    poll_ui_commands,
                ),
            );

        #[cfg(not(target_arch = "wasm32"))]
        {
            app.init_non_send_resource::<webview::WebViewHolder>()
                .add_systems(Update, (
                    webview::create_overlay_webview,
                    webview::resize_overlay_webview,
                    flush_state_to_webview,
                ))
                // Inject webview-forwarded key/mouse events into
                // ButtonInput in PreUpdate, right after Bevy's own
                // input processing.  This makes ButtonInput the
                // single source of truth for all game systems.
                .add_systems(
                    PreUpdate,
                    inject_webview_input
                        .after(bevy::input::InputSystems),
                );
        }
    }
}

// ── Resources ─────────────────────────────────────────────────────

/// Whether the React UI has pointer focus (mouse is over an interactive element).
#[derive(Resource)]
pub struct UiFocused(pub bool);

/// Set by JS when the browser exits Pointer Lock (e.g., Escape key).
/// `sync_cursor` reads and clears this each frame.
#[derive(Resource)]
pub struct PointerLockLost(pub bool);

/// Monotonic toast ID counter.
#[derive(Resource)]
struct ToastIdCounter(u64);

/// Fire this message to show a toast via the React UI.
#[derive(Message, Clone)]
pub struct ToastEvent {
    pub text: String,
}

// ── JS interop ────────────────────────────────────────────────────

#[cfg(target_arch = "wasm32")]
mod js {
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen]
    extern "C" {
        #[wasm_bindgen(js_namespace = window, js_name = "__onGameState")]
        pub fn push_to_js(json: &str);

        #[wasm_bindgen(js_namespace = window, js_name = "__pollUiCommands")]
        pub fn poll_from_js() -> String;
    }
}

// ── Transport: unified push/poll across WASM and native ─────────

#[cfg(target_arch = "wasm32")]
fn push_state(update: &GameStateUpdate) {
    match serde_json::to_string(update) {
        Ok(json) => js::push_to_js(&json),
        Err(e) => warn!("overlay: failed to serialize state update: {e}"),
    }
}

#[cfg(target_arch = "wasm32")]
fn poll_commands() -> Vec<UiCommand> {
    let raw = js::poll_from_js();
    serde_json::from_str(&raw).unwrap_or_default()
}

/// On native, push_state writes JSON to a buffer. A separate system
/// flushes the buffer to the webview via evaluate_script each frame.
#[cfg(not(target_arch = "wasm32"))]
mod native_bridge {
    use std::sync::Mutex;

    static OUTBOX: Mutex<Vec<String>> = Mutex::new(Vec::new());

    pub fn push(json: &str) {
        if let Ok(mut outbox) = OUTBOX.lock() {
            outbox.push(json.to_owned());
        }
    }

    pub fn drain_outbox() -> Vec<String> {
        let Ok(mut outbox) = OUTBOX.lock() else {
            return Vec::new();
        };
        outbox.drain(..).collect()
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn push_state(update: &GameStateUpdate) {
    match serde_json::to_string(update) {
        Ok(json) => native_bridge::push(&json),
        Err(e) => eprintln!("overlay: failed to serialize state update: {e}"),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn poll_commands() -> Vec<UiCommand> {
    let mut cmds = Vec::new();
    for raw in webview::drain_ipc_commands() {
        match serde_json::from_str::<Vec<UiCommand>>(&raw) {
            Ok(batch) => cmds.extend(batch),
            Err(_) => {
                if let Ok(cmd) = serde_json::from_str::<UiCommand>(&raw) {
                    cmds.push(cmd);
                } else {
                    eprintln!("overlay: failed to parse command: {raw}");
                }
            }
        }
    }
    cmds
}

/// Flush buffered state updates to the webview via evaluate_script.
#[cfg(not(target_arch = "wasm32"))]
fn flush_state_to_webview(holder: NonSend<webview::WebViewHolder>) {
    let Some(ref wv) = holder.webview else {
        return;
    };
    for json in native_bridge::drain_outbox() {
        // Calls the same window.__onGameState that the React app listens to
        let js = format!("window.__onGameState && window.__onGameState({})", json);
        let _ = wv.evaluate_script(&js);
    }
}

// ── State push systems ────────────────────────────────────────────

fn push_hotbar(
    hotbar: Res<Hotbar>,
    zoom: Res<CameraZoom>,
    palette: Option<Res<Palette>>,
) {
    if !hotbar.is_changed() && !zoom.is_changed() {
        return;
    }

    let layer_slots = hotbar.slots(zoom.layer);
    let slots: Vec<SlotInfo> = layer_slots
        .iter()
        .map(|item| match item {
            HotbarItem::Block(voxel) => {
                let (name, color) = palette
                    .as_ref()
                    .and_then(|p| p.get(*voxel))
                    .map(|e| {
                        let c = color_to_rgba(e.color);
                        (e.name.clone(), c)
                    })
                    .unwrap_or_else(|| (format!("Block {}", voxel), [0.3, 0.3, 0.3, 1.0]));
                SlotInfo {
                    kind: "block",
                    index: *voxel as u32,
                    name,
                    color,
                }
            }
            HotbarItem::Model(idx) => SlotInfo {
                kind: "model",
                index: *idx as u32,
                name: format!("Mesh #{}", idx),
                color: [0.20, 0.85, 0.75, 0.7],
            },
        })
        .collect();

    push_state(&GameStateUpdate::Hotbar(HotbarState {
        active: hotbar.active,
        slots,
        layer: zoom.layer,
    }));
}

fn push_mode_indicator(zoom: Res<CameraZoom>, save_mode: Res<SaveMode>) {
    if !zoom.is_changed() && !save_mode.is_changed() {
        return;
    }

    push_state(&GameStateUpdate::ModeIndicator(ModeIndicatorStateJs {
        layer: zoom.layer,
        save_mode: save_mode.active,
        save_eligible: save_mode_eligible(zoom.layer),
    }));
}

fn push_inventory(
    inv: Res<InventoryState>,
    palette: Option<Res<Palette>>,
    saved: Res<SavedMeshes>,
    zoom: Res<CameraZoom>,
) {
    if !inv.is_changed() && !saved.is_changed() && !zoom.is_changed() {
        // Also push when palette changes (custom blocks added)
        let palette_changed = palette.as_ref().is_some_and(|p| p.is_changed());
        if !palette_changed {
            return;
        }
    }

    let builtin_count = BlockType::ALL.len() as u8;
    let mut builtin_blocks = Vec::new();
    let mut custom_blocks = Vec::new();

    if let Some(ref pal) = palette {
        for (voxel, entry) in pal.iter() {
            let info = BlockInfo {
                voxel,
                name: entry.name.clone(),
                color: color_to_rgba(entry.color),
            };
            if voxel <= builtin_count {
                builtin_blocks.push(info);
            } else {
                custom_blocks.push(info);
            }
        }
    }

    let target = target_layer_for(zoom.layer);
    let saved_meshes: Vec<MeshInfo> = saved
        .items
        .iter()
        .enumerate()
        .filter(|(_, m)| m.layer == target)
        .map(|(i, m)| MeshInfo {
            index: i,
            layer: m.layer,
        })
        .collect();

    push_state(&GameStateUpdate::Inventory(InventoryStateJs {
        open: inv.open,
        builtin_blocks,
        custom_blocks,
        saved_meshes,
        layer: zoom.layer,
    }));
}

fn push_color_picker(picker: Res<ColorPickerState>) {
    if !picker.is_changed() {
        return;
    }

    push_state(&GameStateUpdate::ColorPicker(ColorPickerStateJs {
        open: picker.open,
        r: picker.r,
        g: picker.g,
        b: picker.b,
    }));
}

fn push_toasts(
    mut events: MessageReader<ToastEvent>,
    mut counter: ResMut<ToastIdCounter>,
) {
    for event in events.read() {
        counter.0 += 1;
        push_state(&GameStateUpdate::Toast(ToastMessageJs {
            text: event.text.clone(),
            id: counter.0,
        }));
    }
}

// ── Command polling ───────────────────────────────────────────────

fn poll_ui_commands(
    mut hotbar: ResMut<Hotbar>,
    zoom: Res<CameraZoom>,
    mut inv: ResMut<InventoryState>,
    mut picker: ResMut<ColorPickerState>,
    mut palette: ResMut<Palette>,
    mut materials: ResMut<Assets<BslMaterial>>,
    mut ui_focused: ResMut<UiFocused>,
    mut lock_lost: ResMut<PointerLockLost>,
) {
    for cmd in poll_commands() {
        match cmd {
            UiCommand::SelectHotbarSlot { slot } => {
                if slot < 10 {
                    hotbar.active = slot;
                }
            }
            UiCommand::AssignBlockToSlot { voxel } => {
                let active = hotbar.active;
                hotbar.slots_mut(zoom.layer)[active] = HotbarItem::Block(voxel);
            }
            UiCommand::AssignMeshToSlot { mesh_index } => {
                let active = hotbar.active;
                hotbar.slots_mut(zoom.layer)[active] = HotbarItem::Model(mesh_index);
            }
            UiCommand::SetColorPickerRgb { r, g, b } => {
                picker.r = r;
                picker.g = g;
                picker.b = b;
            }
            UiCommand::CreateBlock => {
                let name = format!("Custom #{}", palette.len() - BlockType::ALL.len() + 1);
                let color = picker.current_color();
                palette.register(
                    PaletteEntry {
                        name: name.clone(),
                        color,
                        roughness: 0.9,
                        metallic: 0.0,
                        alpha_mode: AlphaMode::Opaque,
                    },
                    &mut materials,
                );
                info!("Created custom block: {name}");
                picker.open = false;
            }
            UiCommand::ToggleInventory => {
                inv.open = !inv.open;
                if inv.open {
                    picker.open = false;
                }
            }
            UiCommand::ToggleColorPicker => {
                picker.open = !picker.open;
                if picker.open {
                    inv.open = false;
                }
            }
            UiCommand::CloseAllPanels => {
                inv.open = false;
                picker.open = false;
            }
            UiCommand::UiFocused { focused } => {
                ui_focused.0 = focused;
            }
            UiCommand::PointerLockLost => {
                lock_lost.0 = true;
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────

fn color_to_rgba(color: Color) -> [f32; 4] {
    let c = color.to_srgba();
    [c.red, c.green, c.blue, c.alpha]
}

// ── Centralised input injection (native only) ────────────────────
//
// The webview forwards every key and mouse-button event via IPC.
// This system drains those queues and injects them into Bevy's
// `ButtonInput`, making it the single source of truth.  It runs in
// `PreUpdate` after `InputSystems` so that all `Update` systems
// see a complete picture.

#[cfg(not(target_arch = "wasm32"))]
fn inject_webview_input(
    mut keyboard: ResMut<ButtonInput<KeyCode>>,
    mut mouse: ResMut<ButtonInput<MouseButton>>,
) {
    for (code, pressed) in webview::drain_forwarded_keys() {
        if let Some(key) = js_code_to_keycode(&code) {
            if pressed {
                keyboard.press(key);
            } else {
                keyboard.release(key);
            }
        }
    }
    for (button, pressed) in webview::drain_forwarded_mouse() {
        if let Some(btn) = js_button_to_mouse(button) {
            if pressed {
                mouse.press(btn);
            } else {
                mouse.release(btn);
            }
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn js_code_to_keycode(code: &str) -> Option<KeyCode> {
    Some(match code {
        "KeyW" => KeyCode::KeyW,
        "KeyA" => KeyCode::KeyA,
        "KeyS" => KeyCode::KeyS,
        "KeyD" => KeyCode::KeyD,
        "KeyE" => KeyCode::KeyE,
        "KeyC" => KeyCode::KeyC,
        "KeyV" => KeyCode::KeyV,
        "KeyQ" => KeyCode::KeyQ,
        "KeyF" => KeyCode::KeyF,
        "Space" => KeyCode::Space,
        "ShiftLeft" => KeyCode::ShiftLeft,
        "ShiftRight" => KeyCode::ShiftRight,
        "Escape" => KeyCode::Escape,
        "Digit1" => KeyCode::Digit1,
        "Digit2" => KeyCode::Digit2,
        "Digit3" => KeyCode::Digit3,
        "Digit4" => KeyCode::Digit4,
        "Digit5" => KeyCode::Digit5,
        "Digit6" => KeyCode::Digit6,
        "Digit7" => KeyCode::Digit7,
        "Digit8" => KeyCode::Digit8,
        "Digit9" => KeyCode::Digit9,
        "Digit0" => KeyCode::Digit0,
        _ => return None,
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn js_button_to_mouse(button: u8) -> Option<MouseButton> {
    match button {
        0 => Some(MouseButton::Left),
        1 => Some(MouseButton::Middle),
        2 => Some(MouseButton::Right),
        _ => None,
    }
}
