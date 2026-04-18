//! WASM bridge: in the browser, React UI lives in the page DOM and
//! talks to WASM through `window.__onGameState` (state) and
//! `window.__pollUiCommands` (commands). The native overlay's
//! WebView IPC isn't used.

use wasm_bindgen::JsValue;
use wasm_bindgen::JsCast;

use crate::bridge::{GameStateUpdate, UiCommand};

/// Push a state update into the React UI by calling
/// `window.__onGameState(json)`. Fail-soft if React hasn't registered
/// the handler yet (e.g. during the WASM bootstrap window before
/// `useTransport` runs).
pub fn push_state(update: &GameStateUpdate) {
    let Ok(json) = serde_json::to_string(update) else { return };
    let Some(window) = web_sys::window() else { return };
    let Ok(handler) = js_sys::Reflect::get(&window, &JsValue::from_str("__onGameState")) else {
        return;
    };
    let Ok(func) = handler.dyn_into::<js_sys::Function>() else { return };
    let _ = func.call1(&window, &JsValue::from_str(&json));
}

/// Native cursor-grab passthrough doesn't apply in the browser —
/// pointer-lock is the equivalent and winit handles it.
pub fn clear_passthrough() {}

/// Drain commands queued by the React UI via `window.__pollUiCommands()`.
/// React pushes UiCommands by calling its own `sendCommand` which
/// queues; this Rust-side poll pulls + parses them per frame.
pub fn poll_commands() -> Vec<UiCommand> {
    let Some(window) = web_sys::window() else { return Vec::new() };
    let Ok(handler) = js_sys::Reflect::get(&window, &JsValue::from_str("__pollUiCommands")) else {
        return Vec::new();
    };
    let Ok(func) = handler.dyn_into::<js_sys::Function>() else { return Vec::new() };
    let Ok(result) = func.call0(&window) else { return Vec::new() };
    let Some(json) = result.as_string() else { return Vec::new() };
    serde_json::from_str(&json).unwrap_or_default()
}
