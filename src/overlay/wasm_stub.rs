//! WASM stub: in the browser, React UI lives in the page DOM and talks
//! to WASM through `crate::bridge` directly. The native overlay's
//! WebView IPC is unnecessary, so these are no-ops.

use crate::bridge::GameStateUpdate;

pub fn push_state(_update: &GameStateUpdate) {}

pub fn clear_passthrough() {}
