//! Native: re-export the wry/macOS WKWebView overlay bridge.
//! WASM: re-export the in-page bridge (window.__onGameState +
//! window.__pollUiCommands).

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(target_arch = "wasm32")]
mod web;
#[cfg(target_arch = "wasm32")]
pub use web::*;
