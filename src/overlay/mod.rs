//! Native: re-export the wry/macOS WKWebView overlay.
//! WASM: stubs that no-op — React UI lives in the page DOM directly.

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::*;

#[cfg(target_arch = "wasm32")]
mod wasm_stub;
#[cfg(target_arch = "wasm32")]
pub use wasm_stub::*;
