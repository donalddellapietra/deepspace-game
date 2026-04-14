//! Platform-specific glue.
//!
//! Centralises OS-specific window/chrome manipulation so the rest of
//! the codebase can call a small cross-platform API without `cfg`
//! branching at each call site.
//!
//! On macOS, this delegates to [`macos`] (unsafe obj-c calls against
//! `NSWindow`). On other targets, the functions are empty stubs so
//! callers compile unchanged.

#[cfg(target_os = "macos")]
pub mod macos;

#[cfg(target_os = "macos")]
pub use macos::*;

/// Cross-platform stub: called once at window creation to perform any
/// OS-specific focus/key-window handshake. No-op on non-macOS.
#[cfg(not(target_os = "macos"))]
pub fn prepare_window(_window: &winit::window::Window) {}

/// Cross-platform stub: called once after the wry overlay WebView has
/// been constructed, to re-claim focus if the OS gave it to the
/// webview. No-op on non-macOS.
#[cfg(not(target_os = "macos"))]
pub fn after_overlay_created(_window: &winit::window::Window) {}

/// Cross-platform stub: refocus the window's content view without
/// reordering the window. Called on each cursor-lock transition on
/// macOS so keyboard events don't stay routed to the webview. No-op
/// on non-macOS.
#[cfg(not(target_os = "macos"))]
pub fn refocus_content_view(_window: &winit::window::Window) {}
