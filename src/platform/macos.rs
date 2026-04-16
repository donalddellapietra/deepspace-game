//! macOS-specific window glue.
//!
//! Two concerns are handled here:
//!
//! 1. **Key-window handshake.** When a winit-created `NSWindow` first
//!    appears, macOS does not always make it the key window (title
//!    bar stays grayed out, keyboard events don't route correctly).
//!    [`prepare_window`] calls `makeKeyAndOrderFront:` + sets the
//!    content view as first responder to force the handshake.
//!
//! 2. **WKWebView stealing first responder.** The wry overlay adds a
//!    `WKWebView` as a subview of our content view. As a side effect
//!    of being inserted into the view hierarchy, the webview often
//!    becomes first responder — so keyboard events go to it instead
//!    of back to winit. [`after_overlay_created`] re-runs the
//!    key-window handshake after the webview is live so focus returns
//!    to the content view.
//!
//! The unsafe obj-c here is deliberately minimal and mirrors what
//! `overlay.rs` historically did inline; it should not be "improved"
//! without a matching audit of the wry/winit interaction.

use objc2_app_kit::NSView;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;

/// Make the `NSWindow` key and set its content view as first
/// responder. Call once at window creation.
pub fn prepare_window(window: &Window) {
    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
        return;
    };
    // SAFETY: The pointers come from winit's live window and are
    // valid for the duration of this call.
    unsafe {
        let ns_view = appkit.ns_view.as_ptr() as *mut NSView;
        let Some(ns_window) = (*ns_view).window() else {
            return;
        };
        ns_window.makeKeyAndOrderFront(None);
        ns_window.makeFirstResponder(Some(&*ns_view));
    }
}

/// Re-claim key status and content-view first responder after the
/// wry `WKWebView` has been added as a subview (inserting it often
/// steals first responder from the content view). Call once after
/// the webview is constructed.
pub fn after_overlay_created(window: &Window) {
    // Same handshake as `prepare_window` — wry's webview insertion can
    // both drop key status and steal first responder, so we redo both.
    prepare_window(window);
}

/// Best-effort: make the `NSWindow`'s content view the first
/// responder, returning keyboard events to winit. Useful when only
/// the responder half of the handshake is needed (e.g. on each
/// cursor-lock transition) without reordering the window.
pub fn refocus_content_view(window: &Window) {
    let Ok(handle) = window.window_handle() else {
        return;
    };
    let RawWindowHandle::AppKit(appkit) = handle.as_raw() else {
        return;
    };
    // SAFETY: The pointers come from winit's live window and are
    // valid for the duration of this call.
    unsafe {
        let ns_view = appkit.ns_view.as_ptr() as *mut NSView;
        let Some(ns_window) = (*ns_view).window() else {
            return;
        };
        ns_window.makeFirstResponder(Some(&*ns_view));
    }
}
