//! Transparent wry WebView overlay on the Bevy game window.
//!
//! IPC uses wry's native channel:
//! - Rust → JS: `evaluate_script("window.__onGameState(json)")`
//! - JS → Rust: `window.ipc.postMessage(json)` → ipc_handler → channel
//!
//! Mouse passthrough: the WKWebView defaults to ignoring mouse events.
//! Each frame, JS checks if the cursor is over an interactive element
//! and toggles `setIgnoresMouseEvents:` accordingly via IPC.
//!
//! Only compiled on non-wasm32 targets.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy::winit::WINIT_WINDOWS;
use objc2::rc::Retained;
use objc2::runtime::Bool;
use objc2::{msg_send, sel};
use objc2_app_kit::NSView;
use raw_window_handle::HasWindowHandle;
use wry::WebViewExtMacOS;
use wry::{dpi::*, Rect, WebViewBuilder};

// ── Global command channel (JS → Rust) ────────────────────────────

static IPC_COMMANDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Whether the cursor is currently over an interactive UI element.
/// Set by JS via IPC, read by the passthrough system.
static CURSOR_OVER_UI: AtomicBool = AtomicBool::new(false);

pub fn drain_ipc_commands() -> Vec<String> {
    let Ok(mut cmds) = IPC_COMMANDS.lock() else {
        return Vec::new();
    };
    cmds.drain(..).collect()
}

// ── WebView holder ────────────────────────────────────────────────

#[derive(Default)]
pub struct WebViewHolder {
    pub webview: Option<wry::WebView>,
    frames_waited: u32,
}

const WAIT_FRAMES: u32 = 10;

// ── Initialization script ─────────────────────────────────────────

const INIT_SCRIPT: &str = r#"
// Force transparent backgrounds
document.documentElement.style.background = 'transparent';
document.addEventListener('DOMContentLoaded', () => {
    document.body.style.background = 'transparent';
});

// Buffer for game state updates that arrive before React mounts.
window.__stateBuffer = [];
window.__onGameState = (data) => {
    const parsed = typeof data === 'string' ? JSON.parse(data) : data;
    window.__stateBuffer.push(parsed);
};

// Mouse passthrough: track whether cursor is over an interactive element.
// This runs on mousemove and sends the result to Rust via IPC.
document.addEventListener('mousemove', (e) => {
    const el = document.elementFromPoint(e.clientX, e.clientY);
    let interactive = false;
    if (el) {
        // Walk up the DOM tree to check pointer-events
        let node = el;
        while (node && node !== document.documentElement) {
            const pe = getComputedStyle(node).pointerEvents;
            if (pe === 'auto') { interactive = true; break; }
            if (pe === 'none') { break; }
            node = node.parentElement;
        }
    }
    window.ipc.postMessage(JSON.stringify({
        __passthrough: !interactive
    }));
}, { passive: true });

// Also handle mouseleave on the document (cursor left the webview)
document.addEventListener('mouseleave', () => {
    window.ipc.postMessage(JSON.stringify({ __passthrough: true }));
}, { passive: true });
"#;

// ── Systems ───────────────────────────────────────────────────────

pub fn create_overlay_webview(
    mut holder: NonSendMut<WebViewHolder>,
    primary: Query<Entity, With<PrimaryWindow>>,
) {
    if holder.webview.is_some() {
        return;
    }

    holder.frames_waited += 1;
    if holder.frames_waited < WAIT_FRAMES {
        return;
    }

    let Ok(entity) = primary.single() else {
        return;
    };

    WINIT_WINDOWS.with_borrow(|winit_windows| {
        let Some(wrapper) = winit_windows.get_window(entity) else {
            return;
        };

        let Ok(handle) = wrapper.window_handle() else {
            return;
        };

        let url = "http://localhost:5173";
        let phys = wrapper.inner_size();
        let scale = wrapper.scale_factor();

        match WebViewBuilder::new()
            .with_transparent(true)
            .with_background_color((0, 0, 0, 0))
            .with_initialization_script(INIT_SCRIPT)
            .with_url(url)
            .with_accept_first_mouse(true)
            .with_ipc_handler(|request| {
                let body = request.body();
                // Check for passthrough messages vs UI commands
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(body) {
                    if let Some(pt) = val.get("__passthrough") {
                        CURSOR_OVER_UI.store(!pt.as_bool().unwrap_or(true), Ordering::Relaxed);
                        return;
                    }
                }
                if let Ok(mut cmds) = IPC_COMMANDS.lock() {
                    cmds.push(body.to_string());
                }
            })
            .with_bounds(Rect {
                position: PhysicalPosition::new(0, 0).into(),
                size: PhysicalSize::new(phys.width, phys.height).into(),
            })
            .build_as_child(&handle)
        {
            Ok(webview) => {
                info!(
                    "overlay: WebView created ({}x{} @{:.1}x) → {url}",
                    phys.width, phys.height, scale
                );
                // Start with mouse events ignored so game receives input
                set_ignores_mouse_events(&webview, true);
                holder.webview = Some(webview);
            }
            Err(e) => {
                warn!("overlay: failed to create WebView: {e}");
            }
        }
    });
}

/// Toggle `setIgnoresMouseEvents:` based on whether the cursor is
/// over an interactive UI element.
pub fn sync_mouse_passthrough(holder: NonSend<WebViewHolder>) {
    let Some(ref webview) = holder.webview else {
        return;
    };

    let over_ui = CURSOR_OVER_UI.load(Ordering::Relaxed);
    // When cursor is over UI, the webview should NOT ignore mouse events.
    // When cursor is over transparent area, it SHOULD ignore them.
    set_ignores_mouse_events(webview, !over_ui);
}

/// Call `[NSView setIgnoresMouseEvents:]` on the WKWebView.
fn set_ignores_mouse_events(webview: &wry::WebView, ignores: bool) {
    let wk: Retained<NSView> = {
        let wry_view = webview.webview();
        // WryWebView inherits from WKWebView which inherits from NSView.
        // We can safely cast via Retained::cast since the inheritance is linear.
        unsafe { Retained::cast(wry_view) }
    };
    let val = Bool::new(ignores);
    let _: () = unsafe { msg_send![&*wk, setIgnoresMouseEvents: val] };
}

pub fn resize_overlay_webview(
    holder: NonSend<WebViewHolder>,
    primary: Query<Entity, With<PrimaryWindow>>,
) {
    let Some(ref webview) = holder.webview else {
        return;
    };

    let Ok(entity) = primary.single() else {
        return;
    };

    WINIT_WINDOWS.with_borrow(|winit_windows| {
        let Some(wrapper) = winit_windows.get_window(entity) else {
            return;
        };
        let phys = wrapper.inner_size();
        let _ = webview.set_bounds(Rect {
            position: PhysicalPosition::new(0, 0).into(),
            size: PhysicalSize::new(phys.width, phys.height).into(),
        });
    });
}
