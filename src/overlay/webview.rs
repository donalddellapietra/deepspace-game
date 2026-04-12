//! Transparent wry WebView overlay on the Bevy game window.
//!
//! IPC uses wry's native channel — no WebSocket needed:
//! - Rust → JS: `evaluate_script("window.__onGameState(json)")`
//! - JS → Rust: `window.ipc.postMessage(json)` → ipc_handler → channel
//!
//! Only compiled on non-wasm32 targets.

use std::sync::Mutex;

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy::winit::WINIT_WINDOWS;
use raw_window_handle::HasWindowHandle;
use wry::{dpi::*, Rect, WebViewBuilder};

// ── Global command channel (JS → Rust) ────────────────────────────

static IPC_COMMANDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

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
//
// Runs before the page loads. Sets up:
// 1. Transparent background
// 2. __onGameState buffer so early state pushes aren't lost
// 3. IPC plumbing

const INIT_SCRIPT: &str = r#"
// Force transparent backgrounds
document.documentElement.style.background = 'transparent';
document.addEventListener('DOMContentLoaded', () => {
    document.body.style.background = 'transparent';
});

// Buffer for game state updates that arrive before React mounts.
// When React calls getTransport().onState(handler), it replaces
// __onGameState with its own handler and replays the buffer.
window.__stateBuffer = [];
window.__onGameState = (data) => {
    const parsed = typeof data === 'string' ? JSON.parse(data) : data;
    window.__stateBuffer.push(parsed);
};
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
                holder.webview = Some(webview);
            }
            Err(e) => {
                warn!("overlay: failed to create WebView: {e}");
            }
        }
    });
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
