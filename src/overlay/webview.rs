//! Transparent wry WebView overlay on the Bevy game window.
//!
//! IPC uses wry's native channel:
//! - Rust → JS: `evaluate_script("window.__onGameState(json)")`
//! - JS → Rust: `window.ipc.postMessage(json)` → ipc_handler → channel
//!
//! Mouse passthrough: hitTest: is swizzled on the WKWebView so that
//! clicks on transparent areas fall through to the Metal/Bevy layer.
//! JS mousemove sets an atomic flag; the swizzled hitTest checks it.
//!
//! ## Centralised input
//!
//! Clicking on the webview makes it macOS first-responder, stealing
//! keyboard (and sometimes mouse) events from Bevy/winit.  Rather
//! than trying to prevent this, the webview forwards **every** key
//! and mouse-button event to Bevy via IPC.  A system in `PreUpdate`
//! injects them into `ButtonInput`, so all game systems see a single,
//! always-correct input state.
//!
//! Only compiled on non-wasm32 targets.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy::winit::WINIT_WINDOWS;
use objc2::runtime::{AnyObject, Imp, Sel};
use objc2::sel;
use objc2_foundation::CGPoint;
use raw_window_handle::HasWindowHandle;
use wry::WebViewExtMacOS;
use wry::{dpi::*, Rect, WebViewBuilder};

// ── Global state (JS → Rust) ────────────────────────────────────

static IPC_COMMANDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Whether the cursor is currently over an interactive UI element.
/// Controls the hitTest: swizzle — true routes clicks to the webview,
/// false lets them fall through to Bevy.
static CURSOR_OVER_UI: AtomicBool = AtomicBool::new(false);

/// Forwarded key events: (js_code, pressed).
static FORWARDED_KEYS: Mutex<Vec<(String, bool)>> = Mutex::new(Vec::new());

/// Forwarded mouse-button events: (js_button, pressed).
static FORWARDED_MOUSE: Mutex<Vec<(u8, bool)>> = Mutex::new(Vec::new());

// ── Drain helpers ────────────────────────────────────────────────

pub fn drain_ipc_commands() -> Vec<String> {
    let Ok(mut cmds) = IPC_COMMANDS.lock() else {
        return Vec::new();
    };
    cmds.drain(..).collect()
}

pub fn drain_forwarded_keys() -> Vec<(String, bool)> {
    let Ok(mut q) = FORWARDED_KEYS.lock() else {
        return Vec::new();
    };
    q.drain(..).collect()
}

pub fn drain_forwarded_mouse() -> Vec<(u8, bool)> {
    let Ok(mut q) = FORWARDED_MOUSE.lock() else {
        return Vec::new();
    };
    q.drain(..).collect()
}

// ── Passthrough control ─────────────────────────────────────────

/// Force the hitTest: swizzle to pass all events through to Bevy.
/// Called when locking the cursor — there are no mousemove events
/// while the cursor is locked, so the JS tracker would never update.
pub fn clear_passthrough() {
    CURSOR_OVER_UI.store(false, Ordering::Relaxed);
}

// ── WebView holder ───────────────────────────────────────────────

#[derive(Default)]
pub struct WebViewHolder {
    pub webview: Option<wry::WebView>,
    frames_waited: u32,
}

const WAIT_FRAMES: u32 = 10;

// ── Initialization script ────────────────────────────────────────

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

// ── Centralised input ───────────────────────────────────────────
// The webview steals macOS first-responder on click.  Instead of
// fighting it, forward EVERY key and mouse event to Bevy via IPC.
// A Rust system in PreUpdate injects them into ButtonInput, making
// it the single source of truth regardless of which view has focus.

// Keys — skip forwarding keydown when a text input is focused so
// typing in the hex field works, but ALWAYS forward keyup so Bevy
// can release the key.
document.addEventListener('keydown', (e) => {
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    window.ipc.postMessage(JSON.stringify({
        __key: { code: e.code, pressed: true }
    }));
}, true);

document.addEventListener('keyup', (e) => {
    window.ipc.postMessage(JSON.stringify({
        __key: { code: e.code, pressed: false }
    }));
}, true);

// Mouse buttons
document.addEventListener('mousedown', (e) => {
    window.ipc.postMessage(JSON.stringify({
        __mouse: { button: e.button, pressed: true }
    }));
    // Prevent focus theft on non-input elements
    const tag = e.target.tagName;
    if (tag !== 'INPUT' && tag !== 'TEXTAREA') {
        e.preventDefault();
    }
}, true);

document.addEventListener('mouseup', (e) => {
    window.ipc.postMessage(JSON.stringify({
        __mouse: { button: e.button, pressed: false }
    }));
}, true);

// Never show the browser context menu — right-click belongs to Bevy.
document.addEventListener('contextmenu', (e) => e.preventDefault(), true);

// Mouse passthrough: track whether cursor is over an interactive element.
// Throttled to avoid flooding IPC with messages on every mouse move.
(function() {
    let lastPassthrough = true;
    let scheduled = false;

    function checkPassthrough(e) {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(() => {
            scheduled = false;
            const el = document.elementFromPoint(e.clientX, e.clientY);
            let interactive = false;
            if (el) {
                let node = el;
                while (node && node !== document.documentElement) {
                    const pe = getComputedStyle(node).pointerEvents;
                    if (pe === 'auto') { interactive = true; break; }
                    if (pe === 'none') { break; }
                    node = node.parentElement;
                }
            }
            const passthrough = !interactive;
            if (passthrough !== lastPassthrough) {
                lastPassthrough = passthrough;
                window.ipc.postMessage(JSON.stringify({ __passthrough: passthrough }));
            }
        });
    }

    document.addEventListener('mousemove', checkPassthrough, { passive: true });
    document.addEventListener('mouseleave', () => {
        if (!lastPassthrough) {
            lastPassthrough = true;
            window.ipc.postMessage(JSON.stringify({ __passthrough: true }));
        }
    }, { passive: true });
})();
"#;

// ── hitTest: swizzle ─────────────────────────────────────────────
//
// We replace the WKWebView's hitTest: method so it returns nil when
// the cursor is over a transparent area (CURSOR_OVER_UI is false).
// This causes macOS to pass the event to the view behind (Metal layer).

/// The original hitTest: IMP, saved during swizzle.
static mut ORIGINAL_HIT_TEST: Option<
    unsafe extern "C" fn(*mut AnyObject, Sel, CGPoint) -> *mut AnyObject,
> = None;

/// Our replacement hitTest: — returns nil to pass through, or calls
/// the original to let the webview handle the event.
unsafe extern "C" fn swizzled_hit_test(
    this: *mut AnyObject,
    cmd: Sel,
    point: CGPoint,
) -> *mut AnyObject {
    if CURSOR_OVER_UI.load(Ordering::Relaxed) {
        if let Some(original) = ORIGINAL_HIT_TEST {
            original(this, cmd, point)
        } else {
            std::ptr::null_mut()
        }
    } else {
        std::ptr::null_mut()
    }
}

/// Swizzle hitTest: on the WKWebView's runtime class.
unsafe fn swizzle_hit_test(webview: &wry::WebView) {
    let wry_view = webview.webview();
    // Get the runtime class of this specific instance (may be a KVO subclass)
    let class = (*wry_view).class();

    let sel = sel!(hitTest:);
    let Some(method) = class.instance_method(sel) else {
        eprintln!("overlay: hitTest: method not found on WryWebView class");
        return;
    };

    let new_imp: Imp = std::mem::transmute(swizzled_hit_test as *const ());
    let original_imp: Imp = unsafe { method.set_implementation(new_imp) };
    ORIGINAL_HIT_TEST = Some(std::mem::transmute(original_imp));

    info!("overlay: hitTest: swizzled for mouse passthrough");
}

// ── Keyboard refocus ─────────────────────────────────────────────

use objc2_app_kit::{NSView, NSWindow};

/// Make the NSWindow's contentView the first responder, returning
/// keyboard events to Bevy/winit.  This is a best-effort
/// optimisation — even if it fails, the IPC forwarding ensures
/// ButtonInput stays correct.
pub fn refocus_content_view(entity: Entity) {
    WINIT_WINDOWS.with_borrow(|winit_windows| {
        let Some(wrapper) = winit_windows.get_window(entity) else {
            return;
        };
        let Ok(handle) = wrapper.window_handle() else {
            return;
        };
        let raw = handle.as_raw();
        let raw_window_handle::RawWindowHandle::AppKit(appkit) = raw else {
            return;
        };
        // SAFETY: The pointers come from winit's live window and are
        // valid for the duration of the borrow.
        unsafe {
            let ns_view = appkit.ns_view.as_ptr() as *mut NSView;
            let Some(ns_window) = (*ns_view).window() else {
                return;
            };
            ns_window.makeFirstResponder(Some(&*ns_view));
        }
    });
}

// ── IPC routing ──────────────────────────────────────────────────

/// Route an IPC message from the webview to the appropriate queue.
fn route_ipc_message(body: &str) {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(body) {
        // Passthrough flag (mouse hit-test)
        if let Some(pt) = val.get("__passthrough") {
            CURSOR_OVER_UI.store(!pt.as_bool().unwrap_or(true), Ordering::Relaxed);
            return;
        }
        // Forwarded key event
        if let Some(key) = val.get("__key") {
            let code = key.get("code").and_then(|c| c.as_str()).unwrap_or("");
            let pressed = key.get("pressed").and_then(|p| p.as_bool()).unwrap_or(false);
            if let Ok(mut q) = FORWARDED_KEYS.lock() {
                q.push((code.to_string(), pressed));
            }
            return;
        }
        // Forwarded mouse-button event
        if let Some(m) = val.get("__mouse") {
            let button = m.get("button").and_then(|b| b.as_u64()).unwrap_or(99) as u8;
            let pressed = m.get("pressed").and_then(|p| p.as_bool()).unwrap_or(false);
            if let Ok(mut q) = FORWARDED_MOUSE.lock() {
                q.push((button, pressed));
            }
            return;
        }
    }
    // Everything else is a UI command (React → Rust).
    if let Ok(mut cmds) = IPC_COMMANDS.lock() {
        cmds.push(body.to_string());
    }
}

// ── Systems ──────────────────────────────────────────────────────

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
                route_ipc_message(request.body());
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
                unsafe { swizzle_hit_test(&webview) };
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
