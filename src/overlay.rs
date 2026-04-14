//! Transparent wry WebView overlay on the winit game window.
//!
//! IPC uses wry's native channel:
//! - Rust → JS: `evaluate_script("window.__onGameState(json)")`
//! - JS → Rust: `window.ipc.postMessage(json)` → ipc_handler → queues
//!
//! Mouse passthrough: hitTest: is swizzled on the WKWebView so that
//! clicks on transparent areas fall through to the Metal layer.
//! JS mousemove sets an atomic flag; the swizzled hitTest checks it.
//!
//! ## Centralised input
//!
//! Clicking on the webview makes it macOS first-responder, stealing
//! keyboard (and sometimes mouse) events from winit.  Rather than
//! trying to prevent this, the webview forwards **every** key and
//! mouse-button event via IPC.  The app drains these queues each
//! frame and applies them to its own input state.
//!
//! Only compiled on non-wasm32 targets.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use objc2::runtime::{AnyObject, Imp, Sel};
use objc2::sel;
use objc2_app_kit::NSView;
use objc2_foundation::CGPoint;
use raw_window_handle::{HasWindowHandle, RawWindowHandle};
use winit::window::Window;
use wry::WebViewExtMacOS;
use wry::{dpi::*, Rect, WebViewBuilder};

use crate::bridge::{GameStateUpdate, UiCommand};

// ── Global state (JS → Rust) ────────────────────────────────────

static IPC_COMMANDS: Mutex<Vec<String>> = Mutex::new(Vec::new());

/// Whether the cursor is currently over an interactive UI element.
/// Controls the hitTest: swizzle — true routes clicks to the webview,
/// false lets them fall through to the game.
static CURSOR_OVER_UI: AtomicBool = AtomicBool::new(false);

/// Forwarded key events: (js_code, pressed).
static FORWARDED_KEYS: Mutex<Vec<(String, bool)>> = Mutex::new(Vec::new());

/// Forwarded mouse-button events: (js_button, pressed).
static FORWARDED_MOUSE: Mutex<Vec<(u8, bool)>> = Mutex::new(Vec::new());

/// Outbox: state updates buffered by push_state(), flushed to the
/// webview each frame via flush_to_webview().
static OUTBOX: Mutex<Vec<String>> = Mutex::new(Vec::new());

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

/// Force the hitTest: swizzle to pass all events through to the game.
/// Called when locking the cursor — there are no mousemove events
/// while the cursor is locked, so the JS tracker would never update.
pub fn clear_passthrough() {
    CURSOR_OVER_UI.store(false, Ordering::Relaxed);
}

// ── State push ──────────────────────────────────────────────────

/// Serialize a state update and buffer it for the next flush.
pub fn push_state(update: &GameStateUpdate) {
    match serde_json::to_string(update) {
        Ok(json) => {
            if let Ok(mut outbox) = OUTBOX.lock() {
                outbox.push(json);
            }
        }
        Err(e) => eprintln!("overlay: failed to serialize state update: {e}"),
    }
}

/// Flush buffered state updates to the webview via evaluate_script.
/// Call this once per frame from the render loop.
pub fn flush_to_webview(webview: &wry::WebView) {
    let jsons: Vec<String> = {
        let Ok(mut outbox) = OUTBOX.lock() else { return };
        outbox.drain(..).collect()
    };
    for json in jsons {
        let js = format!("window.__onGameState && window.__onGameState({})", json);
        let _ = webview.evaluate_script(&js);
    }
}

// ── Command polling ─────────────────────────────────────────────

/// Drain IPC commands and deserialize them into UiCommands.
pub fn poll_commands() -> Vec<UiCommand> {
    let mut cmds = Vec::new();
    for raw in drain_ipc_commands() {
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
// fighting it, forward EVERY key and mouse event via IPC.

// Keys — skip forwarding keydown when a text input is focused so
// typing in the hex field works, but ALWAYS forward keyup so the
// game can release the key.
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

// Never show the browser context menu — right-click belongs to the game.
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
        // SAFETY: ORIGINAL_HIT_TEST is only written once during swizzle
        // (single-threaded init) and read here.
        if let Some(original) = unsafe { ORIGINAL_HIT_TEST } {
            unsafe { original(this, cmd, point) }
        } else {
            std::ptr::null_mut()
        }
    } else {
        std::ptr::null_mut()
    }
}

/// Swizzle hitTest: on the WKWebView's runtime class.
unsafe fn swizzle_hit_test(webview: &wry::WebView) {
    unsafe {
        let wry_view = webview.webview();
        // Get the runtime class of this specific instance (may be a KVO subclass)
        let class = (*wry_view).class();

        let sel = sel!(hitTest:);
        let Some(method) = class.instance_method(sel) else {
            eprintln!("overlay: hitTest: method not found on WryWebView class");
            return;
        };

        let new_imp: Imp = std::mem::transmute(swizzled_hit_test as *const ());
        let original_imp: Imp = method.set_implementation(new_imp);
        ORIGINAL_HIT_TEST = Some(std::mem::transmute(original_imp));

        log::info!("overlay: hitTest: swizzled for mouse passthrough");
    }
}

// ── Keyboard refocus ─────────────────────────────────────────────

/// Make the NSWindow's contentView the first responder, returning
/// keyboard events to winit.  This is a best-effort optimisation —
/// even if it fails, the IPC forwarding ensures input stays correct.
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

// ── WebView creation ─────────────────────────────────────────────

/// Create the wry WebView as a child of the given winit window.
/// Returns `None` if creation fails.
pub fn create_webview(window: &Window) -> Option<wry::WebView> {
    let Ok(handle) = window.window_handle() else {
        eprintln!("overlay: could not get window handle");
        return None;
    };

    let phys = window.inner_size();
    let scale = window.scale_factor();
    let url = "http://localhost:5173";

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
            log::info!(
                "overlay: WebView created ({}x{} @{:.1}x) → {url}",
                phys.width, phys.height, scale
            );
            unsafe { swizzle_hit_test(&webview) };
            Some(webview)
        }
        Err(e) => {
            eprintln!("overlay: failed to create WebView: {e}");
            None
        }
    }
}

/// Resize the webview to match the window's current physical size.
pub fn resize_webview(webview: &wry::WebView, window: &Window) {
    let phys = window.inner_size();
    let _ = webview.set_bounds(Rect {
        position: PhysicalPosition::new(0, 0).into(),
        size: PhysicalSize::new(phys.width, phys.height).into(),
    });
}

// ── JS key/mouse code mapping ────────────────────────────────────

/// Map a JS key code string to a winit KeyCode.
pub fn js_code_to_keycode(code: &str) -> Option<winit::keyboard::KeyCode> {
    use winit::keyboard::KeyCode;
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

/// Map a JS mouse button number to a winit MouseButton.
pub fn js_button_to_mouse(button: u8) -> Option<winit::event::MouseButton> {
    use winit::event::MouseButton;
    match button {
        0 => Some(MouseButton::Left),
        1 => Some(MouseButton::Middle),
        2 => Some(MouseButton::Right),
        _ => None,
    }
}
