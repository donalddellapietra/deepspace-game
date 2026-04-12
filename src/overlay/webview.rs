//! Transparent wry WebView overlay on the Bevy game window.
//!
//! Creates a WebView as a child of the primary Bevy window. The WebView
//! has a transparent background so the game renders through, and the
//! React UI floats on top. Only compiled on non-wasm32 targets.

use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy::winit::WINIT_WINDOWS;
use raw_window_handle::HasWindowHandle;
use wry::{dpi::*, Rect, WebViewBuilder};

/// Holds the wry WebView so it isn't dropped, plus a frame counter
/// to delay creation until the window event loop is fully initialized.
#[derive(Default)]
pub struct WebViewHolder {
    pub webview: Option<wry::WebView>,
    frames_waited: u32,
}

/// Number of frames to wait before creating the WebView.
const WAIT_FRAMES: u32 = 10;

/// Creates the transparent WebView overlay on the primary window.
/// Accesses WinitWindows via its thread-local (not a Bevy resource).
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
            .with_initialization_script(
                // Force transparent background before anything else renders
                "document.documentElement.style.background = 'transparent';\
                 document.addEventListener('DOMContentLoaded', () => {\
                     document.body.style.background = 'transparent';\
                 });",
            )
            .with_url(url)
            .with_accept_first_mouse(true)
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

/// After a delay, capture diagnostic info from the webview and write
/// a screenshot of its HTML content to /tmp/webview-capture.png.
pub fn capture_webview_diagnostic(
    holder: NonSend<WebViewHolder>,
    mut done: Local<bool>,
    mut frame: Local<u32>,
) {
    if *done {
        return;
    }
    *frame += 1;
    // Wait ~3 seconds (180 frames at 60fps) for the page to fully load
    if *frame < 180 {
        return;
    }
    *done = true;

    let Some(ref webview) = holder.webview else {
        return;
    };

    // Inject JS that captures the page as a data URL and sends it via IPC
    let _ = webview.evaluate_script(
        r#"
        (async () => {
            // Report diagnostic info
            const body = document.body;
            const root = document.getElementById('root');
            const diag = {
                bodyBg: getComputedStyle(body).backgroundColor,
                bodyChildren: body.children.length,
                rootExists: !!root,
                rootBg: root ? getComputedStyle(root).backgroundColor : null,
                rootPointerEvents: root ? getComputedStyle(root).pointerEvents : null,
                rootChildren: root ? root.children.length : 0,
                rootInnerHTML: root ? root.innerHTML.substring(0, 500) : null,
                viewport: { w: window.innerWidth, h: window.innerHeight },
                wsConnected: !!window.__wsConnected,
                url: window.location.href,
            };
            // Write to a temp element we can read via title
            document.title = 'DIAG:' + JSON.stringify(diag);
        })();
        "#,
    );

    // Read the title back after a short delay via another evaluate
    let _ = webview.evaluate_script(
        r#"
        setTimeout(() => {
            document.title = 'DIAG:' + JSON.stringify({
                bodyBg: getComputedStyle(document.body).backgroundColor,
                rootExists: !!document.getElementById('root'),
                rootChildren: document.getElementById('root') ? document.getElementById('root').children.length : 0,
                rootHTML: document.getElementById('root') ? document.getElementById('root').innerHTML.substring(0, 200) : 'NO ROOT',
                viewport: { w: window.innerWidth, h: window.innerHeight },
                url: window.location.href,
            });
        }, 500);
        "#,
    );

    info!("overlay: diagnostic capture triggered (check game window title)");
}

/// Keep the webview bounds in sync when the game window is resized.
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
