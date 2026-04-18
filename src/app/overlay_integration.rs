//! Native-only wry webview integration on the `App`.
//!
//! The overlay lives in a transparent WKWebView layered over the
//! main Metal surface. This file wires the webview into the
//! frame loop: creation (deferred a few frames so the NSWindow is
//! fully configured), input injection from forwarded JS events, UI
//! command polling, state flush, and resize.

use super::App;
use crate::overlay;

pub(super) const WAIT_FRAMES: u32 = 10;

impl App {
    pub(super) fn try_create_webview(&mut self) {
        if self.webview.is_some() {
            return;
        }
        self.frames_waited += 1;
        if self.frames_waited < WAIT_FRAMES {
            return;
        }
        let Some(window) = &self.window else { return };
        if let Some(wv) = overlay::create_webview(window) {
            self.webview = Some(wv);
            if let Some(test) = self.test.as_ref() {
                use std::sync::atomic::Ordering;
                test.monitor.webview_created.store(true, Ordering::Relaxed);
            }
            // The webview is a WKWebView added as a subview. It
            // often grabs first responder during creation. Reclaim
            // key + first-responder on the content view so the
            // title bar lights up and clicks land on the game.
            crate::platform::after_overlay_created(window);
        }
    }

    pub(super) fn inject_webview_input(&mut self) {
        for (code, pressed) in overlay::drain_forwarded_keys() {
            if let Some(key) = overlay::js_code_to_keycode(&code) {
                self.apply_key(key, pressed);
            }
        }
        for (button, pressed) in overlay::drain_forwarded_mouse() {
            if pressed {
                if let Some(btn) = overlay::js_button_to_mouse(button) {
                    self.apply_mouse(btn);
                }
            }
        }
    }

    pub(super) fn flush_overlay(&self) {
        if let Some(ref wv) = self.webview {
            overlay::flush_to_webview(wv);
        }
    }

    pub(super) fn resize_overlay(&self) {
        if let Some(ref wv) = self.webview {
            if let Some(window) = &self.window {
                overlay::resize_webview(wv, window);
            }
        }
    }
}
