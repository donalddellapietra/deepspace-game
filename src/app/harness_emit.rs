//! Test-harness stdout protocol.
//!
//! Emits `HARNESS_*` lines the e2e tests parse to verify actions
//! landed correctly. Three record types:
//!
//! - `HARNESS_MARK  label=... ui_layer=... anchor_depth=... frame=...`
//!   Timeline marker. Correlates screenshots/log lines to actions.
//!
//! - `HARNESS_EDIT  action=broke|placed anchor=[...] changed=... ui_layer=... anchor_depth=...`
//!   Emitted from `do_break` / `do_place` after the edit attempt.
//!
//! - `HARNESS_PROBE direction=... hit=... anchor=[...] ui_layer=... anchor_depth=...`
//!   Emitted from the `probe_down` script command. CPU raycast
//!   straight down in world-space; does not affect render orientation.
//!
//! All lines are single-record, whitespace-separated, `key=value`.

use super::App;
use super::test_runner::ScriptCmd;
use crate::world::anchor::{Path, WorldPos, WORLD_SIZE};
use crate::world::edit::HitInfo;
use crate::world::tree::slot_coords;

/// World-space (origin, size) of the cell addressed by `path`.
/// Origin = (0,0,0) corner of the cell; size = side length.
fn cell_origin_size_world(path: &Path) -> ([f32; 3], f32) {
    let mut origin = [0.0_f32; 3];
    let mut size = WORLD_SIZE;
    for k in 0..path.depth() as usize {
        let (sx, sy, sz) = slot_coords(path.slot(k) as usize);
        let child = size / 3.0;
        origin[0] += sx as f32 * child;
        origin[1] += sy as f32 * child;
        origin[2] += sz as f32 * child;
        size = child;
    }
    (origin, size)
}

/// Format a slot path as `[13,13,13,16]`.
fn path_repr(hit: &HitInfo) -> String {
    let mut out = String::from("[");
    for (i, &(_, slot)) in hit.path.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&slot.to_string());
    }
    out.push(']');
    out
}

impl App {
    /// `HARNESS_EDIT action=<str> anchor=[...] changed=<bool> ui_layer=... anchor_depth=...`
    pub(super) fn harness_emit_edit(&self, action: &str, hit: &HitInfo, changed: bool) {
        println!(
            "HARNESS_EDIT action={} anchor={} changed={} ui_layer={} anchor_depth={}",
            action,
            path_repr(hit),
            changed,
            self.zoom_level(),
            self.anchor_depth(),
        );
    }

    /// `HARNESS_MARK label=<str> ui_layer=... anchor_depth=... frame=<u32>`
    pub(super) fn harness_emit_mark(&self, label: &str, frame: u32) {
        println!(
            "HARNESS_MARK label={} ui_layer={} anchor_depth={} frame={}",
            label,
            self.zoom_level(),
            self.anchor_depth(),
            frame,
        );
    }

    /// Raycast straight down in world-space from the camera and emit a
    /// `HARNESS_PROBE` line. Pitch/yaw are temporarily overridden for
    /// the raycast and restored before returning — the next render
    /// sees the camera's original orientation.
    pub(super) fn harness_probe_down(&mut self) {
        use std::f32::consts::FRAC_PI_2;
        let saved_pitch = self.camera.pitch;
        let saved_yaw = self.camera.yaw;
        // -π/2 with yaw 0 points at world -Y regardless of frame.
        self.camera.pitch = -FRAC_PI_2;
        self.camera.yaw = 0.0;
        let hit = self.frame_aware_raycast();
        self.camera.pitch = saved_pitch;
        self.camera.yaw = saved_yaw;

        match hit {
            Some(h) => println!(
                "HARNESS_PROBE direction=down hit=true anchor={} ui_layer={} anchor_depth={}",
                path_repr(&h),
                self.zoom_level(),
                self.anchor_depth(),
            ),
            None => println!(
                "HARNESS_PROBE direction=down hit=false anchor=[] ui_layer={} anchor_depth={}",
                self.zoom_level(),
                self.anchor_depth(),
            ),
        }
    }

    /// Shared script-command dispatcher. Called from both the live
    /// event loop (`event_loop.rs`) and the render-harness loop
    /// (`test_runner.rs`) so new commands only need one handler.
    pub(super) fn handle_script_cmd(&mut self, cmd: ScriptCmd, frame: u32) {
        match cmd {
            ScriptCmd::Break => self.do_break(),
            ScriptCmd::Place => self.do_place(),
            ScriptCmd::Wait(_) => {}
            ScriptCmd::ZoomIn(steps) => {
                for _ in 0..steps {
                    self.zoom_anchor(1);
                }
            }
            ScriptCmd::ZoomOut(steps) => {
                for _ in 0..steps {
                    self.zoom_anchor(-1);
                }
            }
            ScriptCmd::ToggleDebugOverlay => {
                self.debug_overlay_visible = !self.debug_overlay_visible;
            }
            ScriptCmd::Screenshot(path) => {
                if let Some(r) = &mut self.renderer {
                    match r.capture_to_png(&path) {
                        Ok(()) => eprintln!("script screenshot saved to {path}"),
                        Err(e) => eprintln!("script screenshot {path} failed: {e}"),
                    }
                }
            }
            ScriptCmd::Pitch(rad) => {
                self.camera.pitch = rad;
            }
            ScriptCmd::Yaw(rad) => {
                self.camera.yaw = rad;
            }
            ScriptCmd::ProbeDown => self.harness_probe_down(),
            ScriptCmd::Emit(label) => self.harness_emit_mark(&label, frame),
            ScriptCmd::TeleportAboveLastEdit => self.teleport_above_last_edit(),
        }
    }

    /// Position the camera horizontally centered on the last-broken or
    /// last-placed cell, at `y = bottom_of_broken_cell + 0.5 *
    /// current_cell_size`. Intended use: after `zoom_in:1`, this drops
    /// the camera inside the bottom-most child of the previously
    /// broken cell — "one current-layer cell above the new ground" in
    /// the descent flow.
    pub(super) fn teleport_above_last_edit(&mut self) {
        let Some(last) = self.last_edit_slots else {
            eprintln!("teleport_above_last_edit: no last edit recorded; skipping");
            return;
        };
        let (origin, broken_size) = cell_origin_size_world(&last);
        let current_depth = self.anchor_depth() as i32;
        let new_cell_size = WORLD_SIZE * 3.0_f32.powi(-current_depth);
        let target = [
            origin[0] + 0.5 * broken_size,
            origin[1] + 0.5 * new_cell_size,
            origin[2] + 0.5 * broken_size,
        ];
        self.camera.position =
            WorldPos::from_frame_local(&Path::root(), target, current_depth as u8);
        self.apply_zoom();
        eprintln!(
            "teleport_above_last_edit: broken_path={:?} broken_size={} new_cell_size={} target_xyz={:?} current_depth={}",
            last.as_slice(), broken_size, new_cell_size, target, current_depth,
        );
    }
}
