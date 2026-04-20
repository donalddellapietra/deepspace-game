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
//! - `HARNESS_PROBE direction=... hit=... anchor=[...] ui_layer=... anchor_depth=... [t=...]`
//!   Emitted from the `probe_down` / `probe_cursor` script commands.
//!   The cursor probe is the GPU-side `march()` readback; the "down"
//!   probe rotates the camera to point straight down and reads the
//!   same probe, so both sources of truth collapse to one.
//!
//! All lines are single-record, whitespace-separated, `key=value`.

use super::App;
use super::test_runner::ScriptCmd;
use crate::world::anchor::{Path, WorldPos};
use crate::world::edit::HitInfo;
use crate::world::tree::slot_index;

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

    /// Rotate the camera straight down (preserving yaw), wait one
    /// frame for the GPU probe to re-dispatch with the new basis,
    /// then emit the probe's hit. Tests that used to `probe_down`
    /// against a CPU raycast now share the same GPU probe everything
    /// else uses — one source of truth for what the crosshair hits.
    pub(super) fn harness_probe_down(&mut self) {
        let saved_pitch = self.camera.pitch;
        self.camera.pitch = -std::f32::consts::FRAC_PI_2;
        let hit = self.probe_hit();
        self.camera.pitch = saved_pitch;
        match hit {
            Some(h) => println!(
                "HARNESS_PROBE direction=down hit=true anchor={} ui_layer={} anchor_depth={} t={:.6}",
                path_repr(&h),
                self.zoom_level(),
                self.anchor_depth(),
                h.t,
            ),
            None => println!(
                "HARNESS_PROBE direction=down hit=false anchor=[] ui_layer={} anchor_depth={}",
                self.zoom_level(),
                self.anchor_depth(),
            ),
        }
    }

    /// Emit the cursor probe's hit unmodified.
    pub(super) fn harness_probe_cursor(&mut self) {
        let hit = self.probe_hit();
        match hit {
            Some(h) => println!(
                "HARNESS_PROBE direction=cursor hit=true anchor={} ui_layer={} anchor_depth={} t={:.6}",
                path_repr(&h),
                self.zoom_level(),
                self.anchor_depth(),
                h.t,
            ),
            None => println!(
                "HARNESS_PROBE direction=cursor hit=false anchor=[] ui_layer={} anchor_depth={}",
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
            ScriptCmd::ProbeCursor => self.harness_probe_cursor(),
            ScriptCmd::Emit(label) => self.harness_emit_mark(&label, frame),
            ScriptCmd::TeleportAboveLastEdit => self.teleport_above_last_edit(),
            ScriptCmd::RespawnOnSurface => self.respawn_on_surface(),
            ScriptCmd::Step { axis, delta } => {
                let mut d = [0.0f32; 3];
                d[axis as usize] = delta;
                self.camera.position.add_local(d, &self.world.library);
            }
            ScriptCmd::FlyToSurface => self.fly_to_surface(),
        }
    }

    /// Rotate the camera straight down, read the GPU probe, then
    /// reposition the camera a couple of anchor-cells above the hit.
    pub(super) fn fly_to_surface(&mut self) {
        let saved_pitch = self.camera.pitch;
        self.camera.pitch = -std::f32::consts::FRAC_PI_2;
        let hit = self.probe_hit();
        self.camera.pitch = saved_pitch;
        let Some(hit) = hit else {
            eprintln!("fly_to_surface: probe returned no hit");
            return;
        };
        let root_cam = self.camera.position.in_frame(&Path::root());
        // Probe t is in whatever frame the walker used. For a straight-down
        // cursor-center ray in a Cartesian frame the ray direction in root
        // coords is (0, -1, 0), so hit_y = cam_y - t. For frames scaled
        // through pops the magnitude won't match root-frame exactly, but
        // this is just a spawn nudge — landing 2 anchor cells above the
        // coarse surface is tolerant of small errors.
        let hit_y = (root_cam[1] - hit.t).max(0.0);
        let anchor_depth = self.anchor_depth() as u8;
        let cell = 1.0_f32 / 3.0_f32.powi(anchor_depth as i32);
        let above_y = hit_y + 2.0 * cell;
        let new_pos = [root_cam[0], above_y.min(3.0 - cell), root_cam[2]];
        self.camera.position = WorldPos::from_frame_local(&Path::root(), new_pos, anchor_depth);
        eprintln!(
            "fly_to_surface: t={:.6} hit_y={:.6} new_y={:.6} anchor_depth={}",
            hit.t, hit_y, above_y, anchor_depth,
        );
    }

    pub(super) fn respawn_on_surface(&mut self) {
        if self.planet_path.is_none() {
            eprintln!("respawn_on_surface: world has no planet; skipping (use teleport_above_last_edit for plain worlds)");
            return;
        }
        let depth = self.anchor_depth() as u8;
        let pos = crate::world::bootstrap::demo_sphere_surface_spawn(depth);
        self.camera.position = pos;
        self.apply_zoom();
        eprintln!(
            "respawn_on_surface: anchor_depth={} path={:?}",
            depth,
            self.camera.position.anchor.as_slice(),
        );
    }

    pub(super) fn teleport_above_last_edit(&mut self) {
        let Some(last) = self.last_edit_slots else {
            eprintln!("teleport_above_last_edit: no last edit recorded; skipping");
            return;
        };
        let current_depth = self.anchor_depth() as u8;
        if current_depth < last.depth() {
            eprintln!(
                "teleport_above_last_edit: current anchor_depth {} shallower than last edit depth {}; skipping",
                current_depth, last.depth(),
            );
            return;
        }
        let mut anchor = last;
        let bottom_center = slot_index(1, 0, 1) as u8;
        while anchor.depth() < current_depth {
            anchor.push(bottom_center);
        }
        self.camera.position = WorldPos::new(anchor, [0.5, 0.5, 0.5]);
        self.apply_zoom();
        eprintln!(
            "teleport_above_last_edit: broken_path={:?} final_anchor={:?} current_depth={}",
            last.as_slice(), anchor.as_slice(), current_depth,
        );
    }
}
