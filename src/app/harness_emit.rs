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
use crate::world::anchor::WorldPos;
use crate::world::raycast::HitInfo;
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

    /// Raycast from the camera and emit a `HARNESS_PROBE` line.
    ///
    /// Uses the camera's current pitch/yaw verbatim so the probe
    /// direction exactly matches what a subsequent `break`/`place`
    /// would see — any forward-direction reconstruction via the
    /// camera basis is sub-ULP identical. Tests that want "straight
    /// down" should spawn with `--spawn-pitch` near `-π/2`; the
    /// refactored sphere walker is slot-accurate, and the old
    /// tolerance for sub-ULP direction differences (courtesy of the
    /// OLD coarse-step sphere DDA) does not carry over.
    pub(super) fn harness_probe_down(&mut self) {
        let hit = self.frame_aware_raycast();
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

    /// Cast the camera's current forward ray, emit the hit path, the
    /// body-frame AABB computed from that path, and the body-frame hit
    /// point (`cam_body + ray_dir * t`). If `hit_point` is outside the
    /// AABB, the highlight-AABB computation disagrees with the raycast
    /// itself — the visible cursor would land in a different cell
    /// than the one the next break would edit.
    pub(super) fn harness_probe_cursor(&mut self) {
        use crate::app::ActiveFrameKind;
        use crate::world::{aabb, cubesphere_local, sdf};
        let hit = self.frame_aware_raycast();
        let Some(hit) = hit else {
            println!(
                "HARNESS_PROBE direction=cursor hit=false anchor=[] ui_layer={} anchor_depth={}",
                self.zoom_level(),
                self.anchor_depth(),
            );
            return;
        };
        let anchor_path = path_repr(&hit);
        println!(
            "HARNESS_PROBE direction=cursor hit=true anchor={} ui_layer={} anchor_depth={} t={:.6}",
            anchor_path,
            self.zoom_level(),
            self.anchor_depth(),
            hit.t,
        );
        // Unified render-frame-local probe. The camera position is
        // always `in_frame(render_path)`, which is precision-safe for
        // any frame (Cartesian, Body, or Sphere at any face_depth).
        // The ray direction is rotated into face axes when the render
        // frame is inside a face subtree; otherwise world axes.
        let frame_path = self.active_frame.render_path;
        let cam_frame = self.camera.position.in_frame(&frame_path);
        let fwd_world = sdf::normalize(self.camera.forward());
        let ray_dir = match self.active_frame.kind {
            ActiveFrameKind::Sphere(sphere) if sphere.face_depth >= 1 => {
                sdf::normalize(cubesphere_local::world_vec_to_face_axes(
                    fwd_world, sphere.face,
                ))
            }
            _ => fwd_world,
        };
        let (aabb_min, aabb_max) = aabb::hit_aabb_in_frame_local(&hit, &frame_path);
        let hit_point = [
            cam_frame[0] + ray_dir[0] * hit.t,
            cam_frame[1] + ray_dir[1] * hit.t,
            cam_frame[2] + ray_dir[2] * hit.t,
        ];
        let inside = (0..3).all(|i| hit_point[i] >= aabb_min[i] && hit_point[i] <= aabb_max[i]);
        println!(
            "HARNESS_PROBE_AABB direction=cursor anchor={} aabb_min=[{:.5},{:.5},{:.5}] aabb_max=[{:.5},{:.5},{:.5}] hit_point=[{:.5},{:.5},{:.5}] cam_frame=[{:.5},{:.5},{:.5}] inside={}",
            anchor_path,
            aabb_min[0], aabb_min[1], aabb_min[2],
            aabb_max[0], aabb_max[1], aabb_max[2],
            hit_point[0], hit_point[1], hit_point[2],
            cam_frame[0], cam_frame[1], cam_frame[2],
            inside,
        );
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

    /// Raycast straight down in world-space bypassing the normal
    /// interaction-radius cap, then place the camera a couple of
    /// anchor cells above the hit. Used by perf repros that need to
    /// land "on the ground" regardless of spawn coordinates.
    pub(super) fn fly_to_surface(&mut self) {
        use crate::world::anchor::{Path, WorldPos};
        use crate::world::raycast::cpu_raycast;

        let root_cam = self.camera.position.in_frame(&Path::root());
        let ray_dir = [0.0f32, -1.0, 0.0];
        // Raycast the full tree depth so deeply-nested content gets
        // resolved. interaction_radius doesn't apply here — this is
        // explicit teleport, not a cursor hit.
        let max_depth = self.tree_depth.saturating_sub(1).max(1);
        let hit = cpu_raycast(
            &self.world.library, self.world.root, root_cam, ray_dir, max_depth,
        );
        let Some(hit) = hit else {
            eprintln!("fly_to_surface: no hit from root_cam={root_cam:?}");
            return;
        };
        // Position the camera two anchor-cells above the hit's
        // y-coordinate. Reconstruct as a root-frame WorldPos, then
        // deepen to the current anchor depth. f32 precision is fine
        // for this nudge — we just need to land above the surface.
        let hit_y = root_cam[1] + ray_dir[1] * hit.t;
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

    /// Position the camera inside the bottom-center child of the
    /// last-broken cell. Intended use: after `zoom_in:1` following a
    /// break, this drops the camera "one current-layer cell above the
    /// new ground" in the descent flow.
    ///
    /// The path is constructed symbolically rather than via world-xyz
    /// arithmetic. At deep anchors, `1.5 + 3^{-17}` collapses to `1.5`
    /// in f32 (cell size ≈ 2e-8 below f32's ~7-digit precision at
    /// y≈1), which caused `from_frame_local(target_xyz, 17)` to land
    /// in the wrong (solid) cell. Walking `last_edit_slots + [slot
    /// (1,0,1)]*k` is exact at any depth.
    /// Respawn the camera above the sphere surface at the current
    /// `anchor_depth` via `demo_sphere_surface_spawn`. Sphere-only;
    /// emits a warning and no-ops for plain worlds. Use after
    /// `zoom_in:1` in descent flows: the face subtree's slot indices
    /// are `(u, v, r)` whereas `WorldPos` arithmetic is Cartesian,
    /// so the Cartesian `teleport_above_last_edit` drifts horizontally
    /// across the face instead of radially into the sphere. This
    /// respawns on the surface at the new depth so `probe_down` /
    /// `break` find terrain at every layer.
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
        // Camera at the center of the final cell.
        self.camera.position = WorldPos::new(anchor, [0.5, 0.5, 0.5]);
        self.apply_zoom();
        eprintln!(
            "teleport_above_last_edit: broken_path={:?} final_anchor={:?} current_depth={}",
            last.as_slice(), anchor.as_slice(), current_depth,
        );
    }
}
