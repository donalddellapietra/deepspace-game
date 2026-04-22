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
use crate::world::anchor::{Path, WorldPos};
use crate::world::raycast::HitInfo;
use crate::world::tree::{slot_index, Child, NodeId, NodeKind, NodeLibrary};

/// Walk `path` from `root`; return true if any step's node has
/// `CubedSphereBody` kind. Used to detect whether a stored edit path
/// refers to a sphere cell (UVR slots) or a Cartesian cell (XYZ slots).
fn path_crosses_sphere_body(
    library: &NodeLibrary, root: NodeId, path: &Path,
) -> bool {
    let mut node = root;
    for k in 0..path.depth() as usize {
        let Some(n) = library.get(node) else { return false };
        if matches!(n.kind, NodeKind::CubedSphereBody { .. }) {
            return true;
        }
        let slot = path.slot(k) as usize;
        match n.children[slot] {
            Child::Node(next) => node = next,
            _ => return false,
        }
    }
    if let Some(n) = library.get(node) {
        matches!(n.kind, NodeKind::CubedSphereBody { .. })
    } else {
        false
    }
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
            Some(ref h) => {
                eprintln!(
                    "probe_down: hit t={:.6e} path_len={} face={} sphere_cell={:?} path={:?}",
                    h.t,
                    h.path.len(),
                    h.face,
                    h.sphere_cell,
                    h.path.iter().map(|(_, s)| *s as u32).collect::<Vec<_>>(),
                );
                println!(
                    "HARNESS_PROBE direction=down hit=true anchor={} ui_layer={} anchor_depth={}",
                    path_repr(h),
                    self.zoom_level(),
                    self.anchor_depth(),
                );
            }
            None => {
                eprintln!("probe_down: MISS");
                println!(
                    "HARNESS_PROBE direction=down hit=false anchor=[] ui_layer={} anchor_depth={}",
                    self.zoom_level(),
                    self.anchor_depth(),
                );
            }
        }
    }

    /// Shared script-command dispatcher. Called from both the live
    /// event loop (`event_loop.rs`) and the render-harness loop
    /// (`test_runner.rs`) so new commands only need one handler.
    pub(super) fn handle_script_cmd(&mut self, cmd: ScriptCmd, frame: u32) {
        // Helper — emit camera/anchor/frame state at every script cmd so
        // we can see whether commands actually move / zoom / dig.
        let log_state = |label: &str, app: &App, extra: &str| {
            let anchor = app.camera.position.anchor;
            let anchor_slots: Vec<u32> =
                anchor.as_slice().iter().map(|&s| s as u32).collect();
            let offset = app.camera.position.offset;
            let world_pos = app.camera.position.in_frame(
                &crate::world::anchor::Path::root(),
            );
            let sphere_state = app.camera.position.sphere.as_ref().map(|s| {
                (
                    s.face as u32,
                    s.body_path.depth(),
                    s.uvr_path.depth(),
                    s.uvr_offset,
                )
            });
            let kind_str = match app.active_frame.kind {
                crate::app::ActiveFrameKind::Cartesian => "Cartesian".to_string(),
                crate::app::ActiveFrameKind::Body { inner_r, outer_r } => {
                    format!("Body(ir={inner_r},or={outer_r})")
                }
            };
            let render_slots: Vec<u32> = app
                .active_frame
                .render_path
                .as_slice()
                .iter()
                .map(|&s| s as u32)
                .collect();
            eprintln!(
                "SCRIPT_STATE [{label}] frame={frame} anchor_depth={} anchor={:?} offset=[{:.6},{:.6},{:.6}] world_pos=[{:.6},{:.6},{:.6}] zoom_level={} edit_depth={} visual_depth={} sphere={:?} kind={} render_path={:?} {extra}",
                anchor.depth(),
                anchor_slots,
                offset[0], offset[1], offset[2],
                world_pos[0], world_pos[1], world_pos[2],
                app.zoom_level(),
                app.edit_depth(),
                app.visual_depth(),
                sphere_state,
                kind_str,
                render_slots,
            );
        };

        log_state("PRE", self, &format!("cmd={:?}", cmd));
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
            ScriptCmd::TeleportIntoLastEdit => self.teleport_into_last_edit(),
            ScriptCmd::Step { axis, delta } => {
                let mut d = [0.0f32; 3];
                d[axis as usize] = delta;
                self.camera.position.add_local(d, &self.world.library);
            }
            ScriptCmd::FlyToSurface => self.fly_to_surface(),
        }
        log_state("POST", self, "");
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

    /// Position the camera inside the "above the edit" child of the
    /// last-broken cell. Intended use: after `zoom_in:1` following a
    /// break, this drops the camera "one current-layer cell above the
    /// new ground" in the descent flow.
    ///
    /// The path is constructed symbolically rather than via world-xyz
    /// arithmetic. At deep anchors, `1.5 + 3^{-17}` collapses to `1.5`
    /// in f32, so walking `last_edit_slots + [above_slot]*k` is the
    /// only way to land in the right cell at any depth.
    ///
    /// For sphere hits, "above the edit" = +r_local (slot (1, 1, 2)
    /// in UVR), so the camera sits at the outer radial face of the
    /// broken cell looking inward (toward core). For Cartesian hits,
    /// "above" = +y (slot (1, 0, 1) in XYZ's bottom-center, where the
    /// camera then looks -y at the cell it just broke).
    ///
    /// After setting the anchor, `WorldPos::new_with_sphere_resolved`
    /// walks the tree to repopulate `SphereState` so subsequent
    /// zoom_in / probe / render all see the sphere context.
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
        // Detect whether `last` crosses a CubedSphereBody: if so, the
        // descent slots are UVR-semantic and "above" means +r, not -y.
        let in_sphere = path_crosses_sphere_body(
            &self.world.library, self.world.root, &last,
        );
        let mut anchor = last;
        let above_slot = if in_sphere {
            // UVR (u, v, r) = (1, 1, 2) → outer-radial, center-u/v.
            slot_index(1, 1, 2) as u8
        } else {
            // Cartesian bottom-center.
            slot_index(1, 0, 1) as u8
        };
        while anchor.depth() < current_depth {
            anchor.push(above_slot);
        }
        // Camera at the center of the final cell; `new_with_sphere_resolved`
        // walks the tree so deep anchors inside a body re-establish
        // SphereState automatically.
        self.camera.position = if in_sphere {
            WorldPos::new_with_sphere_resolved(
                anchor,
                [0.5, 0.5, 0.5],
                &self.world.library,
                self.world.root,
            )
        } else {
            WorldPos::new(anchor, [0.5, 0.5, 0.5])
        };
        eprintln!(
            "teleport_above_last_edit: broken_path={:?} final_anchor={:?} current_depth={} sphere={} cam_sphere_after={:?}",
            last.as_slice(), anchor.as_slice(), current_depth, in_sphere,
            self.camera.position.sphere.map(|s| (s.face, s.uvr_path.depth())),
        );
        self.apply_zoom();
        eprintln!("teleport_above_last_edit: post_apply_zoom active_frame.kind={:?}", self.active_frame.kind);
    }

    /// Teleport the camera INTO the most recent dug cell, not the
    /// cell above it. Sets anchor = last_edit_slots, offset = (0.5,
    /// 0.5, 0.5). Rays from here look at the pit's walls / floor /
    /// ceiling from inside the empty cell.
    pub(super) fn teleport_into_last_edit(&mut self) {
        let Some(last) = self.last_edit_slots else {
            eprintln!("teleport_into_last_edit: no last edit recorded; skipping");
            return;
        };
        let in_sphere = path_crosses_sphere_body(
            &self.world.library, self.world.root, &last,
        );
        self.camera.position = if in_sphere {
            WorldPos::new_with_sphere_resolved(
                last,
                [0.5, 0.5, 0.5],
                &self.world.library,
                self.world.root,
            )
        } else {
            WorldPos::new(last, [0.5, 0.5, 0.5])
        };
        eprintln!(
            "teleport_into_last_edit: anchor={:?} depth={} sphere={} cam_sphere_after={:?}",
            last.as_slice(), last.depth(), in_sphere,
            self.camera.position.sphere.map(|s| (s.face, s.uvr_path.depth())),
        );
        self.apply_zoom();
        eprintln!("teleport_into_last_edit: post_apply_zoom active_frame.kind={:?}", self.active_frame.kind);
    }
}
