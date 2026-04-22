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

    /// CPU raycast at arbitrary pitch/yaw. Lets us probe specific
    /// rays from the scripted harness — the ray direction is set by
    /// (pitch, yaw), not hardcoded to straight-down. Used for
    /// bit-level pre vs post place diffing of adjacent rays at the
    /// pixel-adjacency scale (~1/400 rad pitch delta per pixel).
    pub(super) fn harness_probe_at(&mut self, pitch: f32, yaw: f32) {
        let saved_pitch = self.camera.pitch;
        let saved_yaw = self.camera.yaw;
        self.camera.pitch = pitch;
        self.camera.yaw = yaw;
        let hit = self.frame_aware_raycast();
        self.camera.pitch = saved_pitch;
        self.camera.yaw = saved_yaw;
        match hit {
            Some(ref h) => {
                eprintln!(
                    "probe_at pitch={:+.4} yaw={:+.4}: hit t={:.6e} path_len={} face={} sphere_cell={:?} path={:?}",
                    pitch, yaw,
                    h.t,
                    h.path.len(),
                    h.face,
                    h.sphere_cell,
                    h.path.iter().map(|(_, s)| *s as u32).collect::<Vec<_>>(),
                );
            }
            None => {
                eprintln!("probe_at pitch={:+.4} yaw={:+.4}: MISS", pitch, yaw);
            }
        }
    }

    /// Enable GPU walker-state probing for pixel (x, y). Forces an
    /// offscreen render so the probe buffer is populated deterministi-
    /// cally (rather than racing the main event-loop frame). Prints
    /// one line with all fields decoded to human-readable format.
    /// `hit_flag == 0` means sphere_in_cell's hit branch never ran
    /// for that pixel — either the ray missed the sphere entirely,
    /// or it traversed all empty cells without finding content.
    pub(super) fn harness_probe_gpu(&mut self, x: u32, y: u32) {
        // First: ensure GPU tree + camera reflect current state. The
        // offscreen render below uses the renderer's current bind
        // group; if the script just performed a place, its tree
        // update only lands on the NEXT main-loop `upload_tree_lod`.
        // Call it explicitly so the probe sees the post-edit tree.
        self.upload_tree_lod();
        let Some(renderer) = self.renderer.as_mut() else {
            eprintln!("probe_gpu: no renderer yet, skipping");
            return;
        };
        renderer.set_walker_probe_pixel(x, y, true);
        let _ = renderer.render_offscreen();
        let probe = renderer.read_walker_probe();
        eprintln!(
            "probe_gpu x={} y={}: hit_flag={} steps={} walker=(depth={} block={} ratio=({},{},{},{}) u_lo={:.6} v_lo={:.6} r_lo={:.6} size={:.6e}) face={} face_node_idx={} final=(winning={} t={:.6e})",
            x, y,
            probe.hit_flag,
            probe.steps,
            probe.walker_depth, probe.walker_block,
            probe.walker_ratio_u, probe.walker_ratio_v,
            probe.walker_ratio_r, probe.walker_ratio_depth,
            probe.walker_u_lo, probe.walker_v_lo,
            probe.walker_r_lo, probe.walker_size,
            probe.face, probe.face_node_idx,
            probe.final_winning, probe.final_t,
        );
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
            ScriptCmd::ProbeAt { pitch, yaw } => self.harness_probe_at(pitch, yaw),
            ScriptCmd::ProbeGpu { x, y } => self.harness_probe_gpu(x, y),
            ScriptCmd::Emit(label) => self.harness_emit_mark(&label, frame),
            ScriptCmd::TeleportAboveLastEdit => self.teleport_above_last_edit(),
            ScriptCmd::TeleportIntoLastEdit => self.teleport_into_last_edit(),
            ScriptCmd::Step { axis, delta } => {
                let mut d = [0.0f32; 3];
                d[axis as usize] = delta;
                self.camera.position.add_local(d, &self.world.library);
            }
            ScriptCmd::FlyToSurface => self.fly_to_surface(),
            ScriptCmd::FlyToSurfaceElevation(cells) => {
                self.fly_to_surface_elevation(cells);
            }
            // Unused — the rich positional stats it was going to
            // emit live in `DebugOverlayStateJs` now (see event_loop's
            // `overlay_active` branch). Kept as a parsed cmd for
            // backwards compat with any scripts that may reference it.
            ScriptCmd::DumpPosition => {}
        }
        log_state("POST", self, "");
    }

    /// Raycast straight down in world-space bypassing the normal
    /// interaction-radius cap, then place the camera EXACTLY ONE
    /// anchor cell above the hit (at the current anchor depth). Used
    /// by the d≥10 repro harness: the camera has to be within one
    /// cell of the surface so the cursor interaction radius reaches
    /// it AND the walker's Nyquist LOD allows descent to full anchor
    /// depth. Sphere worlds: camera stays `sphere=None` (same
    /// convention as the live game's zoom path — sphere state is
    /// populated only by explicit teleports); the body march renders
    /// from the body cell regardless.
    pub(super) fn fly_to_surface(&mut self) {
        // Preserves the original single-cell-above behavior.
        self.fly_to_surface_elevation(1);
    }

    /// Generalized form of `fly_to_surface`: raycast straight down in
    /// world-space bypassing the normal interaction-radius cap, then
    /// place the camera exactly `cells` anchor cells above the hit
    /// (at the current anchor depth). Cell size = `1 / 3^anchor_depth`
    /// in root-frame units, so the VISUAL altitude (cells above
    /// terrain) is independent of anchor depth — "50 cells above at
    /// d=10" and "50 cells above at d=5" look the same, just with
    /// finer subdivision visible at deeper d.
    ///
    /// Sphere worlds: camera stays `sphere=None` (same convention as
    /// `fly_to_surface` / the live game's zoom path); the body march
    /// renders from the body cell regardless. The result is clamped
    /// to stay inside the root cell (`<= 3.0 - cell`) so we never
    /// produce an out-of-frame WorldPos at extreme elevations.
    pub(super) fn fly_to_surface_elevation(&mut self, cells: u32) {
        use crate::world::anchor::{Path, WorldPos};
        use crate::world::raycast::cpu_raycast;

        let root_cam = self.camera.position.in_frame(&Path::root());
        let ray_dir = [0.0f32, -1.0, 0.0];
        let max_depth = self.tree_depth.saturating_sub(1).max(1);
        let hit = cpu_raycast(
            &self.world.library, self.world.root, root_cam, ray_dir, max_depth,
        );
        let Some(hit) = hit else {
            eprintln!(
                "fly_to_surface_elevation: MISS root_cam={root_cam:?} cells={cells}",
            );
            return;
        };
        let hit_y = root_cam[1] + ray_dir[1] * hit.t;
        let anchor_depth = self.anchor_depth() as u8;
        let cell_size = 1.0_f32 / 3.0_f32.powi(anchor_depth as i32);
        // `cells` anchor cells above the hit. At d=10, cell_size ≈
        // 1.69e-5 root units, so 1 cell ≈ the interaction-radius
        // epsilon; 50 cells ≈ 8.5e-4. The upper clamp keeps us inside
        // the root cell so `from_frame_local` doesn't saturate.
        let above_y = (hit_y + cells as f32 * cell_size).min(3.0 - cell_size);
        let cells_above_actual = ((above_y - hit_y) / cell_size).max(0.0);
        let new_pos = [root_cam[0], above_y, root_cam[2]];
        self.camera.position =
            WorldPos::from_frame_local(&Path::root(), new_pos, anchor_depth);
        eprintln!(
            "fly_to_surface_elevation: hit_y={:.6e} cell_size={:.6e} above_y={:.6e} cells_above={:.3} (requested={}) root_cam=[{:.6},{:.6},{:.6}] t={:.6e} new_pos=[{:.6e},{:.6e},{:.6e}] anchor_depth={} anchor={:?}",
            hit_y, cell_size, above_y, cells_above_actual, cells,
            root_cam[0], root_cam[1], root_cam[2],
            hit.t,
            new_pos[0], new_pos[1], new_pos[2],
            anchor_depth,
            self.camera.position.anchor.as_slice(),
        );
        // Refresh `active_frame`, ribbon, pack, GPU upload. Without
        // this the renderer keeps the pre-teleport frame and shows
        // stale content when the camera moved outside its old frame.
        self.apply_zoom();
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
