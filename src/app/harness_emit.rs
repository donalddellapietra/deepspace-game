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
            ScriptCmd::DigStepDown => self.dig_step_down(),
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

    /// Position the camera INSIDE the just-broken cell, at the inner
    /// side of that cell — i.e. just above the "floor" of the hole we
    /// dug. The anchor/UVR depth is unchanged from the break: we are
    /// symbolically standing IN the broken cell, not inside one of its
    /// children. The subsequent `probe_down` raycast exits through the
    /// inner face and hits the cell directly below (the r-1 sibling
    /// inside a sphere, or the -y sibling in Cartesian) at the SAME
    /// total depth.
    ///
    /// Sphere case: `sphere.uvr_path` is set to the broken cell's UVR
    /// descent, `uvr_offset = (0.5, 0.5, 0.05)` — center-u, center-v,
    /// just inside the inner-r face. The Cartesian `anchor` is the
    /// body_path only.
    /// Cartesian case: `anchor` is the broken cell's full path,
    /// `offset = (0.5, 0.05, 0.5)` — center-xz, just above -y face.
    ///
    /// This is the dig-down primitive: each invocation moves the
    /// camera one cell closer to the core (sphere) or ground
    /// (Cartesian) without changing depth. Depth changes only happen
    /// on a separate `zoom_in` — keeping the "dig at this depth" and
    /// "descend one depth" operations orthogonal.
    pub(super) fn dig_step_down(&mut self) {
        let Some(last) = self.last_edit_slots else {
            eprintln!("dig_step_down: no last edit recorded; skipping");
            return;
        };
        let in_sphere = path_crosses_sphere_body(
            &self.world.library, self.world.root, &last,
        );
        if in_sphere {
            // Walk `last` to find the body node + extract inner_r /
            // outer_r. `body_depth` is the anchor-index at which the
            // node.kind == CubedSphereBody. The slot AT that index
            // inside `last` is the face-root slot; slots after that
            // are UVR descent.
            use crate::world::cubesphere::Face;
            let mut node = self.world.root;
            let mut body_info: Option<(u8, f32, f32)> = None;
            for k in 0..last.depth() as usize {
                let Some(n) = self.world.library.get(node) else { break };
                if let NodeKind::CubedSphereBody { inner_r, outer_r } = n.kind {
                    body_info = Some((k as u8, inner_r, outer_r));
                    break;
                }
                let slot = last.slot(k) as usize;
                match n.children[slot] {
                    Child::Node(next) => node = next,
                    _ => break,
                }
            }
            // Check the terminal node too — `last` may END at the body.
            if body_info.is_none() {
                if let Some(n) = self.world.library.get(node) {
                    if let NodeKind::CubedSphereBody { inner_r, outer_r } = n.kind {
                        body_info = Some((last.depth(), inner_r, outer_r));
                    }
                }
            }
            let Some((body_depth, inner_r, outer_r)) = body_info else {
                eprintln!("dig_step_down: sphere path but body node not found; skipping");
                return;
            };
            if (body_depth as usize) >= last.depth() as usize {
                eprintln!(
                    "dig_step_down: last edit does not descend into body (body_depth={} path_len={}); skipping",
                    body_depth, last.depth(),
                );
                return;
            }
            let face_slot = last.slot(body_depth as usize);
            let Some(face) = Face::from_body_slot(face_slot) else {
                eprintln!(
                    "dig_step_down: edit slot {} is not a face slot; skipping",
                    face_slot,
                );
                return;
            };
            let body_path = last.with_truncated(body_depth);
            let mut uvr_path = Path::root();
            for k in ((body_depth as usize) + 1)..(last.depth() as usize) {
                uvr_path.push(last.slot(k));
            }
            // uvr_offset: center-u, center-v, just inside the inner-r
            // face of the broken cell. 0.05 (not 0.0) keeps the camera
            // well clear of the inner-r face for numerical stability
            // when the ray starts its march.
            let uvr_offset = [0.5f32, 0.5, 0.05];
            self.camera.position = crate::world::anchor::WorldPos {
                anchor: body_path,
                offset: uvr_offset,
                sphere: Some(crate::world::anchor::SphereState {
                    body_path,
                    inner_r,
                    outer_r,
                    face,
                    uvr_path,
                    uvr_offset,
                }),
            };
            eprintln!(
                "dig_step_down: sphere broken_path={:?} body_depth={} face={:?} uvr_path_depth={} total_depth={}",
                last.as_slice(),
                body_depth,
                face,
                uvr_path.depth(),
                self.camera.position.total_depth(),
            );
        } else {
            // Cartesian: camera at (0.5, 0.05, 0.5) inside the broken
            // cell. The anchor is the broken path as-is (no extra
            // slot push, unlike teleport_above_last_edit). The next
            // probe_down then exits through the -y face and hits the
            // cell below at the same depth.
            self.camera.position = WorldPos::new(last, [0.5, 0.05, 0.5]);
            eprintln!(
                "dig_step_down: cartesian broken_path={:?} anchor_depth={}",
                last.as_slice(),
                self.camera.position.anchor.depth(),
            );
        }
        self.apply_zoom();
        eprintln!(
            "dig_step_down: post_apply_zoom active_frame.kind={:?} anchor_depth={}",
            self.active_frame.kind,
            self.anchor_depth(),
        );
    }
}
