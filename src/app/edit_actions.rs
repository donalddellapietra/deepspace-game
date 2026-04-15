//! Break / place / highlight / zoom / GPU upload on the `App`.
//!
//! All edits go through the unified `edit::cpu_raycast` →
//! `break_block` / `place_block` pipeline. The planet is part of
//! the same tree, so there's no longer a sphere-specific code
//! path here. (CPU raycast traversal of `CubedSphereBody` cells
//! is a follow-up; for now hits inside the body cell are treated
//! as Cartesian, which means edits on the planet's interior
//! aren't yet supported. Rendering of the planet works.)

use crate::game_state::HotbarItem;
use crate::world::edit;
use crate::world::gpu;

use super::App;

struct BodyInfo {
    body_id: crate::world::tree::NodeId,
    body_path: crate::world::anchor::Path,
    inner_r_world: f32,
    outer_r_world: f32,
}

impl App {
    pub(super) fn edit_depth(&self) -> u32 {
        self.anchor_depth().saturating_sub(1).max(1)
    }

    /// Face-subtree depth at which sphere edits land. Picks a
    /// user-visible cell granularity matching the UI's layer
    /// readout: placing at "Layer N" produces a block whose cell
    /// size is exactly 3^N cells per face axis. The previous
    /// `anchor - 4` formula was off by one — placements at Layer
    /// N landed at face-subtree depth N-1 ("Layer N+1 block").
    ///
    /// No numerical depth cap. Any ceiling would be an admission
    /// the shader's rendering doesn't honor path-anchored
    /// precision (camera in root-scale f32 breaks at anchor ~15,
    /// `cells_d = pow(3, depth)` overflows f32-integer-exact at
    /// depth 16). Those are shader bugs to fix, not budgets to
    /// ration. The lower bound of 1 is the only actual content
    /// constraint — depth 0 would place above the face root.
    pub(super) fn cs_edit_depth(&self) -> u32 {
        ((self.anchor_depth() as i32) - 3).max(1) as u32
    }

    pub(super) fn visual_depth(&self) -> u32 {
        (self.edit_depth() + 3).min(16)
    }

    pub fn apply_zoom(&mut self) {
        self.ui.zoom_level = self.zoom_level();
        let vd = self.visual_depth();
        let world_pos = self.camera.world_pos_f32();
        if let Some(renderer) = &mut self.renderer {
            renderer.set_max_depth(vd);
            renderer.update_camera(&self.camera.gpu_camera_at(world_pos, 1.2));
        }
        // Rebuild the ribbon + tree pack from the new anchor.
        self.upload_tree_lod();
        log::info!(
            "Zoom: {}/{}, edit_depth: {}, visual: {}, anchor_depth: {}",
            self.zoom_level(), self.tree_depth as i32,
            self.edit_depth(), vd, self.anchor_depth(),
        );
    }

    pub(super) fn do_break(&mut self) {
        let ray_dir = self.camera.forward();
        let camera_pos = self.camera.world_pos_f32();
        let hit = edit::cpu_raycast_with_face_depth(
            &self.world.library, self.world.root,
            camera_pos, ray_dir, self.edit_depth(),
            self.cs_edit_depth(),
        );
        eprintln!("do_break: hit={:?}",
            hit.as_ref().map(|h| (h.path.len(), h.face, h.t)));
        let Some(hit) = hit else { return };

        if self.save_mode {
            use crate::world::tree::Child;
            let mut saved_id = None;
            if let Some(&(parent_id, slot)) = hit.path.last() {
                if let Some(node) = self.world.library.get(parent_id) {
                    match node.children[slot] {
                        Child::Node(child_id) => saved_id = Some(child_id),
                        Child::Block(_) | Child::Empty => saved_id = Some(parent_id),
                    }
                }
            }
            if let Some(node_id) = saved_id {
                self.world.library.ref_inc(node_id);
                let idx = self.saved_meshes.save(node_id);
                self.ui.slots[self.ui.active_slot] = HotbarItem::Mesh(idx);
                log::info!("Saved mesh #{idx} (node {node_id})");
            }
            self.save_mode = false;
            return;
        }

        let changed = edit::break_block(&mut self.world, &hit);
        eprintln!("do_break: break_block returned {}", changed);
        if changed {
            self.upload_tree();
        }
    }

    pub(super) fn do_place(&mut self) {
        let ray_dir = self.camera.forward();
        let camera_pos = self.camera.world_pos_f32();
        let hit = edit::cpu_raycast_with_face_depth(
            &self.world.library, self.world.root,
            camera_pos, ray_dir, self.edit_depth(),
            self.cs_edit_depth(),
        );
        eprintln!("do_place: hit={:?}",
            hit.as_ref().map(|h| (h.path.len(), h.face, h.t)));
        let Some(hit) = hit else { return };

        match &self.ui.slots[self.ui.active_slot] {
            HotbarItem::Block(block_type) => {
                if edit::place_block(&mut self.world, &hit, *block_type) {
                    self.upload_tree();
                }
            }
            HotbarItem::Mesh(idx) => {
                let Some(saved) = self.saved_meshes.items.get(*idx) else { return };
                let node_id = saved.node_id;
                if edit::place_child(
                    &mut self.world, &hit,
                    crate::world::tree::Child::Node(node_id),
                ) {
                    self.upload_tree();
                }
            }
        }
    }

    pub(super) fn upload_tree(&mut self) {
        self.tree_depth = self.world.tree_depth();
        self.upload_tree_lod();
    }

    /// Pack the tree from world root into one buffer and build the
    /// ribbon uniform. Every ribbon frame's node is kept walkable by
    /// passing the camera's anchor as the packer's `preserve_path`,
    /// so LOD flattening can't strand a frame root.
    pub(super) fn upload_tree_lod(&mut self) {
        let ribbon = self.render_ribbon();
        let world_pos = self.camera.world_pos_f32();
        let anchor = self.camera.position.anchor;
        let (tree_data, node_kinds, _world_root_idx, visited) = gpu::pack_tree_lod(
            &self.world.library, self.world.root,
            world_pos, 1440.0, 1.2,
            &anchor,
        );

        // Compute the planet uniform. Walks `planet_path` to find the
        // current body node (paths persist across edits via content-
        // addressed rebuild, but the NodeId at the end of the path
        // changes each time the subtree mutates — look it up fresh).
        let planet_gpu = self.compute_planet_uniform(&visited);
        // Map each ribbon frame's NodeId to its buffer index. If a
        // frame's node wasn't reachable (shouldn't happen with the
        // preserve_path pass, but defensive), fall back to the root.
        // For per-frame sphere math we need the body's `(inner_r,
        // outer_r)` in world units and its center as a path-anchored
        // WorldPos. Look these up once per upload.
        let body_info = self.lookup_body_info();

        let mut gpu_ribbon: Vec<gpu::GpuRibbonFrame> = Vec::with_capacity(ribbon.len());
        for f in &ribbon {
            let root_index = visited.get(&f.node_id).copied().unwrap_or(0);
            // Try to compute per-frame sphere data. If the frame's
            // path doesn't go through the body, sphere_active=0.
            let sphere = body_info.as_ref().and_then(|bi| {
                self.compute_sphere_frame_data(&f.path, &visited, bi)
            });
            let mut gf = gpu::GpuRibbonFrame {
                root_index,
                sphere_active: 0,
                world_scale: f.world_scale,
                face: 0,
                camera_local: [f.camera_local[0], f.camera_local[1], f.camera_local[2], 0.0],
                ..Default::default()
            };
            if let Some(s) = sphere {
                gf.sphere_active = 1;
                gf.face = s.face;
                gf.frame_face_node_idx = s.frame_face_node_idx;
                gf.frame_un_size = s.frame_un_size;
                gf.frame_alpha_n_u = s.frame_alpha_n_u;
                gf.frame_alpha_n_v = s.frame_alpha_n_v;
                gf.camera_un_remainder = s.camera_un_remainder;
                gf.camera_vn_remainder = s.camera_vn_remainder;
                gf.camera_rn_remainder = s.camera_rn_remainder;
                gf.frame_alpha_r = s.frame_alpha_r;
                gf.frame_n_u_lo_ref = s.frame_n_u_lo_ref;
                gf.frame_n_v_lo_ref = s.frame_n_v_lo_ref;
                gf.frame_r_lo_world = s.frame_r_lo_world;
                gf.sphere_inner_r_world = s.sphere_inner_r_world;
                gf.sphere_outer_r_world = s.sphere_outer_r_world;
                gf.sphere_shell_world = s.sphere_shell_world;
                gf.face_n_axis = s.face_n_axis;
            }
            gpu_ribbon.push(gf);
        }
        if let Some(renderer) = &mut self.renderer {
            renderer.update_tree(&tree_data, &node_kinds, &gpu_ribbon);
            renderer.set_planet(planet_gpu);
            renderer.update_camera(&self.camera.gpu_camera_at(world_pos, 1.2));
        }
    }

    /// Look up the body's path-anchored position + radii, or None
    /// if the planet hasn't been installed.
    fn lookup_body_info(&self) -> Option<BodyInfo> {
        use crate::world::tree::{Child, NodeKind};
        let mut node_id = self.world.root;
        for k in 0..self.planet_path.depth() as usize {
            let node = self.world.library.get(node_id)?;
            let slot = self.planet_path.slot(k) as usize;
            match node.children[slot] {
                Child::Node(cid) => { node_id = cid; }
                _ => return None,
            }
        }
        let body_node = self.world.library.get(node_id)?;
        let (inner_r_local, outer_r_local) = match body_node.kind {
            NodeKind::CubedSphereBody { inner_r, outer_r } => (inner_r, outer_r),
            _ => return None,
        };
        let body_depth = self.planet_path.depth();
        let body_cell_size = crate::world::anchor::WORLD_SIZE
            / 3.0_f32.powi(body_depth as i32);
        Some(BodyInfo {
            body_id: node_id,
            body_path: self.planet_path,
            inner_r_world: inner_r_local * body_cell_size,
            outer_r_world: outer_r_local * body_cell_size,
        })
    }

    /// Compute the per-frame sphere uniform fields for a ribbon
    /// frame whose path passes through the body. Returns None if
    /// the frame's path doesn't reach into the body's face subtree
    /// (not deep enough, or doesn't go through the body's slot).
    ///
    /// All accumulation is in f64 — the f32 cast at the end keeps
    /// 7 digits of precision for the small remainder values, while
    /// raw f32 subtraction `(camera_un_global - frame_un_lo_global)`
    /// would have lost the low-order bits at deep frame depths.
    fn compute_sphere_frame_data(
        &self,
        frame_path: &crate::world::anchor::Path,
        visited: &std::collections::HashMap<crate::world::tree::NodeId, u32>,
        body_info: &BodyInfo,
    ) -> Option<gpu::SphereFrameData> {
        use crate::world::cubesphere::{Face, FACE_SLOTS};
        use crate::world::tree::{slot_coords, Child, NodeKind};

        // 1. Walk the frame's path until we encounter the body node.
        //    Track current node_id and depth as we descend.
        let mut node_id = self.world.root;
        let mut depth = 0u8;
        let mut body_face: Option<Face> = None;
        let mut face_root_id: Option<crate::world::tree::NodeId> = None;
        while depth < frame_path.depth() {
            let node = self.world.library.get(node_id)?;
            let slot = frame_path.slot(depth as usize) as usize;
            if matches!(node.kind, NodeKind::CubedSphereBody { .. }) {
                // Identify which face the path enters via.
                let f_idx = (0..6).find(|&f| FACE_SLOTS[f] == slot)?;
                body_face = Some(Face::from_index(f_idx as u8));
                if let Child::Node(cid) = node.children[slot] {
                    face_root_id = Some(cid);
                    node_id = cid;
                    depth += 1;
                    break;
                }
                return None;
            }
            // Cartesian step.
            if let Child::Node(cid) = node.children[slot] {
                node_id = cid;
                depth += 1;
            } else {
                return None;
            }
        }
        let face = body_face?;
        let _face_root_id = face_root_id?;

        // 2. Walk remaining face-subtree path, accumulating the
        //    frame's (un_lo, vn_lo, rn_lo, size) in face EA-norm
        //    [0, 1] coords using f64.
        let mut un_lo: f64 = 0.0;
        let mut vn_lo: f64 = 0.0;
        let mut rn_lo: f64 = 0.0;
        let mut size: f64 = 1.0;
        while depth < frame_path.depth() {
            let node = self.world.library.get(node_id)?;
            let slot = frame_path.slot(depth as usize) as usize;
            let (us, vs, rs) = slot_coords(slot);
            let next_size = size / 3.0;
            un_lo += us as f64 * next_size;
            vn_lo += vs as f64 * next_size;
            rn_lo += rs as f64 * next_size;
            size = next_size;
            if let Child::Node(cid) = node.children[slot] {
                node_id = cid;
                depth += 1;
            } else {
                // Path terminates early at a uniform terminal —
                // frame_face_node_idx points at this terminal's
                // parent's slot via `node_id` which is the last
                // valid Node we had.
                break;
            }
        }
        let frame_face_node_idx = visited.get(&node_id).copied()?;

        // 3. Camera in face EA-norm coords (f64).
        let body_center = crate::world::anchor::WorldPos::new(
            body_info.body_path, [0.5, 0.5, 0.5],
        );
        let oc = self.camera.position.offset_from(&body_center);
        let oc_64 = [oc[0] as f64, oc[1] as f64, oc[2] as f64];
        let r_camera = (oc_64[0]*oc_64[0] + oc_64[1]*oc_64[1] + oc_64[2]*oc_64[2]).sqrt();
        if r_camera < 1e-12 { return None; }
        let cam_dir = [oc_64[0]/r_camera, oc_64[1]/r_camera, oc_64[2]/r_camera];

        let face_n = face.normal();
        let (face_u, face_v) = face.tangents();
        let n_axis = [face_n[0] as f64, face_n[1] as f64, face_n[2] as f64];
        let u_axis = [face_u[0] as f64, face_u[1] as f64, face_u[2] as f64];
        let v_axis = [face_v[0] as f64, face_v[1] as f64, face_v[2] as f64];

        let axis_dot = cam_dir[0]*n_axis[0] + cam_dir[1]*n_axis[1] + cam_dir[2]*n_axis[2];
        if axis_dot.abs() < 1e-9 { return None; }
        let cube_u = (cam_dir[0]*u_axis[0] + cam_dir[1]*u_axis[1] + cam_dir[2]*u_axis[2]) / axis_dot;
        let cube_v = (cam_dir[0]*v_axis[0] + cam_dir[1]*v_axis[1] + cam_dir[2]*v_axis[2]) / axis_dot;
        let pi = std::f64::consts::PI;
        let camera_un_global = (cube_u.atan() * 4.0 / pi + 1.0) * 0.5;
        let camera_vn_global = (cube_v.atan() * 4.0 / pi + 1.0) * 0.5;
        let cs_inner_64 = body_info.inner_r_world as f64;
        let cs_outer_64 = body_info.outer_r_world as f64;
        let shell_64 = cs_outer_64 - cs_inner_64;
        let camera_rn_global = (r_camera - cs_inner_64) / shell_64;

        let camera_un_remainder = ((camera_un_global - un_lo) / size) as f32;
        let camera_vn_remainder = ((camera_vn_global - vn_lo) / size) as f32;
        let camera_rn_remainder = ((camera_rn_global - rn_lo) / size) as f32;

        // 4. Reference plane normals at the frame's u_lo / v_lo,
        //    in cube coords composed with face axes (world).
        let u_lo_ea = un_lo * 2.0 - 1.0;
        let v_lo_ea = vn_lo * 2.0 - 1.0;
        let u_lo_cube = (u_lo_ea * pi / 4.0).tan();
        let v_lo_cube = (v_lo_ea * pi / 4.0).tan();
        let frame_n_u_lo = [
            (u_axis[0] - u_lo_cube * n_axis[0]) as f32,
            (u_axis[1] - u_lo_cube * n_axis[1]) as f32,
            (u_axis[2] - u_lo_cube * n_axis[2]) as f32,
            0.0,
        ];
        let frame_n_v_lo = [
            (v_axis[0] - v_lo_cube * n_axis[0]) as f32,
            (v_axis[1] - v_lo_cube * n_axis[1]) as f32,
            (v_axis[2] - v_lo_cube * n_axis[2]) as f32,
            0.0,
        ];

        // 5. Alpha values: rate of cube-coord change with respect
        //    to a unit cell_local offset within the frame.
        let sec_sq_u = 1.0 / (u_lo_ea * pi / 4.0).cos().powi(2);
        let sec_sq_v = 1.0 / (v_lo_ea * pi / 4.0).cos().powi(2);
        let frame_alpha_n_u = (2.0 * size * (pi / 4.0) * sec_sq_u) as f32;
        let frame_alpha_n_v = (2.0 * size * (pi / 4.0) * sec_sq_v) as f32;
        let frame_alpha_r = (size * shell_64) as f32;

        let frame_r_lo_world = (cs_inner_64 + rn_lo * shell_64) as f32;

        Some(gpu::SphereFrameData {
            face: face as u32,
            frame_face_node_idx,
            frame_un_size: size as f32,
            frame_alpha_n_u,
            frame_alpha_n_v,
            camera_un_remainder,
            camera_vn_remainder,
            camera_rn_remainder,
            frame_alpha_r,
            frame_n_u_lo_ref: frame_n_u_lo,
            frame_n_v_lo_ref: frame_n_v_lo,
            frame_r_lo_world,
            sphere_inner_r_world: body_info.inner_r_world,
            sphere_outer_r_world: body_info.outer_r_world,
            sphere_shell_world: shell_64 as f32,
            face_n_axis: [face_n[0], face_n[1], face_n[2], 0.0],
        })
    }

    /// Compute the planet uniform from the current world state and
    /// camera. Uses path-anchored `offset_from` so `oc_world` stays
    /// bounded (magnitude ≤ body cell size in world units)
    /// regardless of the camera's anchor depth — this is the
    /// precision-safe replacement for passing the world-scale
    /// camera/center through a frame-local scaling.
    ///
    /// `active=0` is returned when the planet isn't reachable via
    /// `planet_path` (e.g., the planet hasn't been installed yet or
    /// the pack truncated before the body) — the shader skips its
    /// sphere pass entirely.
    fn compute_planet_uniform(
        &self,
        visited: &std::collections::HashMap<crate::world::tree::NodeId, u32>,
    ) -> gpu::GpuPlanet {
        use crate::world::anchor::WorldPos;
        use crate::world::tree::{Child, NodeKind};

        // Walk planet_path to find the body NodeId.
        let mut node_id = self.world.root;
        for k in 0..self.planet_path.depth() as usize {
            let Some(node) = self.world.library.get(node_id) else {
                return gpu::GpuPlanet::default();
            };
            let slot = self.planet_path.slot(k) as usize;
            match node.children[slot] {
                Child::Node(cid) => node_id = cid,
                _ => return gpu::GpuPlanet::default(),
            }
        }
        let body_id = node_id;
        let Some(body_node) = self.world.library.get(body_id) else {
            return gpu::GpuPlanet::default();
        };
        let (inner_r_local, outer_r_local) = match body_node.kind {
            NodeKind::CubedSphereBody { inner_r, outer_r } => (inner_r, outer_r),
            _ => return gpu::GpuPlanet::default(),
        };

        // Body cell size in world units. At depth 1, cell_size =
        // WORLD_SIZE / 3 = 1.0. More generally: WORLD_SIZE / 3^depth.
        let body_depth = self.planet_path.depth();
        let body_cell_size = crate::world::anchor::WORLD_SIZE
            / (3.0_f32).powi(body_depth as i32);
        let inner_r_world = inner_r_local * body_cell_size;
        let outer_r_world = outer_r_local * body_cell_size;

        // Body center in WorldPos form, then path-anchored offset
        // from the camera. `offset_from` composes positions in their
        // common ancestor's frame — the body node IS the common
        // ancestor when the camera is inside the body's subtree, so
        // precision is bounded by body_cell_size * 1e-7 in that case.
        let body_center = WorldPos::new(self.planet_path, [0.5, 0.5, 0.5]);
        let oc = self.camera.position.offset_from(&body_center);

        let body_node_index = visited.get(&body_id).copied().unwrap_or(u32::MAX);
        if body_node_index == u32::MAX {
            return gpu::GpuPlanet::default();
        }

        gpu::GpuPlanet {
            enabled: 1,
            body_node_index,
            inner_r_world,
            outer_r_world,
            oc_world: [oc[0], oc[1], oc[2], 0.0],
            _reserved: 0,
            _pad: [0; 3],
        }
    }

    pub(super) fn update_highlight(&mut self) {
        if !self.cursor_locked {
            if let Some(renderer) = &mut self.renderer {
                renderer.set_highlight(None);
            }
            return;
        }
        let ray_dir = self.camera.forward();
        let camera_pos = self.camera.world_pos_f32();
        let tree_hit = edit::cpu_raycast_with_face_depth(
            &self.world.library, self.world.root,
            camera_pos, ray_dir, self.edit_depth(),
            self.cs_edit_depth(),
        );
        let aabb = tree_hit.as_ref().map(|h| edit::hit_aabb(&self.world.library, h));
        if let Some((mn, mx)) = &aabb {
            eprintln!("highlight: min={:?} max={:?} size={:?}",
                mn, mx, [mx[0]-mn[0], mx[1]-mn[1], mx[2]-mn[2]]);
        }
        if let Some(renderer) = &mut self.renderer {
            renderer.set_highlight(aabb);
        }
    }
}
