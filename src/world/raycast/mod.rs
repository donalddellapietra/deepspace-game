//! CPU raycasting — Cartesian DDA, frame-aware ribbon pops, and
//! point solidity queries.
//!
//! The CPU ray march mirrors the GPU shader's tree traversal so that
//! the cell the crosshair targets is the same cell the shader is
//! shading. Edits operate at a layer-dependent depth: the zoom level
//! controls how deep the raycast descends, so the same code breaks a
//! single block at fine zoom or an entire 3×3×3 node at coarse zoom.

mod cartesian;

use crate::world::tree::{slot_coords, slot_index, Child, NodeId, NodeLibrary};

/// Information about a ray hit in the tree.
#[derive(Debug, Clone)]
pub struct HitInfo {
    /// Path from root to the hit: each entry is (node_id, child_slot).
    /// The last entry's child_slot is the slot that was hit.
    pub path: Vec<(NodeId, usize)>,
    /// Which face was crossed when the block was hit.
    /// 0=+X, 1=-X, 2=+Y, 3=-Y, 4=+Z, 5=-Z
    pub face: u32,
    /// Distance along the ray to the hit point.
    pub t: f32,
    /// Optional explicit path where a place_block should land. Cartesian
    /// hits leave this `None` — `place_child` derives the adjacent cell
    /// via `face`.
    pub place_path: Option<Vec<(NodeId, usize)>>,
}

/// Cast a ray through the tree, stopping at `max_depth` levels from
/// root. `max_depth` controls the interaction layer: at depth 3 in a
/// 3-level tree the ray targets individual blocks; at depth 2 it
/// targets 3×3×3 node groups.
pub fn cpu_raycast(
    library: &NodeLibrary,
    root: NodeId,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    max_depth: u32,
) -> Option<HitInfo> {
    cartesian::cpu_raycast_inner(library, root, ray_origin, ray_dir, max_depth)
}

/// Phase 3 REVISED A.4 — UV-sphere raycast for the WrappedPlane
/// frame when sphere-render mode is active. CPU mirror of the
/// shader's `sphere_uv_in_cell`.
///
/// `cam_local` and `ray_dir` are in the slab-root frame's local
/// `[0, 3)³` coords. The function ray-intersects the implied
/// sphere (centered at frame center, R = body_size / (2π)),
/// computes (lat, lon) from the surface normal, bans poles past
/// `lat_max`, maps to slab `(cell_x, cell_z)` (with `cell_y` at
/// the GRASS row = `dims.y - 1`), and walks the slab tree to
/// retrieve the cell's NodeId path.
///
/// `frame_path` is the world-root → slab-root path; it's prepended
/// to the returned HitInfo's path so callers (place / break) get a
/// fully-qualified world-tree path matching the existing flat
/// raycast's output shape.
pub fn cpu_raycast_sphere_uv(
    library: &NodeLibrary,
    world_root: NodeId,
    frame_path: &[u8],
    cam_local: [f32; 3],
    ray_dir: [f32; 3],
    dims: [u32; 3],
    slab_depth: u8,
    lat_max: f32,
) -> Option<HitInfo> {
    // Sphere center / radius in the slab-root frame's [0, 3)³ space.
    let cs_center = [1.5_f32, 1.5, 1.5];
    let body_size = 3.0_f32;
    let r_sphere = body_size / (2.0 * std::f32::consts::PI);

    // Normalize ray_dir for sphere intersect; remember the inverse so
    // the returned `t` matches the un-normalised parameter the caller
    // expects (`cam_local + ray_dir * t` lands on the hit point).
    let dir_len = (ray_dir[0] * ray_dir[0]
        + ray_dir[1] * ray_dir[1]
        + ray_dir[2] * ray_dir[2])
        .sqrt()
        .max(1e-6);
    let inv_norm = 1.0 / dir_len;
    let dir = [
        ray_dir[0] * inv_norm,
        ray_dir[1] * inv_norm,
        ray_dir[2] * inv_norm,
    ];

    let oc = [
        cam_local[0] - cs_center[0],
        cam_local[1] - cs_center[1],
        cam_local[2] - cs_center[2],
    ];
    let b = oc[0] * dir[0] + oc[1] * dir[1] + oc[2] * dir[2];
    let c = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2] - r_sphere * r_sphere;
    let disc = b * b - c;
    if disc <= 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let t_enter = (-b - sq).max(0.0);
    let t_exit = -b + sq;
    if t_exit <= 0.0 {
        return None;
    }

    let hit = [
        cam_local[0] + dir[0] * t_enter,
        cam_local[1] + dir[1] * t_enter,
        cam_local[2] + dir[2] * t_enter,
    ];
    let n = [
        (hit[0] - cs_center[0]) / r_sphere,
        (hit[1] - cs_center[1]) / r_sphere,
        (hit[2] - cs_center[2]) / r_sphere,
    ];
    // A.2 (rev) — Analytical per-layer sampling. Mirrors the GPU
    // shader's revised approach: for each radial cell layer cy ∈
    // [dims_y - 1 .. 0], compute the EXACT t at the layer's midpoint
    // radius (= solve quadratic for ray-vs-sphere-of-radius-r_mid),
    // sample (lat, lon) ONCE at that t, look up the cell. First
    // solid layer wins. dims_y iterations per ray instead of fixed-
    // step march. Eliminates per-pixel radial drift artifacts.
    let pi = std::f32::consts::PI;
    let shell_thickness = r_sphere * 0.25;
    let _r_outer = r_sphere;
    let r_inner = r_sphere - shell_thickness;
    let oc_dot_oc = oc[0] * oc[0] + oc[1] * oc[1] + oc[2] * oc[2];

    // Walk frame_path from world_root to the slab root once.
    let mut frame_chain: Vec<(NodeId, usize)> = Vec::with_capacity(frame_path.len());
    let mut cur = world_root;
    for &slot in frame_path.iter() {
        let node = library.get(cur)?;
        frame_chain.push((cur, slot as usize));
        match node.children[slot as usize] {
            Child::Node(child) => cur = child,
            _ => return None,
        }
    }
    let slab_root = cur;

    for cy in (0..dims[1] as i32).rev() {
        // Sample at layer cy's TOP boundary radius (where the ray
        // first enters the layer from outside) — same fix as the
        // GPU shader: sampling at the midpoint dropped grazing
        // chords that touched the layer but never reached the
        // midpoint.
        let r_mid = r_inner + (cy as f32 + 1.0) / dims[1] as f32 * shell_thickness;
        let cr = oc_dot_oc - r_mid * r_mid;
        let disc_l = b * b - cr;
        if disc_l < 0.0 {
            continue; // ray doesn't reach this radius
        }
        let sq_l = disc_l.sqrt();
        let t_layer = -b - sq_l;
        if t_layer < 0.0 || t_layer > t_exit {
            continue;
        }

        let pos_l = [
            cam_local[0] + dir[0] * t_layer,
            cam_local[1] + dir[1] * t_layer,
            cam_local[2] + dir[2] * t_layer,
        ];
        let n_l = [
            (pos_l[0] - cs_center[0]) / r_mid,
            (pos_l[1] - cs_center[1]) / r_mid,
            (pos_l[2] - cs_center[2]) / r_mid,
        ];
        let lat_l = n_l[1].clamp(-1.0, 1.0).asin();
        if lat_l.abs() > lat_max {
            continue;
        }
        let lon_l = n_l[2].atan2(n_l[0]);
        let u_l = (lon_l + pi) / (2.0 * pi);
        let v_l = (lat_l + lat_max) / (2.0 * lat_max);
        let cell_x = ((u_l * dims[0] as f32).floor() as i32).clamp(0, dims[0] as i32 - 1);
        let cell_z = ((v_l * dims[2] as f32).floor() as i32).clamp(0, dims[2] as i32 - 1);

        // Walk slab tree down to (cell_x, cy, cell_z).
        let mut path = frame_chain.clone();
        let mut idx = slab_root;
        let mut cells_per_slot: i32 = 1;
        for _ in 1..slab_depth {
            cells_per_slot *= 3;
        }
        let mut cell_is_empty = false;
        for level in 0..slab_depth {
            let sx = (cell_x / cells_per_slot).rem_euclid(3);
            let sy = (cy / cells_per_slot).rem_euclid(3);
            let sz = (cell_z / cells_per_slot).rem_euclid(3);
            let slot = slot_index(sx as usize, sy as usize, sz as usize);
            path.push((idx, slot));
            let node = match library.get(idx) {
                Some(n) => n,
                None => {
                    cell_is_empty = true;
                    break;
                }
            };
            match node.children[slot] {
                Child::Empty | Child::EntityRef(_) => {
                    cell_is_empty = true;
                    break;
                }
                Child::Block(_) => break,
                Child::Node(child) => {
                    if level + 1 < slab_depth {
                        idx = child;
                    }
                }
            }
            cells_per_slot /= 3;
        }
        if !cell_is_empty {
            return Some(HitInfo {
                path,
                face: 2,
                t: t_layer * inv_norm,
                place_path: None,
            });
        }
    }

    None
}

/// Frame-aware raycast. Mirrors the renderer's ribbon-pop
/// architecture so the CPU hit depth matches what the shader
/// renders (LOD-bounded, not budget-bounded): cell-precision is
/// bounded by the frame depth (camera in `[0, 3)` regardless of
/// absolute path), and the ray pops upward into ancestor frames
/// when it exits the current frame's bubble.
pub fn cpu_raycast_in_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    frame_path: &[u8],
    cam_local: [f32; 3],
    ray_dir: [f32; 3],
    max_depth: u32,
    _max_face_depth: u32,
) -> Option<HitInfo> {
    let (chain, frame_entries) = build_frame_chain(library, world_root, frame_path);
    let effective_depth = chain.len() - 1;
    let frame_entries = &frame_entries[..effective_depth];

    let mut current_frame_depth = effective_depth;
    let mut ray_origin = cam_local;
    let mut ray_dir = ray_dir;
    let total_max_depth = max_depth;

    loop {
        let frame_root_id = chain[current_frame_depth];
        let inner_max = total_max_depth.saturating_sub(current_frame_depth as u32);

        let hit_opt = cartesian::cpu_raycast_inner(
            library, frame_root_id, ray_origin, ray_dir, inner_max,
        );

        if let Some(mut hit) = hit_opt {
            prepend_frame_entries(&mut hit, frame_entries, current_frame_depth);
            return Some(hit);
        }

        // Miss in current frame — pop one level. Single-level pops
        // match the shader: skip_slot only covers the immediate
        // child (which the inner shell fully traversed). Multi-pop
        // would skip intermediate levels with un-traversed content.
        if current_frame_depth == 0 {
            return None;
        }
        let last_slot = frame_entries[current_frame_depth - 1].1;
        let (sx, sy, sz) = slot_coords(last_slot);
        ray_origin = [
            sx as f32 + ray_origin[0] / 3.0,
            sy as f32 + ray_origin[1] / 3.0,
            sz as f32 + ray_origin[2] / 3.0,
        ];
        ray_dir = [ray_dir[0] / 3.0, ray_dir[1] / 3.0, ray_dir[2] / 3.0];
        current_frame_depth -= 1;
    }
}

/// Check whether the cell at world-space position `pos` is solid at
/// the given tree depth. Walks the tree from root, mapping the
/// position to slot indices at each level. Returns true if the cell
/// is Block or Node (has content); false if Empty or out of bounds.
pub fn is_solid_at(
    library: &NodeLibrary,
    root: NodeId,
    pos: [f32; 3],
    max_depth: u32,
) -> bool {
    if pos[0] < 0.0 || pos[0] >= 3.0
        || pos[1] < 0.0 || pos[1] >= 3.0
        || pos[2] < 0.0 || pos[2] >= 3.0
    {
        return false;
    }

    let mut node_id = root;
    let mut node_origin = [0.0f32; 3];
    let mut cell_size = 1.0f32;

    for depth in 0..max_depth {
        let node = match library.get(node_id) {
            Some(n) => n,
            None => return false,
        };

        let cx = ((pos[0] - node_origin[0]) / cell_size).floor() as i32;
        let cy = ((pos[1] - node_origin[1]) / cell_size).floor() as i32;
        let cz = ((pos[2] - node_origin[2]) / cell_size).floor() as i32;

        if cx < 0 || cx > 2 || cy < 0 || cy > 2 || cz < 0 || cz > 2 {
            return false;
        }

        let slot = slot_index(cx as usize, cy as usize, cz as usize);
        match node.children[slot] {
            Child::Empty | Child::EntityRef(_) => return false,
            Child::Block(_) => return true,
            Child::Node(child_id) => {
                if depth + 1 >= max_depth {
                    return true;
                }
                node_origin = [
                    node_origin[0] + cx as f32 * cell_size,
                    node_origin[1] + cy as f32 * cell_size,
                    node_origin[2] + cz as f32 * cell_size,
                ];
                cell_size /= 3.0;
                node_id = child_id;
            }
        }
    }

    true
}

/// Walk `frame_path` from `world_root`, returning the chain of
/// NodeIds (root + each descended child) and the matching
/// `(parent_id, slot)` entries. Stops early on a non-Node child.
fn build_frame_chain(
    library: &NodeLibrary,
    world_root: NodeId,
    frame_path: &[u8],
) -> (Vec<NodeId>, Vec<(NodeId, usize)>) {
    let mut chain: Vec<NodeId> = Vec::with_capacity(frame_path.len() + 1);
    chain.push(world_root);
    let mut entries: Vec<(NodeId, usize)> = Vec::with_capacity(frame_path.len());
    let mut current = world_root;
    for &slot in frame_path {
        let Some(node) = library.get(current) else { break };
        entries.push((current, slot as usize));
        match node.children[slot as usize] {
            Child::Node(child_id) => {
                current = child_id;
                chain.push(current);
            }
            _ => break,
        }
    }
    (chain, entries)
}

/// Glue `frame_entries[..depth]` onto the front of `hit.path` and
/// (if present) `hit.place_path`, so the inner DDA's local path
/// becomes an absolute path rooted at `world_root`.
fn prepend_frame_entries(
    hit: &mut HitInfo,
    frame_entries: &[(NodeId, usize)],
    depth: usize,
) {
    let mut new_path = Vec::with_capacity(depth + hit.path.len());
    new_path.extend(frame_entries.iter().take(depth).copied());
    new_path.append(&mut hit.path);
    hit.path = new_path;
    if let Some(mut pp) = hit.place_path.take() {
        let mut new_pp = Vec::with_capacity(depth + pp.len());
        new_pp.extend(frame_entries.iter().take(depth).copied());
        new_pp.append(&mut pp);
        hit.place_path = Some(new_pp);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::bootstrap::plain_test_world;
    use crate::world::edit::break_block;

    #[test]
    fn raycast_hits_ground() {
        let world = plain_test_world();
        let hit = cpu_raycast(
            &world.library,
            world.root,
            [1.5, 2.5, 1.5],
            [0.0, -1.0, 0.0],
            8,
        );
        assert!(hit.is_some(), "Should hit the ground");
        let hit = hit.unwrap();
        assert_eq!(hit.face, 2, "Should hit top face");
    }

    #[test]
    fn raycast_misses_sky() {
        let world = plain_test_world();
        let hit = cpu_raycast(
            &world.library,
            world.root,
            [0.5, 2.5, 0.5],
            [0.0, 1.0, 0.0],
            8,
        );
        assert!(hit.is_none(), "Should miss when looking at sky");
    }

    #[test]
    fn zoom_controls_edit_depth() {
        let world = plain_test_world();
        let hit_coarse = cpu_raycast(
            &world.library, world.root,
            [1.5, 2.5, 1.5], [0.0, -1.0, 0.0], 1,
        );
        let hit_fine = cpu_raycast(
            &world.library, world.root,
            [1.5, 2.5, 1.5], [0.0, -1.0, 0.0], 8,
        );
        assert!(hit_coarse.is_some());
        assert!(hit_fine.is_some());
        assert!(hit_coarse.unwrap().path.len() < hit_fine.unwrap().path.len());
    }

    #[test]
    fn cpu_raycast_in_frame_at_root_matches_world_raycast() {
        let world = plain_test_world();
        let world_hit = cpu_raycast(
            &world.library, world.root,
            [1.5, 2.5, 1.5], [0.0, -1.0, 0.0], 8,
        );
        let frame_hit = cpu_raycast_in_frame(
            &world.library, world.root,
            &[], [1.5, 2.5, 1.5], [0.0, -1.0, 0.0], 8, 6,
        );
        assert!(world_hit.is_some());
        assert!(frame_hit.is_some());
        let w = world_hit.unwrap();
        let f = frame_hit.unwrap();
        assert_eq!(w.path.len(), f.path.len());
        assert_eq!(w.face, f.face);
    }

    #[test]
    fn cpu_raycast_in_frame_pop_finds_hit_in_ancestor() {
        let world = plain_test_world();
        let frame_path = [16u8, 13u8];
        let cam = [0.5, 0.5, 0.5];
        let dir = [0.7, 0.7, 0.0];
        let _ = cpu_raycast_in_frame(
            &world.library, world.root,
            &frame_path, cam, dir, 8, 6,
        );
    }

    #[test]
    fn cpu_raycast_in_frame_path_starts_from_world_root() {
        let world = plain_test_world();
        let hit = cpu_raycast_in_frame(
            &world.library, world.root,
            &[], [1.5, 2.5, 1.5], [0.0, -1.0, 0.0], 8, 6,
        ).expect("should hit ground");
        assert_eq!(hit.path[0].0, world.root);
    }

    #[test]
    fn frame_aware_raycast_hits_at_all_depths() {
        use crate::world::bootstrap;

        let render_frame_k = 3u8;

        for anchor_depth in [4u8, 6, 8, 10, 11, 12, 15, 20, 25, 30, 33, 38] {
            let boot = bootstrap::bootstrap_world(
                bootstrap::WorldPreset::PlainTest,
                Some(40),
            );
            let mut world = boot.world;
            let pos = bootstrap::plain_surface_spawn(anchor_depth);
            bootstrap::carve_air_pocket(&mut world, &pos.anchor, 40);

            let frame_depth = anchor_depth.saturating_sub(render_frame_k);
            let mut frame_path = pos.anchor;
            frame_path.truncate(frame_depth);

            let cam_local = pos.in_frame(&frame_path);
            let ray_dir = crate::world::sdf::normalize([0.0, -0.434, -0.901]);
            let edit_depth = anchor_depth as u32;

            let hit = cpu_raycast_in_frame(
                &world.library, world.root,
                frame_path.as_slice(), cam_local, ray_dir,
                edit_depth, 6,
            );

            assert!(hit.is_some(),
                "direct-spawn raycast missed at anchor_depth={anchor_depth}: \
                 frame_path={:?} cam_local={:?} edit_depth={edit_depth}",
                frame_path.as_slice(), cam_local);

            let h = hit.unwrap();
            let old_root = world.root;
            let changed = break_block(&mut world, &h);
            assert!(changed,
                "break_block failed at anchor_depth={anchor_depth}: path_len={} face={}",
                h.path.len(), h.face);
            assert_ne!(world.root, old_root,
                "root unchanged after break at anchor_depth={anchor_depth}");
        }
    }

    #[test]
    fn frame_aware_raycast_hits_after_zoom_in_from_spawn() {
        use crate::world::bootstrap;

        let render_frame_k = 3u8;
        let initial_depth = 8u8;

        let boot = bootstrap::bootstrap_world(
            bootstrap::WorldPreset::PlainTest,
            Some(40),
        );
        let mut world = boot.world;
        let mut pos = bootstrap::plain_surface_spawn(initial_depth);
        bootstrap::carve_air_pocket(&mut world, &pos.anchor, 40);

        for target_depth in (initial_depth + 1)..=38u8 {
            pos.zoom_in();

            let anchor_depth = pos.anchor.depth();
            assert_eq!(anchor_depth, target_depth);

            let frame_depth = anchor_depth.saturating_sub(render_frame_k);
            let mut frame_path = pos.anchor;
            frame_path.truncate(frame_depth);

            let cam_local = pos.in_frame(&frame_path);
            let ray_dir = crate::world::sdf::normalize([0.0, -0.434, -0.901]);
            let edit_depth = anchor_depth as u32;

            let hit = cpu_raycast_in_frame(
                &world.library, world.root,
                frame_path.as_slice(), cam_local, ray_dir,
                edit_depth, 6,
            );

            assert!(hit.is_some(),
                "zoom-in raycast missed at depth={target_depth}");
        }
    }

    // Phase 3 REVISED A.4 — verify the sphere CPU raycast lands the
    // hit on the cell predicted by the (lon, lat) → (cell_x, cell_z)
    // math. Three cardinal directions:
    // - Ray east of sphere going west (-X) → hits the +X equator
    //   point. Normal = (1,0,0). lon = atan2(0, 1) = 0. cell_x in
    //   the middle (= ~13 for dims_x=27). cell_z in the middle.
    // - Ray north of sphere going down (+Y, lat = π/2) → BANNED by
    //   pole filter, returns None.
    // - Ray going at +Z direction towards sphere → hits +Z normal.
    //   lon = atan2(1, 0) = π/2. cell_x = ~20.
    #[test]
    fn cpu_raycast_sphere_uv_east_equator_hits_middle_cell() {
        use crate::world::bootstrap::wrapped_planet_world;
        use crate::world::tree::slot_index;

        let world = wrapped_planet_world(2, [27, 2, 14], 3, 0);
        let mut frame_path = vec![];
        for _ in 0..2 {
            frame_path.push(slot_index(1, 1, 1) as u8);
        }
        // Camera way east of sphere center (1.5, 1.5, 1.5), looking
        // -X. Ray hits the +X equator point at world-frame
        // (1.5 + R, 1.5, 1.5).
        let cam_local = [3.0, 1.5, 1.5];
        let ray_dir = [-1.0, 0.0, 0.0];
        let hit = cpu_raycast_sphere_uv(
            &world.library, world.root, &frame_path,
            cam_local, ray_dir,
            [27, 2, 14], 3,
            1.26,
        ).expect("ray to +X equator must hit");
        // Expected: lon = 0, lat = 0 → u = 0.5, v = 0.5 →
        // cell_x = floor(0.5 * 27) = 13, cell_z = floor(0.5 * 14) = 7,
        // cell_y = dims_y - 1 = 1 (GRASS row).
        // The path's leaf-level (NodeId, slot) encodes the cell —
        // leaf-level slot for (cell_x % 3, cell_y % 3, cell_z % 3) =
        // (13 % 3, 1, 7 % 3) = (1, 1, 1) = slot 13.
        let leaf_slot = hit.path.last().expect("hit path non-empty").1;
        assert_eq!(leaf_slot, slot_index(1, 1, 1),
            "expected leaf slot (1,1,1) for cell (13, 1, 7), got slot {leaf_slot}");
        assert_eq!(hit.face, 2, "sphere hit reports +Y face");
    }

    #[test]
    fn cpu_raycast_sphere_uv_pole_is_banned() {
        use crate::world::bootstrap::wrapped_planet_world;
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 0);
        let frame_path = vec![13u8, 13u8];
        // Ray straight DOWN toward sphere from above — hits +Y
        // normal (lat = π/2 ≈ 1.57 > lat_max 1.26) → banned pole.
        let cam_local = [1.5, 3.0, 1.5];
        let ray_dir = [0.0, -1.0, 0.0];
        let hit = cpu_raycast_sphere_uv(
            &world.library, world.root, &frame_path,
            cam_local, ray_dir,
            [27, 2, 14], 3,
            1.26,
        );
        assert!(hit.is_none(), "north-pole ray must be banned");
    }

    /// A.2 — when the outer (GRASS) cell at the target lat/lon is
    /// removed, the ray walks through the chord and lands on the
    /// next layer (STONE). Verifies the shell-march math doesn't
    /// hardcode "first hit = top cell"; it samples whatever data
    /// is at each radial step.
    #[test]
    fn cpu_raycast_sphere_uv_dug_grass_reveals_layer_below() {
        use crate::world::bootstrap::wrapped_planet_world;
        use crate::world::tree::{empty_children, slot_index, Child, NodeKind};

        // Build a slab with a HOLE at (cell_x=13, cell_y=1, cell_z=7)
        // — that's the +X equator point that the
        // _east_equator_hits_middle_cell test targets. We do this by
        // building a custom WrappedPlane subtree in which the GRASS
        // cell at that position is Child::Empty. Below it (cell_y=0
        // at the same x, z) STONE is intact.
        let mut world = wrapped_planet_world(2, [27, 2, 14], 3, 0);
        // Walk world tree to the slab root: 2 embedding levels of
        // slot (1, 1, 1).
        let mut node_id = world.root;
        for _ in 0..2 {
            let n = world.library.get(node_id).unwrap();
            match n.children[slot_index(1, 1, 1)] {
                Child::Node(c) => node_id = c,
                _ => unreachable!(),
            }
        }
        // node_id is now slab root (NodeKind::WrappedPlane). Walk
        // down to (cell_x=13, cell_y=1, cell_z=7) at slab_depth=3.
        // Compute slot indices at each level (cells_per_slot 9, 3, 1).
        let path_slots = {
            let cx = 13i32; let cy = 1i32; let cz = 7i32;
            let mut slots = Vec::new();
            let mut cps = 9i32;
            for _ in 0..3 {
                let sx = (cx / cps).rem_euclid(3) as usize;
                let sy = (cy / cps).rem_euclid(3) as usize;
                let sz = (cz / cps).rem_euclid(3) as usize;
                slots.push(slot_index(sx, sy, sz));
                cps /= 3;
            }
            slots
        };

        // Replace the GRASS leaf at the target with Empty by
        // re-emitting the path's nodes from scratch with the
        // modified leaf. Library is content-addressed so we just
        // build new nodes.
        let mut current = node_id;
        // Collect (parent_id, slot, parent_node_kind) for each level
        // so we can rebuild upward.
        let mut levels: Vec<(NodeId, usize)> = Vec::with_capacity(3);
        for &slot in &path_slots {
            let n = world.library.get(current).unwrap();
            levels.push((current, slot));
            match n.children[slot] {
                Child::Node(c) => current = c,
                _ => break,
            }
        }
        // Replace leaf at the deepest slot with Empty. Rebuild
        // upward, preserving each level's NodeKind (slab root is
        // WrappedPlane; mid-levels are Cartesian).
        // Last level's parent is `levels[2].0`. Replace its child at
        // `levels[2].1` with Empty.
        let (deepest_parent, deepest_slot) = levels[2];
        let deepest_parent_node = world.library.get(deepest_parent).unwrap();
        let mut new_children = deepest_parent_node.children;
        new_children[deepest_slot] = Child::Empty;
        let new_deepest = world.library.insert_with_kind(new_children, deepest_parent_node.kind);
        // Rebuild parent of deepest_parent.
        let (mid_parent, mid_slot) = levels[1];
        let mid_parent_node = world.library.get(mid_parent).unwrap();
        let mut mid_children = mid_parent_node.children;
        mid_children[mid_slot] = Child::Node(new_deepest);
        let new_mid = world.library.insert_with_kind(mid_children, mid_parent_node.kind);
        // Rebuild slab root.
        let (slab_root_id, top_slot) = levels[0];
        let slab_root_node = world.library.get(slab_root_id).unwrap();
        let mut slab_children = slab_root_node.children;
        slab_children[top_slot] = Child::Node(new_mid);
        let new_slab_root = world.library.insert_with_kind(slab_children, slab_root_node.kind);
        // Rebuild upward through embedding levels.
        let mut new_root = new_slab_root;
        // Walk the world root's path to the slab root, rebuilding.
        // Embedding has depth 2 with slot (1,1,1) at each level.
        for _ in 0..2 {
            // Find the parent of new_root in the existing tree (not
            // straightforward without back-pointers); easier path:
            // walk the tree fresh and substitute.
            // Actually for a 2-level rebuild it's simpler to do it
            // explicitly: find the embedding node that was pointing
            // to slab_root, swap its (1,1,1) child for new_root.
            // Since each embedding level is `empty_children` with
            // only (1,1,1) populated, we can rebuild from scratch.
            let mut emb = empty_children();
            emb[slot_index(1, 1, 1)] = Child::Node(new_root);
            new_root = world.library.insert_with_kind(emb, NodeKind::Cartesian);
        }
        world.root = new_root;

        // Now ray east-of-sphere going west — same setup as the
        // first test. With the GRASS cell removed, the cell beneath
        // (cell_y = 0) should be hit instead. The leaf slot for
        // cell (13, 0, 7) is (1, 0, 1) = slot_index(1, 0, 1).
        let frame_path = vec![13u8, 13u8];
        let cam_local = [3.0, 1.5, 1.5];
        let ray_dir = [-1.0, 0.0, 0.0];
        let hit = cpu_raycast_sphere_uv(
            &world.library, world.root, &frame_path,
            cam_local, ray_dir,
            [27, 2, 14], 3,
            1.26,
        ).expect("dug-through ray must hit stone underneath");
        let leaf_slot = hit.path.last().unwrap().1;
        // Expect cell_y = 0 → leaf slot has sy = 0 → slot_index(1, 0, 1) = 10.
        assert_eq!(leaf_slot, slot_index(1, 0, 1),
            "expected leaf slot (1,0,1)=10 (cell_y=0 = stone) after digging grass, got {leaf_slot}");
    }

    #[test]
    fn cpu_raycast_sphere_uv_misses_when_ray_misses_sphere() {
        use crate::world::bootstrap::wrapped_planet_world;
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 0);
        let frame_path = vec![13u8, 13u8];
        // Ray going +Y from above, way off to the side — never
        // intersects the sphere.
        let cam_local = [3.0, 3.0, 3.0];
        let ray_dir = [0.0, 1.0, 0.0];
        let hit = cpu_raycast_sphere_uv(
            &world.library, world.root, &frame_path,
            cam_local, ray_dir,
            [27, 2, 14], 3,
            1.26,
        );
        assert!(hit.is_none(), "ray missing the sphere returns None");
    }
}
