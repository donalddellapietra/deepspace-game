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
    let lat = n[1].clamp(-1.0, 1.0).asin();
    if lat.abs() > lat_max {
        return None; // banned pole — pixel reads as sky
    }
    let lon = n[2].atan2(n[0]);

    // (lon, lat) → slab cell coords. Floor + clamp.
    let pi = std::f32::consts::PI;
    let u = (lon + pi) / (2.0 * pi); // [0, 1)
    let v = (lat + lat_max) / (2.0 * lat_max); // [0, 1]
    let cell_x = ((u * dims[0] as f32).floor() as i32).clamp(0, dims[0] as i32 - 1);
    let cell_z = ((v * dims[2] as f32).floor() as i32).clamp(0, dims[2] as i32 - 1);
    let cell_y = dims[1] as i32 - 1; // GRASS row (top of populated band)

    // Walk frame_path from world_root to the slab root, accumulating
    // the (NodeId, slot) pairs the HitInfo expects.
    let mut path: Vec<(NodeId, usize)> = Vec::with_capacity(frame_path.len() + slab_depth as usize);
    let mut cur = world_root;
    for &slot in frame_path.iter() {
        let node = library.get(cur)?;
        path.push((cur, slot as usize));
        match node.children[slot as usize] {
            Child::Node(child) => cur = child,
            _ => return None, // frame_path doesn't lead to a node
        }
    }
    let slab_root = cur;

    // Walk the slab tree from slab_root down `slab_depth` levels to
    // reach (cell_x, cell_y, cell_z).
    let mut idx = slab_root;
    let mut cells_per_slot: i32 = 1;
    for _ in 1..slab_depth {
        cells_per_slot *= 3;
    }
    for level in 0..slab_depth {
        let sx = (cell_x / cells_per_slot).rem_euclid(3);
        let sy = (cell_y / cells_per_slot).rem_euclid(3);
        let sz = (cell_z / cells_per_slot).rem_euclid(3);
        let slot = slot_index(sx as usize, sy as usize, sz as usize);
        path.push((idx, slot));
        let node = library.get(idx)?;
        match node.children[slot] {
            Child::Block(_) => break,
            Child::Node(child) => {
                if level + 1 < slab_depth {
                    idx = child;
                }
            }
            _ => return None, // empty cell — caller treats as miss
        }
        cells_per_slot /= 3;
    }

    // Hit returned: face = +Y (top face — the GRASS surface is up).
    // place_path: insert above the GRASS row (at cell_y + 1) — since
    // dims.y = 2 and we're already at the top, "above" means we'd
    // need to extend the slab. Leave None for now; A.4+ can wire
    // proper placement once the architecture for "above-slab cells"
    // is decided.
    Some(HitInfo {
        path,
        face: 2,
        t: t_enter * inv_norm,
        place_path: None,
    })
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
