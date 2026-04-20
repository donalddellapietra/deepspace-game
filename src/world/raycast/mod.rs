//! CPU raycasting — Cartesian DDA, cubed-sphere shell march, frame-
//! aware ribbon pops, and point solidity queries.
//!
//! The CPU ray march mirrors the GPU shader's tree traversal so that
//! the cell the crosshair targets is the same cell the shader is
//! shading. Edits operate at a layer-dependent depth: the zoom level
//! controls how deep the raycast descends, so the same code breaks a
//! single block at fine zoom or an entire 3×3×3 node at coarse zoom.

mod cartesian;
mod sphere;

pub use sphere::{FaceWindow, LodParams};

use crate::world::tree::{slot_coords, slot_index, Child, NodeId, NodeKind, NodeLibrary};

pub(super) const MAX_FACE_DEPTH: u32 = crate::world::tree::MAX_DEPTH as u32;

/// Face-space bounds for a sphere cell hit. Used by the highlight
/// AABB and any caller that needs the exact cell the shader
/// terminated at, without reconstructing it from `hit.path`.
///
/// Bounds are in normalized `[0, 1]` face coords; the 8 cell corners
/// map to body-local points via `cubesphere::face_space_to_body_point`.
#[derive(Debug, Clone, Copy)]
pub struct SphereHitCell {
    pub face: u32,
    pub u_lo: f32,
    pub v_lo: f32,
    pub r_lo: f32,
    pub size: f32,
    pub inner_r: f32,
    pub outer_r: f32,
    /// Length of the hit path prefix that lands at the containing
    /// body cell. `hit.path[..body_path_len]` is the world-to-body
    /// chain (Cartesian nodes); `hit.path[body_path_len]` is
    /// `(body_node_id, face_slot)`; the rest is the face-subtree
    /// descent. The AABB uses this to find the body's world-space
    /// origin + size for the body-to-world transform.
    pub body_path_len: usize,
}

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
    /// Optional explicit path where a place_block should land. For
    /// Cartesian hits this is `None` — `place_child` derives the
    /// adjacent cell via `face`. Reserved as a seam for coordinate
    /// systems where face semantics don't map onto an xyz axis.
    pub place_path: Option<Vec<(NodeId, usize)>>,
    /// When the hit lives inside a cubed-sphere body, this carries
    /// the exact terminal cell in face-space — the same cell the
    /// GPU shader rendered (same LOD, same algorithm). Used by the
    /// highlight AABB so the box aligns with the visible cell.
    /// `None` for pure Cartesian hits.
    pub sphere_cell: Option<SphereHitCell>,
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
    cartesian::cpu_raycast_with_face_depth(
        library, root, ray_origin, ray_dir, max_depth,
        LodParams::fixed_max(),
    )
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
    lod: LodParams,
) -> Option<HitInfo> {
    let (chain, frame_entries) = build_frame_chain(library, world_root, frame_path);
    let effective_depth = chain.len() - 1;
    let frame_entries = &frame_entries[..effective_depth];

    let mut current_frame_depth = effective_depth;
    let mut ray_origin = cam_local;
    let mut ray_dir = ray_dir;
    let total_max_depth = max_depth;
    let mut cur_scale: f32 = 1.0;

    loop {
        let frame_root_id = chain[current_frame_depth];
        let inner_max = total_max_depth.saturating_sub(current_frame_depth as u32);

        // Dispatch on the frame root's NodeKind. A CubedSphereBody
        // frame-root hands off to the sphere DDA directly — the
        // body fills the frame's [0, 3)³ bubble.
        //
        // `lod` is scaled by `cur_scale` on ribbon pops: each pop
        // rescales the ray by 1/3, which also scales the effective
        // body-frame distance that feeds `face_lod_depth`. Without
        // scaling, the LOD would shift after a pop, causing the
        // CPU's hit cell to drift from the shader's.
        let scaled_lod = LodParams {
            pixel_density: lod.pixel_density * cur_scale,
            lod_threshold: lod.lod_threshold,
        };
        let frame_kind = library.get(frame_root_id).map(|n| n.kind);
        let hit_opt = if let Some(NodeKind::CubedSphereBody { inner_r, outer_r }) = frame_kind {
            sphere::cs_raycast(
                library, frame_root_id, [0.0; 3], 3.0,
                inner_r, outer_r,
                ray_origin, ray_dir,
                &[], scaled_lod,
                None,
            )
        } else {
            cartesian::cpu_raycast_with_face_depth(
                library, frame_root_id, ray_origin, ray_dir,
                inner_max, scaled_lod,
            )
        };

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
        cur_scale *= 1.0 / 3.0;
        current_frame_depth -= 1;
    }
}

/// Frame-aware raycast for a sphere sub-frame (render root lives
/// inside a face subtree). The linear render frame stays rooted at
/// the containing body cell; the face window tells the sphere DDA
/// which absolute UVR region the render frame covers.
///
/// Camera coords are expressed in the body cell's frame (`[0, 3)³`).
/// `ray_dir` is the world-axis direction in body-local orientation
/// (shared between Cartesian and sphere because a body cell is
/// axis-aligned with its parent's frame).
pub fn cpu_raycast_in_sphere_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    body_path: &[u8],
    cam_body: [f32; 3],
    ray_dir: [f32; 3],
    lod: LodParams,
    window: FaceWindow,
    inner_r: f32,
    outer_r: f32,
) -> Option<HitInfo> {
    let (chain, frame_entries) = build_frame_chain(library, world_root, body_path);
    let effective_depth = chain.len() - 1;
    let body_node_id = chain[effective_depth];
    let hit = sphere::cs_raycast(
        library, body_node_id, [0.0; 3], 3.0,
        inner_r, outer_r,
        cam_body, ray_dir,
        &[], lod,
        Some(window),
    );
    if let Some(mut hit) = hit {
        prepend_frame_entries(&mut hit, &frame_entries[..effective_depth], effective_depth);
        Some(hit)
    } else {
        None
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
            &[], [1.5, 2.5, 1.5], [0.0, -1.0, 0.0], 8, super::LodParams::fixed_max(),
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
            &frame_path, cam, dir, 8, super::LodParams::fixed_max(),
        );
    }

    #[test]
    fn cpu_raycast_in_frame_path_starts_from_world_root() {
        let world = plain_test_world();
        let hit = cpu_raycast_in_frame(
            &world.library, world.root,
            &[], [1.5, 2.5, 1.5], [0.0, -1.0, 0.0], 8, super::LodParams::fixed_max(),
        ).expect("should hit ground");
        assert_eq!(hit.path[0].0, world.root);
    }

    #[test]
    fn planet_world_raycast_hits_sphere() {
        use crate::world::cubesphere::{demo_planet, install_at_root_center};
        use crate::world::worldgen::generate_world;

        let mut world = generate_world();
        let setup = demo_planet();
        let (new_root, _planet_path) =
            install_at_root_center(&mut world.library, world.root, &setup);
        world.swap_root(new_root);

        // Camera just above the planet looking straight down. The
        // sphere is at [1.5, 1.5, 1.5] with outer_r=0.45.
        let hit = cpu_raycast_in_frame(
            &world.library, world.root,
            &[], [1.5, 2.0, 1.5], [0.0, -1.0, 0.0],
            30, super::LodParams::fixed_max(),
        );
        assert!(hit.is_some(), "ray should hit the planet");
        let h = hit.unwrap();
        assert!(h.path.len() >= 2, "hit path should descend into body");
    }

    #[test]
    fn planet_world_cartesian_descend_triggers_sphere_dispatch() {
        use crate::world::cubesphere::{demo_planet, install_at_root_center};
        use crate::world::worldgen::generate_world;

        let mut world = generate_world();
        let setup = demo_planet();
        let (new_root, _planet_path) =
            install_at_root_center(&mut world.library, world.root, &setup);
        world.swap_root(new_root);

        // Start from outside the body cell so the Cartesian DDA must
        // descend into slot 13 and dispatch sphere. Camera [1.5, 2.5,
        // 1.5] is in slot y=2 (above the body slot).
        let hit = cpu_raycast_in_frame(
            &world.library, world.root,
            &[], [1.5, 2.5, 1.5], [0.0, -1.0, 0.0],
            30, super::LodParams::fixed_max(),
        );
        assert!(hit.is_some(), "Cartesian DDA should cross into body cell and dispatch sphere");
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
                edit_depth, super::LodParams::fixed_max(),
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
                edit_depth, super::LodParams::fixed_max(),
            );

            assert!(hit.is_some(),
                "zoom-in raycast missed at depth={target_depth}");
        }
    }
}
