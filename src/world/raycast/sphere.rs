//! CPU UV-sphere raycast for `WrappedPlane` frames.
//!
//! Mirrors `assets/shaders/sphere_dda.wgsl::sphere_uv_in_cell`:
//! ray-intersect the implied sphere of radius `body_size / (2π)`,
//! sample slab cells via `(lon, lat)` at each radial layer, and
//! (when the user's edit depth allows) descend further into a non-
//! uniform anchor's Cartesian subtree using fractional `(lon, lat,
//! r)` slot picks.
//!
//! Sub-frame architecture note: this CPU path stays spherical at
//! all depths (= matches the GPU's `sphere_uv_in_cell` /
//! `sphere_descend_anchor` partition). The sphere sub-frame
//! architecture (`sphere_geom::subframe_range` +
//! `sphere_uv_in_subframe.wgsl`) is a render-only optimization; the
//! tree partition it operates on is identical to the one this
//! function picks.

use crate::world::tree::{slot_index, Child, NodeId, NodeLibrary, REPRESENTATIVE_EMPTY};

use super::HitInfo;

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
    max_depth: u32,
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
        // Track the deepest Node (= anchor block) reached so we can
        // continue descending into its subtree if the user's edit
        // depth is deeper than slab_depth.
        let mut sub_idx: Option<NodeId> = None;
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
                    } else {
                        // Last slab level — child is an anchor block.
                        sub_idx = Some(child);
                    }
                }
            }
            cells_per_slot /= 3;
        }
        if cell_is_empty {
            continue;
        }

        // Sub-slab descent: when the slab cell is an anchor Node and
        // the user's edit depth (max_depth) goes beyond the slab,
        // continue picking sub-cells using the ray's fractional
        // (lon, lat, r) coords WITHIN the slab cell. Each level
        // multiplies the resolution by 3, so the break path lands at
        // the right sub-cell for the camera's anchor depth — same
        // behavior as Cartesian where deeper anchors break smaller
        // cells, and as 2-2-3-2's `walk_face_subtree(max_depth)`.
        if let Some(mut node_idx) = sub_idx {
            let absolute_slab_depth = frame_path.len() as u32 + slab_depth as u32;
            let extra_levels = max_depth.saturating_sub(absolute_slab_depth);
            if extra_levels > 0 {
                // Fractional coords within the hit slab cell. The
                // sample was taken at the layer's TOP boundary (r at
                // the cell's outer face), so r_in_cell ≈ 1.0 — bias
                // it slightly inside so floor-pick lands on the
                // outermost r-slot (sy = 2), which is the cell at
                // the surface the user sees.
                let lon_fine = u_l * dims[0] as f32;
                let lat_fine = v_l * dims[2] as f32;
                let mut frac_x = (lon_fine - cell_x as f32).clamp(0.0, 0.99999);
                let mut frac_z = (lat_fine - cell_z as f32).clamp(0.0, 0.99999);
                let mut frac_y = 0.99999_f32;

                // Per-block hybrid: every slab cell is visually
                // rendered as a tangent-plane CUBE on the sphere
                // (matching the GPU's render_cell_as_tangent_cube /
                // cartesian_voxels_in_cell dispatch). Sub-cell
                // descent must use the CUBE's local hit position
                // (= ray-OBB intersect in cube basis), not the
                // curved sphere's (lon, lat, r) — those differ by
                // O(curvature) for off-center pixels and make
                // break/place land at a DIFFERENT sub-voxel than the
                // cube visually shows.
                if let Some((cx_3, cy_3, cz_3)) = cube_local_hit(
                    cam_local, dir,
                    cs_center, lat_max,
                    cell_x, cy, cell_z,
                    dims, shell_thickness, r_inner,
                ) {
                    // cx_3 / cy_3 / cz_3 are in cube-local
                    // [0, 3)³. Convert to [0, 1) fractions for
                    // the descent loop below.
                    frac_x = (cx_3 / 3.0).clamp(0.0, 0.99999);
                    frac_y = (cy_3 / 3.0).clamp(0.0, 0.99999);
                    frac_z = (cz_3 / 3.0).clamp(0.0, 0.99999);
                }
                for _ in 0..extra_levels {
                    let sx = (frac_x * 3.0).floor() as usize;
                    let sy = (frac_y * 3.0).floor() as usize;
                    let sz = (frac_z * 3.0).floor() as usize;
                    let slot = slot_index(sx, sy, sz);
                    path.push((node_idx, slot));
                    let node = match library.get(node_idx) {
                        Some(n) => n,
                        None => break,
                    };
                    match node.children[slot] {
                        Child::Node(c) => node_idx = c,
                        // Sub-cell terminates: leaf Block, Empty (a
                        // hole someone dug earlier), or EntityRef.
                        // Path now points at this sub-cell — that's
                        // what the caller should break/place.
                        _ => break,
                    }
                    frac_x = (frac_x * 3.0) - sx as f32;
                    frac_y = (frac_y * 3.0) - sy as f32;
                    frac_z = (frac_z * 3.0) - sz as f32;
                }
            }
        }

        // Empty-rep skip — mirrors `cpu_raycast_uv_body::descend`'s
        // `representative_block == REPRESENTATIVE_EMPTY` check (and
        // Cartesian's `march_cartesian` empty-rep fast path). Without
        // this, the path can land on a uniform-empty Node (e.g. an
        // air pocket the user dug earlier, or an unmaterialized empty
        // sub-region) — `break_block` then "modifies" already-empty
        // content, the pack output is unchanged, the render shows no
        // change, and the user sees the break log fire with no visual
        // effect. Skip and continue to the next radial layer so the
        // ray finds genuine solid material to break.
        let &(parent_id, last_slot) = path.last().unwrap();
        let final_is_empty = match library
            .get(parent_id)
            .map(|n| n.children[last_slot])
        {
            Some(Child::Empty) | Some(Child::EntityRef(_)) | None => true,
            Some(Child::Block(_)) => false,
            Some(Child::Node(c)) => library
                .get(c)
                .map_or(true, |n| n.representative_block == REPRESENTATIVE_EMPTY),
        };
        if final_is_empty {
            continue;
        }

        return Some(HitInfo {
            path,
            face: 2,
            t: t_layer * inv_norm,
            place_path: None,
        });
    }

    None
}

/// Hybrid prototype helper: ray-OBB intersect against the slab
/// cell's tangent-plane CUBE (same basis as the GPU's
/// `render_cell_as_tangent_cube` / `cartesian_voxels_in_cell`).
/// Returns the hit position in cube-local [0, 3)³ coords on hit,
/// or `None` if the ray misses the cube.
///
/// CPU mirror of the GPU cube basis:
///   cube.x = lon-tangent
///   cube.y = +radial out  (cube.y=0 inner, cube.y=3 outer)
///   cube.z = lat-tangent
///
/// For ray hits, the (cx_3, cy_3, cz_3) returned is the entry
/// position in cube-local coords. Caller divides by 3 to get
/// frac coords for the sub-slab descent loop.
fn cube_local_hit(
    cam_local: [f32; 3],
    dir_unit: [f32; 3],
    cs_center: [f32; 3],
    lat_max: f32,
    cell_x: i32, cell_y: i32, cell_z: i32,
    dims: [u32; 3],
    shell_thickness: f32,
    r_inner: f32,
) -> Option<(f32, f32, f32)> {
    use std::f32::consts::PI;
    // Cell's (lat, lon, r) range from grid coords (matches the
    // GPU's body-rooted DDA mapping).
    let lon_step = 2.0 * PI / dims[0] as f32;
    let lat_step = 2.0 * lat_max / dims[2] as f32;
    let r_step = shell_thickness / dims[1] as f32;
    let lon_lo = -PI + cell_x as f32 * lon_step;
    let lon_hi = lon_lo + lon_step;
    let lat_lo = -lat_max + cell_z as f32 * lat_step;
    let lat_hi = lat_lo + lat_step;
    let r_lo = r_inner + cell_y as f32 * r_step;
    let r_hi = r_lo + r_step;

    let lat_c = (lat_lo + lat_hi) * 0.5;
    let lon_c = (lon_lo + lon_hi) * 0.5;
    let cl = lat_c.cos();
    let sl = lat_c.sin();
    let co = lon_c.cos();
    let so = lon_c.sin();
    // Cube basis (matches GPU shader exactly).
    let radial = [cl * co, sl, cl * so];
    let u_axis = [-so, 0.0, co];
    let v_axis = radial;
    let w_axis = [-sl * co, cl, -sl * so];

    let r_mid = (r_lo + r_hi) * 0.5;
    let center = [
        cs_center[0] + r_mid * radial[0],
        cs_center[1] + r_mid * radial[1],
        cs_center[2] + r_mid * radial[2],
    ];
    let half_u = (lon_hi - lon_lo) * r_hi * cl * 0.5;
    let half_v = (r_hi - r_lo) * 0.5;
    let half_w = (lat_hi - lat_lo) * r_hi * 0.5;

    // Transform ray into cube-local.
    let dot3 = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let to_origin = [
        cam_local[0] - center[0],
        cam_local[1] - center[1],
        cam_local[2] - center[2],
    ];
    let pos_l = [
        dot3(to_origin, u_axis),
        dot3(to_origin, v_axis),
        dot3(to_origin, w_axis),
    ];
    let dir_l = [
        dot3(dir_unit, u_axis),
        dot3(dir_unit, v_axis),
        dot3(dir_unit, w_axis),
    ];

    // Ray-AABB on [-half, +half] in cube-local.
    let half = [half_u, half_v, half_w];
    let inv_dir = [
        if dir_l[0].abs() > 1e-8 { 1.0 / dir_l[0] } else { 1e10 },
        if dir_l[1].abs() > 1e-8 { 1.0 / dir_l[1] } else { 1e10 },
        if dir_l[2].abs() > 1e-8 { 1.0 / dir_l[2] } else { 1e10 },
    ];
    let mut t_min = f32::NEG_INFINITY;
    let mut t_max = f32::INFINITY;
    for axis in 0..3 {
        let t1 = (-half[axis] - pos_l[axis]) * inv_dir[axis];
        let t2 = (half[axis] - pos_l[axis]) * inv_dir[axis];
        let (lo, hi) = if t1 < t2 { (t1, t2) } else { (t2, t1) };
        t_min = t_min.max(lo);
        t_max = t_max.min(hi);
    }
    if t_min > t_max || t_max <= 0.0 {
        return None;
    }
    let t_hit = t_min.max(0.0);

    // Hit position in cube-local, mapped to [0, 3)³.
    let hit_l = [
        pos_l[0] + dir_l[0] * t_hit,
        pos_l[1] + dir_l[1] * t_hit,
        pos_l[2] + dir_l[2] * t_hit,
    ];
    let cx_3 = (hit_l[0] + half_u) * 3.0 / (2.0 * half_u);
    let cy_3 = (hit_l[1] + half_v) * 3.0 / (2.0 * half_v);
    let cz_3 = (hit_l[2] + half_w) * 3.0 / (2.0 * half_w);
    Some((
        cx_3.clamp(0.0, 2.99999),
        cy_3.clamp(0.0, 2.99999),
        cz_3.clamp(0.0, 2.99999),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

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
            5,
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
            5,
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
            5,
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
            5,
        );
        assert!(hit.is_none(), "ray missing the sphere returns None");
    }

    #[test]
    fn cpu_raycast_sphere_uv_descends_into_anchor_subtree_with_extra_depth() {
        use crate::world::bootstrap::wrapped_planet_world;
        // cell_subtree_depth=3 → each anchor block has 3 more levels
        // of uniform Cartesian subtree. Same +X equator setup as
        // east_equator_hits_middle_cell.
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 3);
        let frame_path = vec![13u8, 13u8];
        let cam_local = [3.0, 1.5, 1.5];
        let ray_dir = [-1.0, 0.0, 0.0];
        // Slab natural depth = 2 (frame) + 3 (slab) = 5. Extra=3 →
        // path should be 8 entries long, descending 3 sub-cell levels
        // into the anchor block's uniform subtree.
        let hit_shallow = cpu_raycast_sphere_uv(
            &world.library, world.root, &frame_path,
            cam_local, ray_dir,
            [27, 2, 14], 3,
            1.26,
            5,
        ).expect("shallow hit");
        let hit_deep = cpu_raycast_sphere_uv(
            &world.library, world.root, &frame_path,
            cam_local, ray_dir,
            [27, 2, 14], 3,
            1.26,
            8,
        ).expect("deep hit");
        assert_eq!(hit_shallow.path.len(), 5,
            "shallow path = frame(2) + slab(3) = 5, got {}", hit_shallow.path.len());
        assert_eq!(hit_deep.path.len(), 8,
            "deep path = frame(2) + slab(3) + extra(3) = 8, got {}", hit_deep.path.len());
        // Deep path's first 5 entries should match the shallow path's
        // (same slab cell hit, just descending further into its subtree).
        for i in 0..5 {
            assert_eq!(
                hit_shallow.path[i].1, hit_deep.path[i].1,
                "slab path slot mismatch at level {i}",
            );
        }
    }

    /// DIAGNOSIS — sphere-mercator-1-1: documents the data-layer
    /// asymmetry that makes deep-zoom breaks invisible on the sphere.
    ///
    /// What the user sees: at default zoom a break makes a hole; at
    /// "more than 1 layer below the top" (i.e. anchor depth > slab
    /// natural depth) the break records in the tree but renders as
    /// solid grass.
    ///
    /// What this test checks: walk the tree to the *slab cell* — the
    /// granularity at which the GPU shader's `sample_slab_cell` reads.
    /// At max_depth=slab_natural the slab cell becomes Empty/None.
    /// At max_depth>slab_natural the slab cell stays as a non-uniform
    /// Node whose `representative_block` is unchanged (still the
    /// original block type, because 1/27 cells empty leaves the
    /// majority unchanged), so `sample_slab_cell` returns the SAME
    /// block_type it returned before the edit — render is identical.
    ///
    /// Cartesian doesn't have this problem: `march_cartesian` walks
    /// `tag=2` Nodes recursively (gated by `LOD_PIXEL_THRESHOLD`), so
    /// sub-cell holes inside a non-uniform anchor are visible at any
    /// zoom where the sub-cell is bigger than a pixel.
    #[test]
    fn diagnosis_sphere_deep_break_invisible_at_slab_granularity() {
        use crate::world::bootstrap::wrapped_planet_world;
        use crate::world::edit::break_block;
        use crate::world::tree::{slot_index, Child, REPRESENTATIVE_EMPTY, UNIFORM_MIXED};

        // cell_subtree_depth=3 so anchor cells have a real 3-level
        // sub-tree the deep-break can carve into. Slab natural depth =
        // embedding(2) + slab(3) = 5.
        let case = |max_depth: u32| {
            let mut world = wrapped_planet_world(2, [27, 2, 14], 3, 3);
            let frame_path = vec![13u8, 13u8];
            let cam_local = [3.0, 1.5, 1.5];
            let ray_dir = [-1.0, 0.0, 0.0];
            let hit = cpu_raycast_sphere_uv(
                &world.library, world.root, &frame_path,
                cam_local, ray_dir,
                [27, 2, 14], 3,
                1.26,
                max_depth,
            ).expect("ray hits +X equator");
            let path_len = hit.path.len();
            let slab_path: Vec<usize> =
                hit.path.iter().take(5).map(|&(_, s)| s).collect();
            assert!(break_block(&mut world, &hit), "break should succeed");

            // Walk to the slab cell at its natural depth (5 levels)
            // and inspect what the GPU shader would read there.
            // sample_slab_cell on the GPU returns block_type from the
            // packed tag at slab_depth - 1; that tag is `tag=1` (Block,
            // for uniform anchors after pack-time uniform-flatten), or
            // `tag=2` with `representative_block` (for non-uniform).
            let mut node = world.root;
            for &slot in &slab_path[..4] {
                match world.library.get(node).unwrap().children[slot] {
                    Child::Node(c) => node = c,
                    other => panic!("expected Node mid-slab, got {other:?}"),
                }
            }
            let last_child =
                world.library.get(node).unwrap().children[slab_path[4]];
            let observed = match last_child {
                Child::Empty => "EMPTY".to_string(),
                Child::Block(bt) => format!("Block(bt={bt})"),
                Child::EntityRef(_) => "EntityRef".to_string(),
                Child::Node(nid) => {
                    let n = world.library.get(nid).unwrap();
                    if n.uniform_type != UNIFORM_MIXED {
                        format!("Uniform-Node(uniform_type={})", n.uniform_type)
                    } else if n.representative_block == REPRESENTATIVE_EMPTY {
                        "Mixed-Node-but-rep-empty".to_string()
                    } else {
                        format!("Mixed-Node(rep_block={})", n.representative_block)
                    }
                }
            };
            (path_len, observed)
        };

        let (len5, slab5) = case(5);
        let (len6, slab6) = case(6);
        let (len7, slab7) = case(7);
        let (len8, slab8) = case(8);

        // Edit-depth=5 reaches slab natural granularity. The slab cell
        // itself is replaced with Empty: the GPU shader's
        // sample_slab_cell would return REPRESENTATIVE_EMPTY (0xFFFE),
        // and the cell renders as a hole. CORRECT.
        assert_eq!(len5, 5, "edit at slab natural takes 5-step path");
        assert_eq!(slab5, "EMPTY",
            "edit-depth=5 (slab natural) makes the slab cell Empty -> shader sees REPRESENTATIVE_EMPTY -> visible hole");

        // Edit-depth=6 reaches 1 sub-cell below the slab cell. The
        // slab cell becomes a non-uniform Node, but its
        // representative_block is still GRASS (1 of 27 sub-cells empty
        // leaves majority untouched). The GPU shader's
        // sample_slab_cell returns this representative -> renders SAME
        // as before edit. INVISIBLE. **This is the bug.**
        assert_eq!(len6, 6, "edit-depth=6 takes 6-step path (1 sub-cell level)");
        assert!(slab6.starts_with("Mixed-Node"),
            "edit-depth=6 leaves slab cell as Mixed-Node (was {slab6})");
        assert!(slab6.contains("rep_block=2") || slab6.contains("rep_block=1"),
            "representative is still grass/dirt/stone (not REPRESENTATIVE_EMPTY) — got {slab6}");

        // Same pattern at edit-depth=7 and 8: each break is 1/27^k of
        // the slab cell, representative stays the dominant terrain.
        assert_eq!(len7, 7);
        assert!(slab7.starts_with("Mixed-Node"));
        assert_eq!(len8, 8);
        assert!(slab8.starts_with("Mixed-Node"));

        eprintln!(
            "DIAGNOSIS — slab cell observation at depth 5 (what shader's sample_slab_cell sees):\n  edit_depth=5 -> {slab5}\n  edit_depth=6 -> {slab6}\n  edit_depth=7 -> {slab7}\n  edit_depth=8 -> {slab8}",
        );

        // Shut up unused warnings if assertions ever loosen.
        let _ = slot_index(0, 0, 0);
    }
}
