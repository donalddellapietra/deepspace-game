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

        // Sub-slab descent — Cartesian-local DDA matching the GPU
        // shader's `sphere_descend_anchor` exactly. The previous
        // version walked sub-cells using EQUAL ANGULAR slices of
        // (lon, lat, r), which diverged from the GPU's CARTESIAN-LOCAL
        // sub-cell partition at descent depth ≥ 2. The mismatch caused
        // breaks at depth ≥ 2 to hit slot path A (CPU) while the GPU
        // rendered slot path B — so a "successful" CPU edit had no
        // visible effect (the GPU's DDA traversed past the break and
        // rendered a different sub-cell that wasn't edited).
        //
        // Both sides now use the same orthonormal local frame at the
        // slab cell's center (e_lon, e_r, e_lat tangents) and walk
        // sub-cells via standard Cartesian DDA in [0, 3)³ with re-zero
        // per push (`new_O = (old_O - cell) * 3`, `new_D = old_D * 3`).
        // Coordinates stay O(1) at every depth — same trick as
        // Cartesian's ribbon-pop, applied in the descent direction.
        if let Some(anchor_idx) = sub_idx {
            let absolute_slab_depth = frame_path.len() as u32 + slab_depth as u32;
            let extra_levels = max_depth.saturating_sub(absolute_slab_depth) as usize;
            if extra_levels > 0 {
                let lon_step = 2.0 * pi / dims[0] as f32;
                let lat_step = 2.0 * lat_max / dims[2] as f32;
                let r_step = shell_thickness / dims[1] as f32;
                let lon_c = -pi + (cell_x as f32 + 0.5) * lon_step;
                let lat_c = -lat_max + (cell_z as f32 + 0.5) * lat_step;
                let r_c   = r_inner + (cy as f32 + 0.5) * r_step;
                descend_anchor_cartesian_local(
                    library,
                    anchor_idx,
                    cam_local,
                    dir,
                    cs_center,
                    lon_c, lat_c, r_c,
                    lon_step, lat_step, r_step,
                    t_layer,
                    extra_levels,
                    &mut path,
                );
            }
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

/// Cartesian-local DDA inside a sphere-render anchor block. CPU mirror
/// of the GPU's `sphere_descend_anchor` (assets/shaders/march_sphere_anchor.wgsl).
///
/// Both walk an orthonormal frame at the slab cell's center
/// (e_lon, e_r, e_lat tangents to the sphere), with the slab cell
/// occupying `[0, 3)³` in local. Each push re-zeros the local frame
/// onto the descended cell:
///     new_O = (old_O - cell) * 3
///     new_D = old_D * 3
/// so coordinates stay O(1) at every depth. The cell index path
/// produced is the slot path through the anchor's tree, which becomes
/// `HitInfo.path` and feeds `propagate_edit` for break/place.
///
/// CPU and GPU MUST agree on the slot path or a CPU edit lands at one
/// cell while the GPU renders a different one. This function is the
/// CPU side of that contract.
fn descend_anchor_cartesian_local(
    library: &NodeLibrary,
    anchor_idx: NodeId,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    cs_center: [f32; 3],
    cell_lon_center: f32, cell_lat_center: f32, cell_r_center: f32,
    cell_lon_step: f32, cell_lat_step: f32, cell_r_step: f32,
    t_in: f32,
    max_extra_levels: usize,
    path: &mut Vec<(NodeId, usize)>,
) {
    // Build orthonormal local frame at slab cell center.
    let cos_lat = cell_lat_center.cos();
    let sin_lat = cell_lat_center.sin();
    let cos_lon = cell_lon_center.cos();
    let sin_lon = cell_lon_center.sin();
    let e_r:   [f32; 3] = [cos_lat * cos_lon,  sin_lat,        cos_lat * sin_lon];
    let e_lon: [f32; 3] = [-sin_lon,           0.0,            cos_lon];
    let e_lat: [f32; 3] = [-sin_lat * cos_lon, cos_lat,       -sin_lat * sin_lon];

    // Slab cell extents in world along the three local axes (slab-axis
    // convention: local x→e_lon, y→e_r, z→e_lat — matches slot layout
    // dims = [lon, r, lat]).
    let ext_x = cell_r_center * cos_lat * cell_lon_step;
    let ext_y = cell_r_step;
    let ext_z = cell_r_center * cell_lat_step;
    let scale_x = 3.0 / ext_x.max(1e-30);
    let scale_y = 3.0 / ext_y.max(1e-30);
    let scale_z = 3.0 / ext_z.max(1e-30);

    let cell_corner = [
        cs_center[0] + cell_r_center * e_r[0]
            - 0.5 * ext_x * e_lon[0]
            - 0.5 * ext_y * e_r[0]
            - 0.5 * ext_z * e_lat[0],
        cs_center[1] + cell_r_center * e_r[1]
            - 0.5 * ext_x * e_lon[1]
            - 0.5 * ext_y * e_r[1]
            - 0.5 * ext_z * e_lat[1],
        cs_center[2] + cell_r_center * e_r[2]
            - 0.5 * ext_x * e_lon[2]
            - 0.5 * ext_y * e_r[2]
            - 0.5 * ext_z * e_lat[2],
    ];

    let dv = [
        ray_origin[0] - cell_corner[0],
        ray_origin[1] - cell_corner[1],
        ray_origin[2] - cell_corner[2],
    ];
    let dot3 = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

    let mut cur_o: [f32; 3] = [
        dot3(dv, e_lon) * scale_x,
        dot3(dv, e_r)   * scale_y,
        dot3(dv, e_lat) * scale_z,
    ];
    let mut cur_d: [f32; 3] = [
        dot3(ray_dir, e_lon) * scale_x,
        dot3(ray_dir, e_r)   * scale_y,
        dot3(ray_dir, e_lat) * scale_z,
    ];

    let inv_or_huge = |v: f32| if v.abs() > 1e-12 { 1.0 / v } else { 1e30 };
    let mut inv_dir = [inv_or_huge(cur_d[0]), inv_or_huge(cur_d[1]), inv_or_huge(cur_d[2])];
    let mut step = [
        if cur_d[0] >= 0.0 { 1i32 } else { -1 },
        if cur_d[1] >= 0.0 { 1i32 } else { -1 },
        if cur_d[2] >= 0.0 { 1i32 } else { -1 },
    ];
    let mut delta_dist = [inv_dir[0].abs(), inv_dir[1].abs(), inv_dir[2].abs()];

    // Initial cell at slab entry (t = t_in is the slab DDA's current
    // t; ray is geometrically inside the curved slab cell at this t).
    let entry_pos = [
        cur_o[0] + cur_d[0] * t_in,
        cur_o[1] + cur_d[1] * t_in,
        cur_o[2] + cur_d[2] * t_in,
    ];
    let mut cur_cell: [i32; 3] = [
        (entry_pos[0].floor() as i32).clamp(0, 2),
        (entry_pos[1].floor() as i32).clamp(0, 2),
        (entry_pos[2].floor() as i32).clamp(0, 2),
    ];
    let mut cur_side_dist: [f32; 3] = [
        if cur_d[0] >= 0.0 { (cur_cell[0] as f32 + 1.0 - cur_o[0]) * inv_dir[0] }
        else                 { (cur_cell[0] as f32       - cur_o[0]) * inv_dir[0] },
        if cur_d[1] >= 0.0 { (cur_cell[1] as f32 + 1.0 - cur_o[1]) * inv_dir[1] }
        else                 { (cur_cell[1] as f32       - cur_o[1]) * inv_dir[1] },
        if cur_d[2] >= 0.0 { (cur_cell[2] as f32 + 1.0 - cur_o[2]) * inv_dir[2] }
        else                 { (cur_cell[2] as f32       - cur_o[2]) * inv_dir[2] },
    ];

    // Stacks: cell index path (for pop's re-zero reversal) and node
    // ids (parent of `cur_node` at each depth). The node-id stack
    // mirrors the GPU's `s_node_idx[depth]`. Walking `path.last()` to
    // recover the parent on pop is wrong — `path` entries are
    // (parent_node_id, slot), and after `path.pop()` the new last
    // entry is the GRANDPARENT's edge, not the parent's.
    //
    // Cap at SPHERE_DESCENT_DEPTH = 24 to mirror the GPU stack
    // ceiling exactly. Past that the GPU splats representative; the
    // CPU also stops at the deepest visible cell so break paths
    // can't outrun what's rendered.
    let cap = max_extra_levels.min(24);
    let mut s_cell_path: Vec<[i32; 3]> = Vec::with_capacity(cap);
    let mut s_node_path: Vec<NodeId> = Vec::with_capacity(cap);
    let mut cur_node = anchor_idx;
    let mut depth: usize = 0;
    let mut iters = 0usize;
    let max_iters = 4096usize;

    let min_axis = |sd: [f32; 3]| -> usize {
        if sd[0] <= sd[1] && sd[0] <= sd[2] { 0 }
        else if sd[1] <= sd[2] { 1 }
        else { 2 }
    };

    loop {
        if iters >= max_iters { break; }
        iters += 1;

        // OOB → pop one frame.
        if cur_cell[0] < 0 || cur_cell[0] > 2
            || cur_cell[1] < 0 || cur_cell[1] > 2
            || cur_cell[2] < 0 || cur_cell[2] > 2
        {
            if depth == 0 { break; }
            depth -= 1;
            // Pop transform reverses the push:
            //   old_O = new_O / 3 + popped_cell
            //   old_D = new_D / 3
            let popped = s_cell_path.pop().unwrap();
            cur_o = [
                cur_o[0] / 3.0 + popped[0] as f32,
                cur_o[1] / 3.0 + popped[1] as f32,
                cur_o[2] / 3.0 + popped[2] as f32,
            ];
            cur_d = [cur_d[0] / 3.0, cur_d[1] / 3.0, cur_d[2] / 3.0];
            inv_dir = [inv_dir[0] * 3.0, inv_dir[1] * 3.0, inv_dir[2] * 3.0];
            delta_dist = [delta_dist[0] * 3.0, delta_dist[1] * 3.0, delta_dist[2] * 3.0];
            // Path entry for the popped level was added at push; the
            // descent didn't terminate inside that subtree, so we
            // remove the path entry that pointed into it.
            path.pop();
            // Determine which face we crossed (in CHILD frame coords)
            // and advance parent's cell on that axis.
            let mut axis_x = 0i32;
            let mut axis_y = 0i32;
            let mut axis_z = 0i32;
            if cur_cell[0] < 0 { axis_x = -1; }
            if cur_cell[0] > 2 { axis_x =  1; }
            if cur_cell[1] < 0 { axis_y = -1; }
            if cur_cell[1] > 2 { axis_y =  1; }
            if cur_cell[2] < 0 { axis_z = -1; }
            if cur_cell[2] > 2 { axis_z =  1; }
            cur_cell = [popped[0] + axis_x, popped[1] + axis_y, popped[2] + axis_z];
            // Restore parent node from the parallel s_node_path stack.
            cur_node = s_node_path.pop().unwrap_or(anchor_idx);
            // Recompute side_dist for the new (parent-frame) cell.
            let cf = [cur_cell[0] as f32, cur_cell[1] as f32, cur_cell[2] as f32];
            cur_side_dist = [
                if cur_d[0] >= 0.0 { (cf[0] + 1.0 - cur_o[0]) * inv_dir[0] }
                else                 { (cf[0]       - cur_o[0]) * inv_dir[0] },
                if cur_d[1] >= 0.0 { (cf[1] + 1.0 - cur_o[1]) * inv_dir[1] }
                else                 { (cf[1]       - cur_o[1]) * inv_dir[1] },
                if cur_d[2] >= 0.0 { (cf[2] + 1.0 - cur_o[2]) * inv_dir[2] }
                else                 { (cf[2]       - cur_o[2]) * inv_dir[2] },
            ];
            continue;
        }

        let slot = slot_index(cur_cell[0] as usize, cur_cell[1] as usize, cur_cell[2] as usize);
        let node = match library.get(cur_node) {
            Some(n) => n,
            None => return,
        };
        let child = node.children[slot];

        match child {
            Child::Empty | Child::EntityRef(_) => {
                // Empty slot: step DDA along smallest side_dist.
                let m = min_axis(cur_side_dist);
                let step_axis = step[m];
                cur_cell[m] += step_axis;
                cur_side_dist[m] += delta_dist[m];
                continue;
            }
            Child::Block(_) => {
                // Hit a Block leaf — terminate descent here.
                path.push((cur_node, slot));
                return;
            }
            Child::Node(child_id) => {
                // Push or terminate at extra-level limit.
                path.push((cur_node, slot));
                if depth + 1 >= cap {
                    return;
                }
                // Re-zero local frame onto the cell. Save parent's
                // node id so pop can restore it without walking the
                // path (path entries point to parents-by-slot, not
                // by-node-id directly).
                s_cell_path.push(cur_cell);
                s_node_path.push(cur_node);
                let cf = [cur_cell[0] as f32, cur_cell[1] as f32, cur_cell[2] as f32];
                cur_o = [
                    (cur_o[0] - cf[0]) * 3.0,
                    (cur_o[1] - cf[1]) * 3.0,
                    (cur_o[2] - cf[2]) * 3.0,
                ];
                cur_d = [cur_d[0] * 3.0, cur_d[1] * 3.0, cur_d[2] * 3.0];
                inv_dir = [inv_dir[0] / 3.0, inv_dir[1] / 3.0, inv_dir[2] / 3.0];
                delta_dist = [delta_dist[0] / 3.0, delta_dist[1] / 3.0, delta_dist[2] / 3.0];

                depth += 1;
                cur_node = child_id;

                // Compute new sub-cell at depth+1 from ray's pos in
                // the new frame. Since the push preserves the world
                // ray, the position at the same world-t maps to a
                // local-pos that lies in [0, 3)³.
                // Use a small offset past the just-crossed face so
                // numerical error doesn't put us back outside.
                let new_pos = [
                    cur_o[0] + cur_d[0] * t_in,
                    cur_o[1] + cur_d[1] * t_in,
                    cur_o[2] + cur_d[2] * t_in,
                ];
                cur_cell = [
                    (new_pos[0].floor() as i32).clamp(0, 2),
                    (new_pos[1].floor() as i32).clamp(0, 2),
                    (new_pos[2].floor() as i32).clamp(0, 2),
                ];
                let cf = [cur_cell[0] as f32, cur_cell[1] as f32, cur_cell[2] as f32];
                cur_side_dist = [
                    if cur_d[0] >= 0.0 { (cf[0] + 1.0 - cur_o[0]) * inv_dir[0] }
                    else                 { (cf[0]       - cur_o[0]) * inv_dir[0] },
                    if cur_d[1] >= 0.0 { (cf[1] + 1.0 - cur_o[1]) * inv_dir[1] }
                    else                 { (cf[1]       - cur_o[1]) * inv_dir[1] },
                    if cur_d[2] >= 0.0 { (cf[2] + 1.0 - cur_o[2]) * inv_dir[2] }
                    else                 { (cf[2]       - cur_o[2]) * inv_dir[2] },
                ];
                continue;
            }
        }
    }
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
