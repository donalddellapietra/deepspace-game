//! CPU mirror of the shader's `march_rotated45y_subtree`.
//!
//! Applies the same `T(p) = (p.x - p.z, p.y, p.x + p.z)` transform
//! to the ray (relative to the parent cell center), then runs the
//! standard cartesian DDA inside the rotated subtree. On hit, the
//! frame-local t is mapped back to a world t by projecting the
//! transformed hit point onto the world ray — non-uniform T means
//! a scalar divide wouldn't be correct.
//!
//! Interior of the rotated subtree is ordinary cartesian, so we
//! simply reuse `cpu_raycast_with_face_depth` on the transformed
//! ray; no separate inner walker is needed.

use super::cartesian::cpu_raycast_inner;
use super::HitInfo;
use crate::world::tree::{NodeId, NodeLibrary};

/// `cell_min` / `cell_size` are the axis-aligned world bounds of the
/// parent cell containing the rotated node. `parent_path` is the
/// path accumulated from the world root down to (and including) the
/// slot entry that points at this rotated child — the returned
/// `HitInfo` prepends it so callers get an absolute path.
pub(super) fn rotated_raycast_in_cell(
    library: &NodeLibrary,
    rotated_node_id: NodeId,
    cell_min: [f32; 3],
    cell_size: f32,
    ray_origin: [f32; 3],
    ray_dir: [f32; 3],
    parent_path: &[(NodeId, usize)],
    max_depth: u32,
) -> Option<HitInfo> {
    let center = [
        cell_min[0] + cell_size * 0.5,
        cell_min[1] + cell_size * 0.5,
        cell_min[2] + cell_size * 0.5,
    ];
    let pc = [
        ray_origin[0] - center[0],
        ray_origin[1] - center[1],
        ray_origin[2] - center[2],
    ];
    // T(p) = (p.x - p.z, p.y, p.x + p.z) — 45° Y rot + √2 XZ stretch.
    let pc_t = [pc[0] - pc[2], pc[1], pc[0] + pc[2]];
    let dc_t = [ray_dir[0] - ray_dir[2], ray_dir[1], ray_dir[0] + ray_dir[2]];
    let s3 = 3.0 / cell_size;
    let local_origin = [
        (pc_t[0] + cell_size * 0.5) * s3,
        (pc_t[1] + cell_size * 0.5) * s3,
        (pc_t[2] + cell_size * 0.5) * s3,
    ];
    let local_dir = [dc_t[0] * s3, dc_t[1] * s3, dc_t[2] * s3];

    let inner_depth = max_depth.saturating_sub(parent_path.len() as u32);
    let inner = cpu_raycast_inner(
        library,
        rotated_node_id,
        local_origin,
        local_dir,
        inner_depth,
    )?;

    // Map the inner frame-local hit point back to world coordinates
    // via T⁻¹, then project onto the world ray to recover world t.
    let p_local_hit = [
        local_origin[0] + local_dir[0] * inner.t,
        local_origin[1] + local_dir[1] * inner.t,
        local_origin[2] + local_dir[2] * inner.t,
    ];
    let p_shifted = [
        p_local_hit[0] / s3 - cell_size * 0.5,
        p_local_hit[1] / s3 - cell_size * 0.5,
        p_local_hit[2] / s3 - cell_size * 0.5,
    ];
    // T⁻¹(u, y, w) = ((u + w) / 2, y, (w - u) / 2).
    let world_off = [
        (p_shifted[0] + p_shifted[2]) * 0.5,
        p_shifted[1],
        (p_shifted[2] - p_shifted[0]) * 0.5,
    ];
    let world_hit = [
        world_off[0] + center[0],
        world_off[1] + center[1],
        world_off[2] + center[2],
    ];
    let dd = ray_dir[0] * ray_dir[0] + ray_dir[1] * ray_dir[1] + ray_dir[2] * ray_dir[2];
    let t_world = if dd > 1e-12 {
        ((world_hit[0] - ray_origin[0]) * ray_dir[0]
            + (world_hit[1] - ray_origin[1]) * ray_dir[1]
            + (world_hit[2] - ray_origin[2]) * ray_dir[2])
            / dd
    } else {
        inner.t
    };

    let mut path = Vec::with_capacity(parent_path.len() + inner.path.len());
    path.extend_from_slice(parent_path);
    path.extend_from_slice(&inner.path);

    let place_path = inner.place_path.map(|pp| {
        let mut v = Vec::with_capacity(parent_path.len() + pp.len());
        v.extend_from_slice(parent_path);
        v.extend_from_slice(&pp);
        v
    });

    Some(HitInfo {
        path,
        // `face` stays in the rotated local frame. Edits treat faces
        // as XYZ-axial (for "place on the +X side" etc.); for the
        // first test pass we accept that a rotated hit reports a
        // local face id. Cursor highlight still works because the
        // path is correct; only the face-based place-direction is
        // in local space.
        face: inner.face,
        t: t_world.max(0.0),
        place_path,
    })
}
