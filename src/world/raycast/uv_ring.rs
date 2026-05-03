//! CPU mirror of the shader's `march_uv_ring`. Renders the `UvRing`
//! frame as a circle of cubes in the ring node's local `[0, 3)³`,
//! one cube per cell of an `[N, 1, 1]` storage slab. Pure cell-local
//! arithmetic — no global coordinates, no `body_origin` parameters.
//!
//! For each `cell_x ∈ 0..N`:
//!   1. Compute the ring tangent basis at θ = -π + (cell_x + 0.5)·2π/N.
//!   2. Lookup the cell's storage child via the slab descent.
//!   3. Transform the ray into the cell's local `[0, 3)³`.
//!   4. Dispatch the standard Cartesian cube DDA on the cell content.
//!   5. Keep the closest hit.
//!
//! The ring centre is fixed at `(1.5, 1.5, 1.5)`, the radius at `1.0`,
//! and the cell side at `arc · 0.95` where `arc = 2π·radius/N`.
//! These constants must stay in lockstep with the shader's
//! `march_uv_ring`; if you change one, change both.

use super::HitInfo;
use crate::world::tree::{slot_index, Child, NodeId, NodeLibrary};

const RING_CENTER: [f32; 3] = [1.5, 1.5, 1.5];
const RING_RADIUS: f32 = 1.0;
const RING_PACKING: f32 = 0.95;

/// Cast a ray through a `UvRing`-rooted slab rendered as a circle of
/// Cartesian cubes.
///
/// `frame_path` is the world-root → ring-root path; it's prepended
/// to the returned `HitInfo`'s path so callers (place / break) get
/// a fully-qualified world-tree path.
///
/// `cam_local` and `ray_dir` are in the ring-root frame's local
/// `[0, 3)³` coords.
pub fn cpu_raycast_uv_ring(
    library: &NodeLibrary,
    world_root: NodeId,
    frame_path: &[u8],
    cam_local: [f32; 3],
    ray_dir: [f32; 3],
    dims: [u32; 3],
    slab_depth: u8,
    max_depth: u32,
) -> Option<HitInfo> {
    if dims[0] == 0 {
        return None;
    }

    // Walk frame_path from world_root to the ring root once.
    let mut frame_chain: Vec<(NodeId, usize)> = Vec::with_capacity(frame_path.len());
    let mut cur = world_root;
    for &slot in frame_path.iter() {
        let node = library.get(cur)?;
        frame_chain.push((cur, slot as usize));
        match node.children[slot as usize] {
            Child::Node(child) => {
                cur = child;
            }
            _ => return None,
        }
    }
    let ring_root = cur;

    let pi = std::f32::consts::PI;
    let two_pi = 2.0 * pi;
    let arc = (two_pi * RING_RADIUS) / dims[0] as f32;
    let cell_side = arc * RING_PACKING;
    let scale = 3.0 / cell_side;

    let mut best: Option<HitInfo> = None;
    let mut best_t = f32::INFINITY;

    for cell_x in 0..dims[0] as i32 {
        let theta = -pi + (cell_x as f32 + 0.5) * (two_pi / dims[0] as f32);
        let (st, ct) = theta.sin_cos();
        let radial = [ct, 0.0, st];
        let cell_center = [
            RING_CENTER[0] + radial[0] * RING_RADIUS,
            RING_CENTER[1],
            RING_CENTER[2] + radial[2] * RING_RADIUS,
        ];

        let Some((cell_path, cell_root)) = walk_ring_storage(
            library, ring_root, &frame_chain, dims[0], slab_depth, cell_x as u32,
        ) else { continue };

        // Translate + scale only — no rotation. The TB head at
        // the cell's storage tip applies the ring tangent basis
        // R^T at descent (handled by `cpu_raycast_inner`'s TB
        // dispatch).
        let local_origin = [
            (cam_local[0] - cell_center[0]) * scale + 1.5,
            (cam_local[1] - cell_center[1]) * scale + 1.5,
            (cam_local[2] - cell_center[2]) * scale + 1.5,
        ];
        let local_dir = [
            ray_dir[0] * scale,
            ray_dir[1] * scale,
            ray_dir[2] * scale,
        ];

        let inner_max = max_depth.saturating_sub(cell_path.len() as u32);
        let Some(mut hit) = super::cartesian::cpu_raycast_inner(
            library, cell_root, local_origin, local_dir, inner_max,
        ) else { continue };

        // hit.t is the parametric `t` for `local_dir`. Because we
        // built `local_dir` by rotating `ray_dir` and scaling by
        // `scale`, the same `t` carries `cam_local` to the same
        // world hit point along `ray_dir`. So we can compare hits
        // across cells directly without a rescale.
        if hit.t >= best_t {
            continue;
        }

        // Prepend the storage path: world_root → ring_root → cell_x slab descent.
        let mut full_path = Vec::with_capacity(cell_path.len() + hit.path.len());
        full_path.extend_from_slice(&cell_path);
        full_path.extend_from_slice(&hit.path);
        hit.path = full_path;
        if let Some(pp) = hit.place_path.as_mut() {
            let mut full_pp = Vec::with_capacity(cell_path.len() + pp.len());
            full_pp.extend_from_slice(&cell_path);
            full_pp.extend_from_slice(pp);
            *pp = full_pp;
        }
        best_t = hit.t;
        best = Some(hit);
    }

    best
}

/// Walk the ring's `[N, 1, 1]` slab storage from `ring_root` to the
/// cell at `cell_x`. Returns the path entries and the cell content's
/// root NodeId, or None if the cell is empty.
fn walk_ring_storage(
    library: &NodeLibrary,
    ring_root: NodeId,
    frame_chain: &[(NodeId, usize)],
    dims_x: u32,
    slab_depth: u8,
    cell_x: u32,
) -> Option<(Vec<(NodeId, usize)>, NodeId)> {
    if cell_x >= dims_x {
        return None;
    }
    let mut path = Vec::with_capacity(frame_chain.len() + slab_depth as usize);
    path.extend_from_slice(frame_chain);

    let mut cur = ring_root;
    let mut cells_per_slot: u32 = 1;
    for _ in 1..slab_depth {
        cells_per_slot *= 3;
    }
    for level in 0..slab_depth {
        let sx = (cell_x / cells_per_slot) % 3;
        let slot = slot_index(sx as usize, 0, 0);
        let node = library.get(cur)?;
        path.push((cur, slot));
        match node.children[slot] {
            Child::Empty | Child::EntityRef(_) => return None,
            Child::Block(_) => {
                // A pack-time uniform-flatten landed at a slab cell.
                // For now treat as opaque; CPU raycast won't descend
                // further. Caller still gets a hit at the cell.
                return Some((path, cur));
            }
            Child::Node(child) => {
                if level + 1 == slab_depth {
                    return Some((path, child));
                }
                cur = child;
            }
        }
        cells_per_slot = (cells_per_slot / 3).max(1);
    }
    None
}
