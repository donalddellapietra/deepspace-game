//! UvRing camera/frame integration.
//!
//! The world's `UvRing` topology stores cells in a flat `[N, 1, 1]`
//! slab but renders them at ring positions in the UvRing root's
//! local `[0, 3)³`. To make the storage and render layouts
//! consistent at the `App` layer we adopt a single invariant:
//!
//! > **When the camera is anywhere near the ring, its `WorldPos`
//! > anchor traverses the slab path of some `cell_x` and its
//! > offset is in the cell's own `[0, 1)³` (post-`R^T`)**.
//!
//! `ensure_uv_ring_camera_anchor_local` enforces this every update
//! by re-anchoring a stray camera through the topology adapter
//! (`UvRingCellFrame::point_to_local`). `exit_uv_ring_cell_if_needed`
//! handles the inverse — when motion takes the offset across a
//! cell boundary, anchor rolls to the neighbour cell and the
//! offset wraps. Together they keep the camera state inside the
//! ring without ever expressing the camera at a slab-storage
//! position (which doesn't correspond to anything visible).
//!
//! Two render kinds fall out:
//!
//! * [`ActiveFrameKind::UvRingCell`]: the user is inside a
//!   specific cell. `render_path` is the slab path so the GPU's
//!   frame root is the cell content. The shader runs the standard
//!   Cartesian DDA on the cell content; no UvRing-specific shader
//!   code is needed at this kind.
//!
//! * [`ActiveFrameKind::UvRing`]: overview. `render_path` is the
//!   world root. The shader runs `march_uv_ring` and the camera
//!   projects through `point_to_world` to its ring topology
//!   position so rays start where the camera visually is.
//!
//! Constants here MUST match the renderer
//! (`assets/shaders/march.wgsl::march_uv_ring`) and the CPU
//! raycast (`src/world/raycast/uv_ring.rs`). If you change a
//! constant in one place, change it in all three.

use crate::app::{frame, ActiveFrame, ActiveFrameKind, App, RENDER_FRAME_CONTEXT};
use crate::world::anchor::{Path, WorldPos, WORLD_SIZE};
use crate::world::tree::{slot_coords, slot_index, NodeKind};

/// Centre of the ring inside a UvRing node's `[0, 3)³`.
pub const UV_RING_CENTER: [f32; 3] = [1.5, 1.5, 1.5];
/// Distance from `UV_RING_CENTER` to each cell's centre.
pub const UV_RING_RADIUS: f32 = 1.0;
/// Per-cell side packing factor: the cube side is
/// `(2π · UV_RING_RADIUS / N) · UV_RING_PACKING`, leaving a small
/// tangential gap between adjacent cells so AABBs don't touch.
pub const UV_RING_PACKING: f32 = 0.95;

/// Topology basis for one ring cell. Maps points and directions
/// between the cell's local `[0, 3)³` (cell-axis-aligned) and the
/// UvRing root's `[0, 3)³`.
///
/// The cell's local axes are
/// `(tangent, radial, up)`: storage `+X` runs along the ring
/// tangent, storage `+Y` points outward from the ring centre,
/// storage `+Z` is the world `+Y` axis. `origin` is the cell
/// centre in the UvRing's frame; `scale = 3 / cell_side` maps
/// world units to cell-local units.
#[derive(Clone, Copy, Debug)]
pub struct UvRingCellFrame {
    pub origin: [f32; 3],
    pub tangent: [f32; 3],
    pub radial: [f32; 3],
    pub up: [f32; 3],
    pub scale: f32,
}

impl UvRingCellFrame {
    /// Project a UvRing-frame point into the cell's local `[0, 3)³`.
    pub fn point_to_local(self, p: [f32; 3]) -> [f32; 3] {
        let d = [
            p[0] - self.origin[0],
            p[1] - self.origin[1],
            p[2] - self.origin[2],
        ];
        [
            (self.tangent[0] * d[0] + self.tangent[1] * d[1] + self.tangent[2] * d[2]) * self.scale + 1.5,
            (self.radial[0] * d[0] + self.radial[1] * d[1] + self.radial[2] * d[2]) * self.scale + 1.5,
            (self.up[0] * d[0] + self.up[1] * d[1] + self.up[2] * d[2]) * self.scale + 1.5,
        ]
    }

    /// Inverse of [`point_to_local`]: cell-local → UvRing frame.
    pub fn point_to_world(self, p: [f32; 3]) -> [f32; 3] {
        let q = [
            (p[0] - 1.5) / self.scale,
            (p[1] - 1.5) / self.scale,
            (p[2] - 1.5) / self.scale,
        ];
        [
            self.origin[0] + self.tangent[0] * q[0] + self.radial[0] * q[1] + self.up[0] * q[2],
            self.origin[1] + self.tangent[1] * q[0] + self.radial[1] * q[1] + self.up[1] * q[2],
            self.origin[2] + self.tangent[2] * q[0] + self.radial[2] * q[1] + self.up[2] * q[2],
        ]
    }

    /// Project a UvRing-frame direction into the cell's local axes.
    /// Same scale factor as `point_to_local` (no translation).
    pub fn dir_to_local(self, d: [f32; 3]) -> [f32; 3] {
        [
            (self.tangent[0] * d[0] + self.tangent[1] * d[1] + self.tangent[2] * d[2]) * self.scale,
            (self.radial[0] * d[0] + self.radial[1] * d[1] + self.radial[2] * d[2]) * self.scale,
            (self.up[0] * d[0] + self.up[1] * d[1] + self.up[2] * d[2]) * self.scale,
        ]
    }
}

/// Build the topology basis for `cell_x` of a `[N, 1, 1]` UvRing.
pub fn uv_ring_cell_frame(dims: [u32; 3], _slab_depth: u8, cell_x: u32) -> UvRingCellFrame {
    let pi = std::f32::consts::PI;
    let two_pi = std::f32::consts::TAU;
    let theta = -pi + (cell_x as f32 + 0.5) * two_pi / dims[0] as f32;
    let (st, ct) = theta.sin_cos();
    let radial = [ct, 0.0_f32, st];
    let tangent = [-st, 0.0_f32, ct];
    let up = [0.0_f32, 1.0, 0.0];
    let origin = [
        UV_RING_CENTER[0] + radial[0] * UV_RING_RADIUS,
        UV_RING_CENTER[1],
        UV_RING_CENTER[2] + radial[2] * UV_RING_RADIUS,
    ];
    let arc = two_pi * UV_RING_RADIUS / dims[0] as f32;
    let cell_side = arc * UV_RING_PACKING;
    UvRingCellFrame {
        origin,
        tangent,
        radial,
        up,
        scale: 3.0 / cell_side,
    }
}

/// Build the slab descent path for `cell_x`. For an `[N, 1, 1]`
/// slab the storage path is `slab_depth` slots all of the form
/// `(sx, 0, 0)` where the `sx` digits decompose `cell_x` in
/// ternary, MSB-first.
pub fn uv_ring_cell_path(cell_x: u32, slab_depth: u8) -> Path {
    let mut path = Path::root();
    let mut cells_per_slot = 1u32;
    for _ in 1..slab_depth {
        cells_per_slot *= 3;
    }
    for _ in 0..slab_depth {
        let sx = (cell_x / cells_per_slot) % 3;
        path.push(slot_index(sx as usize, 0, 0) as u8);
        cells_per_slot = (cells_per_slot / 3).max(1);
    }
    path
}

/// Inverse of [`uv_ring_cell_path`]: read `cell_x` from the first
/// `slab_depth` slots of `path`. Returns `None` if any of those
/// slots have non-zero `sy` or `sz` (i.e. the path doesn't
/// traverse the `[N, 1, 1]` slab).
pub fn uv_ring_cell_x_from_path(path: &Path, slab_depth: u8) -> Option<u32> {
    if path.depth() < slab_depth {
        return None;
    }
    let mut cell_x = 0u32;
    for k in 0..slab_depth as usize {
        let (sx, sy, sz) = slot_coords(path.slot(k) as usize);
        if sy != 0 || sz != 0 {
            return None;
        }
        cell_x = cell_x * 3 + sx as u32;
    }
    Some(cell_x)
}

/// True when `cell_local` lands close enough to the cell's
/// content `[0, 3)³` for inside-cell rendering. Camera positions
/// far above/below the ring shouldn't trigger the cell-anchored
/// frame — they belong to the overview view.
///
/// Tangent (X) bounds are intentionally wider than the cell's
/// `[0, 3)`: motion across cell boundaries is allowed (the wrap
/// happens in `exit_uv_ring_cell_if_needed`). Radial (Y) and up
/// (Z) bounds are tight — only one cell's worth of margin on
/// either side.
fn uv_ring_cell_local_is_near_ring_slab(p: [f32; 3]) -> bool {
    let radial_margin = 1.0_f32;
    p.iter().all(|v| v.is_finite())
        && (-1.5..4.5).contains(&p[0])
        && (-radial_margin..(3.0 + radial_margin)).contains(&p[1])
        && (-radial_margin..(3.0 + radial_margin)).contains(&p[2])
}

/// Build a `WorldPos` from `(cell_path, cell_local)`, inflating to
/// `anchor_depth` by repeatedly picking the geometric Cartesian
/// slot at each level. Used by `exit_uv_ring_cell_if_needed` and
/// `ensure_uv_ring_camera_anchor_local` to land the camera at a
/// specific cell with a known local offset, preserving anchor
/// depth across the topology re-anchor.
fn world_pos_from_cell_local(
    cell_path: &Path,
    mut local: [f32; 3],
    anchor_depth: u8,
) -> WorldPos {
    let mut anchor = *cell_path;
    while anchor.depth() < anchor_depth && (anchor.depth() as usize) < crate::world::tree::MAX_DEPTH {
        let sx = local[0].floor().clamp(0.0, 2.0) as usize;
        let sy = local[1].floor().clamp(0.0, 2.0) as usize;
        let sz = local[2].floor().clamp(0.0, 2.0) as usize;
        anchor.push(slot_index(sx, sy, sz) as u8);
        local = [
            (local[0] - sx as f32) * WORLD_SIZE,
            (local[1] - sy as f32) * WORLD_SIZE,
            (local[2] - sz as f32) * WORLD_SIZE,
        ];
    }
    WorldPos::new_unchecked(
        anchor,
        [
            local[0] / WORLD_SIZE,
            local[1] / WORLD_SIZE,
            local[2] / WORLD_SIZE,
        ],
    )
}

impl App {
    /// Try to build an [`ActiveFrameKind::UvRingCell`] frame for
    /// the camera's current cell anchor. Returns `None` when the
    /// world isn't a UvRing, or the anchor doesn't traverse a
    /// slab path, or the camera's cell-local position is outside
    /// the ring band.
    pub(super) fn uv_ring_cell_render_frame(
        &self,
        desired_depth: u8,
    ) -> Option<ActiveFrame> {
        let root = self.world.library.get(self.world.root)?;
        let NodeKind::UvRing { dims, slab_depth } = root.kind else {
            return None;
        };
        if dims[0] == 0 || dims[1] != 1 || dims[2] != 1 {
            return None;
        }
        if desired_depth < slab_depth {
            return None;
        }
        let cell_x = uv_ring_cell_x_from_path(&self.camera.position.anchor, slab_depth)?;
        let cell_path = uv_ring_cell_path(cell_x, slab_depth);
        let cell_local = self.camera.position.in_frame_rot(
            &self.world.library, self.world.root, &cell_path,
        );
        if !uv_ring_cell_local_is_near_ring_slab(cell_local) {
            return None;
        }
        let mut logical_path = self.camera.position.anchor;
        logical_path.truncate(desired_depth);
        if logical_path.depth() < slab_depth {
            logical_path = cell_path;
        }
        let mut render_path = logical_path;
        let render_depth = render_path
            .depth()
            .saturating_sub(RENDER_FRAME_CONTEXT)
            .max(slab_depth);
        render_path.truncate(render_depth);
        let render = frame::compute_render_frame(
            &self.world.library, self.world.root, &render_path, render_depth,
        );
        Some(ActiveFrame {
            render_path,
            logical_path,
            node_id: render.node_id,
            kind: ActiveFrameKind::UvRingCell { dims, slab_depth, cell_x },
        })
    }

    /// Overview frame: render frame is the UvRing root itself.
    pub(super) fn uv_ring_overview_frame(&self) -> Option<ActiveFrame> {
        let root = self.world.library.get(self.world.root)?;
        let NodeKind::UvRing { dims, slab_depth } = root.kind else {
            return None;
        };
        Some(ActiveFrame {
            render_path: Path::root(),
            logical_path: Path::root(),
            node_id: self.world.root,
            kind: ActiveFrameKind::UvRing { dims, slab_depth },
        })
    }

    /// XZ-projection of the camera onto the ring; returns the
    /// nearest `cell_x`.
    fn nearest_uv_ring_cell_x(&self, dims: [u32; 3]) -> u32 {
        let pi = std::f32::consts::PI;
        let two_pi = std::f32::consts::TAU;
        let cam = self.camera.position.in_frame_rot(
            &self.world.library, self.world.root, &Path::root(),
        );
        let dx = cam[0] - UV_RING_CENTER[0];
        let dz = cam[2] - UV_RING_CENTER[2];
        let mut u = (dz.atan2(dx) + pi) / two_pi;
        if !u.is_finite() {
            u = 0.0;
        }
        let cell = (u * dims[0] as f32).floor() as i32;
        cell.rem_euclid(dims[0] as i32) as u32
    }

    /// Re-anchor the camera through the ring topology when it has
    /// drifted far enough into a cell band. Idempotent — a no-op
    /// when the anchor already traverses a slab path or when the
    /// camera is too far from the ring.
    pub(super) fn ensure_uv_ring_camera_anchor_local(&mut self) {
        let Some(root) = self.world.library.get(self.world.root) else { return };
        let NodeKind::UvRing { dims, slab_depth } = root.kind else { return };
        if dims[0] == 0 || dims[1] != 1 || dims[2] != 1 {
            return;
        }
        if uv_ring_cell_x_from_path(&self.camera.position.anchor, slab_depth).is_some() {
            return;
        }
        let cell_x = self.nearest_uv_ring_cell_x(dims);
        let cam_in_root = self.camera.position.in_frame_rot(
            &self.world.library, self.world.root, &Path::root(),
        );
        let cell_frame = uv_ring_cell_frame(dims, slab_depth, cell_x);
        let cell_local = cell_frame.point_to_local(cam_in_root);
        if !uv_ring_cell_local_is_near_ring_slab(cell_local) {
            return;
        }
        let cell_path = uv_ring_cell_path(cell_x, slab_depth);
        self.camera.position = world_pos_from_cell_local(
            &cell_path,
            cell_local,
            self.camera.position.anchor.depth(),
        );
    }

    /// After motion: if the camera was anchored in a cell on the
    /// previous frame but isn't now, walk the cell-local offset to
    /// the neighbour cell `cell_x ± 1` (with wrap) and re-anchor
    /// there. This is what makes "walking around the ring" work
    /// without leaving cell-local representation.
    pub(super) fn exit_uv_ring_cell_if_needed(&mut self, previous_position: WorldPos) {
        let Some(root) = self.world.library.get(self.world.root) else { return };
        let NodeKind::UvRing { dims, slab_depth } = root.kind else { return };
        if dims[0] == 0 || dims[1] != 1 || dims[2] != 1 {
            return;
        }
        let Some(prev_cell_x) =
            uv_ring_cell_x_from_path(&previous_position.anchor, slab_depth)
        else {
            return;
        };
        if uv_ring_cell_x_from_path(&self.camera.position.anchor, slab_depth).is_some() {
            return;
        }
        let prev_cell_path = uv_ring_cell_path(prev_cell_x, slab_depth);
        let mut cell_local = self.camera.position.in_frame_rot(
            &self.world.library, self.world.root, &prev_cell_path,
        );
        let mut cell_x = prev_cell_x as i32;
        while cell_local[0] < 0.0 {
            cell_local[0] += WORLD_SIZE;
            cell_x -= 1;
        }
        while cell_local[0] >= WORLD_SIZE {
            cell_local[0] -= WORLD_SIZE;
            cell_x += 1;
        }
        let cell_x = cell_x.rem_euclid(dims[0] as i32) as u32;
        let cell_path = uv_ring_cell_path(cell_x, slab_depth);
        self.camera.position = world_pos_from_cell_local(
            &cell_path,
            cell_local,
            self.camera.position.anchor.depth(),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Encoding `cell_x` as a slab path then decoding it should be
    /// the identity for every cell of the canonical `[27, 1, 1]`
    /// slab.
    #[test]
    fn cell_x_path_round_trip() {
        for cell_x in 0..27u32 {
            let path = uv_ring_cell_path(cell_x, 3);
            assert_eq!(path.depth(), 3);
            let recovered = uv_ring_cell_x_from_path(&path, 3);
            assert_eq!(recovered, Some(cell_x), "cell_x={cell_x}");
        }
    }

    /// Every slab slot picked by `uv_ring_cell_path` lies on the
    /// X-axis row `(sx, 0, 0)`. If this ever started picking
    /// `(sx, sy, sz)` with `sy != 0` or `sz != 0`, the storage
    /// path would target an empty UvRing slot and the renderer
    /// would never reach the cell content.
    #[test]
    fn cell_path_slots_lie_on_slab_x_axis() {
        for cell_x in 0..27u32 {
            let path = uv_ring_cell_path(cell_x, 3);
            for k in 0..3 {
                let slot = path.slot(k) as usize;
                let (_, sy, sz) = slot_coords(slot);
                assert_eq!(sy, 0);
                assert_eq!(sz, 0);
            }
        }
    }

    /// `point_to_local(point_to_world(p)) ≈ p` and the centre of
    /// the cell content frame `(1.5, 1.5, 1.5)` maps to the
    /// cell's ring origin.
    #[test]
    fn cell_frame_round_trip_and_centre() {
        let dims = [27u32, 1, 1];
        let slab_depth = 3u8;
        for cell_x in 0..dims[0] {
            let cell = uv_ring_cell_frame(dims, slab_depth, cell_x);

            let centre = cell.point_to_world([1.5, 1.5, 1.5]);
            for axis in 0..3 {
                assert!(
                    (centre[axis] - cell.origin[axis]).abs() < 1e-5,
                    "centre mismatch cell_x={cell_x} axis={axis}: got {} want {}",
                    centre[axis],
                    cell.origin[axis],
                );
            }

            for sample in [
                [1.5_f32, 1.5, 1.5],
                [0.5, 2.0, 1.5],
                [2.7, 1.0, 0.3],
            ] {
                let world = cell.point_to_world(sample);
                let local = cell.point_to_local(world);
                for axis in 0..3 {
                    assert!(
                        (local[axis] - sample[axis]).abs() < 1e-4,
                        "round-trip drift cell_x={cell_x} axis={axis}: {local:?} vs {sample:?}",
                    );
                }
            }
        }
    }

    /// The ring tangent basis: tangent · radial = 0, both
    /// orthogonal to up = (0, 1, 0), and `radial` points from the
    /// ring centre to the cell origin.
    #[test]
    fn cell_frame_basis_is_orthonormal_and_radial_aligned() {
        let dims = [27u32, 1, 1];
        for cell_x in 0..dims[0] {
            let cell = uv_ring_cell_frame(dims, 3, cell_x);
            let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            assert!(dot(cell.tangent, cell.radial).abs() < 1e-5);
            assert!(dot(cell.tangent, cell.up).abs() < 1e-5);
            assert!(dot(cell.radial, cell.up).abs() < 1e-5);
            // origin = centre + radial · UV_RING_RADIUS.
            let expected_origin = [
                UV_RING_CENTER[0] + cell.radial[0] * UV_RING_RADIUS,
                UV_RING_CENTER[1] + cell.radial[1] * UV_RING_RADIUS,
                UV_RING_CENTER[2] + cell.radial[2] * UV_RING_RADIUS,
            ];
            for axis in 0..3 {
                assert!(
                    (cell.origin[axis] - expected_origin[axis]).abs() < 1e-5,
                    "origin axis {axis}: {} vs {}",
                    cell.origin[axis],
                    expected_origin[axis],
                );
            }
        }
    }
}
