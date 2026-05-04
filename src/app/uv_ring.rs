use crate::app::{frame, ActiveFrame, ActiveFrameKind, App, RENDER_FRAME_CONTEXT};
use crate::world::anchor::{Path, WorldPos, WORLD_SIZE};

impl App {
    pub(super) fn uv_ring_cell_render_frame(&self, desired_depth: u8) -> Option<ActiveFrame> {
        let root = self.world.library.get(self.world.root)?;
        let crate::world::tree::NodeKind::UvRing { dims, slab_depth } = root.kind else {
            return None;
        };
        if dims[0] == 0 || dims[1] != 1 || dims[2] == 0 {
            return None;
        }
        let (cell_x, cell_z) =
            uv_ring_cell_coords_from_path(&self.camera.position.anchor, dims, slab_depth)?;
        let cell_path = uv_ring_cell_path(cell_x, cell_z, slab_depth);
        let cell_local =
            self.camera
                .position
                .in_frame_rot(&self.world.library, self.world.root, &cell_path);
        if !uv_ring_cell_local_is_near_ring_slab(cell_local, slab_depth) {
            return None;
        }
        let mut logical_path = self.camera.position.anchor;
        logical_path.truncate(desired_depth.max(slab_depth));
        let logical_path = if logical_path.depth() < slab_depth {
            uv_ring_cell_path(cell_x, cell_z, slab_depth)
        } else {
            logical_path
        };
        let mut render_path = logical_path;
        render_path.truncate(
            render_path
                .depth()
                .saturating_sub(RENDER_FRAME_CONTEXT)
                .max(slab_depth),
        );
        let render = frame::compute_render_frame(
            &self.world.library,
            self.world.root,
            &render_path,
            render_path.depth(),
        );
        Some(ActiveFrame {
            render_path,
            logical_path,
            node_id: render.node_id,
            kind: ActiveFrameKind::UvRingCell {
                dims,
                slab_depth,
                cell_x,
                cell_z,
            },
        })
    }

    pub(super) fn uv_ring_overview_frame(&self) -> Option<ActiveFrame> {
        let root = self.world.library.get(self.world.root)?;
        let crate::world::tree::NodeKind::UvRing { dims, slab_depth } = root.kind else {
            return None;
        };
        Some(ActiveFrame {
            render_path: Path::root(),
            logical_path: Path::root(),
            node_id: self.world.root,
            kind: ActiveFrameKind::UvRing { dims, slab_depth },
        })
    }

    fn nearest_uv_ring_cell_x(&self, dims: [u32; 3]) -> u32 {
        let root_cam =
            self.camera
                .position
                .in_frame_rot(&self.world.library, self.world.root, &Path::root());
        let dx = root_cam[0] - 1.5;
        let dz = root_cam[2] - 1.5;
        let mut u = (dz.atan2(dx) + std::f32::consts::PI) / (2.0 * std::f32::consts::PI);
        if !u.is_finite() {
            u = 0.0;
        }
        let cell = (u * dims[0] as f32).floor() as i32;
        cell.rem_euclid(dims[0] as i32) as u32
    }

    fn nearest_uv_ring_cell_z(&self, dims: [u32; 3]) -> Option<u32> {
        let root_cam =
            self.camera
                .position
                .in_frame_rot(&self.world.library, self.world.root, &Path::root());
        let side = uv_ring_cell_side(dims);
        let height = side * dims[2].max(1) as f32;
        let height_lo = 1.5 - height * 0.5;
        let cell_f = (root_cam[1] - height_lo) / side;
        if !(0.0..dims[2].max(1) as f32).contains(&cell_f) {
            return None;
        }
        Some(cell_f.floor() as u32)
    }

    pub(super) fn ensure_uv_ring_camera_anchor_local(&mut self) {
        let Some(root) = self.world.library.get(self.world.root) else {
            return;
        };
        let crate::world::tree::NodeKind::UvRing { dims, slab_depth } = root.kind else {
            return;
        };
        if dims[0] == 0 || dims[1] != 1 || dims[2] == 0 {
            return;
        }
        if uv_ring_cell_coords_from_path(&self.camera.position.anchor, dims, slab_depth).is_some() {
            return;
        }
        let cell_x = self.nearest_uv_ring_cell_x(dims);
        let Some(cell_z) = self.nearest_uv_ring_cell_z(dims) else {
            return;
        };
        let cell_local = self.camera_root_to_uv_ring_cell_local(dims, slab_depth, cell_x, cell_z);
        if !uv_ring_cell_local_is_near_ring_slab(cell_local, slab_depth) {
            return;
        }
        self.camera.position = world_pos_from_frame_local_unclamped(
            &uv_ring_cell_path(cell_x, cell_z, slab_depth),
            cell_local,
            self.camera.position.anchor.depth(),
        );
    }

    pub(super) fn exit_uv_ring_cell_if_needed(&mut self, previous_position: WorldPos) {
        let Some(root) = self.world.library.get(self.world.root) else {
            return;
        };
        let crate::world::tree::NodeKind::UvRing { dims, slab_depth } = root.kind else {
            return;
        };
        if dims[0] == 0 || dims[1] != 1 || dims[2] == 0 {
            return;
        }
        let Some((previous_cell_x, previous_cell_z)) =
            uv_ring_cell_coords_from_path(&previous_position.anchor, dims, slab_depth)
        else {
            return;
        };
        if uv_ring_cell_coords_from_path(&self.camera.position.anchor, dims, slab_depth).is_some() {
            return;
        }

        let previous_cell_path = uv_ring_cell_path(previous_cell_x, previous_cell_z, slab_depth);
        let mut cell_local = self.camera.position.in_frame_rot(
            &self.world.library,
            self.world.root,
            &previous_cell_path,
        );
        let mut cell_x = previous_cell_x as i32;
        let mut cell_z = previous_cell_z as i32;
        while cell_local[0] < 0.0 {
            cell_local[0] += WORLD_SIZE;
            cell_x -= 1;
        }
        while cell_local[0] >= WORLD_SIZE {
            cell_local[0] -= WORLD_SIZE;
            cell_x += 1;
        }
        while cell_local[2] < 0.0 && cell_z > 0 {
            cell_local[2] += WORLD_SIZE;
            cell_z -= 1;
        }
        while cell_local[2] >= WORLD_SIZE && cell_z + 1 < dims[2] as i32 {
            cell_local[2] -= WORLD_SIZE;
            cell_z += 1;
        }
        let cell_x = cell_x.rem_euclid(dims[0] as i32) as u32;
        let cell_z = cell_z as u32;
        self.camera.position = world_pos_from_frame_local_unclamped(
            &uv_ring_cell_path(cell_x, cell_z, slab_depth),
            cell_local,
            self.camera.position.anchor.depth(),
        );
    }

    fn camera_root_to_uv_ring_cell_local(
        &self,
        dims: [u32; 3],
        slab_depth: u8,
        cell_x: u32,
        cell_z: u32,
    ) -> [f32; 3] {
        let root_cam =
            self.camera
                .position
                .in_frame_rot(&self.world.library, self.world.root, &Path::root());
        uv_ring_cell_frame(dims, slab_depth, cell_x, cell_z).point_to_local(root_cam)
    }

    pub(super) fn continuous_uv_ring_cell_frame(
        &self,
        dims: [u32; 3],
        slab_depth: u8,
        cell_x: u32,
        cell_z: u32,
    ) -> UvRingCellFrame {
        let cell_path = uv_ring_cell_path(cell_x, cell_z, slab_depth);
        let cell_local =
            self.camera
                .position
                .in_frame_rot(&self.world.library, self.world.root, &cell_path);
        uv_ring_cell_frame_at_local_x(dims, slab_depth, cell_x, cell_z, cell_local[0])
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct UvRingCellFrame {
    origin: [f32; 3],
    pub(super) tangent: [f32; 3],
    pub(super) radial: [f32; 3],
    pub(super) up: [f32; 3],
    scale: f32,
}

impl UvRingCellFrame {
    fn point_to_local(self, p: [f32; 3]) -> [f32; 3] {
        let d = crate::world::sdf::sub(p, self.origin);
        [
            crate::world::sdf::dot(self.tangent, d) * self.scale + 1.5,
            crate::world::sdf::dot(self.radial, d) * self.scale + 1.5,
            crate::world::sdf::dot(self.up, d) * self.scale + 1.5,
        ]
    }

    pub(super) fn dir_to_local(self, d: [f32; 3]) -> [f32; 3] {
        [
            crate::world::sdf::dot(self.tangent, d) * self.scale,
            crate::world::sdf::dot(self.radial, d) * self.scale,
            crate::world::sdf::dot(self.up, d) * self.scale,
        ]
    }

    pub(super) fn point_to_ring_world(self, p: [f32; 3]) -> [f32; 3] {
        let radial_offset = (p[1] - 1.5) / self.scale;
        let up_offset = (p[2] - 1.5) / self.scale;
        [
            self.origin[0] + self.radial[0] * radial_offset + self.up[0] * up_offset,
            self.origin[1] + self.radial[1] * radial_offset + self.up[1] * up_offset,
            self.origin[2] + self.radial[2] * radial_offset + self.up[2] * up_offset,
        ]
    }
}

pub(super) fn uv_ring_cell_frame(
    dims: [u32; 3],
    _slab_depth: u8,
    cell_x: u32,
    cell_z: u32,
) -> UvRingCellFrame {
    let angle_step = 2.0 * std::f32::consts::PI / dims[0] as f32;
    let angle = -std::f32::consts::PI + (cell_x as f32 + 0.5) * angle_step;
    uv_ring_cell_frame_at_angle(dims, angle, cell_z)
}

pub(super) fn uv_ring_cell_frame_at_local_x(
    dims: [u32; 3],
    _slab_depth: u8,
    cell_x: u32,
    cell_z: u32,
    local_x: f32,
) -> UvRingCellFrame {
    let angle_step = 2.0 * std::f32::consts::PI / dims[0] as f32;
    let cell_center_angle = -std::f32::consts::PI + (cell_x as f32 + 0.5) * angle_step;
    let angle = cell_center_angle + ((local_x - 1.5) / WORLD_SIZE) * angle_step;
    uv_ring_cell_frame_at_angle(dims, angle, cell_z)
}

fn uv_ring_cell_frame_at_angle(dims: [u32; 3], angle: f32, cell_z: u32) -> UvRingCellFrame {
    let body_size = 3.0_f32;
    let center = [body_size * 0.5; 3];
    let radius = body_size * 0.38;
    let side = uv_ring_cell_side(dims);
    let height = side * dims[2].max(1) as f32;
    let height_lo = center[1] - height * 0.5;
    let y = height_lo + (cell_z as f32 + 0.5) * side;
    let (sa, ca) = angle.sin_cos();
    let radial = [ca, 0.0, sa];
    let tangent = [-sa, 0.0, ca];
    let up = [0.0, 1.0, 0.0];
    let origin = [
        center[0] + radial[0] * radius,
        y,
        center[2] + radial[2] * radius,
    ];
    UvRingCellFrame {
        origin,
        tangent,
        radial,
        up,
        scale: 3.0 / side,
    }
}

fn uv_ring_cell_side(dims: [u32; 3]) -> f32 {
    let body_size = 3.0_f32;
    let radius = body_size * 0.38;
    ((2.0 * std::f32::consts::PI * radius / dims[0].max(1) as f32) * 0.95).max(body_size / 27.0)
}

pub(super) fn uv_ring_cell_path(cell_x: u32, cell_z: u32, slab_depth: u8) -> Path {
    let mut path = Path::root();
    let mut cells_per_slot = 1u32;
    for _ in 1..slab_depth {
        cells_per_slot *= 3;
    }
    for _ in 0..slab_depth {
        let sx = (cell_x / cells_per_slot) % 3;
        let sz = (cell_z / cells_per_slot) % 3;
        let slot = crate::world::tree::slot_index(sx as usize, 0, sz as usize) as u8;
        path.push(slot);
        cells_per_slot = (cells_per_slot / 3).max(1);
    }
    path
}

pub(super) fn cartesian_dir_in_path(mut d: [f32; 3], path: &Path) -> [f32; 3] {
    for _ in 0..path.depth() as usize {
        d = [d[0] * 3.0, d[1] * 3.0, d[2] * 3.0];
    }
    d
}

fn world_pos_from_frame_local_unclamped(
    frame: &Path,
    mut local: [f32; 3],
    anchor_depth: u8,
) -> WorldPos {
    let mut anchor = *frame;
    while anchor.depth() < anchor_depth {
        let sx = local[0].floor().clamp(0.0, 2.0) as usize;
        let sy = local[1].floor().clamp(0.0, 2.0) as usize;
        let sz = local[2].floor().clamp(0.0, 2.0) as usize;
        anchor.push(crate::world::tree::slot_index(sx, sy, sz) as u8);
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

pub(super) fn uv_ring_cell_dir_to_stick_root(dims: [u32; 3], cell_dir: [f32; 3]) -> [f32; 3] {
    let cell_size = crate::world::anchor::WORLD_SIZE / dims[0] as f32;
    let scale = cell_size / crate::world::anchor::WORLD_SIZE;
    [
        cell_dir[0] * scale,
        cell_dir[1] * scale,
        cell_dir[2] * scale,
    ]
}

pub(super) fn uv_ring_cell_coords_from_path(
    path: &Path,
    dims: [u32; 3],
    slab_depth: u8,
) -> Option<(u32, u32)> {
    if path.depth() < slab_depth {
        return None;
    }
    let mut x = 0u32;
    let mut z = 0u32;
    for k in 0..slab_depth as usize {
        let (sx, sy, sz) = crate::world::tree::slot_coords(path.slot(k) as usize);
        if sy != 0 {
            return None;
        }
        x = x * 3 + sx as u32;
        z = z * 3 + sz as u32;
    }
    if x >= dims[0] || z >= dims[2].max(1) {
        return None;
    }
    Some((x, z))
}

fn uv_ring_cell_local_is_near_ring_slab(p: [f32; 3], slab_depth: u8) -> bool {
    const ENTRY_DEPTH: u8 = 7;
    let extra_depth = ENTRY_DEPTH.saturating_sub(slab_depth) as i32;
    let entry_margin = 3.0 / 3.0_f32.powi(extra_depth);
    p.iter().all(|v| v.is_finite())
        && (-1.5..4.5).contains(&p[0])
        && (-entry_margin..(3.0 + entry_margin)).contains(&p[1])
        && (-entry_margin..(3.0 + entry_margin)).contains(&p[2])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_vec3_close(a: [f32; 3], b: [f32; 3]) {
        for i in 0..3 {
            assert!((a[i] - b[i]).abs() < 1e-5, "{a:?} != {b:?}");
        }
    }

    #[test]
    fn continuous_cell_frame_matches_at_shared_ring_boundary() {
        let dims = [27, 1, 1];
        let slab_depth = 3;
        let left_cell = uv_ring_cell_frame_at_local_x(dims, slab_depth, 20, 0, 0.0);
        let right_cell = uv_ring_cell_frame_at_local_x(dims, slab_depth, 19, 0, WORLD_SIZE);

        assert_vec3_close(left_cell.tangent, right_cell.tangent);
        assert_vec3_close(left_cell.radial, right_cell.radial);
        assert_vec3_close(left_cell.up, right_cell.up);
    }

    #[test]
    fn curved_ring_position_matches_at_shared_ring_boundary() {
        let dims = [27, 1, 1];
        let slab_depth = 3;
        let left_cell = uv_ring_cell_frame_at_local_x(dims, slab_depth, 20, 0, 0.0);
        let right_cell = uv_ring_cell_frame_at_local_x(dims, slab_depth, 19, 0, WORLD_SIZE);
        let left = left_cell.point_to_ring_world([0.0, 1.48, 2.91]);
        let right = right_cell.point_to_ring_world([WORLD_SIZE, 1.48, 2.91]);

        assert_vec3_close(left, right);
    }

    #[test]
    fn cell_path_round_trips_x_and_z() {
        let dims = [27, 1, 2];
        let slab_depth = 3;
        let path = uv_ring_cell_path(21, 1, slab_depth);
        assert_eq!(
            uv_ring_cell_coords_from_path(&path, dims, slab_depth),
            Some((21, 1))
        );
    }

    #[test]
    fn cell_path_rejects_z_outside_ring_height() {
        let dims = [27, 1, 2];
        let slab_depth = 3;
        let path = uv_ring_cell_path(21, 2, slab_depth);
        assert_eq!(uv_ring_cell_coords_from_path(&path, dims, slab_depth), None);
    }

    #[test]
    fn one_layer_vertical_exit_keeps_boundary_cell_with_outside_local_z() {
        let dims = [27, 1, 1];
        let slab_depth = 3;
        let pos = world_pos_from_frame_local_unclamped(
            &uv_ring_cell_path(13, 0, slab_depth),
            [1.5, 1.5, WORLD_SIZE + 0.25],
            5,
        );
        assert_eq!(
            uv_ring_cell_coords_from_path(&pos.anchor, dims, slab_depth),
            Some((13, 0)),
        );
        let local = pos.in_frame(&uv_ring_cell_path(13, 0, slab_depth));
        assert!(local[2] > WORLD_SIZE, "local should remain above the only row: {local:?}");
    }

    #[test]
    fn two_layer_vertical_exit_can_enter_neighbor_row() {
        let dims = [27, 1, 2];
        let slab_depth = 3;
        let mut cell_local = [1.5, 1.5, WORLD_SIZE + 0.25];
        let mut cell_z = 0i32;
        while cell_local[2] >= WORLD_SIZE && cell_z + 1 < dims[2] as i32 {
            cell_local[2] -= WORLD_SIZE;
            cell_z += 1;
        }
        let pos = world_pos_from_frame_local_unclamped(
            &uv_ring_cell_path(13, cell_z as u32, slab_depth),
            cell_local,
            5,
        );
        assert_eq!(
            uv_ring_cell_coords_from_path(&pos.anchor, dims, slab_depth),
            Some((13, 1)),
        );
    }

    #[test]
    fn two_layer_vertical_exit_past_top_keeps_top_row_with_outside_local_z() {
        let dims = [27, 1, 2];
        let slab_depth = 3;
        let mut cell_local = [1.5, 1.5, WORLD_SIZE + 0.25];
        let mut cell_z = 1i32;
        while cell_local[2] >= WORLD_SIZE && cell_z + 1 < dims[2] as i32 {
            cell_local[2] -= WORLD_SIZE;
            cell_z += 1;
        }
        let pos = world_pos_from_frame_local_unclamped(
            &uv_ring_cell_path(13, cell_z as u32, slab_depth),
            cell_local,
            5,
        );
        assert_eq!(
            uv_ring_cell_coords_from_path(&pos.anchor, dims, slab_depth),
            Some((13, 1)),
        );
        let local = pos.in_frame(&uv_ring_cell_path(13, 1, slab_depth));
        assert!(local[2] > WORLD_SIZE, "local should remain above top row: {local:?}");
    }
}
