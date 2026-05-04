use crate::world::anchor::{Path, WORLD_SIZE};

pub const UV_RING_BODY_SIZE: f32 = 3.0;
pub const UV_RING_BODY_CENTER: [f32; 3] = [1.5, 1.5, 1.5];

#[derive(Clone, Copy, Debug)]
pub struct UvRingCellFrame {
    origin: [f32; 3],
    pub tangent: [f32; 3],
    pub radial: [f32; 3],
    pub up: [f32; 3],
    scale: f32,
}

impl UvRingCellFrame {
    pub fn point_to_local(self, p: [f32; 3]) -> [f32; 3] {
        let d = crate::world::sdf::sub(p, self.origin);
        [
            crate::world::sdf::dot(self.tangent, d) * self.scale + 1.5,
            crate::world::sdf::dot(self.radial, d) * self.scale + 1.5,
            crate::world::sdf::dot(self.up, d) * self.scale + 1.5,
        ]
    }

    pub fn dir_to_local(self, d: [f32; 3]) -> [f32; 3] {
        [
            crate::world::sdf::dot(self.tangent, d) * self.scale,
            crate::world::sdf::dot(self.radial, d) * self.scale,
            crate::world::sdf::dot(self.up, d) * self.scale,
        ]
    }

    pub fn point_to_ring_world(self, p: [f32; 3]) -> [f32; 3] {
        let radial_offset = (p[1] - 1.5) / self.scale;
        let up_offset = (p[2] - 1.5) / self.scale;
        [
            self.origin[0] + self.radial[0] * radial_offset + self.up[0] * up_offset,
            self.origin[1] + self.radial[1] * radial_offset + self.up[1] * up_offset,
            self.origin[2] + self.radial[2] * radial_offset + self.up[2] * up_offset,
        ]
    }

    pub fn scale(self) -> f32 {
        self.scale
    }
}

pub fn uv_ring_angle_step(dims: [u32; 3]) -> f32 {
    2.0 * std::f32::consts::PI / dims[0].max(1) as f32
}

pub fn uv_ring_cell_frame(
    dims: [u32; 3],
    _slab_depth: u8,
    cell_x: u32,
    cell_y: u32,
    cell_z: u32,
) -> UvRingCellFrame {
    let angle = -std::f32::consts::PI + (cell_x as f32 + 0.5) * uv_ring_angle_step(dims);
    uv_ring_cell_frame_at_angle(dims, angle, cell_y, cell_z)
}

pub fn uv_ring_cell_frame_at_local_x(
    dims: [u32; 3],
    _slab_depth: u8,
    cell_x: u32,
    cell_y: u32,
    cell_z: u32,
    local_x: f32,
) -> UvRingCellFrame {
    let angle_step = uv_ring_angle_step(dims);
    let cell_center_angle = -std::f32::consts::PI + (cell_x as f32 + 0.5) * angle_step;
    let angle = cell_center_angle + ((local_x - 1.5) / WORLD_SIZE) * angle_step;
    uv_ring_cell_frame_at_angle(dims, angle, cell_y, cell_z)
}

pub fn uv_ring_cell_frame_at_angle(
    dims: [u32; 3],
    angle: f32,
    cell_y: u32,
    cell_z: u32,
) -> UvRingCellFrame {
    let side = uv_ring_cell_side(dims);
    let radius = uv_ring_radial_lo(dims) + (cell_y as f32 + 0.5) * side;
    let y = uv_ring_height_lo(dims) + (cell_z as f32 + 0.5) * side;
    let (sa, ca) = angle.sin_cos();
    let radial = [ca, 0.0, sa];
    let tangent = [-sa, 0.0, ca];
    let up = [0.0, 1.0, 0.0];
    let origin = [
        UV_RING_BODY_CENTER[0] + radial[0] * radius,
        y,
        UV_RING_BODY_CENTER[2] + radial[2] * radius,
    ];
    UvRingCellFrame {
        origin,
        tangent,
        radial,
        up,
        scale: UV_RING_BODY_SIZE / side,
    }
}

pub fn uv_ring_cell_side(dims: [u32; 3]) -> f32 {
    let radius = UV_RING_BODY_SIZE * 0.38;
    ((2.0 * std::f32::consts::PI * radius / dims[0].max(1) as f32) * 0.95)
        .max(UV_RING_BODY_SIZE / 27.0)
}

pub fn uv_ring_radial_lo(dims: [u32; 3]) -> f32 {
    let radius = UV_RING_BODY_SIZE * 0.38;
    radius - uv_ring_cell_side(dims) * dims[1].max(1) as f32 * 0.5
}

pub fn uv_ring_height_lo(dims: [u32; 3]) -> f32 {
    UV_RING_BODY_CENTER[1] - uv_ring_cell_side(dims) * dims[2].max(1) as f32 * 0.5
}

pub fn uv_ring_cell_path(cell_x: u32, cell_y: u32, cell_z: u32, slab_depth: u8) -> Path {
    let mut path = Path::root();
    let mut cells_per_slot = 1u32;
    for _ in 1..slab_depth {
        cells_per_slot *= 3;
    }
    for _ in 0..slab_depth {
        let sx = (cell_x / cells_per_slot) % 3;
        let sy = (cell_y / cells_per_slot) % 3;
        let sz = (cell_z / cells_per_slot) % 3;
        let slot = crate::world::tree::slot_index(sx as usize, sy as usize, sz as usize) as u8;
        path.push(slot);
        cells_per_slot = (cells_per_slot / 3).max(1);
    }
    path
}

pub fn cartesian_dir_in_path(mut d: [f32; 3], path: &Path) -> [f32; 3] {
    for _ in 0..path.depth() as usize {
        d = [d[0] * 3.0, d[1] * 3.0, d[2] * 3.0];
    }
    d
}

pub fn uv_ring_cell_dir_to_stick_root(dims: [u32; 3], cell_dir: [f32; 3]) -> [f32; 3] {
    let cell_size = WORLD_SIZE / dims[0] as f32;
    let scale = cell_size / WORLD_SIZE;
    [
        cell_dir[0] * scale,
        cell_dir[1] * scale,
        cell_dir[2] * scale,
    ]
}

pub fn uv_ring_cell_coords_from_path(
    path: &Path,
    dims: [u32; 3],
    slab_depth: u8,
) -> Option<(u32, u32, u32)> {
    if path.depth() < slab_depth {
        return None;
    }
    let mut x = 0u32;
    let mut y = 0u32;
    let mut z = 0u32;
    for k in 0..slab_depth as usize {
        let (sx, sy, sz) = crate::world::tree::slot_coords(path.slot(k) as usize);
        x = x * 3 + sx as u32;
        y = y * 3 + sy as u32;
        z = z * 3 + sz as u32;
    }
    if x >= dims[0] || y >= dims[1].max(1) || z >= dims[2].max(1) {
        return None;
    }
    Some((x, y, z))
}
