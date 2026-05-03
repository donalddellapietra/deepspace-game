//! UV-sphere range helpers for paths under `NodeKind::WrappedPlane`.
//!
//! `WrappedPlane` remains the canonical storage layout. These helpers
//! only interpret slots below that node as recursive `(lon, r, lat)`
//! parameter-space thirds so render/raycast code can project the same
//! tree path into a sphere-local frame.

use std::f32::consts::PI;

use crate::world::anchor::{Path, WorldPos};
use crate::world::tree::{slot_coords, Child, NodeId, NodeKind, NodeLibrary};

pub const SPHERE_SHELL_THICKNESS_FRAC: f32 = 0.25;
pub const DEFAULT_SPHERE_LAT_MAX: f32 = 1.26;
pub const SPHERE_BODY_SIZE: f32 = 3.0;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereRange {
    pub lon_lo: f32,
    pub lon_hi: f32,
    pub lat_lo: f32,
    pub lat_hi: f32,
    pub r_lo: f32,
    pub r_hi: f32,
    /// Number of slots from world root to the `WrappedPlane` node.
    pub wp_path_depth: u8,
    pub dims: [u32; 3],
    pub slab_depth: u8,
}

impl SphereRange {
    #[inline]
    pub fn lon_center(self) -> f32 {
        (self.lon_lo + self.lon_hi) * 0.5
    }

    #[inline]
    pub fn lat_center(self) -> f32 {
        (self.lat_lo + self.lat_hi) * 0.5
    }

    #[inline]
    pub fn r_center(self) -> f32 {
        (self.r_lo + self.r_hi) * 0.5
    }

    #[inline]
    pub fn lon_extent(self) -> f32 {
        self.lon_hi - self.lon_lo
    }

    #[inline]
    pub fn lat_extent(self) -> f32 {
        self.lat_hi - self.lat_lo
    }

    #[inline]
    pub fn r_extent(self) -> f32 {
        self.r_hi - self.r_lo
    }

    #[inline]
    pub fn slab_footprint(self) -> [f32; 3] {
        let subgrid = 3.0_f32.powi(self.slab_depth as i32);
        [
            SPHERE_BODY_SIZE * self.dims[0] as f32 / subgrid,
            SPHERE_BODY_SIZE * self.dims[1] as f32 / subgrid,
            SPHERE_BODY_SIZE * self.dims[2] as f32 / subgrid,
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereSubFrameCamera {
    pub origin: [f32; 3],
    pub forward: [f32; 3],
    pub right: [f32; 3],
    pub up: [f32; 3],
    pub r_center: f32,
    pub lon_center: f32,
    pub lat_center: f32,
}

pub fn sphere_radius(body_size: f32) -> f32 {
    body_size / (2.0 * PI)
}

pub fn sphere_shell_bounds(body_size: f32) -> (f32, f32) {
    let r_outer = sphere_radius(body_size);
    let r_inner = r_outer - r_outer * SPHERE_SHELL_THICKNESS_FRAC;
    (r_inner, r_outer)
}

pub(crate) fn wrapped_plane_local_to_params(local: [f32; 3], range: SphereRange) -> [f32; 3] {
    let footprint = range.slab_footprint();
    let safe = |v: f32| v.max(1e-6);
    let (r_inner, r_outer) = sphere_shell_bounds(SPHERE_BODY_SIZE);
    let shell = r_outer - r_inner;
    [
        -PI + (local[0] / safe(footprint[0])) * (2.0 * PI),
        r_inner + (local[1] / safe(footprint[1])) * shell,
        -DEFAULT_SPHERE_LAT_MAX
            + (local[2] / safe(footprint[2])) * (2.0 * DEFAULT_SPHERE_LAT_MAX),
    ]
}

pub(crate) fn wrapped_plane_dir_to_param_delta(dir: [f32; 3], range: SphereRange) -> [f32; 3] {
    let footprint = range.slab_footprint();
    let safe = |v: f32| v.max(1e-6);
    let (r_inner, r_outer) = sphere_shell_bounds(SPHERE_BODY_SIZE);
    let shell = r_outer - r_inner;
    [
        dir[0] * (2.0 * PI) / safe(footprint[0]),
        dir[1] * shell / safe(footprint[1]),
        dir[2] * (2.0 * DEFAULT_SPHERE_LAT_MAX) / safe(footprint[2]),
    ]
}

pub(crate) fn sphere_basis(lon: f32, lat: f32) -> ([f32; 3], [f32; 3], [f32; 3]) {
    let (sl, cl) = lat.sin_cos();
    let (so, co) = lon.sin_cos();
    let radial = [cl * co, sl, cl * so];
    let east = [-so, 0.0, co];
    let north = [-sl * co, cl, -sl * so];
    (east, north, radial)
}

pub(crate) fn params_to_sphere(params: [f32; 3]) -> [f32; 3] {
    let (_east, _north, radial) = sphere_basis(params[0], params[2]);
    [
        radial[0] * params[1],
        radial[1] * params[1],
        radial[2] * params[1],
    ]
}

fn param_delta_to_sphere_delta(params: [f32; 3], delta: [f32; 3]) -> [f32; 3] {
    let (east, north, radial) = sphere_basis(params[0], params[2]);
    let r = params[1];
    let lat = params[2];
    let d_lon = r * lat.cos().max(1e-4) * delta[0];
    let d_r = delta[1];
    let d_lat = r * delta[2];
    [
        east[0] * d_lon + radial[0] * d_r + north[0] * d_lat,
        east[1] * d_lon + radial[1] * d_r + north[1] * d_lat,
        east[2] * d_lon + radial[2] * d_r + north[2] * d_lat,
    ]
}

pub(crate) fn project_sphere_point_to_subframe(point: [f32; 3], range: SphereRange) -> [f32; 3] {
    let center = params_to_sphere([range.lon_center(), range.r_center(), range.lat_center()]);
    let (east, north, radial) = sphere_basis(range.lon_center(), range.lat_center());
    let delta = [
        point[0] - center[0],
        point[1] - center[1],
        point[2] - center[2],
    ];
    [
        dot(delta, east),
        dot(delta, north),
        dot(delta, radial),
    ]
}

pub(crate) fn project_sphere_vec_to_subframe(vec: [f32; 3], range: SphereRange) -> [f32; 3] {
    let (east, north, radial) = sphere_basis(range.lon_center(), range.lat_center());
    [dot(vec, east), dot(vec, north), dot(vec, radial)]
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Interpret `path` as a sphere parameter-space range once it enters
/// a `WrappedPlane` node. Returns `None` if the path never reaches one.
pub fn sphere_range_for_path(
    library: &NodeLibrary,
    world_root: NodeId,
    path: &Path,
    lat_max: f32,
) -> Option<SphereRange> {
    let (r_inner, r_outer) = sphere_shell_bounds(SPHERE_BODY_SIZE);

    let mut wp_dims = None;
    let mut wp_slab_depth = 0u8;
    let mut wp_path_depth = 0u8;

    if let Some(root_node) = library.get(world_root) {
        if let NodeKind::WrappedPlane { dims, slab_depth } = root_node.kind {
            wp_dims = Some(dims);
            wp_slab_depth = slab_depth;
        }
    }

    if wp_dims.is_none() {
        let mut node_id = world_root;
        for (i, &slot) in path.as_slice().iter().enumerate() {
            let node = library.get(node_id)?;
            match node.children[slot as usize] {
                Child::Node(child) => {
                    node_id = child;
                    if let Some(child_node) = library.get(child) {
                        if let NodeKind::WrappedPlane { dims, slab_depth } = child_node.kind {
                            wp_dims = Some(dims);
                            wp_slab_depth = slab_depth;
                            wp_path_depth = (i + 1) as u8;
                            break;
                        }
                    }
                }
                Child::Empty | Child::Block(_) | Child::EntityRef(_) => return None,
            }
        }
    }

    let dims = wp_dims?;
    let mut range = SphereRange {
        lon_lo: -PI,
        lon_hi: PI,
        lat_lo: -lat_max,
        lat_hi: lat_max,
        r_lo: r_inner,
        r_hi: r_outer,
        wp_path_depth,
        dims,
        slab_depth: wp_slab_depth,
    };

    for &slot in &path.as_slice()[wp_path_depth as usize..] {
        let (sx, sy, sz) = slot_coords(slot as usize);
        let lon_step = range.lon_extent() / 3.0;
        let r_step = range.r_extent() / 3.0;
        let lat_step = range.lat_extent() / 3.0;

        range.lon_lo += sx as f32 * lon_step;
        range.lon_hi = range.lon_lo + lon_step;
        range.r_lo += sy as f32 * r_step;
        range.r_hi = range.r_lo + r_step;
        range.lat_lo += sz as f32 * lat_step;
        range.lat_hi = range.lat_lo + lat_step;
    }

    Some(range)
}

/// Project the canonical camera into a sphere subframe. This is a pure
/// projection; callers must not write the result back into `WorldPos`.
pub fn camera_in_sphere_subframe(
    cam_pos: &WorldPos,
    cam_forward: [f32; 3],
    cam_right: [f32; 3],
    cam_up: [f32; 3],
    wp_path: &Path,
    range: SphereRange,
) -> SphereSubFrameCamera {
    let cam_wp = cam_pos.in_frame(wp_path);
    let cam_params = wrapped_plane_local_to_params(cam_wp, range);
    let project_dir = |v: [f32; 3]| {
        let param_delta = wrapped_plane_dir_to_param_delta(v, range);
        let sphere_delta = param_delta_to_sphere_delta(cam_params, param_delta);
        project_sphere_vec_to_subframe(sphere_delta, range)
    };

    SphereSubFrameCamera {
        origin: project_sphere_point_to_subframe(params_to_sphere(cam_params), range),
        forward: project_dir(cam_forward),
        right: project_dir(cam_right),
        up: project_dir(cam_up),
        r_center: range.r_center(),
        lon_center: range.lon_center(),
        lat_center: range.lat_center(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::anchor::WorldPos;
    use crate::world::bootstrap::wrapped_planet_world;
    use crate::world::tree::slot_index;

    fn wp_path(embedding_depth: u8) -> Path {
        let mut path = Path::root();
        for _ in 0..embedding_depth {
            path.push(slot_index(1, 1, 1) as u8);
        }
        path
    }

    #[test]
    fn range_is_full_body_at_wrapped_plane_root() {
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 1);
        let path = wp_path(2);
        let range =
            sphere_range_for_path(&world.library, world.root, &path, DEFAULT_SPHERE_LAT_MAX)
                .expect("wp path has sphere range");
        let (r_inner, r_outer) = sphere_shell_bounds(SPHERE_BODY_SIZE);

        assert_eq!(range.wp_path_depth, 2);
        assert_eq!(range.dims, [27, 2, 14]);
        assert_eq!(range.slab_depth, 3);
        assert!((range.lon_lo + PI).abs() < 1e-6);
        assert!((range.lon_hi - PI).abs() < 1e-6);
        assert!((range.lat_lo + DEFAULT_SPHERE_LAT_MAX).abs() < 1e-6);
        assert!((range.lat_hi - DEFAULT_SPHERE_LAT_MAX).abs() < 1e-6);
        assert!((range.r_lo - r_inner).abs() < 1e-6);
        assert!((range.r_hi - r_outer).abs() < 1e-6);
    }

    #[test]
    fn slot_below_wp_maps_to_lon_r_lat_thirds() {
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 1);
        let mut path = wp_path(2);
        path.push(slot_index(2, 1, 0) as u8);
        let range =
            sphere_range_for_path(&world.library, world.root, &path, DEFAULT_SPHERE_LAT_MAX)
                .expect("wp child path has sphere range");
        let (r_inner, r_outer) = sphere_shell_bounds(SPHERE_BODY_SIZE);
        let r_step = (r_outer - r_inner) / 3.0;
        let lat_step = 2.0 * DEFAULT_SPHERE_LAT_MAX / 3.0;

        assert!((range.lon_lo - (PI / 3.0)).abs() < 1e-6);
        assert!((range.lon_hi - PI).abs() < 1e-6);
        assert!((range.r_lo - (r_inner + r_step)).abs() < 1e-6);
        assert!((range.r_hi - (r_inner + 2.0 * r_step)).abs() < 1e-6);
        assert!((range.lat_lo + DEFAULT_SPHERE_LAT_MAX).abs() < 1e-6);
        assert!((range.lat_hi - (-DEFAULT_SPHERE_LAT_MAX + lat_step)).abs() < 1e-6);
    }

    #[test]
    fn camera_projection_is_centered_at_subframe_origin() {
        let world = wrapped_planet_world(2, [27, 2, 14], 3, 1);
        let mut path = wp_path(2);
        path.push(slot_index(1, 1, 1) as u8);
        let range =
            sphere_range_for_path(&world.library, world.root, &path, DEFAULT_SPHERE_LAT_MAX)
                .expect("wp child path has sphere range");
        let wp = wp_path(2);
        let footprint = range.slab_footprint();
        let center_wp = [footprint[0] * 0.5, footprint[1] * 0.5, footprint[2] * 0.5];
        let cam = WorldPos::from_frame_local(&wp, center_wp, wp.depth());
        let projected = camera_in_sphere_subframe(
            &cam,
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            &wp,
            range,
        );

        for v in projected.origin {
            assert!(
                v.abs() < 1e-5,
                "expected centered origin, got {projected:?}"
            );
        }
    }
}
