//! Sphere/wrapped-planet helpers shared between worldgen, the CPU
//! raycaster, and tests. Geometry conventions (lon/lat → unit
//! position) match the WGSL marcher's `sphere_uv_in_cell`.

pub mod range;
pub mod tangent;

pub use range::{
    camera_in_sphere_subframe, sphere_range_for_path, SphereRange, SphereSubFrameCamera,
    DEFAULT_SPHERE_LAT_MAX, SPHERE_SHELL_THICKNESS_FRAC,
};
