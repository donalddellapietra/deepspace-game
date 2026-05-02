//! Sphere/wrapped-planet helpers shared between worldgen, the CPU
//! raycaster, and tests. Geometry conventions (lon/lat → unit
//! position) match the WGSL marcher's `sphere_uv_in_cell`.

pub mod tangent;
pub mod uv_lens;
