pub mod gltf;
mod models;
mod palette;

pub use models::*;

pub const MAX_STORAGE_BUFFER_BINDING_SIZE: u32 = 1024 * 1024 * 1024;

/// Voxel-model output extension: the game's custom sparse voxel format,
/// read by `src/import/vxs.rs`.
pub const MODEL_FILE_EXT: &str = "vxs";
