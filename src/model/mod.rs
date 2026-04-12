pub mod mesher;

use bevy::prelude::*;

/// One sub-mesh per voxel type present in a baked volume.
#[derive(Clone)]
pub struct BakedSubMesh {
    pub mesh: Handle<Mesh>,
    pub voxel: u8,
}
