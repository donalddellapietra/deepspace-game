pub mod mesher;

use bevy::prelude::*;

use crate::block::BlockType;

/// One sub-mesh per block type present in a baked volume.
#[derive(Clone)]
pub struct BakedSubMesh {
    pub mesh: Handle<Mesh>,
    pub block_type: BlockType,
}
