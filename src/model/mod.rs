pub mod mesher;

use bevy::prelude::*;

use crate::block::{BlockType, MODEL_SIZE};

/// Identifies a model in the registry.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ModelId(pub usize);

/// One sub-mesh per block type present in the model.
#[derive(Clone)]
pub struct BakedSubMesh {
    pub mesh: Handle<Mesh>,
    pub block_type: BlockType,
}

/// A saved model: raw voxel data + pre-baked rendering data.
#[derive(Clone)]
pub struct VoxelModel {
    pub name: String,
    pub blocks: [[[Option<BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
    pub baked: Vec<BakedSubMesh>,
}

/// All saved models.
#[derive(Resource, Default)]
pub struct ModelRegistry {
    pub models: Vec<VoxelModel>,
}

impl ModelRegistry {
    pub fn register(&mut self, model: VoxelModel) -> ModelId {
        let id = ModelId(self.models.len());
        self.models.push(model);
        id
    }

    pub fn get(&self, id: ModelId) -> Option<&VoxelModel> {
        self.models.get(id.0)
    }

    pub fn get_mut(&mut self, id: ModelId) -> Option<&mut VoxelModel> {
        self.models.get_mut(id.0)
    }

    /// Re-bake a model's mesh after edits.
    pub fn rebake(&mut self, id: ModelId, meshes: &mut Assets<Mesh>) {
        if let Some(model) = self.models.get_mut(id.0) {
            model.baked = mesher::bake_model(&model.blocks, meshes);
        }
    }
}

pub struct ModelPlugin;

impl Plugin for ModelPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ModelRegistry>();
    }
}
