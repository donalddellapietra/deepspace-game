use bevy::prelude::*;
use super::BlockType;

/// One StandardMaterial per block type, shared across all rendering.
#[derive(Resource)]
pub struct BlockMaterials {
    pub handles: [Handle<StandardMaterial>; 10],
}

impl BlockMaterials {
    pub fn get(&self, bt: BlockType) -> Handle<StandardMaterial> {
        self.handles[bt as usize].clone()
    }
}

pub fn init_block_materials(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let handles = BlockType::ALL.map(|bt| {
        materials.add(StandardMaterial {
            base_color: bt.color(),
            perceptual_roughness: bt.roughness(),
            metallic: bt.metallic(),
            alpha_mode: bt.alpha_mode(),
            ..default()
        })
    });
    commands.insert_resource(BlockMaterials { handles });
}
