use bevy::{
    asset::RenderAssetUsages,
    mesh::Indices,
    prelude::*,
    render::render_resource::PrimitiveTopology,
};

use crate::block::{BlockType, MODEL_SIZE};

use super::BakedSubMesh;

/// Face definitions: (neighbor offset, quad vertices in CCW winding, normal).
/// Each quad verified: (V1-V0) × (V2-V0) == normal direction.
const FACES: [(IVec3, [Vec3; 4], Vec3); 6] = [
    // +X
    (IVec3::X, [
        Vec3::new(1.0, 0.0, 0.0), Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 1.0), Vec3::new(1.0, 0.0, 1.0),
    ], Vec3::X),
    // -X
    (IVec3::NEG_X, [
        Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 1.0),
        Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 0.0),
    ], Vec3::NEG_X),
    // +Y
    (IVec3::Y, [
        Vec3::new(0.0, 1.0, 1.0), Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, 1.0, 0.0), Vec3::new(0.0, 1.0, 0.0),
    ], Vec3::Y),
    // -Y
    (IVec3::NEG_Y, [
        Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 1.0), Vec3::new(0.0, 0.0, 1.0),
    ], Vec3::NEG_Y),
    // +Z
    (IVec3::Z, [
        Vec3::new(0.0, 0.0, 1.0), Vec3::new(1.0, 0.0, 1.0),
        Vec3::new(1.0, 1.0, 1.0), Vec3::new(0.0, 1.0, 1.0),
    ], Vec3::Z),
    // -Z
    (IVec3::NEG_Z, [
        Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0), Vec3::new(1.0, 1.0, 0.0),
    ], Vec3::NEG_Z),
];

/// Bake any cubic voxel volume of `size^3` cells into per-block-type sub-meshes
/// with face culling. The data source is given as a closure so the caller can
/// stitch together a FlatWorld, a 5x5x5 model array, or anything else.
pub fn bake_volume<F: Fn(i32, i32, i32) -> Option<BlockType>>(
    size: i32,
    get: F,
    meshes: &mut Assets<Mesh>,
) -> Vec<BakedSubMesh> {
    let mut groups: std::collections::HashMap<BlockType, FaceCollector> =
        std::collections::HashMap::new();

    for y in 0..size {
        for z in 0..size {
            for x in 0..size {
                let Some(block) = get(x, y, z) else { continue };
                let collector = groups.entry(block).or_default();

                for &(dir, ref quad, normal) in &FACES {
                    let nx = x + dir.x;
                    let ny = y + dir.y;
                    let nz = z + dir.z;
                    if get(nx, ny, nz).is_some() { continue; }

                    let offset = Vec3::new(x as f32, y as f32, z as f32);
                    collector.add_quad(quad, normal, offset);
                }
            }
        }
    }

    groups
        .into_iter()
        .map(|(block_type, collector)| BakedSubMesh {
            mesh: meshes.add(collector.build()),
            block_type,
        })
        .collect()
}

/// Bake a 5x5x5 voxel grid into per-block-type sub-meshes with face culling.
pub fn bake_model(
    blocks: &[[[Option<BlockType>; MODEL_SIZE]; MODEL_SIZE]; MODEL_SIZE],
    meshes: &mut Assets<Mesh>,
) -> Vec<BakedSubMesh> {
    let s = MODEL_SIZE as i32;
    bake_volume(
        s,
        |x, y, z| {
            if x < 0 || y < 0 || z < 0 || x >= s || y >= s || z >= s { return None; }
            blocks[y as usize][z as usize][x as usize]
        },
        meshes,
    )
}

#[derive(Default)]
struct FaceCollector {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    indices: Vec<u32>,
}

impl FaceCollector {
    fn add_quad(&mut self, quad: &[Vec3; 4], normal: Vec3, offset: Vec3) {
        let base = self.positions.len() as u32;

        for &vert in quad {
            self.positions.push((vert + offset).to_array());
            self.normals.push(normal.to_array());
        }

        self.indices.push(base);
        self.indices.push(base + 1);
        self.indices.push(base + 2);
        self.indices.push(base);
        self.indices.push(base + 2);
        self.indices.push(base + 3);
    }

    fn build(self) -> Mesh {
        if self.positions.is_empty() {
            return Mesh::new(
                PrimitiveTopology::TriangleList,
                RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
            );
        }

        Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_indices(Indices::U32(self.indices))
    }
}
