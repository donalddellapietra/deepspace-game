use bevy::{
    asset::RenderAssetUsages,
    mesh::Indices,
    prelude::*,
    render::render_resource::PrimitiveTopology,
};

use crate::block::BlockType;

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

/// AO brightness curve: 0 = fully occluded, 3 = no occlusion.
const AO_CURVE: [f32; 4] = [0.6, 0.75, 0.9, 1.0];

/// Compute per-vertex ambient occlusion for a quad face.
///
/// For each of the 4 vertices on the face, we check the 3 neighboring voxels
/// diagonal to that corner (on the outside of the face). The AO level is:
///   - If both side neighbors are solid: ao = 0 (darkest)
///   - Otherwise: ao = 3 - (side1 + side2 + corner)
fn compute_face_ao<F: Fn(i32, i32, i32) -> bool>(
    bx: i32,
    by: i32,
    bz: i32,
    normal: IVec3,
    quad: &[Vec3; 4],
    is_solid: &F,
) -> [u8; 4] {
    let sample_base = IVec3::new(bx, by, bz) + normal;
    let mut ao = [3u8; 4];

    for (i, vert) in quad.iter().enumerate() {
        // For each vertex, determine the two tangent-axis directions.
        // On the face plane (excluding the normal axis), each vertex coordinate
        // is either 0 or 1. We map 0 -> -1, 1 -> +1 to get the side direction.
        let mut sides = [IVec3::ZERO; 2];
        let mut idx = 0;
        for axis in 0..3 {
            if normal[axis] != 0 {
                continue;
            }
            let v = vert[axis] as i32; // 0 or 1
            let dir = if v == 0 { -1 } else { 1 };
            sides[idx][axis] = dir;
            idx += 1;
        }

        let s1 = is_solid(
            (sample_base + sides[0]).x,
            (sample_base + sides[0]).y,
            (sample_base + sides[0]).z,
        ) as u8;
        let s2 = is_solid(
            (sample_base + sides[1]).x,
            (sample_base + sides[1]).y,
            (sample_base + sides[1]).z,
        ) as u8;
        let c = if s1 == 1 && s2 == 1 {
            1 // Both sides solid → corner is fully occluded regardless
        } else {
            is_solid(
                (sample_base + sides[0] + sides[1]).x,
                (sample_base + sides[0] + sides[1]).y,
                (sample_base + sides[0] + sides[1]).z,
            ) as u8
        };

        ao[i] = 3 - (s1 + s2 + c);
    }

    ao
}

/// Bake any cubic voxel volume of `size^3` cells into per-block-type sub-meshes
/// with face culling. The data source is given as a closure so the caller can
/// feed it any voxel grid.
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

                    let ao = compute_face_ao(x, y, z, dir, quad, &|ax, ay, az| {
                        get(ax, ay, az).is_some()
                    });
                    let offset = Vec3::new(x as f32, y as f32, z as f32);
                    collector.add_quad(quad, normal, offset, ao);
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

#[derive(Default)]
struct FaceCollector {
    positions: Vec<[f32; 3]>,
    normals: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    indices: Vec<u32>,
}

impl FaceCollector {
    fn add_quad(&mut self, quad: &[Vec3; 4], normal: Vec3, offset: Vec3, ao: [u8; 4]) {
        let base = self.positions.len() as u32;

        for (i, &vert) in quad.iter().enumerate() {
            self.positions.push((vert + offset).to_array());
            self.normals.push(normal.to_array());
            let brightness = AO_CURVE[ao[i] as usize];
            self.colors.push([brightness, brightness, brightness, 1.0]);
        }

        // Flip the quad diagonal when needed to avoid AO interpolation artifacts.
        // Split across the brighter diagonal so the dark corner doesn't bleed.
        if ao[0] + ao[2] > ao[1] + ao[3] {
            // Standard winding: triangles (0,1,2) and (0,2,3)
            self.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        } else {
            // Flipped winding: triangles (1,2,3) and (1,3,0)
            self.indices.extend_from_slice(&[
                base + 1,
                base + 2,
                base + 3,
                base + 1,
                base + 3,
                base,
            ]);
        }
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
        .with_inserted_attribute(Mesh::ATTRIBUTE_COLOR, self.colors)
        .with_inserted_indices(Indices::U32(self.indices))
    }
}
