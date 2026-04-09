use bevy::{
    asset::RenderAssetUsages,
    mesh::Indices,
    prelude::*,
    render::render_resource::PrimitiveTopology,
};

use super::terrain::TerrainGenerator;

/// Number of quads per chunk side. 32x32 = 1024 quads = 2048 triangles.
pub const CHUNK_SIZE: usize = 32;

/// World-space size of one chunk side in units.
pub const CHUNK_WORLD_SIZE: f32 = 32.0;

#[derive(Component)]
pub struct Chunk {
    pub coord: IVec2,
}

/// Build a heightmap mesh for a chunk at the given coordinate.
/// Heights are sampled from the terrain generator using world-space positions.
/// Normals are computed from finite differences for smooth cross-chunk lighting.
pub fn generate_chunk_mesh(coord: IVec2, terrain: &TerrainGenerator) -> Mesh {
    let verts_per_side = CHUNK_SIZE + 1;
    let vertex_count = verts_per_side * verts_per_side;
    let index_count = CHUNK_SIZE * CHUNK_SIZE * 6;
    let step = CHUNK_WORLD_SIZE / CHUNK_SIZE as f32;

    let world_x = coord.x as f32 * CHUNK_WORLD_SIZE;
    let world_z = coord.y as f32 * CHUNK_WORLD_SIZE;

    let mut positions = Vec::with_capacity(vertex_count);
    let mut normals = Vec::with_capacity(vertex_count);
    let mut uvs = Vec::with_capacity(vertex_count);

    for gz in 0..verts_per_side {
        for gx in 0..verts_per_side {
            let local_x = gx as f32 * step;
            let local_z = gz as f32 * step;
            let wx = world_x + local_x;
            let wz = world_z + local_z;
            let height = terrain.height_at(wx, wz);

            positions.push([local_x, height, local_z]);
            uvs.push([gx as f32 / CHUNK_SIZE as f32, gz as f32 / CHUNK_SIZE as f32]);

            // Normals via central differences of the height function
            let h_left = terrain.height_at(wx - step, wz);
            let h_right = terrain.height_at(wx + step, wz);
            let h_down = terrain.height_at(wx, wz - step);
            let h_up = terrain.height_at(wx, wz + step);
            let normal = Vec3::new(h_left - h_right, 2.0 * step, h_down - h_up).normalize();
            normals.push([normal.x, normal.y, normal.z]);
        }
    }

    // Two triangles per quad, counter-clockwise winding
    let mut indices = Vec::with_capacity(index_count);
    let row = verts_per_side as u32;
    for gz in 0..CHUNK_SIZE as u32 {
        for gx in 0..CHUNK_SIZE as u32 {
            let i = gz * row + gx;
            indices.push(i);
            indices.push(i + row);
            indices.push(i + 1);
            indices.push(i + 1);
            indices.push(i + row);
            indices.push(i + row + 1);
        }
    }

    Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    )
    .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, positions)
    .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, normals)
    .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, uvs)
    .with_inserted_indices(Indices::U32(indices))
}
