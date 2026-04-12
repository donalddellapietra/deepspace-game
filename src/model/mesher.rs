use bevy::{
    asset::RenderAssetUsages,
    mesh::Indices,
    prelude::*,
    render::render_resource::PrimitiveTopology,
};

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
        let mut sides = [IVec3::ZERO; 2];
        let mut idx = 0;
        for axis in 0..3 {
            if normal[axis] != 0 {
                continue;
            }
            let v = vert[axis] as i32;
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
            1
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

/// For each face direction, the axes used by greedy meshing:
/// (normal_axis, u_axis, v_axis). The grid sweeps slices along the
/// normal axis and merges rectangles on the (u, v) plane.
const FACE_AXES: [(usize, usize, usize); 6] = [
    (0, 2, 1), // +X: sweep x, grid (z, y)
    (0, 2, 1), // -X: sweep x, grid (z, y)
    (1, 0, 2), // +Y: sweep y, grid (x, z)
    (1, 0, 2), // -Y: sweep y, grid (x, z)
    (2, 0, 1), // +Z: sweep z, grid (x, y)
    (2, 0, 1), // -Z: sweep z, grid (x, y)
];

/// Build a world-space position from (d, u, v) given axis mappings.
#[inline]
fn make_pos(d: i32, u: i32, v: i32, axes: (usize, usize, usize)) -> [i32; 3] {
    let mut pos = [0i32; 3];
    pos[axes.0] = d;
    pos[axes.1] = u;
    pos[axes.2] = v;
    pos
}

/// Bake any cubic voxel volume of `size^3` cells into per-voxel-type
/// sub-meshes with greedy face merging.
///
/// ## Greedy meshing
///
/// The naive approach emits one quad per exposed voxel face. For a
/// flat 125×125 grass surface, that's 15,625 quads — most of which
/// are coplanar, same-material, and visually identical. At low zoom
/// layers with ~1,100 chunks in the cull sphere, this produces ~88M
/// triangles for a scene that looks the same as layer 12's ~80k.
///
/// Greedy meshing fixes this by merging adjacent coplanar faces into
/// larger rectangles. The algorithm sweeps each 2D slice of the
/// volume (one per face direction per depth), finds maximal
/// rectangles of mergeable faces, and emits one quad per rectangle.
/// A flat grass surface collapses to ~1 quad. Total triangles drop
/// from 88M to ~40k.
///
/// ## Merge safety
///
/// Two faces merge only when the result is **pixel-identical** to
/// separate quads. The merge condition:
///
/// - Same voxel type (material)
/// - **Uniform AO**: all 4 vertex AO values are the same number
///   (e.g., `[3,3,3,3]`)
///
/// The uniform-AO rule prevents interpolation artifacts. When a
/// merged quad's interior vertices are removed, the GPU interpolates
/// AO linearly between the corners. This is only correct when AO is
/// constant across the merged region. Faces near placed blocks
/// typically have non-uniform AO (e.g., `[3,3,2,3]`) and stay as
/// individual quads with the original AO diagonal flip logic.
///
/// ## Future considerations
///
/// If per-voxel visual variation is added (textures, tints), the
/// merge eligibility check should be extended to exclude faces with
/// varying attributes. Non-eligible faces fall through to the
/// individual quad path automatically.
pub fn bake_volume<F: Fn(i32, i32, i32) -> Option<u8>>(
    size: i32,
    get: F,
    meshes: &mut Assets<Mesh>,
) -> Vec<BakedSubMesh> {
    let mut groups: std::collections::HashMap<u8, FaceCollector> =
        std::collections::HashMap::new();

    let sz = size as usize;

    for (face_idx, &(dir, ref quad, normal)) in FACES.iter().enumerate() {
        let axes = FACE_AXES[face_idx];

        for d in 0..size {
            // Build 2D grid of exposed faces in this slice.
            // Each cell is either:
            //   Some((voxel, ao_level)) — uniform AO, eligible for merging
            //   None — no face, or non-uniform AO (emitted immediately)
            let mut grid: Vec<Option<(u8, u8)>> = vec![None; sz * sz];

            for v in 0..size {
                for u in 0..size {
                    let pos = make_pos(d, u, v, axes);
                    let Some(voxel) = get(pos[0], pos[1], pos[2]) else {
                        continue;
                    };

                    let npos = [pos[0] + dir.x, pos[1] + dir.y, pos[2] + dir.z];
                    if get(npos[0], npos[1], npos[2]).is_some() {
                        continue;
                    }

                    let ao = compute_face_ao(
                        pos[0], pos[1], pos[2], dir, quad,
                        &|ax, ay, az| get(ax, ay, az).is_some(),
                    );

                    if ao[0] == ao[1] && ao[1] == ao[2] && ao[2] == ao[3] {
                        // Uniform AO → eligible for greedy merge.
                        grid[(v as usize) * sz + (u as usize)] =
                            Some((voxel, ao[0]));
                    } else {
                        // Non-uniform AO → emit as individual quad now.
                        let offset = Vec3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32);
                        groups.entry(voxel).or_default().add_quad(
                            quad, normal, offset, ao,
                        );
                    }
                }
            }

            // Greedy merge the grid.
            let mut visited = vec![false; sz * sz];

            for v in 0..size {
                for u in 0..size {
                    let idx = (v as usize) * sz + (u as usize);
                    if visited[idx] {
                        continue;
                    }
                    let Some((voxel, ao_level)) = grid[idx] else {
                        continue;
                    };

                    // Extend width (u direction).
                    let mut w = 1i32;
                    while u + w < size {
                        let ni = (v as usize) * sz + ((u + w) as usize);
                        if visited[ni] || grid[ni] != Some((voxel, ao_level)) {
                            break;
                        }
                        w += 1;
                    }

                    // Extend height (v direction).
                    let mut h = 1i32;
                    'outer: while v + h < size {
                        for du in 0..w {
                            let ni = ((v + h) as usize) * sz + ((u + du) as usize);
                            if visited[ni]
                                || grid[ni] != Some((voxel, ao_level))
                            {
                                break 'outer;
                            }
                        }
                        h += 1;
                    }

                    // Mark visited.
                    for dv in 0..h {
                        for du in 0..w {
                            visited[((v + dv) as usize) * sz
                                + ((u + du) as usize)] = true;
                        }
                    }

                    // Emit the merged quad.
                    let collector = groups.entry(voxel).or_default();
                    emit_merged_quad(
                        collector, face_idx, d, u, v, w, h, ao_level, axes,
                    );
                }
            }
        }
    }

    groups
        .into_iter()
        .map(|(voxel, collector)| BakedSubMesh {
            mesh: meshes.add(collector.build()),
            voxel,
        })
        .collect()
}

/// Emit a merged quad covering a (w × h) rectangle on the (u, v) plane
/// at depth `d` along the normal axis. The AO is uniform across the
/// quad (all 4 corners have `ao_level`).
fn emit_merged_quad(
    collector: &mut FaceCollector,
    face_idx: usize,
    d: i32,
    u: i32,
    v: i32,
    w: i32,
    h: i32,
    ao_level: u8,
    axes: (usize, usize, usize),
) {
    let (_, ref quad, normal) = FACES[face_idx];

    // The original 1×1 quad vertices are at positions 0 or 1 on each
    // tangent axis. For the merged quad, scale 0→0 and 1→w (or h)
    // on the appropriate axis.
    let base = collector.positions.len() as u32;
    let brightness = AO_CURVE[ao_level as usize];
    let normal_arr = normal.to_array();

    for &vert in quad {
        // Map vertex (0 or 1) on each tangent axis to the merged range.
        let mut pos = [0.0f32; 3];
        pos[axes.0] = d as f32 + vert[axes.0]; // normal axis: d or d+1
        pos[axes.1] = u as f32 + vert[axes.1] * (w as f32); // u axis: u..u+w
        pos[axes.2] = v as f32 + vert[axes.2] * (h as f32); // v axis: v..v+h

        collector.positions.push(pos);
        collector.normals.push(normal_arr);
        collector.colors.push([brightness, brightness, brightness, 1.0]);
    }

    // Uniform AO → no diagonal flip needed. Standard winding.
    collector.indices.extend_from_slice(&[
        base,
        base + 1,
        base + 2,
        base,
        base + 2,
        base + 3,
    ]);
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
        if ao[0] + ao[2] > ao[1] + ao[3] {
            self.indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        } else {
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
