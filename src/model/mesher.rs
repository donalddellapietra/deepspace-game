use std::collections::HashMap;

use bevy::{
    asset::RenderAssetUsages,
    mesh::Indices,
    prelude::*,
    render::render_resource::PrimitiveTopology,
};

use super::BakedSubMesh;

/// Face definitions: (neighbor offset, quad vertices in CCW winding, normal).
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

/// For each face direction, the axes used by greedy meshing:
/// (normal_axis, u_axis, v_axis).
const FACE_AXES: [(usize, usize, usize); 6] = [
    (0, 2, 1), // +X
    (0, 2, 1), // -X
    (1, 0, 2), // +Y
    (1, 0, 2), // -Y
    (2, 0, 1), // +Z
    (2, 0, 1), // -Z
];

// ---------------------------------------------------------------- AO

fn compute_face_ao<F: Fn(i32, i32, i32) -> bool>(
    bx: i32, by: i32, bz: i32,
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
            if normal[axis] != 0 { continue; }
            let v = vert[axis] as i32;
            sides[idx][axis] = if v == 0 { -1 } else { 1 };
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

// --------------------------------------------------------- face data

/// Raw face data for one voxel type, before conversion to a Mesh.
/// Stored per-child so edits only re-bake the affected child.
#[derive(Clone, Default)]
pub struct FaceData {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
}

impl FaceData {
    fn add_quad(&mut self, quad: &[Vec3; 4], normal: Vec3, offset: Vec3, ao: [u8; 4]) {
        let base = self.positions.len() as u32;
        for (i, &vert) in quad.iter().enumerate() {
            self.positions.push((vert + offset).to_array());
            self.normals.push(normal.to_array());
            let brightness = AO_CURVE[ao[i] as usize];
            self.colors.push([brightness, brightness, brightness, 1.0]);
        }
        if ao[0] + ao[2] > ao[1] + ao[3] {
            self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        } else {
            self.indices.extend_from_slice(&[base + 1, base + 2, base + 3, base + 1, base + 3, base]);
        }
    }

    fn add_merged_quad(
        &mut self,
        face_idx: usize,
        d: i32, u: i32, v: i32,
        w: i32, h: i32,
        ao_level: u8,
        axes: (usize, usize, usize),
    ) {
        let (_, ref quad, normal) = FACES[face_idx];
        let base = self.positions.len() as u32;
        let brightness = AO_CURVE[ao_level as usize];
        let normal_arr = normal.to_array();
        for &vert in quad {
            let mut pos = [0.0f32; 3];
            pos[axes.0] = d as f32 + vert[axes.0];
            pos[axes.1] = u as f32 + vert[axes.1] * (w as f32);
            pos[axes.2] = v as f32 + vert[axes.2] * (h as f32);
            self.positions.push(pos);
            self.normals.push(normal_arr);
            self.colors.push([brightness, brightness, brightness, 1.0]);
        }
        self.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
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

/// Per-child baked face data, keyed by voxel type.
pub type ChildFaces = HashMap<u8, FaceData>;

// --------------------------------------------------- greedy bake core

#[inline]
fn make_pos(d: i32, u: i32, v: i32, axes: (usize, usize, usize)) -> [i32; 3] {
    let mut pos = [0i32; 3];
    pos[axes.0] = d;
    pos[axes.1] = u;
    pos[axes.2] = v;
    pos
}

/// Greedy-bake a region of `size³` voxels into per-voxel-type
/// `FaceData`. The `get` closure provides voxel lookups (may extend
/// beyond the region for neighbor checks). Coordinates passed to
/// `get` are in the caller's frame — the caller controls the offset.
fn bake_faces<F: Fn(i32, i32, i32) -> Option<u8>>(
    size: i32,
    get: &F,
    offset: [i32; 3],
) -> ChildFaces {
    let mut groups: HashMap<u8, FaceData> = HashMap::new();
    let sz = size as usize;

    for (face_idx, &(dir, ref quad, normal)) in FACES.iter().enumerate() {
        let axes = FACE_AXES[face_idx];

        for d_local in 0..size {
            let d = d_local + offset[axes.0];
            let mut grid: Vec<Option<(u8, u8)>> = vec![None; sz * sz];

            for v_local in 0..size {
                let v = v_local + offset[axes.2];
                for u_local in 0..size {
                    let u = u_local + offset[axes.1];
                    let pos = make_pos(d, u, v, axes);
                    let Some(voxel) = get(pos[0], pos[1], pos[2]) else { continue };
                    let npos = [pos[0] + dir.x, pos[1] + dir.y, pos[2] + dir.z];
                    if get(npos[0], npos[1], npos[2]).is_some() { continue; }

                    let ao = compute_face_ao(
                        pos[0], pos[1], pos[2], dir, quad,
                        &|ax, ay, az| get(ax, ay, az).is_some(),
                    );

                    if ao[0] == ao[1] && ao[1] == ao[2] && ao[2] == ao[3] {
                        grid[(v_local as usize) * sz + (u_local as usize)] =
                            Some((voxel, ao[0]));
                    } else {
                        let world_offset = Vec3::new(pos[0] as f32, pos[1] as f32, pos[2] as f32);
                        groups.entry(voxel).or_default().add_quad(quad, normal, world_offset, ao);
                    }
                }
            }

            // Greedy merge within this slice.
            let mut visited = vec![false; sz * sz];
            for v_local in 0..size {
                let v = v_local + offset[axes.2];
                for u_local in 0..size {
                    let u = u_local + offset[axes.1];
                    let idx = (v_local as usize) * sz + (u_local as usize);
                    if visited[idx] { continue; }
                    let Some((voxel, ao_level)) = grid[idx] else { continue; };

                    let mut w = 1i32;
                    while u_local + w < size {
                        let ni = (v_local as usize) * sz + ((u_local + w) as usize);
                        if visited[ni] || grid[ni] != Some((voxel, ao_level)) { break; }
                        w += 1;
                    }

                    let mut h = 1i32;
                    'outer: while v_local + h < size {
                        for du in 0..w {
                            let ni = ((v_local + h) as usize) * sz + ((u_local + du) as usize);
                            if visited[ni] || grid[ni] != Some((voxel, ao_level)) { break 'outer; }
                        }
                        h += 1;
                    }

                    for dv in 0..h {
                        for du in 0..w {
                            visited[((v_local + dv) as usize) * sz + ((u_local + du) as usize)] = true;
                        }
                    }

                    groups.entry(voxel).or_default().add_merged_quad(
                        face_idx, d, u, v, w, h, ao_level, axes,
                    );
                }
            }
        }
    }

    groups
}

// ----------------------------------------------- public API

/// Bake a `size³` volume into per-voxel-type sub-meshes with greedy
/// meshing. Used for leaf nodes and any case where incremental
/// baking isn't needed.
pub fn bake_volume<F: Fn(i32, i32, i32) -> Option<u8>>(
    size: i32,
    get: F,
    meshes: &mut Assets<Mesh>,
) -> Vec<BakedSubMesh> {
    let faces = bake_faces(size, &get, [0, 0, 0]);
    faces
        .into_iter()
        .map(|(voxel, data)| BakedSubMesh {
            mesh: meshes.add(data.build()),
            voxel,
        })
        .collect()
}

/// Bake one child's 25³ region within a parent's 125³ grid.
///
/// `get` is the same closure used for the full 125³ bake — it
/// provides voxel lookups across the entire parent volume (including
/// neighbors in adjacent children, needed for face culling and AO).
///
/// `child_slot` (0..125) determines which 25³ sub-region to bake.
/// The returned `ChildFaces` contains face data in parent-local
/// coordinates, ready to be merged with other children's faces.
pub fn bake_child_faces<F: Fn(i32, i32, i32) -> Option<u8>>(
    get: &F,
    child_slot: usize,
    child_size: i32,
    branch_factor: usize,
) -> ChildFaces {
    let sx = child_slot % branch_factor;
    let sy = (child_slot / branch_factor) % branch_factor;
    let sz = child_slot / (branch_factor * branch_factor);
    let offset = [
        (sx as i32) * child_size,
        (sy as i32) * child_size,
        (sz as i32) * child_size,
    ];
    bake_faces(child_size, get, offset)
}

/// Voxel classification for a child slot: empty, uniform, or mixed.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ChildClass {
    /// EMPTY_NODE — no voxels at all.
    Empty,
    /// Every voxel in the 25³ grid is the same value.
    Uniform(u8),
    /// The grid contains multiple voxel types.
    Mixed,
}

/// Classify each of `n_children` children as empty, uniform, or mixed.
/// `get_voxels` returns the 25³ grid for a slot, or `None` for EMPTY_NODE.
pub fn classify_children<F>(n_children: usize, get_voxels: F) -> Vec<ChildClass>
where
    F: Fn(usize) -> Option<&'static [u8]>,
{
    (0..n_children)
        .map(|slot| {
            let Some(voxels) = get_voxels(slot) else {
                return ChildClass::Empty;
            };
            let first = voxels[0];
            if voxels.iter().all(|&v| v == first) {
                ChildClass::Uniform(first)
            } else {
                ChildClass::Mixed
            }
        })
        .collect()
}

/// Check whether a uniform child at `slot` can be skipped entirely
/// (all 6 neighbors are the same uniform value).
///
/// `neighbor_same` indicates whether the neighboring emit-layer
/// parent in each direction (-x, +x, -y, +y, -z, +z) has the same
/// NodeId as the current parent. If true, boundary children in that
/// direction are treated as having the same content (instead of
/// assuming empty). This lets underground chunks skip all boundary
/// children when surrounded by identical neighbors.
pub fn uniform_child_skippable(
    slot: usize,
    v: u8,
    child_class: &[ChildClass],
    branch_factor: usize,
    empty_voxel: u8,
    neighbor_same: [bool; 6],
) -> bool {
    let sx = slot % branch_factor;
    let sy = (slot / branch_factor) % branch_factor;
    let sz = slot / (branch_factor * branch_factor);
    // (delta, direction_index) for each of 6 neighbors.
    // Direction indices: 0=-x, 1=+x, 2=-y, 3=+y, 4=-z, 5=+z.
    let neighbors = [
        (sx.wrapping_sub(1), sy, sz, 0usize), // -x
        (sx + 1, sy, sz, 1),                  // +x
        (sx, sy.wrapping_sub(1), sz, 2),       // -y
        (sx, sy + 1, sz, 3),                   // +y
        (sx, sy, sz.wrapping_sub(1), 4),       // -z
        (sx, sy, sz + 1, 5),                   // +z
    ];
    neighbors.iter().all(|&(nx, ny, nz, dir)| {
        if nx >= branch_factor || ny >= branch_factor || nz >= branch_factor {
            // Outside the parent. If the neighbor parent has the same
            // NodeId, its content is identical → treat as same type.
            if neighbor_same[dir] {
                return true;
            }
            // Otherwise conservatively assume empty.
            return v == empty_voxel;
        }
        let nslot = nz * branch_factor * branch_factor + ny * branch_factor + nx;
        child_class[nslot] == ChildClass::Uniform(v)
    })
}

/// Flatten children's voxel grids into a contiguous `size³` array,
/// using child classification to skip empty children and memset
/// uniform ones.
pub fn flatten_children(
    children_voxels: &[Option<&[u8]>],
    child_class: &[ChildClass],
    branch_factor: usize,
    child_size: usize,
    empty_voxel: u8,
) -> Vec<u8> {
    let size = branch_factor * child_size;
    let mut flat = vec![empty_voxel; size * size * size];

    for slot in 0..children_voxels.len() {
        let sx = slot % branch_factor;
        let sy = (slot / branch_factor) % branch_factor;
        let sz = slot / (branch_factor * branch_factor);
        let bx = sx * child_size;
        let by = sy * child_size;
        let bz = sz * child_size;

        match child_class[slot] {
            ChildClass::Empty => {}
            ChildClass::Uniform(v) if v == empty_voxel => {}
            ChildClass::Uniform(v) => {
                for z in 0..child_size {
                    for y in 0..child_size {
                        for x in 0..child_size {
                            flat[((bz + z) * size + (by + y)) * size + (bx + x)] = v;
                        }
                    }
                }
            }
            ChildClass::Mixed => {
                if let Some(voxels) = children_voxels[slot] {
                    for z in 0..child_size {
                        for y in 0..child_size {
                            for x in 0..child_size {
                                let v = voxels[z * child_size * child_size + y * child_size + x];
                                if v != empty_voxel {
                                    flat[((bz + z) * size + (by + y)) * size + (bx + x)] = v;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    flat
}

/// Merge 125 children's `ChildFaces` into final `BakedSubMesh`es.
/// Each child's face data is concatenated (with index offsets) into
/// one mesh per voxel type.
pub fn merge_child_faces(
    children: &[ChildFaces],
    meshes: &mut Assets<Mesh>,
) -> Vec<BakedSubMesh> {
    // Collect all voxel types across all children.
    let mut all_voxels: Vec<u8> = Vec::new();
    for child in children {
        for &v in child.keys() {
            if !all_voxels.contains(&v) {
                all_voxels.push(v);
            }
        }
    }

    all_voxels
        .into_iter()
        .filter_map(|voxel| {
            let mut merged = FaceData::default();
            for child in children {
                if let Some(data) = child.get(&voxel) {
                    let base = merged.positions.len() as u32;
                    merged.positions.extend_from_slice(&data.positions);
                    merged.normals.extend_from_slice(&data.normals);
                    merged.colors.extend_from_slice(&data.colors);
                    for &idx in &data.indices {
                        merged.indices.push(base + idx);
                    }
                }
            }
            if merged.is_empty() {
                return None;
            }

            // Smooth AO at child boundaries: the per-child greedy merge
            // can't merge across 25-voxel child edges, so adjacent
            // children's quads can have different AO levels at their
            // shared edge, creating visible dark lines. Setting
            // boundary vertices to full brightness removes the
            // discontinuity without affecting interior AO or the
            // incremental baking cache.
            let child_size: f32 = 25.0;
            let eps = 0.01;
            for (i, pos) in merged.positions.iter().enumerate() {
                let on_child_edge =
                    (pos[0] % child_size).abs() < eps
                    || (child_size - pos[0] % child_size).abs() < eps
                    || (pos[1] % child_size).abs() < eps
                    || (child_size - pos[1] % child_size).abs() < eps
                    || (pos[2] % child_size).abs() < eps
                    || (child_size - pos[2] % child_size).abs() < eps;
                if on_child_edge {
                    merged.colors[i] = [1.0, 1.0, 1.0, 1.0];
                }
            }

            Some(BakedSubMesh {
                mesh: meshes.add(merged.build()),
                voxel,
            })
        })
        .collect()
}
