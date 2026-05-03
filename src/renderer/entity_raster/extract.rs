//! Extract a triangle mesh from a subtree in the NodeLibrary.
//!
//! The subtree is flattened into a dense voxel grid at its deepest
//! resolution; then a **greedy mesher** emits as few quads as
//! possible by merging co-planar exposed faces of the same block
//! type into maximal rectangles. A 10×10×10 solid cube becomes 6
//! quads (24 vertices), not 600.
//!
//! The raw per-voxel extractor — used briefly during bring-up —
//! emitted ~64k vertices for a soldier. At 10k instances that's
//! 640M vertices per frame, which drowned the GPU's vertex shader
//! stage and made the raster path slower than the ray-march
//! fallback. Greedy meshing cuts vertex count by ~15–50× depending
//! on block homogeneity; after this the raster path scales as
//! fragment work, not vertex work.
//!
//! Content-addressing: the result depends only on `NodeId`, so the
//! mesh cache stays free on edits (a modified subtree gets a new
//! NodeId and a new mesh).

use bytemuck::{Pod, Zeroable};

use crate::world::tree::{slot_coords, Child, NodeId, NodeLibrary};

/// Per-vertex GPU layout for the entity raster shader.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

pub struct EntityMesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
}

impl EntityMesh {
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty() || self.indices.is_empty()
    }
}

/// Build a 6-face cube spanning [0, 3]^3 in subtree-local coords,
/// flat-shaded white. Used as the LOD-terminal impostor for
/// distant entities: the instance shader multiplies vertex color
/// by the per-instance tint (set to the entity subtree's
/// representative block color at upload time), so one shared mesh
/// colors every small-on-screen entity correctly.
///
/// 24 vertices, 12 triangles — ~2000× cheaper than the full
/// per-voxel extract for a soldier-sized subtree. The raster pass
/// already scales fragment cost with projected pixel area, so
/// swapping to a cube at sub-20px coverage barely changes visual
/// fidelity while collapsing vertex cost from 45k to 24 per instance.
pub fn unit_cube_mesh() -> EntityMesh {
    let mut vertices: Vec<MeshVertex> = Vec::with_capacity(24);
    let mut indices: Vec<u32> = Vec::with_capacity(36);
    let s = 3.0;
    // Six faces: +X, -X, +Y, -Y, +Z, -Z.
    let faces: [([[f32; 3]; 4], [f32; 3]); 6] = [
        ([[s, 0.0, 0.0], [s, s, 0.0], [s, s, s], [s, 0.0, s]],     [1.0, 0.0, 0.0]),
        ([[0.0, 0.0, s], [0.0, s, s], [0.0, s, 0.0], [0.0, 0.0, 0.0]], [-1.0, 0.0, 0.0]),
        ([[0.0, s, 0.0], [0.0, s, s], [s, s, s], [s, s, 0.0]],     [0.0, 1.0, 0.0]),
        ([[s, 0.0, 0.0], [s, 0.0, s], [0.0, 0.0, s], [0.0, 0.0, 0.0]], [0.0, -1.0, 0.0]),
        ([[s, 0.0, s], [s, s, s], [0.0, s, s], [0.0, 0.0, s]],     [0.0, 0.0, 1.0]),
        ([[0.0, 0.0, 0.0], [0.0, s, 0.0], [s, s, 0.0], [s, 0.0, 0.0]], [0.0, 0.0, -1.0]),
    ];
    for (corners, normal) in &faces {
        let base = vertices.len() as u32;
        for c in corners {
            vertices.push(MeshVertex {
                position: *c,
                normal: *normal,
                color: [1.0, 1.0, 1.0],
            });
        }
        indices.extend_from_slice(&[
            base, base + 1, base + 2,
            base, base + 2, base + 3,
        ]);
    }
    EntityMesh { vertices, indices }
}

/// Extract a mesh from `root` in `library`. Returns `None` when the
/// subtree is entirely empty (no solid voxels — no mesh to draw).
pub fn extract(
    library: &NodeLibrary,
    root: NodeId,
    palette: &[[f32; 4]],
) -> Option<EntityMesh> {
    let node = library.get(root)?;
    let max_depth = node.depth as usize;
    let res = 3usize.pow(max_depth as u32);
    let mut grid = vec![0u16; res * res * res];
    fill_dense(library, root, &mut grid, res, 0, 0, 0, res);
    if grid.iter().all(|&c| c == 0) {
        return None;
    }

    let mut vertices: Vec<MeshVertex> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let cell_size = 3.0 / res as f32;

    // Greedy mesher runs once per face direction (6 total). The
    // mask is a 2D rectangle of (blocktype+facing) tags; a zero tag
    // means "no face here." Two cells can be merged if they share
    // the same tag — which embeds both block-type and normal-facing
    // into one value.
    for axis in 0..3 {
        for sign in [false, true] {
            greedy_axis(
                &grid, res, cell_size, axis, sign,
                palette, &mut vertices, &mut indices,
            );
        }
    }

    if vertices.is_empty() {
        return None;
    }
    Some(EntityMesh { vertices, indices })
}

/// Emit quads for all faces on one axis-sign pair (e.g. +X faces).
/// Slices along `axis`, then scans 2D slabs greedily.
fn greedy_axis(
    grid: &[u16], res: usize, cell_size: f32,
    axis: usize, positive: bool,
    palette: &[[f32; 4]],
    vertices: &mut Vec<MeshVertex>,
    indices: &mut Vec<u32>,
) {
    let (u_axis, v_axis) = match axis {
        0 => (1, 2), // X-normal plane: u=Y, v=Z
        1 => (2, 0), // Y-normal plane: u=Z, v=X
        2 => (0, 1), // Z-normal plane: u=X, v=Y
        _ => unreachable!(),
    };
    let normal = {
        let mut n = [0.0_f32; 3];
        n[axis] = if positive { 1.0 } else { -1.0 };
        n
    };

    // Build and greedy-mesh one slice at a time. Slice index = the
    // `axis` coordinate of the cells whose `positive`-side face
    // we're processing.
    let mut mask = vec![0u16; res * res];
    for slab in 0..res {
        mask.fill(0);
        // Populate mask[u, v] with block type + 1 for each solid
        // cell whose outward face is exposed on this side.
        for v in 0..res {
            for u in 0..res {
                let mut coords = [0usize; 3];
                coords[axis] = slab;
                coords[u_axis] = u;
                coords[v_axis] = v;
                let c = grid[coords[2] * res * res + coords[1] * res + coords[0]];
                if c == 0 { continue; }
                // Neighbor coords on the outward side.
                let mut nb = coords;
                let nb_ok = if positive {
                    nb[axis] += 1;
                    nb[axis] < res
                } else if nb[axis] == 0 {
                    false
                } else {
                    nb[axis] -= 1;
                    true
                };
                let occluded = nb_ok
                    && grid[nb[2] * res * res + nb[1] * res + nb[0]] != 0;
                if !occluded {
                    mask[v * res + u] = c;
                }
            }
        }

        // Greedy-scan the 2D mask, emitting one quad per maximal
        // rectangle of matching tags.
        let mut v = 0;
        while v < res {
            let mut u = 0;
            while u < res {
                let tag = mask[v * res + u];
                if tag == 0 {
                    u += 1;
                    continue;
                }
                // Extend right.
                let mut w = 1;
                while u + w < res && mask[v * res + u + w] == tag {
                    w += 1;
                }
                // Extend down: each row must have the same tag run
                // of length w starting at column u.
                let mut h = 1;
                'extend: while v + h < res {
                    for du in 0..w {
                        if mask[(v + h) * res + u + du] != tag {
                            break 'extend;
                        }
                    }
                    h += 1;
                }

                // Emit a quad for (u, v, u+w, v+h) on this slab.
                let bt = (tag - 1) as usize;
                let pal = palette.get(bt).copied().unwrap_or([0.5, 0.5, 0.5, 1.0]);
                let color = [pal[0], pal[1], pal[2]];
                let slab_coord = if positive {
                    (slab + 1) as f32 * cell_size
                } else {
                    slab as f32 * cell_size
                };
                let u0 = u as f32 * cell_size;
                let u1 = (u + w) as f32 * cell_size;
                let v0 = v as f32 * cell_size;
                let v1 = (v + h) as f32 * cell_size;
                let corners = quad_corners(axis, slab_coord, u_axis, u0, u1, v_axis, v0, v1, positive);
                emit_quad(vertices, indices, corners, normal, color);

                // Clear the consumed rectangle from the mask.
                for dv in 0..h {
                    for du in 0..w {
                        mask[(v + dv) * res + u + du] = 0;
                    }
                }
                u += w;
            }
            v += 1;
        }
    }
}

/// Build the 4 corner positions of a quad in 3D given the plane
/// coordinate, the two in-plane axis extents, and the winding side.
/// Returns them in CCW order when viewed from the +normal side.
fn quad_corners(
    axis: usize, slab: f32,
    u_axis: usize, u0: f32, u1: f32,
    v_axis: usize, v0: f32, v1: f32,
    positive: bool,
) -> [[f32; 3]; 4] {
    let mk = |u: f32, v: f32| -> [f32; 3] {
        let mut p = [0.0_f32; 3];
        p[axis] = slab;
        p[u_axis] = u;
        p[v_axis] = v;
        p
    };
    // When normal points +axis, the quad's CCW winding (viewed from
    // +normal) runs (u0,v0) → (u1,v0) → (u1,v1) → (u0,v1). When the
    // normal points -axis, flip the winding.
    if positive {
        [mk(u0, v0), mk(u1, v0), mk(u1, v1), mk(u0, v1)]
    } else {
        [mk(u0, v0), mk(u0, v1), mk(u1, v1), mk(u1, v0)]
    }
}

fn emit_quad(
    vertices: &mut Vec<MeshVertex>,
    indices: &mut Vec<u32>,
    corners: [[f32; 3]; 4],
    normal: [f32; 3],
    color: [f32; 3],
) {
    let base = vertices.len() as u32;
    for c in &corners {
        vertices.push(MeshVertex { position: *c, normal, color });
    }
    indices.extend_from_slice(&[
        base, base + 1, base + 2,
        base, base + 2, base + 3,
    ]);
}

fn fill_dense(
    library: &NodeLibrary,
    node_id: NodeId,
    grid: &mut [u16],
    res: usize,
    ox: usize, oy: usize, oz: usize,
    cell_res: usize,
) {
    let Some(node) = library.get(node_id) else { return };
    let child_res = cell_res / 3;
    for slot in 0..27 {
        let (sx, sy, sz) = slot_coords(slot);
        let cx = ox + sx * child_res;
        let cy = oy + sy * child_res;
        let cz = oz + sz * child_res;
        match node.children[slot] {
            Child::Empty => {}
            Child::Block(bt) => fill_block(grid, res, cx, cy, cz, child_res, bt),
            Child::Node(nid) | Child::PlacedNode { node: nid, .. } => {
                if child_res == 0 { continue; }
                fill_dense(library, nid, grid, res, cx, cy, cz, child_res);
            }
            Child::EntityRef(_) => {}
        }
    }
}

fn fill_block(
    grid: &mut [u16], res: usize,
    ox: usize, oy: usize, oz: usize, cell_res: usize, bt: u16,
) {
    // Grid uses 0 as "empty" sentinel, so block-type BT is stored as
    // BT+1. Saturating add keeps us in range for palette indices near
    // u16::MAX (fractal's widened palette). BT=u16::MAX → stored as
    // u16::MAX, indistinguishable from BT=u16::MAX-1 for lookup, which
    // we accept: real palette sizes stay well below that.
    let val = bt.saturating_add(1);
    for z in oz..(oz + cell_res).min(res) {
        for y in oy..(oy + cell_res).min(res) {
            let row_start = z * res * res + y * res;
            for x in ox..(ox + cell_res).min(res) {
                grid[row_start + x] = val;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, uniform_children};

    #[test]
    fn empty_subtree_returns_none() {
        let mut lib = NodeLibrary::default();
        let root = lib.insert(empty_children());
        let palette: Vec<[f32; 4]> = vec![[0.0; 4]; 256];
        assert!(extract(&lib, root, &palette).is_none());
    }

    #[test]
    fn single_solid_block_emits_6_faces() {
        let mut lib = NodeLibrary::default();
        let mut children = empty_children();
        children[13] = Child::Block(1);
        let root = lib.insert(children);
        let mut palette: Vec<[f32; 4]> = vec![[0.0; 4]; 256];
        palette[0] = [1.0, 0.0, 0.0, 1.0];
        let mesh = extract(&lib, root, &palette).expect("mesh emitted");
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn uniform_subtree_merges_to_6_quads() {
        let mut lib = NodeLibrary::default();
        let root = lib.insert(uniform_children(Child::Block(1)));
        let palette: Vec<[f32; 4]> = vec![[1.0; 4]; 256];
        let mesh = extract(&lib, root, &palette).expect("mesh emitted");
        // Greedy: 6 external faces, each one big quad = 24 verts
        // total. (Per-voxel would have emitted 216.)
        assert_eq!(mesh.vertices.len(), 24);
    }
}
