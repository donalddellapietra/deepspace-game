//! CPU mirror of the shader's `march_uv_ring`.

use super::HitInfo;
use crate::world::tree::{slot_index, Child, NodeId, NodeLibrary};

pub fn cpu_raycast_uv_ring(
    library: &NodeLibrary,
    world_root: NodeId,
    frame_path: &[u8],
    cam_local: [f32; 3],
    ray_dir: [f32; 3],
    dims: [u32; 3],
    slab_depth: u8,
    max_depth: u32,
) -> Option<HitInfo> {
    let mut frame_chain: Vec<(NodeId, usize)> = Vec::with_capacity(frame_path.len());
    let mut cur = world_root;
    for &slot in frame_path {
        let node = library.get(cur)?;
        frame_chain.push((cur, slot as usize));
        match node.children[slot as usize] {
            Child::Node(child) | Child::PlacedNode { node: child, .. } => cur = child,
            _ => return None,
        }
    }
    let ring_root = cur;
    let body_center = [1.5_f32, 1.5, 1.5];
    let body_size = 3.0_f32;
    let radius = body_size * 0.38;
    let angle_step = 2.0 * std::f32::consts::PI / dims[0] as f32;
    let side = (radius * angle_step).max(body_size / 27.0);

    let mut best: Option<HitInfo> = None;
    for cell_x in 0..dims[0] as i32 {
        let mut path = frame_chain.clone();
        let mut idx = ring_root;
        let mut cells_per_slot = 1i32;
        for _ in 1..slab_depth {
            cells_per_slot *= 3;
        }
        let mut anchor = None;
        let mut uniform_block = false;
        let mut empty = false;
        for level in 0..slab_depth {
            let sx = (cell_x / cells_per_slot).rem_euclid(3);
            let sy = 0;
            let sz = 0;
            let slot = slot_index(sx as usize, sy as usize, sz as usize);
            path.push((idx, slot));
            let Some(node) = library.get(idx) else {
                empty = true;
                break;
            };
            match node.children[slot] {
                Child::Empty | Child::EntityRef(_) => {
                    empty = true;
                    break;
                }
                Child::Block(_) => {
                    uniform_block = true;
                    break;
                }
                Child::Node(child) | Child::PlacedNode { node: child, .. } => {
                    if level + 1 < slab_depth {
                        idx = child;
                    } else {
                        anchor = Some(child);
                    }
                }
            }
            cells_per_slot /= 3;
        }
        if empty {
            continue;
        }

        let angle = -std::f32::consts::PI + (cell_x as f32 + 0.5) * angle_step;
        let (sa, ca) = angle.sin_cos();
        let radial = [ca, 0.0, sa];
        let tangent = [-sa, 0.0, ca];
        let up = [0.0, 1.0, 0.0];
        let cube_origin = [
            body_center[0] + radial[0] * radius,
            body_center[1],
            body_center[2] + radial[2] * radius,
        ];
        let scale = 3.0 / side;
        let d_origin = [
            cam_local[0] - cube_origin[0],
            cam_local[1] - cube_origin[1],
            cam_local[2] - cube_origin[2],
        ];
        let local_origin = [
            dot(tangent, d_origin) * scale + 1.5,
            dot(radial, d_origin) * scale + 1.5,
            dot(up, d_origin) * scale + 1.5,
        ];
        let local_dir = [
            dot(tangent, ray_dir) * scale,
            dot(radial, ray_dir) * scale,
            dot(up, ray_dir) * scale,
        ];

        let candidate = if let Some(anchor_id) = anchor {
            let absolute_slab_depth = frame_path.len() as u32 + slab_depth as u32;
            let cube_max_depth = max_depth.saturating_sub(absolute_slab_depth).max(1);
            super::cartesian::cpu_raycast_inner(
                library,
                anchor_id,
                local_origin,
                local_dir,
                cube_max_depth,
            )
            .map(|sub| {
                for &(parent, slot) in &sub.path {
                    path.push((parent, slot));
                }
                HitInfo {
                    path,
                    face: sub.face,
                    t: sub.t,
                    place_path: None,
                }
            })
        } else if uniform_block {
            cube_aabb_hit(local_origin, local_dir).map(|t| HitInfo {
                path,
                face: 2,
                t,
                place_path: None,
            })
        } else {
            None
        };

        if let Some(hit) = candidate {
            if best.as_ref().map(|b| hit.t < b.t).unwrap_or(true) {
                best = Some(hit);
            }
        }
    }

    best
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cube_aabb_hit(origin: [f32; 3], dir: [f32; 3]) -> Option<f32> {
    let inv = [
        if dir[0].abs() > 1e-8 { 1.0 / dir[0] } else { 1e10 },
        if dir[1].abs() > 1e-8 { 1.0 / dir[1] } else { 1e10 },
        if dir[2].abs() > 1e-8 { 1.0 / dir[2] } else { 1e10 },
    ];
    let t1 = [
        (0.0 - origin[0]) * inv[0],
        (0.0 - origin[1]) * inv[1],
        (0.0 - origin[2]) * inv[2],
    ];
    let t2 = [
        (3.0 - origin[0]) * inv[0],
        (3.0 - origin[1]) * inv[1],
        (3.0 - origin[2]) * inv[2],
    ];
    let t_enter = t1[0].min(t2[0])
        .max(t1[1].min(t2[1]))
        .max(t1[2].min(t2[2]));
    let t_exit = t1[0].max(t2[0])
        .min(t1[1].max(t2[1]))
        .min(t1[2].max(t2[2]));
    if t_enter < t_exit && t_exit > 0.0 {
        Some(t_enter.max(0.0))
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::palette::block;
    use crate::world::tree::{empty_children, uniform_children, NodeKind, NodeLibrary};

    fn ring_world() -> (NodeLibrary, NodeId) {
        let mut library = NodeLibrary::default();
        let content = Child::Node(library.insert(uniform_children(Child::Block(block::GRASS))));
        let mut leaves = vec![Child::Empty; 27 * 27 * 27];
        for x in 0..27 {
            leaves[(0 * 27 + 0) * 27 + x] = content;
        }
        let mut layer = leaves;
        let mut size = 27;
        while size > 1 {
            let next_size = size / 3;
            let mut next = vec![Child::Empty; next_size * next_size * next_size];
            for z in 0..next_size {
                for y in 0..next_size {
                    for x in 0..next_size {
                        let mut children = empty_children();
                        for dz in 0..3 {
                            for dy in 0..3 {
                                for dx in 0..3 {
                                    let src_x = x * 3 + dx;
                                    let src_y = y * 3 + dy;
                                    let src_z = z * 3 + dz;
                                    children[slot_index(dx, dy, dz)] =
                                        layer[(src_z * size + src_y) * size + src_x];
                                }
                            }
                        }
                        let kind = if next_size == 1 {
                            NodeKind::UvRing { dims: [27, 1, 1], slab_depth: 3 }
                        } else {
                            NodeKind::Cartesian
                        };
                        next[(z * next_size + y) * next_size + x] =
                            Child::Node(library.insert_with_kind(children, kind));
                    }
                }
            }
            layer = next;
            size = next_size;
        }
        let Child::Node(root) = layer[0] else {
            panic!("ring root should be node");
        };
        library.ref_inc(root);
        (library, root)
    }

    #[test]
    fn ray_to_positive_x_ring_hits_middle_uv_cell() {
        let (library, root) = ring_world();
        let hit = cpu_raycast_uv_ring(
            &library,
            root,
            &[],
            [3.0, 1.5, 1.5],
            [-1.0, 0.0, 0.0],
            [27, 1, 1],
            3,
            5,
        )
        .expect("ray toward +X ring segment must hit");
        assert!(
            hit.path.len() >= 3,
            "path should include slab descent, got {:?}",
            hit.path,
        );
        assert_eq!(
            hit.path[2].1,
            slot_index(1, 0, 0),
            "slab leaf must select UV cell x=13, y=0, z=0",
        );
    }
}
