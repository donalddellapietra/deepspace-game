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
    _max_depth: u32,
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
    let side = ((2.0 * std::f32::consts::PI * radius / dims[0] as f32) * 0.95)
        .max(body_size / 27.0);

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

        let candidate = if anchor.is_some() || uniform_block {
            curved_cell_hit(
                body_center,
                radius,
                side,
                angle_step,
                cell_x,
                cam_local,
                ray_dir,
            )
            .map(|(t, face)| HitInfo {
                path,
                face,
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

fn curved_cell_hit(
    center: [f32; 3],
    radius: f32,
    side: f32,
    angle_step: f32,
    cell_x: i32,
    origin: [f32; 3],
    dir: [f32; 3],
) -> Option<(f32, u32)> {
    let half_side = side * 0.5;
    let oc = [
        origin[0] - center[0],
        origin[1] - center[1],
        origin[2] - center[2],
    ];
    let mut best = f32::INFINITY;
    consider_curved_t(
        0.0, center, radius, half_side, angle_step, cell_x, origin, dir, &mut best,
    );

    for r in [radius + half_side, (radius - half_side).max(1e-5)] {
        let roots = cylinder_roots_y(oc, dir, r);
        for t in roots.into_iter().flatten() {
            consider_curved_t(
                t, center, radius, half_side, angle_step, cell_x, origin, dir, &mut best,
            );
        }
    }

    if dir[1].abs() > 1e-8 {
        for y in [center[1] - half_side, center[1] + half_side] {
            let t = (y - origin[1]) / dir[1];
            consider_curved_t(
                t, center, radius, half_side, angle_step, cell_x, origin, dir, &mut best,
            );
        }
    }

    let lo = -std::f32::consts::PI + cell_x as f32 * angle_step;
    let hi = lo + angle_step;
    for angle in [lo, hi] {
        if let Some(t) = meridian_t(oc, dir, angle) {
            consider_curved_t(
                t, center, radius, half_side, angle_step, cell_x, origin, dir, &mut best,
            );
        }
    }

    best.is_finite().then(|| {
        let p = [
            origin[0] + dir[0] * best,
            origin[1] + dir[1] * best,
            origin[2] + dir[2] * best,
        ];
        (best, curved_face(center, radius, half_side, angle_step, cell_x, p))
    })
}

fn consider_curved_t(
    t: f32,
    center: [f32; 3],
    radius: f32,
    half_side: f32,
    angle_step: f32,
    cell_x: i32,
    origin: [f32; 3],
    dir: [f32; 3],
    best: &mut f32,
) {
    if t < -1e-5 || t >= *best {
        return;
    }
    let probe_t = t.max(0.0) + 1e-5;
    let p = [
        origin[0] + dir[0] * probe_t,
        origin[1] + dir[1] * probe_t,
        origin[2] + dir[2] * probe_t,
    ];
    if point_in_curved_cell(center, radius, half_side, angle_step, cell_x, p) {
        *best = t.max(0.0);
    }
}

fn point_in_curved_cell(
    center: [f32; 3],
    radius: f32,
    half_side: f32,
    angle_step: f32,
    cell_x: i32,
    p: [f32; 3],
) -> bool {
    let dx = p[0] - center[0];
    let dy = p[1] - center[1];
    let dz = p[2] - center[2];
    let rho = (dx * dx + dz * dz).sqrt();
    let angle = dz.atan2(dx);
    let lo = -std::f32::consts::PI + cell_x as f32 * angle_step;
    let hi = lo + angle_step;
    rho >= radius - half_side - 1e-5
        && rho <= radius + half_side + 1e-5
        && dy >= -half_side - 1e-5
        && dy <= half_side + 1e-5
        && angle >= lo - 1e-5
        && angle <= hi + 1e-5
}

fn cylinder_roots_y(oc: [f32; 3], dir: [f32; 3], radius: f32) -> [Option<f32>; 2] {
    let a = dir[0] * dir[0] + dir[2] * dir[2];
    if a < 1e-12 {
        return [None, None];
    }
    let b = 2.0 * (oc[0] * dir[0] + oc[2] * dir[2]);
    let c = oc[0] * oc[0] + oc[2] * oc[2] - radius * radius;
    let disc = b * b - 4.0 * a * c;
    if disc < 0.0 {
        return [None, None];
    }
    let sq = disc.sqrt();
    let inv_2a = 0.5 / a;
    [Some((-b - sq) * inv_2a), Some((-b + sq) * inv_2a)]
}

fn meridian_t(oc: [f32; 3], dir: [f32; 3], angle: f32) -> Option<f32> {
    let n = [-angle.sin(), 0.0, angle.cos()];
    let denom = dot(dir, n);
    if denom.abs() < 1e-12 {
        return None;
    }
    Some(-dot(oc, n) / denom)
}

fn curved_face(
    center: [f32; 3],
    radius: f32,
    half_side: f32,
    angle_step: f32,
    cell_x: i32,
    p: [f32; 3],
) -> u32 {
    let dx = p[0] - center[0];
    let dy = p[1] - center[1];
    let dz = p[2] - center[2];
    let rho = (dx * dx + dz * dz).sqrt();
    let angle = dz.atan2(dx);
    let lo = -std::f32::consts::PI + cell_x as f32 * angle_step;
    let hi = lo + angle_step;
    let candidates = [
        ((rho - (radius + half_side)).abs(), if dx >= 0.0 { 0 } else { 1 }),
        ((rho - (radius - half_side)).abs(), if dx >= 0.0 { 1 } else { 0 }),
        ((dy - half_side).abs(), 2),
        ((dy + half_side).abs(), 3),
        ((angle - lo).abs() * radius, 5),
        ((angle - hi).abs() * radius, 4),
    ];
    candidates
        .into_iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, face)| face)
        .unwrap_or(2)
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
