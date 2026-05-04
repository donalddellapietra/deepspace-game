//! CPU mirror of the shader's `march_uv_ring`.

use super::HitInfo;
use crate::world::tree::{slot_index, Child, NodeId, NodeLibrary};

const BODY_CENTER: [f32; 3] = [1.5, 1.5, 1.5];
const BODY_SIZE: f32 = 3.0;
const EPS: f32 = 1e-5;

#[derive(Debug, Clone, Copy)]
struct RingGeom {
    center: [f32; 3],
    radial_lo: f32,
    side: f32,
    angle_step: f32,
    height_lo: f32,
}

#[derive(Debug, Clone, Copy)]
struct RingCell {
    tangent: [f32; 3],
    radial: [f32; 3],
    up: [f32; 3],
    origin: [f32; 3],
    scale: f32,
    theta_lo: f32,
    theta_hi: f32,
    radial_lo: f32,
    radial_hi: f32,
    height_lo: f32,
    height_hi: f32,
}

enum ResolvedCell {
    Anchor {
        path: Vec<(NodeId, usize)>,
        node: NodeId,
    },
    Uniform {
        path: Vec<(NodeId, usize)>,
    },
}

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
    if dims[0] == 0 || slab_depth == 0 {
        return None;
    }

    let mut frame_chain = Vec::with_capacity(frame_path.len());
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
    let geom = ring_geom(dims);
    let dims_y = dims[1].max(1) as i32;
    let dims_z = dims[2].max(1) as i32;
    let mut best: Option<HitInfo> = None;

    for cell_z in 0..dims_z {
        for cell_y in 0..dims_y {
            for cell_x in 0..dims[0] as i32 {
                let Some(resolved) = resolve_uv_cell(
                    library,
                    ring_root,
                    &frame_chain,
                    [cell_x, cell_y, cell_z],
                    slab_depth,
                ) else {
                    continue;
                };

                let cell = ring_cell(&geom, cell_x, cell_y, cell_z);
                let Some((t, face)) = curved_cell_hit(cell, geom.center, cam_local, ray_dir) else {
                    continue;
                };

                if let ResolvedCell::Anchor { mut path, node } = resolved {
                    let absolute_slab_depth = frame_path.len() as u32 + slab_depth as u32;
                    let cube_max_depth = _max_depth.saturating_sub(absolute_slab_depth).max(1);
                    let entry = add(cam_local, mul(ray_dir, t + EPS));
                    let mut local_origin = ring_point_to_cell_local(cell, entry);
                    local_origin = [
                        local_origin[0].clamp(EPS, 3.0 - EPS),
                        local_origin[1].clamp(EPS, 3.0 - EPS),
                        local_origin[2].clamp(EPS, 3.0 - EPS),
                    ];
                    let local_dir = ring_dir_to_cell_local(cell, ray_dir);
                    if let Some(sub_hit) =
                        super::cpu_raycast(library, node, local_origin, local_dir, cube_max_depth)
                    {
                        path.extend(sub_hit.path);
                        let hit = HitInfo {
                            path,
                            face: sub_hit.face,
                            t,
                            place_path: None,
                        };
                        if best.as_ref().map(|old| hit.t < old.t).unwrap_or(true) {
                            best = Some(hit);
                        }
                    }
                    continue;
                }

                let ResolvedCell::Uniform { path } = resolved else {
                    unreachable!("anchor cells continue above");
                };
                let hit = HitInfo {
                    path,
                    face,
                    t,
                    place_path: None,
                };
                if best.as_ref().map(|old| hit.t < old.t).unwrap_or(true) {
                    best = Some(hit);
                }
            }
        }
    }

    best
}

fn ring_geom(dims: [u32; 3]) -> RingGeom {
    let dims_x = dims[0].max(1) as f32;
    let dims_y = dims[1].max(1) as f32;
    let dims_z = dims[2].max(1) as f32;
    let radius = BODY_SIZE * 0.38;
    let side = ((2.0 * std::f32::consts::PI * radius / dims_x) * 0.95).max(BODY_SIZE / 27.0);
    let angle_step = 2.0 * std::f32::consts::PI / dims_x;
    let height = side * dims_z;
    RingGeom {
        center: BODY_CENTER,
        radial_lo: radius - side * dims_y * 0.5,
        side,
        angle_step,
        height_lo: BODY_CENTER[1] - height * 0.5,
    }
}

fn ring_cell(geom: &RingGeom, cell_x: i32, cell_y: i32, cell_z: i32) -> RingCell {
    let radial_lo = geom.radial_lo + cell_y as f32 * geom.side;
    let height_lo = geom.height_lo + cell_z as f32 * geom.side;
    let theta_lo = -std::f32::consts::PI + cell_x as f32 * geom.angle_step;
    let theta_mid = theta_lo + geom.angle_step * 0.5;
    let (sa, ca) = theta_mid.sin_cos();
    let tangent = [-sa, 0.0, ca];
    let radial = [ca, 0.0, sa];
    let up = [0.0, 1.0, 0.0];
    let radial_mid = (radial_lo + geom.side * 0.5).max(EPS);
    let height_mid = height_lo + geom.side * 0.5;
    let origin = [
        geom.center[0] + radial[0] * radial_mid,
        height_mid,
        geom.center[2] + radial[2] * radial_mid,
    ];
    RingCell {
        tangent,
        radial,
        up,
        origin,
        scale: 3.0 / geom.side,
        theta_lo,
        theta_hi: theta_lo + geom.angle_step,
        radial_lo: radial_lo.max(EPS),
        radial_hi: radial_lo + geom.side,
        height_lo,
        height_hi: height_lo + geom.side,
    }
}

fn resolve_uv_cell(
    library: &NodeLibrary,
    ring_root: NodeId,
    frame_chain: &[(NodeId, usize)],
    cell: [i32; 3],
    slab_depth: u8,
) -> Option<ResolvedCell> {
    let mut path = frame_chain.to_vec();
    let mut node_id = ring_root;
    let mut cells_per_slot = 3_i32.pow(slab_depth.saturating_sub(1) as u32);

    for level in 0..slab_depth {
        let sx = ((cell[0] / cells_per_slot) % 3) as usize;
        let sy = ((cell[1] / cells_per_slot) % 3) as usize;
        let sz = ((cell[2] / cells_per_slot) % 3) as usize;
        let slot = slot_index(sx, sy, sz);
        path.push((node_id, slot));

        let node = library.get(node_id)?;
        match node.children[slot] {
            Child::Empty | Child::EntityRef(_) => return None,
            Child::Block(_) => return Some(ResolvedCell::Uniform { path }),
            Child::Node(child) | Child::PlacedNode { node: child, .. } => {
                if level + 1 == slab_depth {
                    return Some(ResolvedCell::Anchor { path, node: child });
                }
                node_id = child;
            }
        }
        cells_per_slot = (cells_per_slot / 3).max(1);
    }

    Some(ResolvedCell::Uniform { path })
}

fn ring_point_to_cell_local(cell: RingCell, p: [f32; 3]) -> [f32; 3] {
    let d = sub(p, cell.origin);
    [
        dot(cell.tangent, d) * cell.scale + 1.5,
        dot(cell.radial, d) * cell.scale + 1.5,
        dot(cell.up, d) * cell.scale + 1.5,
    ]
}

fn ring_dir_to_cell_local(cell: RingCell, d: [f32; 3]) -> [f32; 3] {
    [
        dot(cell.tangent, d) * cell.scale,
        dot(cell.radial, d) * cell.scale,
        dot(cell.up, d) * cell.scale,
    ]
}

fn curved_cell_hit(
    cell: RingCell,
    center: [f32; 3],
    origin: [f32; 3],
    dir: [f32; 3],
) -> Option<(f32, u32)> {
    let oc = sub(origin, center);
    let mut best = f32::INFINITY;

    consider_t(0.0, cell, center, origin, dir, &mut best);

    for radius in [cell.radial_lo, cell.radial_hi] {
        for t in cylinder_roots_y(oc, dir, radius).into_iter().flatten() {
            consider_t(t, cell, center, origin, dir, &mut best);
        }
    }

    if dir[1].abs() > 1e-8 {
        for y in [cell.height_lo, cell.height_hi] {
            consider_t(
                (y - origin[1]) / dir[1],
                cell,
                center,
                origin,
                dir,
                &mut best,
            );
        }
    }

    for theta in [cell.theta_lo, cell.theta_hi] {
        if let Some(t) = meridian_t(oc, dir, theta) {
            consider_t(t, cell, center, origin, dir, &mut best);
        }
    }

    best.is_finite().then(|| {
        let p = add(origin, mul(dir, best));
        (best, curved_face(cell, center, p))
    })
}

fn consider_t(
    t: f32,
    cell: RingCell,
    center: [f32; 3],
    origin: [f32; 3],
    dir: [f32; 3],
    best: &mut f32,
) {
    if t < -EPS || t >= *best {
        return;
    }
    let probe_t = t.max(0.0) + EPS;
    let p = add(origin, mul(dir, probe_t));
    if point_in_cell(cell, center, p) {
        *best = t.max(0.0);
    }
}

fn point_in_cell(cell: RingCell, center: [f32; 3], p: [f32; 3]) -> bool {
    let local = sub(p, center);
    let rho = (local[0] * local[0] + local[2] * local[2]).sqrt();
    let angle = local[2].atan2(local[0]);
    rho >= cell.radial_lo - EPS
        && rho <= cell.radial_hi + EPS
        && p[1] >= cell.height_lo - EPS
        && p[1] <= cell.height_hi + EPS
        && angle_in_cell(angle, cell.theta_lo, cell.theta_hi)
}

fn angle_in_cell(angle: f32, lo: f32, hi: f32) -> bool {
    angle >= lo - EPS && angle <= hi + EPS
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

fn meridian_t(oc: [f32; 3], dir: [f32; 3], theta: f32) -> Option<f32> {
    let n = [-theta.sin(), 0.0, theta.cos()];
    let denom = dot(dir, n);
    if denom.abs() < 1e-12 {
        return None;
    }
    Some(-dot(oc, n) / denom)
}

fn curved_face(cell: RingCell, center: [f32; 3], p: [f32; 3]) -> u32 {
    let local = sub(p, center);
    let rho = (local[0] * local[0] + local[2] * local[2]).sqrt();
    let angle = local[2].atan2(local[0]);
    let candidates = [
        (
            (rho - cell.radial_hi).abs(),
            if local[0] >= 0.0 { 0 } else { 1 },
        ),
        (
            (rho - cell.radial_lo).abs(),
            if local[0] >= 0.0 { 1 } else { 0 },
        ),
        ((p[1] - cell.height_hi).abs(), 2),
        ((p[1] - cell.height_lo).abs(), 3),
        ((angle - cell.theta_hi).abs() * cell.radial_hi, 4),
        ((angle - cell.theta_lo).abs() * cell.radial_hi, 5),
    ];
    candidates
        .into_iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(_, face)| face)
        .unwrap_or(2)
}

#[inline]
fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

#[inline]
fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

#[inline]
fn mul(a: [f32; 3], s: f32) -> [f32; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

#[inline]
fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::palette::block;
    use crate::world::tree::{empty_children, uniform_children, NodeKind, NodeLibrary};

    fn uniform_subtree(library: &mut NodeLibrary, depth: u8) -> Child {
        if depth == 0 {
            return Child::Block(block::GRASS);
        }
        let child = uniform_subtree(library, depth - 1);
        Child::Node(library.insert(uniform_children(child)))
    }

    fn ring_world(dims: [u32; 3]) -> (NodeLibrary, NodeId) {
        ring_world_with_content_depth(dims, 1)
    }

    fn ring_world_with_content_depth(dims: [u32; 3], content_depth: u8) -> (NodeLibrary, NodeId) {
        let mut library = NodeLibrary::default();
        let content = uniform_subtree(&mut library, content_depth);
        let mut leaves = vec![Child::Empty; 27 * 27 * 27];
        for z in 0..dims[2] as usize {
            for y in 0..dims[1].max(1) as usize {
                for x in 0..dims[0] as usize {
                    leaves[(z * 27 + y) * 27 + x] = content;
                }
            }
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
                            NodeKind::UvRing {
                                dims,
                                slab_depth: 3,
                            }
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
        let dims = [27, 1, 1];
        let (library, root) = ring_world(dims);
        let hit = cpu_raycast_uv_ring(
            &library,
            root,
            &[],
            [3.0, 1.5, 1.5],
            [-1.0, 0.0, 0.0],
            dims,
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

    #[test]
    fn ray_to_upper_cylinder_row_hits_upper_uv_cell() {
        let dims = [27, 1, 2];
        let (library, root) = ring_world(dims);
        let geom = ring_geom(dims);
        let hit = cpu_raycast_uv_ring(
            &library,
            root,
            &[],
            [3.0, 1.5 + geom.side * 0.25, 1.5],
            [-1.0, 0.0, 0.0],
            dims,
            3,
            5,
        )
        .expect("ray toward upper cylinder row must hit");
        assert_eq!(
            hit.path[2].1,
            slot_index(1, 0, 1),
            "slab leaf must select UV cell x=13, y=0, z=1",
        );
    }

    #[test]
    fn ray_to_outer_radial_shell_hits_outer_uv_cell() {
        let dims = [27, 2, 2];
        let (library, root) = ring_world(dims);
        let geom = ring_geom(dims);
        let hit = cpu_raycast_uv_ring(
            &library,
            root,
            &[],
            [3.0, 1.5 + geom.side * 0.25, 1.5],
            [-1.0, 0.0, 0.0],
            dims,
            3,
            5,
        )
        .expect("ray toward outer radial shell must hit");
        assert_eq!(
            hit.path[2].1,
            slot_index(1, 1, 1),
            "slab leaf must select UV cell x=13, y=1, z=1",
        );
    }

    #[test]
    fn ray_to_lower_cylinder_row_hits_lower_uv_cell() {
        let dims = [27, 1, 2];
        let (library, root) = ring_world(dims);
        let geom = ring_geom(dims);
        let hit = cpu_raycast_uv_ring(
            &library,
            root,
            &[],
            [3.0, 1.5 - geom.side * 0.25, 1.5],
            [-1.0, 0.0, 0.0],
            dims,
            3,
            5,
        )
        .expect("ray toward lower cylinder row must hit");
        assert_eq!(
            hit.path[2].1,
            slot_index(1, 0, 0),
            "slab leaf must select UV cell x=13, y=0, z=0",
        );
    }

    #[test]
    fn outside_ring_hit_descends_to_requested_edit_depth() {
        let dims = [27, 1, 2];
        let (library, root) = ring_world_with_content_depth(dims, 20);
        let hit = cpu_raycast_uv_ring(
            &library,
            root,
            &[],
            [3.0, 1.5, 1.5],
            [-1.0, 0.0, 0.0],
            dims,
            3,
            18,
        )
        .expect("outside ray must hit deep ring content");
        assert_eq!(
            hit.path.len(),
            18,
            "edit depth 18 should not return the whole anchor subtree",
        );
    }
}
