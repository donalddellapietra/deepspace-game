//! CPU mirror of the shader's `march_uv_ring`.

use super::HitInfo;
use crate::world::tree::{slot_index, Child, NodeId, NodeLibrary};
use crate::world::uv_ring::{
    uv_ring_angle_step, uv_ring_cell_side, uv_ring_height_lo, uv_ring_radial_lo,
    UV_RING_BODY_CENTER,
};

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
    theta_lo: f32,
    theta_hi: f32,
    radial_lo: f32,
    radial_hi: f32,
    height_lo: f32,
    height_hi: f32,
}

#[derive(Debug, Clone, Copy)]
struct RingShell {
    theta_lo: f32,
    theta_hi: f32,
    radial_lo: f32,
    radial_hi: f32,
    height_lo: f32,
    height_hi: f32,
}

#[derive(Debug, Clone, Copy)]
struct AnchorFrame {
    node_id: NodeId,
    cell: [i32; 3],
    theta_org: f32,
    r_org: f32,
    y_org: f32,
    theta_step: f32,
    r_step: f32,
    y_step: f32,
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
    let shell = RingShell {
        theta_lo: -std::f32::consts::PI,
        theta_hi: std::f32::consts::PI,
        radial_lo: geom.radial_lo,
        radial_hi: geom.radial_lo + geom.side * dims_y as f32,
        height_lo: geom.height_lo,
        height_hi: geom.height_lo + geom.side * dims_z as f32,
    };
    let oc = sub(cam_local, geom.center);
    let mut t = shell_entry_after(cam_local, ray_dir, geom.center, shell, -EPS)?;

    for _ in 0..512 {
        if !t.is_finite() {
            return None;
        }
        let probe_t = t.max(0.0) + EPS;
        let probe = add(cam_local, mul(ray_dir, probe_t));
        if !point_in_shell(probe, geom.center, shell) {
            t = shell_entry_after(cam_local, ray_dir, geom.center, shell, probe_t)?;
            continue;
        }

        let ring_uv = ring_coords(probe, geom.center);
        let cell_x =
            (((ring_uv[0] + std::f32::consts::PI) / geom.angle_step).floor() as i32)
                .clamp(0, dims[0] as i32 - 1);
        let cell_y = ((ring_uv[1] - geom.radial_lo) / geom.side)
            .floor()
            .clamp(0.0, dims_y as f32 - 1.0) as i32;
        let cell_z = ((ring_uv[2] - geom.height_lo) / geom.side)
            .floor()
            .clamp(0.0, dims_z as f32 - 1.0) as i32;
        let cell = ring_cell(&geom, cell_x, cell_y, cell_z);
        let t_next = next_cell_t(cam_local, ray_dir, oc, cell, probe_t);

        let Some(resolved) = resolve_uv_cell(
            library,
            ring_root,
            &frame_chain,
            [cell_x, cell_y, cell_z],
            slab_depth,
        ) else {
            t = t_next + EPS;
            continue;
        };

        match resolved {
            ResolvedCell::Anchor { path, node } => {
                let absolute_slab_depth = frame_path.len() as u32 + slab_depth as u32;
                let remaining_depth = _max_depth.saturating_sub(absolute_slab_depth);
                if remaining_depth == 0 {
                    return Some(HitInfo {
                        path,
                        face: curved_face(cell, geom.center, probe),
                        t: probe_t,
                        place_path: None,
                    });
                }
                if let Some(hit) = raycast_uv_anchor(
                    library,
                    node,
                    path,
                    geom.center,
                    oc,
                    ray_dir,
                    probe_t,
                    t_next,
                    cell,
                    geom.angle_step,
                    geom.side,
                    geom.side,
                    remaining_depth,
                ) {
                    return Some(hit);
                }
            }
            ResolvedCell::Uniform { path } => {
                return Some(HitInfo {
                    path,
                    face: curved_face(cell, geom.center, probe),
                    t: probe_t,
                    place_path: None,
                });
            }
        }

        t = t_next + EPS;
    }

    None
}

fn raycast_uv_anchor(
    library: &NodeLibrary,
    node: NodeId,
    mut path: Vec<(NodeId, usize)>,
    center: [f32; 3],
    oc: [f32; 3],
    ray_dir: [f32; 3],
    t_in: f32,
    t_exit: f32,
    slab_cell: RingCell,
    slab_theta_step: f32,
    slab_r_step: f32,
    slab_y_step: f32,
    max_depth: u32,
) -> Option<HitInfo> {
    let prefix_len = path.len();
    let ray_origin = add(center, oc);
    let mut t = t_in.max(0.0) + EPS;
    let mut stack = Vec::with_capacity(max_depth as usize + 1);
    stack.push(AnchorFrame {
        node_id: node,
        cell: clamp_cell(ring_recompute_cell(
            ray_origin,
            center,
            ray_dir,
            t,
            slab_cell.theta_lo,
            slab_cell.radial_lo,
            slab_cell.height_lo,
            slab_theta_step / 3.0,
            slab_r_step / 3.0,
            slab_y_step / 3.0,
        )),
        theta_org: slab_cell.theta_lo,
        r_org: slab_cell.radial_lo,
        y_org: slab_cell.height_lo,
        theta_step: slab_theta_step / 3.0,
        r_step: slab_r_step / 3.0,
        y_step: slab_y_step / 3.0,
    });

    let mut iterations = 0u32;
    let max_iterations = (max_depth.max(1) * 4096).max(8192);
    loop {
        if iterations >= max_iterations || stack.is_empty() || t > t_exit {
            break;
        }
        iterations += 1;

        let depth = stack.len() - 1;
        let frame = stack[depth];
        let cell = frame.cell;
        if cell[0] < 0 || cell[0] > 2 || cell[1] < 0 || cell[1] > 2 || cell[2] < 0 || cell[2] > 2 {
            stack.pop();
            path.truncate(prefix_len + stack.len());
            if stack.is_empty() {
                break;
            }
            let parent_depth = stack.len() - 1;
            stack[parent_depth].cell = ring_recompute_cell(
                ray_origin,
                center,
                ray_dir,
                t,
                stack[parent_depth].theta_org,
                stack[parent_depth].r_org,
                stack[parent_depth].y_org,
                stack[parent_depth].theta_step,
                stack[parent_depth].r_step,
                stack[parent_depth].y_step,
            );
            continue;
        }

        let cur_cell = RingCell {
            theta_lo: frame.theta_org + cell[0] as f32 * frame.theta_step,
            theta_hi: frame.theta_org + (cell[0] as f32 + 1.0) * frame.theta_step,
            radial_lo: frame.r_org + cell[1] as f32 * frame.r_step,
            radial_hi: frame.r_org + (cell[1] as f32 + 1.0) * frame.r_step,
            height_lo: frame.y_org + cell[2] as f32 * frame.y_step,
            height_hi: frame.y_org + (cell[2] as f32 + 1.0) * frame.y_step,
        };
        let t_next = next_cell_t(ray_origin, ray_dir, oc, cur_cell, t).min(t_exit);
        let slot = slot_index(cell[0] as usize, cell[1] as usize, cell[2] as usize);
        let node_ref = library.get(frame.node_id)?;

        if path.len() > prefix_len + depth {
            path[prefix_len + depth] = (frame.node_id, slot);
        } else {
            path.push((frame.node_id, slot));
        }

        match node_ref.children[slot] {
            Child::Empty | Child::EntityRef(_) => {
                t = advance_anchor_t(t_next, frame);
                stack[depth].cell = ring_recompute_cell(
                    ray_origin,
                    center,
                    ray_dir,
                    t,
                    frame.theta_org,
                    frame.r_org,
                    frame.y_org,
                    frame.theta_step,
                    frame.r_step,
                    frame.y_step,
                );
            }
            Child::Block(_) => {
                let p = add(center, add(oc, mul(ray_dir, t)));
                return Some(HitInfo {
                    path,
                    face: curved_face(cur_cell, center, p),
                    t,
                    place_path: None,
                });
            }
            Child::Node(child) | Child::PlacedNode { node: child, .. } => {
                let child_node = library.get(child)?;
                if child_node.representative_block == crate::world::tree::REPRESENTATIVE_EMPTY {
                    t = advance_anchor_t(t_next, frame);
                    stack[depth].cell = ring_recompute_cell(
                        ray_origin,
                        center,
                        ray_dir,
                        t,
                        frame.theta_org,
                        frame.r_org,
                        frame.y_org,
                        frame.theta_step,
                        frame.r_step,
                        frame.y_step,
                    );
                    continue;
                }
                if depth as u32 + 1 >= max_depth {
                    let p = add(center, add(oc, mul(ray_dir, t)));
                    return Some(HitInfo {
                        path,
                        face: curved_face(cur_cell, center, p),
                        t,
                        place_path: None,
                    });
                }
                stack.push(AnchorFrame {
                    node_id: child,
                    cell: clamp_cell(ring_recompute_cell(
                        ray_origin,
                        center,
                        ray_dir,
                        t,
                        cur_cell.theta_lo,
                        cur_cell.radial_lo,
                        cur_cell.height_lo,
                        frame.theta_step / 3.0,
                        frame.r_step / 3.0,
                        frame.y_step / 3.0,
                    )),
                    theta_org: cur_cell.theta_lo,
                    r_org: cur_cell.radial_lo,
                    y_org: cur_cell.height_lo,
                    theta_step: frame.theta_step / 3.0,
                    r_step: frame.r_step / 3.0,
                    y_step: frame.y_step / 3.0,
                });
            }
        }
    }

    None
}

fn ring_recompute_cell(
    ray_origin: [f32; 3],
    center: [f32; 3],
    ray_dir: [f32; 3],
    t: f32,
    theta_org: f32,
    r_org: f32,
    y_org: f32,
    theta_step: f32,
    r_step: f32,
    y_step: f32,
) -> [i32; 3] {
    let p = add(ray_origin, mul(ray_dir, t));
    let uv = ring_coords(p, center);
    [
        ((uv[0] - theta_org) / theta_step).floor() as i32,
        ((uv[1] - r_org) / r_step).floor() as i32,
        ((uv[2] - y_org) / y_step).floor() as i32,
    ]
}

fn clamp_cell(cell: [i32; 3]) -> [i32; 3] {
    [
        cell[0].clamp(0, 2),
        cell[1].clamp(0, 2),
        cell[2].clamp(0, 2),
    ]
}

fn advance_anchor_t(t_next: f32, frame: AnchorFrame) -> f32 {
    let eps = frame.theta_step.min(frame.r_step).min(frame.y_step) * 1e-4;
    t_next + eps.max(1e-6)
}

fn ring_geom(dims: [u32; 3]) -> RingGeom {
    let side = uv_ring_cell_side(dims);
    RingGeom {
        center: UV_RING_BODY_CENTER,
        radial_lo: uv_ring_radial_lo(dims),
        side,
        angle_step: uv_ring_angle_step(dims),
        height_lo: uv_ring_height_lo(dims),
    }
}

fn ring_cell(geom: &RingGeom, cell_x: i32, cell_y: i32, cell_z: i32) -> RingCell {
    let radial_lo = geom.radial_lo + cell_y as f32 * geom.side;
    let height_lo = geom.height_lo + cell_z as f32 * geom.side;
    let theta_lo = -std::f32::consts::PI + cell_x as f32 * geom.angle_step;
    RingCell {
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

fn ring_coords(p: [f32; 3], center: [f32; 3]) -> [f32; 3] {
    let d = sub(p, center);
    [(d[2]).atan2(d[0]), (d[0] * d[0] + d[2] * d[2]).sqrt(), p[1]]
}

fn point_in_shell(p: [f32; 3], center: [f32; 3], shell: RingShell) -> bool {
    let uv = ring_coords(p, center);
    uv[0] >= shell.theta_lo - EPS
        && uv[0] <= shell.theta_hi + EPS
        && uv[1] >= shell.radial_lo - EPS
        && uv[1] <= shell.radial_hi + EPS
        && uv[2] >= shell.height_lo - EPS
        && uv[2] <= shell.height_hi + EPS
}

fn shell_entry_after(
    origin: [f32; 3],
    dir: [f32; 3],
    center: [f32; 3],
    shell: RingShell,
    after: f32,
) -> Option<f32> {
    let start_t = after.max(0.0);
    if point_in_shell(add(origin, mul(dir, start_t)), center, shell) {
        return Some(start_t);
    }
    let oc = sub(origin, center);
    let mut best = f32::INFINITY;

    for radius in [shell.radial_lo.max(EPS), shell.radial_hi] {
        for t in cylinder_roots_y(oc, dir, radius).into_iter().flatten() {
            shell_consider_t(origin, dir, center, shell, t, after, &mut best);
        }
    }
    for theta in [shell.theta_lo, shell.theta_hi] {
        if let Some(t) = meridian_t(oc, dir, theta) {
            shell_consider_t(origin, dir, center, shell, t, after, &mut best);
        }
    }
    if dir[1].abs() > 1e-8 {
        for y in [shell.height_lo, shell.height_hi] {
            shell_consider_t(origin, dir, center, shell, (y - origin[1]) / dir[1], after, &mut best);
        }
    }

    best.is_finite().then_some(best)
}

fn shell_consider_t(
    origin: [f32; 3],
    dir: [f32; 3],
    center: [f32; 3],
    shell: RingShell,
    t: f32,
    after: f32,
    best: &mut f32,
) {
    if t <= after || t >= *best {
        return;
    }
    let p = add(origin, mul(dir, t + EPS));
    if point_in_shell(p, center, shell) {
        *best = t;
    }
}

fn next_cell_t(
    origin: [f32; 3],
    dir: [f32; 3],
    oc: [f32; 3],
    cell: RingCell,
    after: f32,
) -> f32 {
    let mut best = f32::INFINITY;
    for theta in [cell.theta_lo, cell.theta_hi] {
        if let Some(t) = meridian_t(oc, dir, theta) {
            if t > after && t < best {
                best = t;
            }
        }
    }
    for radius in [cell.radial_lo.max(EPS), cell.radial_hi] {
        for t in cylinder_roots_y(oc, dir, radius).into_iter().flatten() {
            if t > after && t < best {
                best = t;
            }
        }
    }
    if dir[1].abs() > 1e-8 {
        for y in [cell.height_lo, cell.height_hi] {
            let t = (y - origin[1]) / dir[1];
            if t > after && t < best {
                best = t;
            }
        }
    }
    best
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
