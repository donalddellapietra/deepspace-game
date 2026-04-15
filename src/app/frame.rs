//! Render-frame helpers: walking the camera path to find the
//! active frame, transforming positions/AABBs into that frame.
//!
//! The "render frame" is the GPU's view of the world. The shader
//! starts ray marching at a frame root, with the camera expressed
//! in that frame's coordinates. Cartesian frames are linear
//! `[0, 3)³`; sphere frames stay rooted at the containing body
//! cell but carry an explicit cubed-sphere face-cell window.
//!
//! All functions here are **pure** — no `App` state — for direct
//! unit testing.

use crate::world::anchor::{Path, WORLD_SIZE};
use crate::world::cubesphere::Face;
use crate::world::cubesphere_local;
use crate::world::tree::{slot_coords, Child, NodeId, NodeKind, NodeLibrary};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereFrame {
    pub body_path: Path,
    pub body_node_id: NodeId,
    pub face_root_id: NodeId,
    pub face: Face,
    pub inner_r: f32,
    pub outer_r: f32,
    pub face_u_min: f32,
    pub face_v_min: f32,
    pub face_r_min: f32,
    pub face_size: f32,
    pub frame_path: Path,
    pub face_depth: u32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActiveFrameKind {
    Cartesian,
    Body { inner_r: f32, outer_r: f32 },
    Sphere(SphereFrame),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveFrame {
    /// Path used by the linear ribbon / camera-local transforms.
    pub render_path: Path,
    /// Logical interaction/render layer path. For Cartesian this is
    /// identical to `render_path`; for sphere frames it continues
    /// through the face subtree while the linear render root stays
    /// at the containing body.
    pub logical_path: Path,
    pub node_id: NodeId,
    pub kind: ActiveFrameKind,
}

/// World-space origin and side length of the cell at `path`.
/// Origin is the world XYZ of the cell's `(0, 0, 0)` corner; size
/// is the cell's side length in world units.
///
/// Used to transform between world coords and frame-local shader
/// coords (which map the cell to `[0, WORLD_SIZE)³`).
pub fn frame_origin_size_world(path: &Path) -> ([f32; 3], f32) {
    let mut origin = [0.0f32; 3];
    let mut size = WORLD_SIZE;
    for k in 0..path.depth() as usize {
        let (sx, sy, sz) = slot_coords(path.slot(k) as usize);
        let child = size / 3.0;
        origin[0] += sx as f32 * child;
        origin[1] += sy as f32 * child;
        origin[2] += sz as f32 * child;
        size = child;
    }
    (origin, size)
}

/// Transform an axis-aligned bounding box from world coordinates
/// into the shader-frame coords for `frame_path`. The frame's cell
/// maps to the shader's `[0, 3)³`, so the scale factor is
/// `WORLD_SIZE / frame_cell_size_world`.
pub fn aabb_world_to_frame(
    frame: &ActiveFrame,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let corners = [
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
    ];
    let mut out_min = [f32::INFINITY; 3];
    let mut out_max = [f32::NEG_INFINITY; 3];
    for corner in corners {
        let p = point_world_to_frame(frame, corner);
        for axis in 0..3 {
            out_min[axis] = out_min[axis].min(p[axis]);
            out_max[axis] = out_max[axis].max(p[axis]);
        }
    }
    (out_min, out_max)
}

pub fn point_world_to_frame(frame: &ActiveFrame, point: [f32; 3]) -> [f32; 3] {
    match frame.kind {
        ActiveFrameKind::Cartesian | ActiveFrameKind::Body { .. } => {
            let (frame_origin, frame_size) = frame_origin_size_world(&frame.render_path);
            let scale = WORLD_SIZE / frame_size;
            [
                (point[0] - frame_origin[0]) * scale,
                (point[1] - frame_origin[1]) * scale,
                (point[2] - frame_origin[2]) * scale,
            ]
        }
        ActiveFrameKind::Sphere(sphere) => {
            let body_point = point_world_to_body_frame(&sphere, point);
            let face_point = cubesphere_local::body_point_to_face_space(
                body_point,
                sphere.inner_r,
                sphere.outer_r,
                WORLD_SIZE,
            );
            let (un_abs, vn_abs, rn_abs) = face_point
                .map(|p| (p.un, p.vn, p.rn))
                .unwrap_or((0.5, 0.5, 0.0));
            let scale = WORLD_SIZE / sphere.face_size;
            [
                (un_abs - sphere.face_u_min) * scale,
                (vn_abs - sphere.face_v_min) * scale,
                (rn_abs - sphere.face_r_min) * scale,
            ]
        }
    }
}

pub fn point_world_to_body_frame(sphere: &SphereFrame, point: [f32; 3]) -> [f32; 3] {
    let (body_origin, body_size) = frame_origin_size_world(&sphere.body_path);
    let scale = WORLD_SIZE / body_size;
    [
        (point[0] - body_origin[0]) * scale,
        (point[1] - body_origin[1]) * scale,
        (point[2] - body_origin[2]) * scale,
    ]
}

pub fn aabb_world_to_body_frame(
    sphere: &SphereFrame,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let corners = [
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
    ];
    let mut out_min = [f32::INFINITY; 3];
    let mut out_max = [f32::NEG_INFINITY; 3];
    for corner in corners {
        let p = point_world_to_body_frame(sphere, corner);
        for axis in 0..3 {
            out_min[axis] = out_min[axis].min(p[axis]);
            out_max[axis] = out_max[axis].max(p[axis]);
        }
    }
    (out_min, out_max)
}

pub fn frame_point_to_body(point: [f32; 3], sphere: &SphereFrame) -> [f32; 3] {
    let un = (sphere.face_u_min + (point[0] / WORLD_SIZE) * sphere.face_size)
        .clamp(0.0, 1.0 - f32::EPSILON);
    let vn = (sphere.face_v_min + (point[1] / WORLD_SIZE) * sphere.face_size)
        .clamp(0.0, 1.0 - f32::EPSILON);
    let rn = (sphere.face_r_min + (point[2] / WORLD_SIZE) * sphere.face_size)
        .clamp(0.0, 1.0 - f32::EPSILON);
    cubesphere_local::face_space_to_body_point(
        sphere.face,
        un,
        vn,
        rn,
        sphere.inner_r,
        sphere.outer_r,
        WORLD_SIZE,
    )
}

/// Build a `Path` from the slot prefix the GPU ribbon walker
/// actually reached. This is the renderer's effective frame.
pub fn frame_from_slots(slots: &[u8]) -> Path {
    let mut frame = Path::root();
    for &slot in slots {
        frame.push(slot);
    }
    frame
}

/// Resolve the active frame. Cartesian zones use a normal linear
/// frame root. Sphere zones keep the linear root at the containing
/// body cell but continue the logical frame through the face
/// subtree, carrying explicit face-cell bounds so render/edit can
/// operate at the same layer depth without absolute-depth caps.
pub fn compute_render_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    camera_anchor: &Path,
    desired_depth: u8,
) -> ActiveFrame {
    let mut target = *camera_anchor;
    target.truncate(desired_depth);
    let mut node_id = world_root;
    let mut reached = Path::root();
    let mut body_info: Option<(Path, NodeId, f32, f32)> = None;
    let mut sphere_info: Option<(Face, NodeId, f32, f32, f32, f32)> = None;
    for k in 0..target.depth() as usize {
        let Some(node) = library.get(node_id) else { break };
        let slot = target.slot(k) as usize;
        match node.children[slot] {
            Child::Node(child_id) => {
                let Some(child) = library.get(child_id) else { break };
                reached.push(slot as u8);
                match child.kind {
                    NodeKind::Cartesian => {
                        node_id = child_id;
                        if let Some((_, _, ref mut u_min, ref mut v_min, ref mut r_min, ref mut size)) = sphere_info {
                            let (us, vs, rs) = slot_coords(slot);
                            let child_size = *size / 3.0;
                            *u_min += us as f32 * child_size;
                            *v_min += vs as f32 * child_size;
                            *r_min += rs as f32 * child_size;
                            *size = child_size;
                        }
                    }
                    NodeKind::CubedSphereBody { inner_r, outer_r } => {
                        node_id = child_id;
                        body_info = Some((reached, child_id, inner_r, outer_r));
                    }
                    NodeKind::CubedSphereFace { face } => {
                        node_id = child_id;
                        if let Some((body_path, body_node_id, inner_r, outer_r)) = body_info {
                            sphere_info = Some((face, child_id, 0.0, 0.0, 0.0, 1.0));
                            body_info = Some((body_path, body_node_id, inner_r, outer_r));
                        }
                    }
                }
            }
            Child::Block(_) | Child::Empty => break,
        }
    }
    if let Some((face, face_root_id, face_u_min, face_v_min, face_r_min, face_size)) = sphere_info {
        let (body_path, body_node_id, inner_r, outer_r) =
            body_info.expect("sphere frame requires containing body");
        ActiveFrame {
            render_path: reached,
            logical_path: reached,
            node_id,
            kind: ActiveFrameKind::Sphere(SphereFrame {
                body_path,
                body_node_id,
                face_root_id,
                face,
                inner_r,
                outer_r,
                face_u_min,
                face_v_min,
                face_r_min,
                face_size,
                frame_path: reached,
                face_depth: reached.depth().saturating_sub(body_path.depth() + 1) as u32,
            }),
        }
    } else {
        let kind = library.get(node_id).map(|n| n.kind).unwrap_or(NodeKind::Cartesian);
        ActiveFrame {
            render_path: reached,
            logical_path: reached,
            node_id,
            kind: match kind {
                NodeKind::CubedSphereBody { inner_r, outer_r } => {
                    ActiveFrameKind::Body { inner_r, outer_r }
                }
                _ => ActiveFrameKind::Cartesian,
            },
        }
    }
}

pub fn with_render_margin(
    library: &NodeLibrary,
    world_root: NodeId,
    logical_path: &Path,
    render_margin: u8,
) -> ActiveFrame {
    let logical = compute_render_frame(library, world_root, logical_path, logical_path.depth());
    let min_render_depth = match logical.kind {
        ActiveFrameKind::Sphere(sphere) => (sphere.body_path.depth() + 1).min(logical.logical_path.depth()),
        ActiveFrameKind::Body { .. } => logical.logical_path.depth(),
        ActiveFrameKind::Cartesian => 0,
    };
    let render_depth = logical
        .logical_path
        .depth()
        .saturating_sub(render_margin)
        .max(min_render_depth);
    if render_depth == logical.logical_path.depth() {
        return logical;
    }

    let mut render_path = logical.logical_path;
    render_path.truncate(render_depth);
    let render = compute_render_frame(library, world_root, &render_path, render_depth);
    ActiveFrame {
        render_path: render.render_path,
        logical_path: logical.logical_path,
        node_id: render.node_id,
        kind: render.kind,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, slot_index, uniform_children};

    fn cartesian_frame(path: Path, node_id: NodeId) -> ActiveFrame {
        ActiveFrame {
            render_path: path,
            logical_path: path,
            node_id,
            kind: ActiveFrameKind::Cartesian,
        }
    }

    fn cartesian_chain(depth: u8) -> (NodeLibrary, NodeId) {
        let mut lib = NodeLibrary::default();
        let mut node = lib.insert(empty_children());
        for _ in 1..depth {
            node = lib.insert(uniform_children(Child::Node(node)));
        }
        lib.ref_inc(node);
        (lib, node)
    }

    // --------- compute_render_frame ---------

    #[test]
    fn render_frame_root_when_desired_depth_zero() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..3 { anchor.push(13); }
        let frame = compute_render_frame(&lib, root, &anchor, 0);
        assert_eq!(frame.render_path.depth(), 0);
        assert_eq!(frame.node_id, root);
    }

    #[test]
    fn render_frame_descends_through_cartesian() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..4 { anchor.push(13); }
        let frame = compute_render_frame(&lib, root, &anchor, 3);
        assert_eq!(frame.render_path.depth(), 3);
    }

    #[test]
    fn render_frame_enters_sphere_face_logically() {
        let mut lib = NodeLibrary::default();
        let face = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereFace {
                face: crate::world::cubesphere::Face::PosX,
            },
        );
        let mut body_children = empty_children();
        body_children[14] = Child::Node(face);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(13);
        anchor.push(14);
        let frame = compute_render_frame(&lib, root, &anchor, 2);
        assert_eq!(frame.render_path.depth(), 2);
        assert_eq!(frame.logical_path.depth(), 2);
        assert_eq!(frame.node_id, face);
        assert!(matches!(frame.kind, ActiveFrameKind::Sphere(_)));
    }

    #[test]
    fn render_frame_descends_into_body() {
        let mut lib = NodeLibrary::default();
        let body = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(13);
        let frame = compute_render_frame(&lib, root, &anchor, 1);
        assert_eq!(frame.render_path.depth(), 1);
        assert_eq!(frame.node_id, body);
    }

    #[test]
    fn render_frame_truncates_when_camera_anchor_shallow() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        anchor.push(13);
        let frame = compute_render_frame(&lib, root, &anchor, 5);
        assert!(frame.render_path.depth() <= 1);
    }

    #[test]
    fn render_frame_stops_when_path_misses_node() {
        // Build root with a Block at slot 5 (not a Node).
        let mut lib = NodeLibrary::default();
        let mut root_children = empty_children();
        root_children[5] = Child::Block(crate::world::palette::block::STONE);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        let mut anchor = Path::root();
        anchor.push(5);
        anchor.push(0);
        let frame = compute_render_frame(&lib, root, &anchor, 2);
        assert_eq!(frame.render_path.depth(), 0, "Block child terminates descent");
    }

    // --------- frame_origin_size_world ---------

    #[test]
    fn origin_size_root() {
        let p = Path::root();
        let (origin, size) = frame_origin_size_world(&p);
        assert_eq!(origin, [0.0, 0.0, 0.0]);
        assert!((size - WORLD_SIZE).abs() < 1e-6);
    }

    #[test]
    fn origin_size_center_slot() {
        let mut p = Path::root();
        p.push(slot_index(1, 1, 1) as u8);
        let (origin, size) = frame_origin_size_world(&p);
        assert!((origin[0] - 1.0).abs() < 1e-6);
        assert!((origin[1] - 1.0).abs() < 1e-6);
        assert!((origin[2] - 1.0).abs() < 1e-6);
        assert!((size - 1.0).abs() < 1e-6);
    }

    #[test]
    fn origin_size_corner_slot() {
        let mut p = Path::root();
        p.push(slot_index(0, 0, 0) as u8);
        let (origin, size) = frame_origin_size_world(&p);
        assert_eq!(origin, [0.0, 0.0, 0.0]);
        assert!((size - 1.0).abs() < 1e-6);
    }

    #[test]
    fn origin_size_two_levels_deep() {
        let mut p = Path::root();
        p.push(slot_index(2, 2, 2) as u8);
        p.push(slot_index(1, 1, 1) as u8);
        let (_origin, size) = frame_origin_size_world(&p);
        assert!((size - (1.0 / 3.0)).abs() < 1e-6);
    }

    #[test]
    fn origin_size_decreases_geometrically() {
        // Each level shrinks size by factor of 3.
        let mut p = Path::root();
        for k in 0..10u8 {
            let (_, size) = frame_origin_size_world(&p);
            let expected = WORLD_SIZE / 3f32.powi(k as i32);
            assert!((size - expected).abs() < 1e-6 * expected.max(1e-9));
            p.push(13);
        }
    }

    // --------- aabb_world_to_frame ---------

    #[test]
    fn aabb_root_frame_is_identity() {
        let p = Path::root();
        let frame = cartesian_frame(p, 0);
        let (mn, mx) = aabb_world_to_frame(&frame, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);
        assert!((mn[0] - 0.5).abs() < 1e-6);
        assert!((mx[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn aabb_body_frame_scales_3x() {
        let mut p = Path::root();
        p.push(slot_index(1, 1, 1) as u8);
        let frame = cartesian_frame(p, 0);
        let (mn, mx) = aabb_world_to_frame(&frame, [1.4, 1.4, 1.4], [1.6, 1.6, 1.6]);
        assert!((mn[0] - 1.2).abs() < 1e-5);
        assert!((mx[0] - 1.8).abs() < 1e-5);
    }

    #[test]
    fn aabb_round_trip_at_arbitrary_depth() {
        let mut p = Path::root();
        p.push(slot_index(2, 2, 2) as u8);
        let world_min = [2.5, 2.5, 2.5];
        let world_max = [2.7, 2.7, 2.7];
        let frame = cartesian_frame(p, 0);
        let (mn, mx) = aabb_world_to_frame(&frame, world_min, world_max);
        let (frame_origin, frame_size) = frame_origin_size_world(&p);
        let back_min = [
            mn[0] * frame_size / WORLD_SIZE + frame_origin[0],
            mn[1] * frame_size / WORLD_SIZE + frame_origin[1],
            mn[2] * frame_size / WORLD_SIZE + frame_origin[2],
        ];
        let back_max = [
            mx[0] * frame_size / WORLD_SIZE + frame_origin[0],
            mx[1] * frame_size / WORLD_SIZE + frame_origin[1],
            mx[2] * frame_size / WORLD_SIZE + frame_origin[2],
        ];
        for i in 0..3 {
            assert!((back_min[i] - world_min[i]).abs() < 1e-5);
            assert!((back_max[i] - world_max[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn aabb_deep_frame_inflates_proportionally() {
        // A frame at depth 3 has cell_size 1/9. AABB of size 0.01
        // in world maps to size 0.01 * 9 = 0.09 in frame.
        let mut p = Path::root();
        p.push(slot_index(1, 1, 1) as u8);
        p.push(slot_index(1, 1, 1) as u8);
        p.push(slot_index(1, 1, 1) as u8);
        let frame = cartesian_frame(p, 0);
        let (frame_origin, _) = frame_origin_size_world(&p);
        let world_min = frame_origin;
        let world_max = [
            frame_origin[0] + 0.01,
            frame_origin[1] + 0.01,
            frame_origin[2] + 0.01,
        ];
        let (mn, mx) = aabb_world_to_frame(&frame, world_min, world_max);
        for i in 0..3 {
            assert!(mn[i].abs() < 1e-5);
            assert!((mx[i] - (0.01 * 27.0)).abs() < 1e-4);
        }
    }

    #[test]
    fn frame_from_slots_builds_exact_prefix() {
        let slots = [13u8, 16u8, 4u8];
        let p = frame_from_slots(&slots);
        assert_eq!(p.depth(), slots.len() as u8);
        assert_eq!(p.as_slice(), &slots);
    }
}
