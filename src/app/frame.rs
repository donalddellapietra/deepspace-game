//! Render-frame helpers: walking the camera path to find the
//! frame root, transforming AABBs between world and frame coords.
//!
//! The "render frame" is the GPU's view of the world. The shader
//! starts ray marching at the frame root, with the camera in
//! frame-local `[0, 3)³` coordinates. The `WorldPos.in_frame`
//! method gives the camera those coords; this module gives the
//! tree walker that picks the frame and the AABB transforms used
//! by the highlight overlay.
//!
//! All functions here are **pure** — no `App` state — for direct
//! unit testing.

use crate::world::anchor::{Path, WORLD_SIZE};
use crate::world::cubesphere::{self as cs, Face, FACE_SLOTS};
use crate::world::tree::{slot_coords, Child, NodeId, NodeKind, NodeLibrary};
use crate::world::sdf;

/// Per-frame kind info: at each frame level, what NodeKind did we
/// pass through? Used by `cam_local_in_frame` to dispatch position
/// projection correctly when the frame descends through body and
/// face cells.
#[derive(Clone, Debug)]
pub struct FrameKindChain {
    /// NodeId at each depth (0 = world_root; last = frame root).
    pub node_ids: Vec<NodeId>,
    /// NodeKind at each node_id. Same length.
    pub kinds: Vec<NodeKind>,
}

#[derive(Clone, Copy, Debug)]
pub struct FaceFrameInfo {
    pub face: Face,
    pub body_depth: usize,
    pub subtree_depth: usize,
    pub inner_r: f32,
    pub outer_r: f32,
    pub body_center_world: [f32; 3],
    pub body_size_world: f32,
    pub u_lo: f32,
    pub v_lo: f32,
    pub r_lo: f32,
    pub size: f32,
}

impl FrameKindChain {
    /// Walk `frame_path` from `world_root`, recording node IDs and
    /// kinds at each level. Returns the chain (length = frame_path.len() + 1).
    pub fn build(library: &NodeLibrary, world_root: NodeId, frame_path: &Path) -> Self {
        let mut node_ids = Vec::with_capacity(frame_path.depth() as usize + 1);
        let mut kinds = Vec::with_capacity(frame_path.depth() as usize + 1);
        let mut current = world_root;
        if let Some(node) = library.get(current) {
            node_ids.push(current);
            kinds.push(node.kind);
        } else {
            return FrameKindChain { node_ids, kinds };
        }
        for k in 0..frame_path.depth() as usize {
            let slot = frame_path.slot(k) as usize;
            let Some(node) = library.get(current) else { break };
            match node.children[slot] {
                Child::Node(child_id) => {
                    if let Some(child) = library.get(child_id) {
                        current = child_id;
                        node_ids.push(current);
                        kinds.push(child.kind);
                    } else { break; }
                }
                _ => break,
            }
        }
        FrameKindChain { node_ids, kinds }
    }

    /// Face identity of the frame root, if the path passes through
    /// a body then a face. Returns `(body_depth, face_id)` — the
    /// depth at which the body sits in the chain, and which face
    /// the path descended into from the body.
    pub fn body_and_face(&self, frame_path: &Path) -> Option<(usize, Face)> {
        for k in 0..self.kinds.len() {
            if let NodeKind::CubedSphereBody { .. } = self.kinds[k] {
                // Check the slot at frame_path[k] — this is the
                // slot from the body down (to a face or interior).
                if k < frame_path.depth() as usize {
                    let slot = frame_path.slot(k) as usize;
                    for (fi, fs) in FACE_SLOTS.iter().enumerate() {
                        if *fs == slot {
                            return Some((k, Face::from_index(fi as u8)));
                        }
                    }
                }
            }
        }
        None
    }

    /// The deepest kind in the chain (kind of the frame root itself).
    pub fn frame_kind(&self) -> NodeKind {
        self.kinds.last().copied().unwrap_or(NodeKind::Cartesian)
    }

    /// Body radii — walks the chain to find the CubedSphereBody.
    pub fn body_radii(&self) -> Option<(f32, f32)> {
        for k in &self.kinds {
            if let NodeKind::CubedSphereBody { inner_r, outer_r } = k {
                return Some((*inner_r, *outer_r));
            }
        }
        None
    }
}

/// Walk a face subtree path (slots after the body's face slot) and
/// accumulate the face cell's (u_lo, v_lo, r_lo, size) in
/// normalized face coords ∈ [0, 1]³. Kahan compensation for
/// precision at any depth.
pub fn face_cell_bounds_from_path(face_subtree_slots: &[u8]) -> (f32, f32, f32, f32) {
    let (mut u_sum, mut u_comp) = (0.0f32, 0.0f32);
    let (mut v_sum, mut v_comp) = (0.0f32, 0.0f32);
    let (mut r_sum, mut r_comp) = (0.0f32, 0.0f32);
    let mut size = 1.0f32;
    for &slot in face_subtree_slots {
        let (us, vs, rs) = slot_coords(slot as usize);
        let step = size / 3.0;
        let u_add = step * us as f32;
        let v_add = step * vs as f32;
        let r_add = step * rs as f32;
        // Kahan sum
        let yu = u_add - u_comp; let tu = u_sum + yu;
        u_comp = (tu - u_sum) - yu; u_sum = tu;
        let yv = v_add - v_comp; let tv = v_sum + yv;
        v_comp = (tv - v_sum) - yv; v_sum = tv;
        let yr = r_add - r_comp; let tr = r_sum + yr;
        r_comp = (tr - r_sum) - yr; r_sum = tr;
        size = step;
    }
    (u_sum + u_comp, v_sum + v_comp, r_sum + r_comp, size)
}

/// Project a camera's world-XYZ position into a face-cell frame's
/// local `[0, 3)³` coordinates. The frame's cell is the sub-region
/// of the face's (u_norm, v_norm, r_norm) ∈ [0, 1]³ box mapped to
/// [0, 3)³ shader coords. Returns Some(cam_local) when camera is
/// on/near the same face as the frame; None if camera is on a
/// different face (different quadrant of the sphere).
pub fn cam_local_in_face_frame(
    camera_world: [f32; 3],
    body_center_world: [f32; 3],
    body_size_world: f32,
    inner_r_local: f32,
    outer_r_local: f32,
    face: Face,
    u_lo: f32, v_lo: f32, r_lo: f32, size: f32,
) -> [f32; 3] {
    // Camera vector from body center.
    let p = sdf::sub(camera_world, body_center_world);
    let r_world = sdf::length(p);
    let cs_outer = outer_r_local * body_size_world;
    let cs_inner = inner_r_local * body_size_world;
    let shell = cs_outer - cs_inner;
    // Face EA projection:
    let n_axis = face.normal();
    let (u_axis, v_axis) = face.tangents();
    let axis_dot = sdf::dot(sdf::scale(p, 1.0 / r_world.max(1e-30)), n_axis);
    let p_norm = sdf::scale(p, 1.0 / r_world.max(1e-30));
    let cube_u = sdf::dot(p_norm, u_axis) / axis_dot.max(1e-30);
    let cube_v = sdf::dot(p_norm, v_axis) / axis_dot.max(1e-30);
    let u_ea = cs::cube_to_ea(cube_u);
    let v_ea = cs::cube_to_ea(cube_v);
    let un = (u_ea + 1.0) * 0.5;
    let vn = (v_ea + 1.0) * 0.5;
    let rn = (r_world - cs_inner) / shell.max(1e-30);
    // Map (un, vn, rn) from cell's [u_lo, u_lo+size] → [0, 3).
    let scale = WORLD_SIZE / size.max(1e-30);
    [
        (un - u_lo) * scale,
        (vn - v_lo) * scale,
        (rn - r_lo) * scale,
    ]
}

pub fn face_frame_info(frame: &Path, chain: &FrameKindChain) -> Option<FaceFrameInfo> {
    let (body_depth, face) = chain.body_and_face(frame)?;
    let (inner_r, outer_r) = chain.body_radii()?;
    let (body_center_world, body_size_world) = body_center_world(frame, body_depth);
    let face_subtree_slots = &frame.as_slice()[(body_depth + 1)..];
    let (u_lo, v_lo, r_lo, size) = face_cell_bounds_from_path(face_subtree_slots);
    Some(FaceFrameInfo {
        face,
        body_depth,
        subtree_depth: frame.depth() as usize - (body_depth + 1),
        inner_r,
        outer_r,
        body_center_world,
        body_size_world,
        u_lo,
        v_lo,
        r_lo,
        size,
    })
}

/// World-space body center for the given body depth in the frame
/// path. Body's cell origin in world via Cartesian walk (body is
/// a child of a Cartesian ancestor), center = origin + size/2.
pub fn body_center_world(frame_path: &Path, body_depth: usize) -> ([f32; 3], f32) {
    let mut origin = [0.0f32; 3];
    let mut size = WORLD_SIZE;
    // Walk to body_depth (inclusive of the body cell itself).
    for k in 0..=body_depth.min(frame_path.depth() as usize) {
        if k > 0 {
            let (sx, sy, sz) = slot_coords(frame_path.slot(k - 1) as usize);
            let child = size / 3.0;
            origin[0] += sx as f32 * child;
            origin[1] += sy as f32 * child;
            origin[2] += sz as f32 * child;
            size = child;
        }
    }
    let center = [
        origin[0] + size * 0.5,
        origin[1] + size * 0.5,
        origin[2] + size * 0.5,
    ];
    (center, size)
}

/// Compute cam_local for the frame, kind-aware: dispatches on the
/// frame chain's kinds. Use this anywhere camera position is needed
/// in shader/frame coords when the frame may descend into faces.
pub fn cam_local_in_frame(
    camera_world: [f32; 3],
    frame: &Path,
    chain: &FrameKindChain,
) -> [f32; 3] {
    if let Some(info) = face_frame_info(frame, chain) {
        return cam_local_in_face_frame(
            camera_world,
            info.body_center_world,
            info.body_size_world,
            info.inner_r,
            info.outer_r,
            info.face,
            info.u_lo,
            info.v_lo,
            info.r_lo,
            info.size,
        );
    }
    let (frame_origin, frame_size) = frame_origin_size_world(frame);
    let scale = WORLD_SIZE / frame_size;
    [
        (camera_world[0] - frame_origin[0]) * scale,
        (camera_world[1] - frame_origin[1]) * scale,
        (camera_world[2] - frame_origin[2]) * scale,
    ]
}

pub fn world_point_to_frame(
    frame: &Path,
    chain: &FrameKindChain,
    world: [f32; 3],
) -> [f32; 3] {
    cam_local_in_frame(world, frame, chain)
}

pub fn frame_point_to_body(point: [f32; 3], info: &FaceFrameInfo) -> [f32; 3] {
    let un = (info.u_lo + (point[0] / WORLD_SIZE) * info.size).clamp(0.0, 1.0 - f32::EPSILON);
    let vn = (info.v_lo + (point[1] / WORLD_SIZE) * info.size).clamp(0.0, 1.0 - f32::EPSILON);
    let rn = (info.r_lo + (point[2] / WORLD_SIZE) * info.size).clamp(0.0, 1.0 - f32::EPSILON);
    let world = cs::coord_to_world(
        info.body_center_world,
        cs::CubeSphereCoord {
            face: info.face,
            u: un * 2.0 - 1.0,
            v: vn * 2.0 - 1.0,
            r: (info.inner_r + rn * (info.outer_r - info.inner_r)) * info.body_size_world,
        },
    );
    [
        (world[0] - (info.body_center_world[0] - info.body_size_world * 0.5)) * WORLD_SIZE,
        (world[1] - (info.body_center_world[1] - info.body_size_world * 0.5)) * WORLD_SIZE,
        (world[2] - (info.body_center_world[2] - info.body_size_world * 0.5)) * WORLD_SIZE,
    ]
}

pub fn face_frame_dir_to_body(point: [f32; 3], dir: [f32; 3], info: &FaceFrameInfo) -> [f32; 3] {
    let eps = (info.size * info.body_size_world * 1e-3).max(1e-5);
    let p0 = frame_point_to_body(point, info);
    let p1 = frame_point_to_body([
        point[0] + dir[0] * eps,
        point[1] + dir[1] * eps,
        point[2] + dir[2] * eps,
    ], info);
    sdf::normalize([
        p1[0] - p0[0],
        p1[1] - p0[1],
        p1[2] - p0[2],
    ])
}

pub fn world_dir_to_frame(
    frame: &Path,
    chain: &FrameKindChain,
    camera_world: [f32; 3],
    world_dir: [f32; 3],
) -> [f32; 3] {
    let eps = if let Some(info) = face_frame_info(frame, chain) {
        (info.size * info.body_size_world * 1e-3).max(1e-5)
    } else {
        frame_origin_size_world(frame).1 * 1e-3
    };
    let base = world_point_to_frame(frame, chain, camera_world);
    let shifted = world_point_to_frame(
        frame,
        chain,
        [
            camera_world[0] + world_dir[0] * eps,
            camera_world[1] + world_dir[1] * eps,
            camera_world[2] + world_dir[2] * eps,
        ],
    );
    sdf::normalize([
        shifted[0] - base[0],
        shifted[1] - base[1],
        shifted[2] - base[2],
    ])
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
    frame_path: &Path,
    chain: &FrameKindChain,
    aabb_min: [f32; 3],
    aabb_max: [f32; 3],
) -> ([f32; 3], [f32; 3]) {
    let corners = [
        [aabb_min[0], aabb_min[1], aabb_min[2]],
        [aabb_max[0], aabb_min[1], aabb_min[2]],
        [aabb_min[0], aabb_max[1], aabb_min[2]],
        [aabb_max[0], aabb_max[1], aabb_min[2]],
        [aabb_min[0], aabb_min[1], aabb_max[2]],
        [aabb_max[0], aabb_min[1], aabb_max[2]],
        [aabb_min[0], aabb_max[1], aabb_max[2]],
        [aabb_max[0], aabb_max[1], aabb_max[2]],
    ];
    let mut mn = [f32::INFINITY; 3];
    let mut mx = [f32::NEG_INFINITY; 3];
    for corner in corners {
        let p = world_point_to_frame(frame_path, chain, corner);
        for axis in 0..3 {
            mn[axis] = mn[axis].min(p[axis]);
            mx[axis] = mx[axis].max(p[axis]);
        }
    }
    (mn, mx)
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

/// Render-frame walker: descends the camera's path from
/// `world_root` all the way to `desired_depth`, including face
/// subtrees. Any `Node` child on the camera path is a legal frame.
///
/// Returns `(frame_path, frame_node_id)`.
pub fn compute_render_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    camera_anchor: &Path,
    desired_depth: u8,
) -> (Path, NodeId) {
    let mut frame = *camera_anchor;
    frame.truncate(desired_depth);
    let mut node_id = world_root;
    let mut reached = 0u8;
    for k in 0..frame.depth() as usize {
        let Some(node) = library.get(node_id) else { break };
        let slot = frame.slot(k) as usize;
        match node.children[slot] {
            Child::Node(child_id) => {
                let Some(child) = library.get(child_id) else { break };
                let _ = child.kind;
                node_id = child_id;
                reached = (k as u8) + 1;
            }
            Child::Block(_) | Child::Empty => break,
        }
    }
    frame.truncate(reached);
    (frame, node_id)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::tree::{empty_children, slot_index, uniform_children};

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
        let (frame, node_id) = compute_render_frame(&lib, root, &anchor, 0);
        assert_eq!(frame.depth(), 0);
        assert_eq!(node_id, root);
    }

    #[test]
    fn render_frame_descends_through_cartesian() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..4 { anchor.push(13); }
        let (frame, _node_id) = compute_render_frame(&lib, root, &anchor, 3);
        assert_eq!(frame.depth(), 3);
    }

    #[test]
    fn render_frame_stops_at_face_cell() {
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
        let (frame, node_id) = compute_render_frame(&lib, root, &anchor, 2);
        assert_eq!(frame.depth(), 2, "descends into face cell");
        assert_eq!(node_id, face);
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
        let (frame, node_id) = compute_render_frame(&lib, root, &anchor, 1);
        assert_eq!(frame.depth(), 1);
        assert_eq!(node_id, body);
    }

    #[test]
    fn render_frame_truncates_when_camera_anchor_shallow() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        anchor.push(13);
        let (frame, _node_id) = compute_render_frame(&lib, root, &anchor, 5);
        assert!(frame.depth() <= 1);
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
        let (frame, _) = compute_render_frame(&lib, root, &anchor, 2);
        assert_eq!(frame.depth(), 0, "Block child terminates descent");
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
        let chain = FrameKindChain::build(&NodeLibrary::default(), 0, &p);
        let (mn, mx) = aabb_world_to_frame(&p, &chain, [0.5, 0.5, 0.5], [1.5, 1.5, 1.5]);
        assert!((mn[0] - 0.5).abs() < 1e-6);
        assert!((mx[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn aabb_body_frame_scales_3x() {
        let mut p = Path::root();
        p.push(slot_index(1, 1, 1) as u8);
        let chain = FrameKindChain::build(&NodeLibrary::default(), 0, &p);
        let (mn, mx) = aabb_world_to_frame(&p, &chain, [1.4, 1.4, 1.4], [1.6, 1.6, 1.6]);
        assert!((mn[0] - 1.2).abs() < 1e-5);
        assert!((mx[0] - 1.8).abs() < 1e-5);
    }

    #[test]
    fn aabb_round_trip_at_arbitrary_depth() {
        let mut p = Path::root();
        p.push(slot_index(2, 2, 2) as u8);
        let world_min = [2.5, 2.5, 2.5];
        let world_max = [2.7, 2.7, 2.7];
        let chain = FrameKindChain::build(&NodeLibrary::default(), 0, &p);
        let (mn, mx) = aabb_world_to_frame(&p, &chain, world_min, world_max);
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
        let (frame_origin, _) = frame_origin_size_world(&p);
        let world_min = frame_origin;
        let world_max = [
            frame_origin[0] + 0.01,
            frame_origin[1] + 0.01,
            frame_origin[2] + 0.01,
        ];
        let chain = FrameKindChain::build(&NodeLibrary::default(), 0, &p);
        let (mn, mx) = aabb_world_to_frame(&p, &chain, world_min, world_max);
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
