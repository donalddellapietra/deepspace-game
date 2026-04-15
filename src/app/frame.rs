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

use crate::world::anchor::{Path, WorldPos, WORLD_SIZE};
use crate::world::cubesphere::{self as cs, Face, FACE_SLOTS};
use crate::world::tree::{slot_coords, Child, NodeId, NodeKind, NodeLibrary};
use crate::world::sdf;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
struct BodyWorldBounds {
    center: [f32; 3],
    outer_r: f32,
}

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
    pub surface_r: f32,
    pub noise_scale: f32,
    pub noise_freq: f32,
    pub noise_seed: u32,
    pub surface_block: u8,
    pub core_block: u8,
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
            if let NodeKind::CubedSphereBody { inner_r, outer_r, .. } = k {
                return Some((*inner_r, *outer_r));
            }
        }
        None
    }

    pub fn body_kind(&self) -> Option<NodeKind> {
        self.kinds.iter().copied().find(|k| matches!(k, NodeKind::CubedSphereBody { .. }))
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

pub fn face_frame_info(frame: &Path, chain: &FrameKindChain) -> Option<FaceFrameInfo> {
    let (body_depth, face) = chain.body_and_face(frame)?;
    let NodeKind::CubedSphereBody {
        inner_r,
        outer_r,
        surface_r,
        noise_scale,
        noise_freq,
        noise_seed,
        surface_block,
        core_block,
    } = chain.body_kind()? else {
        return None;
    };
    let face_subtree_slots = &frame.as_slice()[(body_depth + 1)..];
    let (u_lo, v_lo, r_lo, size) = face_cell_bounds_from_path(face_subtree_slots);
    Some(FaceFrameInfo {
        face,
        body_depth,
        subtree_depth: frame.depth() as usize - (body_depth + 1),
        inner_r,
        outer_r,
        surface_r,
        noise_scale,
        noise_freq,
        noise_seed,
        surface_block,
        core_block,
        u_lo,
        v_lo,
        r_lo,
        size,
    })
}

pub fn position_in_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    frame: &Path,
    position: &WorldPos,
) -> [f32; 3] {
    position.in_frame_in(library, world_root, frame)
}

fn body_local_point_to_face_frame(point_body: [f32; 3], info: &FaceFrameInfo) -> [f32; 3] {
    let point_local = [
        point_body[0] / WORLD_SIZE,
        point_body[1] / WORLD_SIZE,
        point_body[2] / WORLD_SIZE,
    ];
    let coord = cs::world_to_coord([0.5, 0.5, 0.5], point_local)
        .unwrap_or(cs::CubeSphereCoord {
            face: info.face,
            u: 0.0,
            v: 0.0,
            r: info.inner_r,
        });
    let un = (coord.u + 1.0) * 0.5;
    let vn = (coord.v + 1.0) * 0.5;
    let rn = (coord.r - info.inner_r) / (info.outer_r - info.inner_r).max(1e-30);
    let scale = WORLD_SIZE / info.size.max(1e-30);
    [
        (un - info.u_lo) * scale,
        (vn - info.v_lo) * scale,
        (rn - info.r_lo) * scale,
    ]
}

pub fn frame_point_to_body(point: [f32; 3], info: &FaceFrameInfo) -> [f32; 3] {
    let un = (info.u_lo + (point[0] / WORLD_SIZE) * info.size).clamp(0.0, 1.0 - f32::EPSILON);
    let vn = (info.v_lo + (point[1] / WORLD_SIZE) * info.size).clamp(0.0, 1.0 - f32::EPSILON);
    let rn = (info.r_lo + (point[2] / WORLD_SIZE) * info.size).clamp(0.0, 1.0 - f32::EPSILON);
    let point_local = cs::coord_to_world(
        [0.5, 0.5, 0.5],
        cs::CubeSphereCoord {
            face: info.face,
            u: un * 2.0 - 1.0,
            v: vn * 2.0 - 1.0,
            r: info.inner_r + rn * (info.outer_r - info.inner_r),
        },
    );
    [
        point_local[0] * WORLD_SIZE,
        point_local[1] * WORLD_SIZE,
        point_local[2] * WORLD_SIZE,
    ]
}

pub fn face_frame_dir_to_body(point: [f32; 3], dir: [f32; 3], info: &FaceFrameInfo) -> [f32; 3] {
    let eps = (info.size * WORLD_SIZE * 1e-3).max(1e-5);
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

pub fn body_dir_to_frame(point_body: [f32; 3], body_dir: [f32; 3], info: &FaceFrameInfo) -> [f32; 3] {
    let eps = (info.size * WORLD_SIZE * 1e-3).max(1e-5);
    let p0 = body_local_point_to_face_frame(point_body, info);
    let p1 = body_local_point_to_face_frame([
        point_body[0] + body_dir[0] * eps,
        point_body[1] + body_dir[1] * eps,
        point_body[2] + body_dir[2] * eps,
    ], info);
    sdf::normalize([
        p1[0] - p0[0],
        p1[1] - p0[1],
        p1[2] - p0[2],
    ])
}

fn body_center_world(frame_path: &Path, body_depth: usize) -> ([f32; 3], f32) {
    let mut origin = [0.0f32; 3];
    let mut size = WORLD_SIZE;
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
    ([
        origin[0] + size * 0.5,
        origin[1] + size * 0.5,
        origin[2] + size * 0.5,
    ], size)
}

fn collect_body_bounds(
    library: &NodeLibrary,
    node_id: NodeId,
    origin: [f32; 3],
    size: f32,
    contains_body: &mut HashMap<NodeId, bool>,
    out: &mut Vec<BodyWorldBounds>,
) {
    let Some(node) = library.get(node_id) else { return };
    if let NodeKind::CubedSphereBody { outer_r, .. } = node.kind {
        out.push(BodyWorldBounds {
            center: [
                origin[0] + size * 0.5,
                origin[1] + size * 0.5,
                origin[2] + size * 0.5,
            ],
            outer_r: outer_r * size,
        });
    }
    let child_size = size / 3.0;
    for (slot, child) in node.children.iter().enumerate() {
        let Child::Node(child_id) = child else { continue };
        if !subtree_contains_body(library, *child_id, contains_body) {
            continue;
        }
        let (sx, sy, sz) = slot_coords(slot);
        let child_origin = [
            origin[0] + sx as f32 * child_size,
            origin[1] + sy as f32 * child_size,
            origin[2] + sz as f32 * child_size,
        ];
        collect_body_bounds(
            library,
            *child_id,
            child_origin,
            child_size,
            contains_body,
            out,
        );
    }
}

fn subtree_contains_body(
    library: &NodeLibrary,
    node_id: NodeId,
    memo: &mut HashMap<NodeId, bool>,
) -> bool {
    if let Some(found) = memo.get(&node_id) {
        return *found;
    }
    let Some(node) = library.get(node_id) else {
        memo.insert(node_id, false);
        return false;
    };
    let found = matches!(node.kind, NodeKind::CubedSphereBody { .. }) || node.children.iter().any(|child| {
        match child {
            Child::Node(child_id) => subtree_contains_body(library, *child_id, memo),
            _ => false,
        }
    });
    memo.insert(node_id, found);
    found
}

fn visible_hit_on_body(
    bodies: &[BodyWorldBounds],
    camera_world: [f32; 3],
    dir: [f32; 3],
) -> Option<[f32; 3]> {
    let mut best_t = f32::INFINITY;
    let mut best_hit = None;
    for body in bodies {
        let oc = sdf::sub(camera_world, body.center);
        let b = sdf::dot(oc, dir);
        let c = sdf::dot(oc, oc) - body.outer_r * body.outer_r;
        let disc = b * b - c;
        if disc <= 0.0 {
            continue;
        }
        let sq = disc.sqrt();
        let mut t = -b - sq;
        if t <= 0.0 {
            t = -b + sq;
        }
        if t <= 0.0 || t >= best_t {
            continue;
        }
        best_t = t;
        best_hit = Some([
            camera_world[0] + dir[0] * t,
            camera_world[1] + dir[1] * t,
            camera_world[2] + dir[2] * t,
        ]);
    }
    best_hit
}

fn cartesian_frame_contains_view(
    library: &NodeLibrary,
    world_root: NodeId,
    frame: &Path,
    chain: &FrameKindChain,
    camera_world: [f32; 3],
    forward: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    fov: f32,
    aspect: f32,
) -> bool {
    let mut bodies = Vec::new();
    let mut contains_body = HashMap::new();
    collect_body_bounds(
        library,
        world_root,
        [0.0, 0.0, 0.0],
        WORLD_SIZE,
        &mut contains_body,
        &mut bodies,
    );
    if bodies.is_empty() {
        return true;
    }
    let half_fov_tan = (fov * 0.5).tan();
    let samples = [
        (0.0f32, 0.0f32),
        (-0.9, -0.9),
        (0.9, -0.9),
        (-0.9, 0.9),
        (0.9, 0.9),
    ];
    let margin = 0.1f32;
    for (sx, sy) in samples {
        let dir = sdf::normalize([
            forward[0] + right[0] * sx * aspect * half_fov_tan + up[0] * sy * half_fov_tan,
            forward[1] + right[1] * sx * aspect * half_fov_tan + up[1] * sy * half_fov_tan,
            forward[2] + right[2] * sx * aspect * half_fov_tan + up[2] * sy * half_fov_tan,
        ]);
        let Some(hit) = visible_hit_on_body(&bodies, camera_world, dir) else {
            continue;
        };
        let local = world_point_to_frame(frame, chain, hit);
        if local[0] < -margin || local[0] > WORLD_SIZE + margin
            || local[1] < -margin || local[1] > WORLD_SIZE + margin
            || local[2] < -margin || local[2] > WORLD_SIZE + margin
        {
            return false;
        }
    }
    true
}

pub fn frame_contains_view(
    library: &NodeLibrary,
    world_root: NodeId,
    frame: &Path,
    chain: &FrameKindChain,
    camera_world: [f32; 3],
    forward: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    fov: f32,
    aspect: f32,
) -> bool {
    let Some(info) = face_frame_info(frame, chain) else {
        return cartesian_frame_contains_view(
            library, world_root, frame, chain,
            camera_world, forward, right, up, fov, aspect,
        );
    };
    let (body_center, body_size) = body_center_world(frame, info.body_depth);
    let outer_r = info.outer_r * body_size;
    let half_fov_tan = (fov * 0.5).tan();
    let samples = [
        (0.0f32, 0.0f32),
        (-0.9, -0.9),
        ( 0.9, -0.9),
        (-0.9,  0.9),
        ( 0.9,  0.9),
    ];
    let margin = info.size * 0.05;
    for (sx, sy) in samples {
        let dir = sdf::normalize([
            forward[0] + right[0] * sx * aspect * half_fov_tan + up[0] * sy * half_fov_tan,
            forward[1] + right[1] * sx * aspect * half_fov_tan + up[1] * sy * half_fov_tan,
            forward[2] + right[2] * sx * aspect * half_fov_tan + up[2] * sy * half_fov_tan,
        ]);
        let oc = sdf::sub(camera_world, body_center);
        let b = sdf::dot(oc, dir);
        let c = sdf::dot(oc, oc) - outer_r * outer_r;
        let disc = b * b - c;
        if disc <= 0.0 {
            continue;
        }
        let sq = disc.sqrt();
        let mut t = -b - sq;
        if t <= 0.0 {
            t = -b + sq;
        }
        if t <= 0.0 {
            continue;
        }
        let hit = [
            camera_world[0] + dir[0] * t,
            camera_world[1] + dir[1] * t,
            camera_world[2] + dir[2] * t,
        ];
        let coord = cs::world_to_coord(body_center, hit).unwrap_or(cs::CubeSphereCoord {
            face: info.face,
            u: 0.0,
            v: 0.0,
            r: outer_r,
        });
        if coord.face != info.face {
            return false;
        }
        let un = (coord.u + 1.0) * 0.5;
        let vn = (coord.v + 1.0) * 0.5;
        if un < info.u_lo - margin
            || un > info.u_lo + info.size + margin
            || vn < info.v_lo - margin
            || vn > info.v_lo + info.size + margin
        {
            return false;
        }
    }
    true
}

pub fn world_point_to_frame(
    frame: &Path,
    chain: &FrameKindChain,
    world: [f32; 3],
) -> [f32; 3] {
    if let Some(info) = face_frame_info(frame, chain) {
        let (body_center, body_size) = body_center_world(frame, info.body_depth);
        let point_body = [
            (world[0] - (body_center[0] - body_size * 0.5)) * WORLD_SIZE,
            (world[1] - (body_center[1] - body_size * 0.5)) * WORLD_SIZE,
            (world[2] - (body_center[2] - body_size * 0.5)) * WORLD_SIZE,
        ];
        return body_local_point_to_face_frame(point_body, &info);
    }
    let (frame_origin, frame_size) = frame_origin_size_world(frame);
    let scale = WORLD_SIZE / frame_size;
    [
        (world[0] - frame_origin[0]) * scale,
        (world[1] - frame_origin[1]) * scale,
        (world[2] - frame_origin[2]) * scale,
    ]
}

pub fn world_dir_to_frame(
    frame: &Path,
    chain: &FrameKindChain,
    camera_world: [f32; 3],
    world_dir: [f32; 3],
) -> [f32; 3] {
    let eps = if let Some(info) = face_frame_info(frame, chain) {
        (info.size * WORLD_SIZE * 1e-3).max(1e-5)
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
            NodeKind::CubedSphereBody {
                inner_r: 0.1,
                outer_r: 0.4,
                surface_r: 0.3,
                noise_scale: 0.0,
                noise_freq: 1.0,
                noise_seed: 0,
                surface_block: 1,
                core_block: 2,
            },
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
            NodeKind::CubedSphereBody {
                inner_r: 0.1,
                outer_r: 0.4,
                surface_r: 0.3,
                noise_scale: 0.0,
                noise_freq: 1.0,
                noise_seed: 0,
                surface_block: 1,
                core_block: 2,
            },
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

    #[test]
    fn cartesian_frame_must_contain_visible_planet() {
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
            NodeKind::CubedSphereBody {
                inner_r: 0.3,
                outer_r: 0.49,
                surface_r: 0.4,
                noise_scale: 0.0,
                noise_freq: 1.0,
                noise_seed: 0,
                surface_block: 1,
                core_block: 2,
            },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let camera_world = [1.5, 2.0, 1.5];
        let forward = [0.0, -1.0, 0.0];
        let right = [1.0, 0.0, 0.0];
        let up = [0.0, 0.0, 1.0];

        let mut off_planet = Path::root();
        off_planet.push(slot_index(0, 0, 0) as u8);
        let off_chain = FrameKindChain::build(&lib, root, &off_planet);
        assert!(
            !frame_contains_view(
                &lib,
                root,
                &off_planet,
                &off_chain,
                camera_world,
                forward,
                right,
                up,
                1.2,
                16.0 / 9.0,
            ),
            "deep empty cartesian frame should be rejected when the planet is visible elsewhere",
        );

        let root_frame = Path::root();
        let root_chain = FrameKindChain::build(&lib, root, &root_frame);
        assert!(
            frame_contains_view(
                &lib,
                root,
                &root_frame,
                &root_chain,
                camera_world,
                forward,
                right,
                up,
                1.2,
                16.0 / 9.0,
            ),
            "root frame should contain the visible planet footprint",
        );
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
