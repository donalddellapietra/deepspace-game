//! Render-frame resolution.
//!
//! `compute_render_frame(world, camera, desired_depth)` returns the
//! frame the shader + CPU raycast operate in. Three kinds:
//!
//!   * `Cartesian` — slot-XYZ descent through Cartesian nodes.
//!   * `Body` — render root IS a `CubedSphereBody` cell; shader
//!     dispatches the exact whole-sphere march.
//!   * `SphereSub` — render root is a face-subtree cell at
//!     face-subtree depth ≥ `MIN_SPHERE_SUB_DEPTH`. Shader runs
//!     the linearized ribbon-pop DDA in the frame's local
//!     `[0, 3)³`. See `docs/design/sphere-ribbon-pop-impl-plan.md`
//!     and `docs/design/sphere-ribbon-pop-uvr-state.md`.
//!
//! The `SphereSub` path depends on the camera carrying symbolic UVR
//! state (`WorldPos.sphere`). When the camera is inside a body,
//! `compute_render_frame` reads `sphere.uvr_path` directly to build
//! the sub-frame — no attempt to decode UVR from Cartesian anchor
//! slots. The caller passes the `WorldPos`; this module does not
//! touch the `NodeLibrary` for sphere frames at all (body metadata
//! is cached on `SphereState`).
//!
//! Cartesian descent still walks the tree to check that anchor slots
//! point at real nodes.
//!
//! Pure functions; no `App` state.

use crate::world::anchor::{Path, WorldPos};
use crate::world::cubesphere::{face_frame_jacobian, mat3_inv, Face, Mat3, FACE_SLOTS};
use crate::world::sdf::Vec3;
use crate::world::tree::{Child, NodeId, NodeKind, NodeLibrary};

/// Sphere sub-frame: the render root is a face-subtree cell at face
/// depth ≥ `MIN_SPHERE_SUB_DEPTH`. Ray-march runs in the frame's
/// local `[0, 3)³`. `c_body`, `J`, `J_inv` map local ↔ body-XYZ via
/// the linearized face transform at the frame's corner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereSubFrame {
    /// Path from world root to the containing `CubedSphereBody`.
    pub body_path: Path,
    /// Full path world-root → body → face-root → UVR descent. This
    /// is the **render path** consumed by CPU raycast / ribbon-pop
    /// infrastructure — the last entry is the sub-frame's node.
    pub render_path: Path,
    /// Face this sub-frame lives on.
    pub face: Face,
    /// Absolute face-normalized coords of the sub-frame's corner —
    /// local (0,0,0) in face `[0, 1)³`.
    pub un_corner: f32,
    pub vn_corner: f32,
    pub rn_corner: f32,
    /// Size in face-normalized coords. Equals `1/3^M` where M is the
    /// face-subtree depth. Carried as f32 — only precision-critical
    /// combined with `un_corner`, which is itself f32; together they
    /// represent the corner well enough for the Jacobian evaluation.
    /// Sub-cell precision lives in `WorldPos.sphere.uvr_offset`, not
    /// in this field.
    pub frame_size: f32,
    pub inner_r: f32,
    pub outer_r: f32,
    pub c_body: Vec3,
    pub j: Mat3,
    pub j_inv: Mat3,
}

impl SphereSubFrame {
    /// How many UVR descent levels inside the face subtree this
    /// sub-frame represents.
    pub fn depth_levels(&self) -> u32 {
        // render_path = body_path + face_root_slot + uvr_slots.
        // Length − body_path.depth() − 1 (the face-root slot) = UVR depth.
        (self.render_path.depth() as u32)
            .saturating_sub(self.body_path.depth() as u32)
            .saturating_sub(1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActiveFrameKind {
    Cartesian,
    /// Render root is a `CubedSphereBody` cell. Shader runs the full
    /// whole-sphere march in body-local coords.
    Body { inner_r: f32, outer_r: f32 },
    /// Render root is a face-subtree cell at depth ≥ `MIN_SPHERE_SUB_DEPTH`.
    SphereSub(SphereSubFrame),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveFrame {
    /// Path from world root to the render frame's root node. For
    /// `Cartesian` / `Body` this is just the Cartesian descent. For
    /// `SphereSub` this ends at the sub-frame's node.
    pub render_path: Path,
    /// Logical interaction layer — the user's edit-depth anchor
    /// (may include symbolic UVR descent through sphere state).
    pub logical_path: Path,
    pub node_id: NodeId,
    pub kind: ActiveFrameKind,
}

/// Build a `Path` from the slot prefix the GPU ribbon walker
/// actually reached.
pub fn frame_from_slots(slots: &[u8]) -> Path {
    let mut frame = Path::root();
    for &slot in slots {
        frame.push(slot);
    }
    frame
}

/// Face-subtree depth at which the linearized `SphereSub` frame
/// kicks in. Shallower face descents use the exact `Body` march
/// because linearization error is perceptible there.
pub const MIN_SPHERE_SUB_DEPTH: u8 = 3;

/// Walker-budget kept between a `SphereSub` render frame and the
/// camera's logical depth. The sub-frame DDA descends this many
/// levels below the render cell to find edit-depth blocks.
/// `cpu_raycast_in_sub_frame` returns `None` when
/// `edit_depth == render_path.depth`, so a budget of at least 1 is
/// required. For `m_logical < MIN_SPHERE_SUB_DEPTH + SPHERE_WALKER_BUDGET`
/// the frame falls back to Body, whose march can handle shallow
/// sphere interaction without the linearization.
pub const SPHERE_WALKER_BUDGET: u32 = 1;

/// Resolve the active render frame from a camera WorldPos +
/// desired anchor depth.
///
/// * If the camera is in sphere state with enough UVR depth →
///   `SphereSub(sub)` with precomputed Jacobian.
/// * Else if the anchor lands on a `CubedSphereBody` cell →
///   `Body { inner_r, outer_r }`.
/// * Else Cartesian.
pub fn compute_render_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    camera: &WorldPos,
    desired_depth: u8,
) -> ActiveFrame {
    // Sphere case: symbolic UVR state provides everything we need.
    // Build the sub-frame directly, no tree-walk decoding.
    if let Some(sphere) = camera.sphere.as_ref() {
        let logical_m = sphere.uvr_path.depth() as u32;
        // Target face-subtree depth after `desired_depth` truncation.
        // `desired_depth` is the OVERALL anchor depth; subtract the
        // body-path length + 1 for the face-root slot to get the
        // UVR depth we want to render at.
        let body_depth = sphere.body_path.depth() as u32;
        let total_anchor_depth = (body_depth + 1 + logical_m) as i32;
        // Reserve `SPHERE_WALKER_BUDGET` levels between the render
        // frame's uvr depth and the camera's logical uvr depth, so
        // `cs_raycast_local`'s walker can descend into the sub-frame
        // to locate edit-depth cells. Without this reservation the
        // walker_limit collapses to 0 and every raycast misses.
        let m_ceiling = logical_m.saturating_sub(SPHERE_WALKER_BUDGET);
        let m_desired = (desired_depth as u32).saturating_sub(body_depth + 1);
        let m_truncated = m_desired.min(m_ceiling);

        // Accumulate un/vn/rn from the first `m_truncated` UVR slots.
        let (un_corner, vn_corner, rn_corner, frame_size) =
            uvr_corner(&sphere.uvr_path, m_truncated as usize);

        if m_truncated >= MIN_SPHERE_SUB_DEPTH as u32 {
            // Build render_path = body_path + face_root_slot +
            // uvr_path[..m_truncated].
            let mut render_path = sphere.body_path;
            render_path.push(FACE_SLOTS[sphere.face as usize] as u8);
            for k in 0..m_truncated as usize {
                render_path.push(sphere.uvr_path.slot(k));
            }
            // Walk the tree to resolve the sub-frame's node id.
            let Some(node_id) = resolve_node(library, world_root, &render_path) else {
                // Tree doesn't reach this depth (e.g., uniform chain
                // ended early). Fall back to Body.
                let logical_path = build_logical_path(sphere, desired_depth);
                return body_frame(library, world_root, sphere, logical_path);
            };
            let (c_body, j) = face_frame_jacobian(
                sphere.face,
                un_corner, vn_corner, rn_corner, frame_size,
                sphere.inner_r, sphere.outer_r,
                3.0, // body_size in the render-frame convention
            );
            let j_inv = mat3_inv(&j);
            let sub = SphereSubFrame {
                body_path: sphere.body_path,
                render_path,
                face: sphere.face,
                un_corner, vn_corner, rn_corner, frame_size,
                inner_r: sphere.inner_r,
                outer_r: sphere.outer_r,
                c_body, j, j_inv,
            };
            let logical_path = build_logical_path(sphere, desired_depth);
            return ActiveFrame {
                render_path: sub.render_path,
                logical_path,
                node_id,
                kind: ActiveFrameKind::SphereSub(sub),
            };
        }
        // Shallow face-subtree depth → Body march handles it.
        let logical_path = build_logical_path(sphere, desired_depth);
        let _ = total_anchor_depth;
        return body_frame(library, world_root, sphere, logical_path);
    }

    // Cartesian case. Walk the anchor slot by slot, checking each
    // descent resolves to a real Node. Stop early at non-Node
    // children. If we hit a CubedSphereBody, pick Body kind.
    let mut target = camera.anchor;
    target.truncate(desired_depth);

    let mut node_id = world_root;
    let mut reached = Path::root();
    let mut body_meta: Option<(f32, f32)> = None;
    for k in 0..target.depth() as usize {
        let Some(node) = library.get(node_id) else { break };
        let slot = target.slot(k) as usize;
        let Child::Node(child_id) = node.children[slot] else { break };
        let Some(child) = library.get(child_id) else { break };
        match child.kind {
            NodeKind::Cartesian => {
                node_id = child_id;
                reached.push(slot as u8);
            }
            NodeKind::CubedSphereBody { inner_r, outer_r } => {
                node_id = child_id;
                reached.push(slot as u8);
                body_meta = Some((inner_r, outer_r));
                break;
            }
            NodeKind::CubedSphereFace { .. } => {
                // Face root reachable via Cartesian anchor only if
                // the user's XYZ slots coincidentally match a face
                // slot inside a body — but compute_render_frame
                // doesn't consume further UVR semantics from XYZ
                // slots (that's what stage 5 fixes). Stop at body.
                break;
            }
        }
    }

    let kind = match body_meta {
        Some((inner_r, outer_r)) => ActiveFrameKind::Body { inner_r, outer_r },
        None => ActiveFrameKind::Cartesian,
    };
    ActiveFrame {
        render_path: reached,
        logical_path: reached,
        node_id,
        kind,
    }
}

/// Walk `path` from `world_root` through each slot; return the
/// terminal node id if every step resolves to a Node child.
fn resolve_node(
    library: &NodeLibrary,
    world_root: NodeId,
    path: &Path,
) -> Option<NodeId> {
    let mut node = world_root;
    for k in 0..path.depth() as usize {
        let n = library.get(node)?;
        let slot = path.slot(k) as usize;
        match n.children[slot] {
            Child::Node(next) => node = next,
            _ => return None,
        }
    }
    Some(node)
}

/// Sum UVR-slot coord contributions from `uvr_path[..m]` into
/// (un_corner, vn_corner, rn_corner, frame_size). This is the
/// symbolic → f32 conversion that produces the Jacobian's evaluation
/// point. f32 precision is limited here, but only the Jacobian cares,
/// and the Jacobian is a linearization reference — the sub-cell
/// position (what matters for rendering) comes from
/// `WorldPos.sphere.uvr_offset` elsewhere.
fn uvr_corner(uvr_path: &Path, m: usize) -> (f32, f32, f32, f32) {
    let mut un = 0.0_f32;
    let mut vn = 0.0_f32;
    let mut rn = 0.0_f32;
    let mut size = 1.0_f32;
    for k in 0..m {
        let slot = uvr_path.slot(k) as usize;
        let (us, vs, rs) = crate::world::tree::slot_coords(slot);
        size /= 3.0;
        un += us as f32 * size;
        vn += vs as f32 * size;
        rn += rs as f32 * size;
    }
    (un, vn, rn, size)
}

fn build_logical_path(
    sphere: &crate::world::anchor::SphereState,
    desired_depth: u8,
) -> Path {
    let body_depth = sphere.body_path.depth();
    let mut p = sphere.body_path;
    p.push(FACE_SLOTS[sphere.face as usize] as u8);
    let want = desired_depth.saturating_sub(body_depth + 1);
    let take = (want as usize).min(sphere.uvr_path.depth() as usize);
    for k in 0..take {
        p.push(sphere.uvr_path.slot(k));
    }
    p
}

fn body_frame(
    library: &NodeLibrary,
    world_root: NodeId,
    sphere: &crate::world::anchor::SphereState,
    logical_path: Path,
) -> ActiveFrame {
    // Render frame is the body cell itself. Resolve body node.
    let node_id = resolve_node(library, world_root, &sphere.body_path)
        .unwrap_or(world_root);
    ActiveFrame {
        render_path: sphere.body_path,
        logical_path,
        node_id,
        kind: ActiveFrameKind::Body {
            inner_r: sphere.inner_r,
            outer_r: sphere.outer_r,
        },
    }
}

pub fn with_render_margin(
    library: &NodeLibrary,
    world_root: NodeId,
    camera: &WorldPos,
    logical_depth: u8,
    render_margin: u8,
) -> ActiveFrame {
    let target_depth = logical_depth.saturating_sub(render_margin);
    let logical = compute_render_frame(library, world_root, camera, logical_depth);
    if target_depth >= logical_depth {
        return logical;
    }
    let render = compute_render_frame(library, world_root, camera, target_depth);
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

    fn cartesian_chain(depth: u8) -> (NodeLibrary, NodeId) {
        let mut lib = NodeLibrary::default();
        let mut node = lib.insert(empty_children());
        for _ in 1..depth {
            node = lib.insert(uniform_children(Child::Node(node)));
        }
        lib.ref_inc(node);
        (lib, node)
    }

    #[test]
    fn cartesian_descends_linearly() {
        let (lib, root) = cartesian_chain(5);
        let mut anchor = Path::root();
        for _ in 0..4 {
            anchor.push(13);
        }
        let camera = WorldPos::new(anchor, [0.5; 3]);
        let f = compute_render_frame(&lib, root, &camera, 3);
        assert_eq!(f.render_path.depth(), 3);
        assert!(matches!(f.kind, ActiveFrameKind::Cartesian));
    }

    #[test]
    fn body_kind_when_anchor_ends_on_body() {
        let mut lib = NodeLibrary::default();
        let body = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereBody { inner_r: 0.12, outer_r: 0.45 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(slot_index(1, 1, 1) as u8);
        let camera = WorldPos::new(anchor, [0.5; 3]);
        let f = compute_render_frame(&lib, root, &camera, 1);
        assert!(matches!(f.kind, ActiveFrameKind::Body { .. }));
    }

    #[test]
    fn sphere_sub_from_symbolic_uvr_state() {
        use crate::world::anchor::SphereState;
        use crate::world::cubesphere::{Face, FACE_SLOTS};
        // `compute_render_frame` reserves `SPHERE_WALKER_BUDGET`
        // levels between render depth and logical uvr depth, so
        // exercising SphereSub at render uvr_depth = MIN_SPHERE_SUB_DEPTH
        // requires the camera's uvr_path to run deeper.
        let uvr_len = MIN_SPHERE_SUB_DEPTH as u32 + SPHERE_WALKER_BUDGET;
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(empty_children());
        let mut chain = leaf;
        for _ in 0..uvr_len {
            chain = lib.insert(uniform_children(Child::Node(chain)));
        }
        let face = lib.insert_with_kind(
            uniform_children(Child::Node(chain)),
            NodeKind::CubedSphereFace { face: Face::PosY },
        );
        let mut body_children = empty_children();
        body_children[FACE_SLOTS[Face::PosY as usize]] = Child::Node(face);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody { inner_r: 0.12, outer_r: 0.45 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let body_path = {
            let mut p = Path::root();
            p.push(slot_index(1, 1, 1) as u8);
            p
        };
        let mut uvr_path = Path::root();
        for _ in 0..uvr_len {
            uvr_path.push(slot_index(1, 1, 1) as u8);
        }

        let camera = WorldPos {
            anchor: body_path,
            offset: [0.5; 3],
            sphere: Some(SphereState {
                body_path,
                inner_r: 0.12,
                outer_r: 0.45,
                face: Face::PosY,
                uvr_path,
                uvr_offset: [0.5; 3],
            }),
        };
        let desired = body_path.depth() + 1 + MIN_SPHERE_SUB_DEPTH;
        let f = compute_render_frame(&lib, root, &camera, desired);
        match f.kind {
            ActiveFrameKind::SphereSub(sub) => {
                assert_eq!(sub.face, Face::PosY);
                assert_eq!(sub.depth_levels(), MIN_SPHERE_SUB_DEPTH as u32);
                let expected = (1.0_f32 / 3.0).powi(MIN_SPHERE_SUB_DEPTH as i32);
                assert!(
                    (sub.frame_size - expected).abs() < 1e-6,
                    "frame_size {} ≠ {}", sub.frame_size, expected
                );
            }
            k => panic!("expected SphereSub, got {k:?}"),
        }
    }

    #[test]
    fn shallow_sphere_falls_back_to_body() {
        use crate::world::anchor::SphereState;
        use crate::world::cubesphere::Face;
        let mut lib = NodeLibrary::default();
        let body = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereBody { inner_r: 0.12, outer_r: 0.45 },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let body_path = {
            let mut p = Path::root();
            p.push(slot_index(1, 1, 1) as u8);
            p
        };
        let mut uvr_path = Path::root();
        uvr_path.push(slot_index(1, 1, 1) as u8); // depth 1, below MIN

        let camera = WorldPos {
            anchor: body_path,
            offset: [0.5; 3],
            sphere: Some(SphereState {
                body_path,
                inner_r: 0.12,
                outer_r: 0.45,
                face: Face::PosY,
                uvr_path,
                uvr_offset: [0.5; 3],
            }),
        };
        let f = compute_render_frame(&lib, root, &camera, body_path.depth() + 1 + 1);
        assert!(matches!(f.kind, ActiveFrameKind::Body { .. }));
    }
}
