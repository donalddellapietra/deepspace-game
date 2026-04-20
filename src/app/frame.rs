//! Render-frame helpers: walking the camera path to find the active
//! frame, transforming positions/AABBs into that frame.
//!
//! Cartesian frames are linear `[0, 3)³`. A cubed-sphere `Body` frame
//! renders the full 6-face shell in body-local coords with the exact
//! (non-linearized) `cs_raycast_body` march. A `SphereSub` frame
//! lives deeper inside a face subtree: the render frame is a single
//! face-cell, and ray-march runs in *local* `[0, 3)³` coords of that
//! cell via the linearized ribbon-pop scheme
//! (`docs/design/sphere-ribbon-pop-impl-plan.md`).
//!
//! `SphereSub` carries a precomputed `(c_body, J, J_inv)` so the
//! sphere DDA can operate entirely in the local frame — plane
//! boundaries become axis-aligned in local coords, and ray-sphere
//! intersection against radial shells reduces to a well-conditioned
//! quadratic in local `t`. This preserves f32 precision at arbitrary
//! face-subtree depth.
//!
//! Slot-semantics note: inside a face subtree, a node's 27 children
//! are indexed by `(u_slot, v_slot, r_slot)` under the face-root's
//! UVR convention. `compute_render_frame` reinterprets path slots in
//! UVR coords once it crosses a `CubedSphereFace` boundary.
//!
//! Pure functions; no `App` state; unit-testable.

use crate::world::anchor::Path;
use crate::world::cubesphere::{
    face_frame_jacobian, mat3_inv, Face, Mat3, FACE_SLOTS,
};
use crate::world::sdf::Vec3;
use crate::world::tree::{slot_coords, Child, NodeId, NodeKind, NodeLibrary};

/// Sphere sub-frame: the render root is a face-subtree cell at depth
/// ≥ 1 within the body. Ray-march runs in the frame's local
/// `[0, 3)³` coords; `c_body`, `J`, `J_inv` map local ↔ body-XYZ via
/// the linearized face transform at the frame's corner.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereSubFrame {
    /// Path from world root to the containing `CubedSphereBody`.
    pub body_path: Path,
    /// Face this sub-frame lives on.
    pub face: Face,
    /// Absolute face-normalized coords of the frame's corner (local
    /// (0,0,0) in body `[0, 1)³` face-normalized coords).
    pub un_corner: f32,
    pub vn_corner: f32,
    pub rn_corner: f32,
    /// Size of the frame in face-normalized coords — `1/3^M` where M
    /// is face-subtree depth.
    pub frame_size: f32,
    pub inner_r: f32,
    pub outer_r: f32,
    /// Precomputed linearization at frame corner. `c_body` is the
    /// body-XYZ of local (0,0,0); `J` maps local → body-XYZ offset;
    /// `J_inv` maps body-XYZ offset → local.
    pub c_body: Vec3,
    pub j: Mat3,
    pub j_inv: Mat3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ActiveFrameKind {
    Cartesian,
    /// Render root IS a cubed-sphere body cell. No face window — the
    /// sphere DDA covers all 6 faces in body-local coords.
    Body { inner_r: f32, outer_r: f32 },
    /// Render root is a face-subtree cell at face-subtree depth ≥ 1.
    /// The sphere DDA runs in the frame's local coords.
    SphereSub(SphereSubFrame),
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ActiveFrame {
    /// Path from world root to the render frame's root node. For
    /// `Cartesian` / `Body` this equals `logical_path`. For
    /// `SphereSub` this ends at the face-subtree cell that IS the
    /// render frame.
    pub render_path: Path,
    /// Logical interaction layer — the user's edit-depth anchor.
    /// Equals `render_path` once `compute_render_frame` has consumed
    /// the full `desired_depth`.
    pub logical_path: Path,
    pub node_id: NodeId,
    pub kind: ActiveFrameKind,
}

/// Build a `Path` from the slot prefix the GPU ribbon walker actually
/// reached.
pub fn frame_from_slots(slots: &[u8]) -> Path {
    let mut frame = Path::root();
    for &slot in slots {
        frame.push(slot);
    }
    frame
}

/// Minimum face-subtree depth at which the linearized `SphereSub`
/// frame kicks in. Shallower depths (face root + 1 level down) use
/// the exact `Body` march because linearization error is
/// perceptible there.
pub const MIN_SPHERE_SUB_DEPTH: u8 = 3;

/// Resolve the active render frame. Walks the camera's anchor path
/// down to `desired_depth`. Descent rules:
///
///   * Cartesian parent ×→ Cartesian child: standard, slot index is
///     XYZ.
///   * Cartesian parent ×→ `CubedSphereBody` child: stop here with
///     `Body` kind unless deeper descent is available through a face
///     subtree.
///   * `CubedSphereBody` →`CubedSphereFace`: re-interpret subsequent
///     slots as UVR. Begin accumulating `un_corner, vn_corner,
///     rn_corner, frame_size` — initially (0, 0, 0, 1).
///   * Face-subtree Cartesian descent: slot is UVR, frame_size *=
///     1/3 each level.
///
/// When the descended face-subtree depth is ≥ `MIN_SPHERE_SUB_DEPTH`,
/// materialize `SphereSub` with precomputed `(c_body, J, J_inv)`.
/// Shallower face-subtree depths revert to `Body` (the exact march
/// handles those levels without linearization).
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

    // Body-subtree state, populated on `CubedSphereBody` descent.
    let mut body_state: Option<BodyDescendState> = None;

    for k in 0..target.depth() as usize {
        let Some(node) = library.get(node_id) else { break };
        let slot = target.slot(k) as usize;
        let Child::Node(child_id) = node.children[slot] else { break };
        let Some(child) = library.get(child_id) else { break };

        match (child.kind, &mut body_state) {
            (NodeKind::Cartesian, None) => {
                node_id = child_id;
                reached.push(slot as u8);
            }
            (NodeKind::CubedSphereBody { inner_r, outer_r }, None) => {
                node_id = child_id;
                reached.push(slot as u8);
                body_state = Some(BodyDescendState {
                    body_path: reached,
                    inner_r,
                    outer_r,
                    face: None,
                    un_corner: 0.0,
                    vn_corner: 0.0,
                    rn_corner: 0.0,
                    frame_size: 1.0,
                });
            }
            (NodeKind::CubedSphereFace { face }, Some(state)) if state.face.is_none() => {
                // Descending body → face root. Slot must match the
                // face's conventional slot. We don't push this slot
                // onto `reached` as a UVR slot — it's the body's
                // child slot (XYZ slot index for the face root).
                node_id = child_id;
                reached.push(slot as u8);
                state.face = Some(face);
                // Face root starts covering the whole face in UVR.
                state.un_corner = 0.0;
                state.vn_corner = 0.0;
                state.rn_corner = 0.0;
                state.frame_size = 1.0;
            }
            (NodeKind::Cartesian, Some(state)) if state.face.is_some() => {
                // Inside a face subtree — interpret slot as UVR.
                let (us, vs, rs) = slot_coords(slot);
                state.frame_size /= 3.0;
                state.un_corner += us as f32 * state.frame_size;
                state.vn_corner += vs as f32 * state.frame_size;
                state.rn_corner += rs as f32 * state.frame_size;
                node_id = child_id;
                reached.push(slot as u8);
            }
            (NodeKind::CubedSphereBody { .. }, Some(_))
            | (NodeKind::CubedSphereFace { .. }, _) => {
                // Malformed: nested body, or face root outside a body
                // context. Stop descent safely.
                break;
            }
            (NodeKind::Cartesian, Some(_)) => {
                // State flagged for body but face not yet entered.
                // Body's 27 children are partitioned by cubemap
                // geometry, not XYZ. Don't descend further through
                // it — stop with `Body` kind.
                break;
            }
        }
    }

    let kind = resolve_kind(library, node_id, body_state);
    ActiveFrame {
        render_path: reached,
        logical_path: reached,
        node_id,
        kind,
    }
}

struct BodyDescendState {
    body_path: Path,
    inner_r: f32,
    outer_r: f32,
    face: Option<Face>,
    un_corner: f32,
    vn_corner: f32,
    rn_corner: f32,
    frame_size: f32,
}

fn resolve_kind(
    library: &NodeLibrary,
    node_id: NodeId,
    body_state: Option<BodyDescendState>,
) -> ActiveFrameKind {
    let Some(state) = body_state else {
        return match library.get(node_id).map(|n| n.kind) {
            Some(NodeKind::Cartesian) | None => ActiveFrameKind::Cartesian,
            _ => ActiveFrameKind::Cartesian,
        };
    };

    match state.face {
        None => ActiveFrameKind::Body {
            inner_r: state.inner_r,
            outer_r: state.outer_r,
        },
        Some(face) => {
            // Face-subtree depth is the number of 1/3-factors applied
            // to `frame_size`. At `face_root` alone, frame_size = 1.
            let face_depth = frame_size_to_depth(state.frame_size);
            if face_depth < MIN_SPHERE_SUB_DEPTH {
                // Shallow face levels: defer to exact body march.
                // body_path is the full path to the body (the render
                // frame's effective root for the Body march).
                ActiveFrameKind::Body {
                    inner_r: state.inner_r,
                    outer_r: state.outer_r,
                }
            } else {
                // Body-local convention used by the renderer is
                // `[0, 3)³` (the body cell fills one render frame),
                // so `body_size = 3.0`. Keeps c_body / J in the same
                // coordinate scale the caller passes camera in.
                let (c_body, j) = face_frame_jacobian(
                    face,
                    state.un_corner,
                    state.vn_corner,
                    state.rn_corner,
                    state.frame_size,
                    state.inner_r,
                    state.outer_r,
                    3.0,
                );
                let j_inv = mat3_inv(&j);
                ActiveFrameKind::SphereSub(SphereSubFrame {
                    body_path: state.body_path,
                    face,
                    un_corner: state.un_corner,
                    vn_corner: state.vn_corner,
                    rn_corner: state.rn_corner,
                    frame_size: state.frame_size,
                    inner_r: state.inner_r,
                    outer_r: state.outer_r,
                    c_body,
                    j,
                    j_inv,
                })
            }
        }
    }
}

fn frame_size_to_depth(frame_size: f32) -> u8 {
    // frame_size = 1/3^d → d = -log3(frame_size). Use integer rounding
    // (exact for the powers of 3 we actually produce).
    let inv = 1.0 / frame_size;
    let d = inv.ln() / 3.0_f32.ln();
    d.round() as u8
}

/// Produce an anchor path that descends into a face subtree. Used by
/// callers (tests, zoom logic) that need to represent a camera
/// parked inside a specific face-subtree cell.
pub fn face_subtree_anchor(
    body_path: Path,
    face: Face,
    u_slots: &[u8],
    v_slots: &[u8],
    r_slots: &[u8],
) -> Path {
    debug_assert_eq!(u_slots.len(), v_slots.len());
    debug_assert_eq!(u_slots.len(), r_slots.len());
    let mut path = body_path;
    path.push(FACE_SLOTS[face as usize] as u8);
    for i in 0..u_slots.len() {
        let us = u_slots[i] as usize;
        let vs = v_slots[i] as usize;
        let rs = r_slots[i] as usize;
        path.push(crate::world::tree::slot_index(us, vs, rs) as u8);
    }
    path
}

pub fn with_render_margin(
    library: &NodeLibrary,
    world_root: NodeId,
    logical_path: &Path,
    render_margin: u8,
) -> ActiveFrame {
    let logical = compute_render_frame(library, world_root, logical_path, logical_path.depth());
    let target_depth = logical
        .logical_path
        .depth()
        .saturating_sub(render_margin);
    if target_depth >= logical.logical_path.depth() {
        return logical;
    }

    let mut render_path = logical.logical_path;
    render_path.truncate(target_depth);
    let render = compute_render_frame(library, world_root, &render_path, target_depth);
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
        let f = compute_render_frame(&lib, root, &anchor, 3);
        assert_eq!(f.render_path.depth(), 3);
        assert!(matches!(f.kind, ActiveFrameKind::Cartesian));
    }

    #[test]
    fn sphere_body_enters_body_kind() {
        let mut lib = NodeLibrary::default();
        let body = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereBody {
                inner_r: 0.12,
                outer_r: 0.45,
            },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(13);
        let f = compute_render_frame(&lib, root, &anchor, 1);
        assert!(matches!(f.kind, ActiveFrameKind::Body { .. }));
    }

    #[test]
    fn deep_face_subtree_enters_sphere_sub() {
        // Build: root → body → face(PosX) → face-cell × MIN_SPHERE_SUB_DEPTH.
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(empty_children());
        let mut chain = leaf;
        for _ in 0..MIN_SPHERE_SUB_DEPTH {
            chain = lib.insert(uniform_children(Child::Node(chain)));
        }
        let face = lib.insert_with_kind(
            uniform_children(Child::Node(chain)),
            NodeKind::CubedSphereFace { face: Face::PosX },
        );
        let mut body_children = empty_children();
        body_children[FACE_SLOTS[Face::PosX as usize]] = Child::Node(face);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody {
                inner_r: 0.12,
                outer_r: 0.45,
            },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        // Anchor descends: slot 13 (body), slot 14 (PosX face root),
        // then MIN_SPHERE_SUB_DEPTH steps through UVR-semantic slots.
        let mut anchor = Path::root();
        anchor.push(slot_index(1, 1, 1) as u8);
        anchor.push(FACE_SLOTS[Face::PosX as usize] as u8);
        for _ in 0..MIN_SPHERE_SUB_DEPTH {
            anchor.push(slot_index(1, 1, 1) as u8);
        }

        let f = compute_render_frame(&lib, root, &anchor, anchor.depth());
        match f.kind {
            ActiveFrameKind::SphereSub(sub) => {
                assert_eq!(sub.face, Face::PosX);
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
    fn shallow_face_subtree_stays_body_kind() {
        // 2 face-subtree levels is below MIN_SPHERE_SUB_DEPTH = 3 →
        // resolves to Body.
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(empty_children());
        let inner = lib.insert(uniform_children(Child::Node(leaf)));
        let face = lib.insert_with_kind(
            uniform_children(Child::Node(inner)),
            NodeKind::CubedSphereFace { face: Face::PosX },
        );
        let mut body_children = empty_children();
        body_children[FACE_SLOTS[Face::PosX as usize]] = Child::Node(face);
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody {
                inner_r: 0.12,
                outer_r: 0.45,
            },
        );
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(body);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let mut anchor = Path::root();
        anchor.push(slot_index(1, 1, 1) as u8);
        anchor.push(FACE_SLOTS[Face::PosX as usize] as u8);
        anchor.push(slot_index(1, 1, 1) as u8);
        anchor.push(slot_index(1, 1, 1) as u8);

        let f = compute_render_frame(&lib, root, &anchor, anchor.depth());
        assert!(
            matches!(f.kind, ActiveFrameKind::Body { .. }),
            "expected Body at face-depth 2, got {:?}", f.kind
        );
    }

    #[test]
    fn frame_size_to_depth_round_trip() {
        for d in 0..15_u8 {
            let size = (1.0_f32 / 3.0).powi(d as i32);
            assert_eq!(frame_size_to_depth(size), d);
        }
    }
}
