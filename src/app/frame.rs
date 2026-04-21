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
//!     `[0, 3)³`. The sub-frame is always built at the camera's
//!     deep `m_truncated` UVR depth so the Jacobian's evaluation
//!     point sits at the true render corner; pre-descent through
//!     `Child::Empty` links is handled symbolically by the walker,
//!     and rays that exit the sub-frame's local box transition to
//!     the neighbor sub-frame via `with_neighbor_stepped` — symbolic
//!     UVR-path stepping + fresh Jacobian. Cross-face transitions
//!     terminate the DDA (deferred to a follow-up commit).
//!     See `docs/design/sphere-ribbon-pop-two-step.md`.
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
use crate::world::cubesphere::{
    face_frame_jacobian, face_space_to_body_point, mat3_inv, Face, Mat3, FACE_SLOTS,
};
use crate::world::sdf::Vec3;
use crate::world::tree::{slot_coords, Child, NodeId, NodeKind, NodeLibrary};

/// Maximum depth at which we still evaluate the face-space Jacobian
/// directly. Past this, the root-relative face-normalized corner
/// `(un, vn, rn)` loses f32 precision (ULP ~1e-7 near magnitude 0.5),
/// and the neighbor-step increment `un += frame_size = 1/3^deep_m`
/// silently drops, freezing the Jacobian's eval point across
/// transitions. We evaluate J at `eval_m = min(deep_m, MAX_EVAL_M)`
/// and scale its columns by `deep_scale = 1/3^(deep_m - eval_m)` to
/// represent the deep cell's tinier per-local-unit size. `J_eval`
/// stays f32-stable at any deep_m because `eval_frame_size` never
/// drops below `1/3^12 ≈ 1.88e-6`, well above ULP.
///
/// At depth `m_eval = 12`, the linearization error from using
/// `N_eval` instead of `N_deep` is `O(eval_frame_size · deep cell
/// distance from eval cell center) = O(1/3^12 · 1/3^12) ≈ 3.5e-12`
/// — orders of magnitude below the f32 noise floor of the ray
/// parameters. Scaling `J_eval · deep_scale` yields `J` with column
/// magnitudes `O(1/3^deep_m)` — f32-representable down to `deep_m =
/// 25` and beyond (`1/3^25 ≈ 1.2e-12`, well above f32 subnormal
/// `~1.2e-38`).
pub const MAX_EVAL_M: u8 = 12;

/// Sphere sub-frame: the render root is a face-subtree cell at face
/// depth ≥ `MIN_SPHERE_SUB_DEPTH`. Ray-march runs in the frame's
/// local `[0, 3)³`. `c_body`, `J`, `J_inv` map local ↔ body-XYZ via
/// the linearized face transform at the frame's corner.
///
/// The Jacobian is evaluated at a SHALLOW anchor (at `eval_m =
/// min(deep_m, MAX_EVAL_M)`), where the face-normalized corner
/// `(un_corner, vn_corner, rn_corner)` is f32-precise, and scaled by
/// `deep_scale = 1/3^(deep_m − eval_m)` to represent the deeper
/// cell's tinier per-local-unit size. This keeps J stable at
/// arbitrary `deep_m` — the previous scheme stored corners at the
/// full deep_m and collapsed at `deep_m ≥ 15` when `un += frame_size`
/// became a no-op in f32.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SphereSubFrame {
    /// Path from world root to the containing `CubedSphereBody`.
    pub body_path: Path,
    /// Face this sub-frame lives on.
    pub face: Face,
    /// NodeId of the face-subtree root — always `body + face_slot`,
    /// so this resolves deterministically regardless of how deep the
    /// UVR descent reaches into the tree (the UVR descent itself may
    /// cross `Child::Empty` links for dug-out regions). The local
    /// walker pre-descends from this root along `uvr_path_prefix`
    /// before starting intra-cell DDA.
    pub face_root_id: NodeId,
    /// Full path world-root → body → face-root → UVR descent. This
    /// is the **render path** consumed by CPU raycast / ribbon-pop
    /// infrastructure — the last entry is the sub-frame's deep cell.
    /// May contain slots whose tree link is `Child::Empty`; the
    /// walker treats those as uniform-empty sub-frame content.
    pub render_path: Path,
    /// Depth at which the Jacobian is evaluated. `eval_m ≤
    /// MAX_EVAL_M` guarantees `(un_corner, vn_corner, rn_corner,
    /// frame_size)` are f32-stable.
    pub eval_m: u8,
    /// Face-normalized corner at the EVAL anchor (depth `eval_m`),
    /// in face `[0, 1)³`. Precise in f32 because `eval_m ≤
    /// MAX_EVAL_M`.
    pub un_corner: f32,
    pub vn_corner: f32,
    pub rn_corner: f32,
    /// Size of the EVAL cell in face-normalized coords. Equals
    /// `1/3^eval_m`. The deep sub-frame's cell size is
    /// `frame_size * deep_scale`.
    pub frame_size: f32,
    /// `3^(eval_m − deep_m)` = deep_cell_size / eval_cell_size. Equals
    /// 1.0 when `eval_m == deep_m` (shallow sub-frame path); less
    /// than 1.0 when the sub-frame is deeper than `MAX_EVAL_M`.
    pub deep_scale: f32,
    pub inner_r: f32,
    pub outer_r: f32,
    pub c_body: Vec3,
    pub j: Mat3,
    pub j_inv: Mat3,
}

impl SphereSubFrame {
    /// How many UVR descent levels inside the face subtree this
    /// sub-frame represents (the "deep" depth).
    pub fn depth_levels(&self) -> u32 {
        // render_path = body_path + face_root_slot + uvr_slots.
        // Length − body_path.depth() − 1 (the face-root slot) = UVR depth.
        (self.render_path.depth() as u32)
            .saturating_sub(self.body_path.depth() as u32)
            .saturating_sub(1)
    }

    /// Face-normalized cell size at `deep_m` (the DDA's per-cell
    /// absolute step). Always f32-representable for `deep_m` up to
    /// ~25 even though the eval corner is clamped shallower.
    pub fn deep_frame_size(&self) -> f32 {
        self.frame_size * self.deep_scale
    }

    /// Construct the neighbor sub-frame one local-axis step away along
    /// `axis ∈ {0, 1, 2}` in direction `direction ∈ {+1, −1}`. Used by
    /// `cs_raycast_local` (CPU) + the shader mirror when a ray exits
    /// the current sub-frame's local `[0, 3)³` box and needs to keep
    /// marching into the adjacent deep cell.
    ///
    /// Path stepping uses `Path::step_neighbor_cartesian` — the slot
    /// packing is the same for UVR as it is for XYZ (the semantic
    /// difference is handled by `face_frame_jacobian`, which reads the
    /// UVR-corner coords we adjust here).
    ///
    /// Returns `None` if the step would bubble past the face-root slot
    /// (the slot at `body_path.depth()` in `render_path`) — i.e., the
    /// step touched a path level at or above the face root. That's a
    /// cross-face transition and is deferred to a future commit; the
    /// CPU / GPU callers terminate their DDA in that case.
    ///
    /// `Path::step_neighbor_cartesian` preserves the path's depth via
    /// unwind-and-repush; we detect the boundary crossing by checking
    /// whether the face-root slot at `body_path.depth()` still holds
    /// the expected `FACE_SLOTS[face]` value.
    pub fn with_neighbor_stepped(&self, axis: usize, direction: i32) -> Option<Self> {
        debug_assert!(axis < 3);
        debug_assert!(direction == 1 || direction == -1);

        // Pre-check: can the step land within the face subtree?
        // `step_neighbor_cartesian` bubbles on overflow, and when the
        // bubble reaches `depth == 0` it silently returns (no-op)
        // while still wrapping child slots on unwind — producing a
        // "fake" step that doesn't actually advance the path. That
        // would mis-report a cross-face situation as success. We
        // detect it by checking whether any UVR level (past the face
        // root) has an axis-coord NOT at the overflow boundary; if
        // every UVR level is boundary-pinned for this direction, the
        // step would bubble past the face root.
        let face_root_idx = self.body_path.depth() as usize;
        if (self.render_path.depth() as usize) <= face_root_idx + 1 {
            return None; // no UVR descent to step
        }
        let boundary = if direction > 0 { 2usize } else { 0usize };
        let mut can_step_in_face = false;
        for level in (face_root_idx + 1..self.render_path.depth() as usize).rev() {
            let slot = self.render_path.slot(level) as usize;
            let (us, vs, rs) = crate::world::tree::slot_coords(slot);
            let coord = match axis { 0 => us, 1 => vs, _ => rs };
            if coord != boundary {
                can_step_in_face = true;
                break;
            }
        }
        if !can_step_in_face {
            // Would bubble past face root — cross-face transition
            // (deferred to a future commit).
            return None;
        }

        let mut new_path = self.render_path;
        new_path.step_neighbor_cartesian(axis, direction);

        // Belt-and-suspenders: after the step, the face-root slot
        // must still hold the expected `FACE_SLOTS[face]` value.
        if new_path.depth() as usize <= face_root_idx
            || new_path.slot(face_root_idx) != FACE_SLOTS[self.face as usize] as u8
        {
            return None;
        }

        // Recompute the EVAL corner fresh from the new render_path.
        // A neighbor step may bubble through multiple UVR levels; some
        // of those may be at or shallower than `eval_m`, in which case
        // the eval corner shifts. Walking just the first `eval_m`
        // slots from the face root keeps this O(MAX_EVAL_M) cheap
        // while handling the bubble-up correctly. If all the bubbling
        // happened below `eval_m`, the eval corner is unchanged and
        // the Jacobian is frozen — that's the physically-correct
        // approximation at `deep_m > eval_m` (face curvature between
        // adjacent deep cells is O(1/3^deep_m · 1/3^eval_m), far
        // below f32 noise).
        let uvr_start = face_root_idx + 1;
        let new_deep_m = (new_path.depth() as usize).saturating_sub(uvr_start);
        let new_eval_m = new_deep_m.min(MAX_EVAL_M as usize);
        let eval_slots =
            &new_path.as_slice()[uvr_start..uvr_start + new_eval_m];
        let (un, vn, rn, eval_frame_size) = uvr_corner_from_slots(eval_slots);
        let mut deep_scale = 1.0_f32;
        for _ in new_eval_m..new_deep_m {
            deep_scale /= 3.0;
        }

        let (_c_body_eval, j_eval) = face_frame_jacobian(
            self.face,
            un, vn, rn, eval_frame_size,
            self.inner_r, self.outer_r,
            3.0,
        );
        let j = scale_mat3_cols(j_eval, deep_scale);
        let j_inv = mat3_inv(&j);

        // c_body at the DEEP corner. `face_space_to_body_point` is
        // smooth, so feeding it the sum-over-deep_m un/vn/rn (imprecise
        // at deep_m ≥ 15) still yields body-XYZ within O(1e-7) of the
        // true corner — negligible at body-scale O(1).
        let (un_deep, vn_deep, rn_deep, _deep_fs) = uvr_corner_from_slots(
            &new_path.as_slice()[uvr_start..uvr_start + new_deep_m],
        );
        let c_body = face_space_to_body_point(
            self.face,
            un_deep, vn_deep, rn_deep,
            self.inner_r, self.outer_r,
            3.0,
        );

        Some(SphereSubFrame {
            body_path: self.body_path,
            face: self.face,
            face_root_id: self.face_root_id,
            render_path: new_path,
            eval_m: new_eval_m as u8,
            un_corner: un,
            vn_corner: vn,
            rn_corner: rn,
            frame_size: eval_frame_size,
            deep_scale,
            inner_r: self.inner_r,
            outer_r: self.outer_r,
            c_body,
            j,
            j_inv,
        })
    }
}

/// Column-wise scale: `out[c][r] = m[c][r] * s`. Keeps the column-
/// major storage convention.
#[inline]
fn scale_mat3_cols(m: Mat3, s: f32) -> Mat3 {
    [
        [m[0][0] * s, m[0][1] * s, m[0][2] * s],
        [m[1][0] * s, m[1][1] * s, m[1][2] * s],
        [m[2][0] * s, m[2][1] * s, m[2][2] * s],
    ]
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
/// kicks in. Any UVR descent is eligible — the linearization error
/// drops geometrically with depth, and the body march has a hard
/// precision wall past ~8 UVR levels, so SphereSub is always at
/// least as good when a face-subtree path is available.
pub const MIN_SPHERE_SUB_DEPTH: u8 = 1;

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
    // Build the sub-frame ALWAYS at the camera's deep `m_truncated`
    // UVR depth. The face-subtree root resolves via `body_path +
    // face_slot` — that chain is guaranteed to be `Child::Node` at
    // every step by world construction, so `face_root_id` is always
    // present. The UVR pre-descent past the face root may hit
    // `Child::Empty` for dug regions; the walker handles those
    // symbolically without requiring a real Node at the deep path.
    if let Some(sphere) = camera.sphere.as_ref() {
        let logical_m = sphere.uvr_path.depth();
        let body_depth = sphere.body_path.depth() as i32;
        let m_logical = logical_m as i32;
        let m_truncated = (desired_depth as i32 - body_depth - 1).clamp(0, m_logical) as u32;

        eprintln!(
            "CRF sphere body_depth={} logical_m={} desired={} m_truncated={} MIN={}",
            body_depth, logical_m, desired_depth, m_truncated, MIN_SPHERE_SUB_DEPTH,
        );

        if m_truncated >= MIN_SPHERE_SUB_DEPTH as u32 {
            // Resolve the face-subtree root. Always reachable for a
            // valid sphere body + face; degeneracy (missing face
            // root) falls back to Body march so the renderer has
            // something to dispatch.
            let mut face_root_path = sphere.body_path;
            face_root_path.push(FACE_SLOTS[sphere.face as usize] as u8);
            let Some(face_root_id) = resolve_node(library, world_root, &face_root_path)
            else {
                let logical_path = build_logical_path(sphere, desired_depth);
                return body_frame(library, world_root, sphere, logical_path);
            };

            // Build render_path = body_path + face_root_slot +
            // uvr_path[..m_truncated]. Unlike the old implementation
            // this does NOT require the deep path to resolve to a
            // real Node — `Child::Empty` mid-descent is legal and
            // represents dug content that the walker renders as
            // empty.
            let mut render_path = sphere.body_path;
            render_path.push(FACE_SLOTS[sphere.face as usize] as u8);
            for k in 0..m_truncated as usize {
                render_path.push(sphere.uvr_path.slot(k));
            }

            let deep_m = m_truncated as usize;
            let eval_m = deep_m.min(MAX_EVAL_M as usize);
            let (un_corner, vn_corner, rn_corner, eval_frame_size) =
                uvr_corner(&sphere.uvr_path, eval_m);
            let mut deep_scale = 1.0_f32;
            for _ in eval_m..deep_m {
                deep_scale /= 3.0;
            }
            // J evaluated at the shallow eval corner is f32-stable
            // and differs from the exact deep-corner J only by face
            // curvature across the eval cell (O((1/3^eval_m)²)).
            let (_c_body_eval, j_eval) = face_frame_jacobian(
                sphere.face,
                un_corner, vn_corner, rn_corner, eval_frame_size,
                sphere.inner_r, sphere.outer_r,
                3.0,
            );
            // Deep J: scale columns by `deep_scale` to reflect the
            // tinier per-local-unit cell size at `deep_m`.
            let j = scale_mat3_cols(j_eval, deep_scale);
            let j_inv = mat3_inv(&j);
            // c_body at the deep corner. face_space_to_body_point is
            // smooth, so the corner coords' ~1e-7 imprecision at
            // deep_m ≥ 15 propagates as O(1e-7) body-XYZ error —
            // fine at body-scale O(1).
            let (un_deep, vn_deep, rn_deep, _deep_fs) =
                uvr_corner(&sphere.uvr_path, deep_m);
            let c_body = face_space_to_body_point(
                sphere.face,
                un_deep, vn_deep, rn_deep,
                sphere.inner_r, sphere.outer_r,
                3.0,
            );
            let sub = SphereSubFrame {
                body_path: sphere.body_path,
                face: sphere.face,
                face_root_id,
                render_path,
                eval_m: eval_m as u8,
                un_corner, vn_corner, rn_corner,
                frame_size: eval_frame_size,
                deep_scale,
                inner_r: sphere.inner_r,
                outer_r: sphere.outer_r,
                c_body, j, j_inv,
            };
            let logical_path = build_logical_path(sphere, desired_depth);
            return ActiveFrame {
                render_path: sub.render_path,
                logical_path,
                // The walker starts from `face_root_id` and descends
                // symbolically via `uvr_path` prefix; that's the
                // node the GPU shader indexes into too.
                node_id: face_root_id,
                kind: ActiveFrameKind::SphereSub(sub),
            };
        }
        // Shallow face-subtree depth → Body march handles it.
        let logical_path = build_logical_path(sphere, desired_depth);
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
/// point. At `m ≤ MAX_EVAL_M` (the only call site that matters for
/// J evaluation), `frame_size = 1/3^m ≥ 1/3^12 ≈ 1.88e-6` — well
/// above the ULP of `un` (~1e-7 near magnitude 0.5), so the
/// cumulative sum is f32-stable. Callers passing `m > MAX_EVAL_M`
/// (e.g., for `c_body` at the deep corner) accept the ~1e-7
/// imprecision, which is harmless for smooth body-XYZ values.
fn uvr_corner(uvr_path: &Path, m: usize) -> (f32, f32, f32, f32) {
    let mut un = 0.0_f32;
    let mut vn = 0.0_f32;
    let mut rn = 0.0_f32;
    let mut size = 1.0_f32;
    for k in 0..m {
        let slot = uvr_path.slot(k) as usize;
        let (us, vs, rs) = slot_coords(slot);
        size /= 3.0;
        un += us as f32 * size;
        vn += vs as f32 * size;
        rn += rs as f32 * size;
    }
    (un, vn, rn, size)
}

/// Same as `uvr_corner` but reads slots from a `&[u8]` slice —
/// useful post-neighbor-step when the updated `render_path`'s UVR
/// tail is different from the camera's `uvr_path`.
fn uvr_corner_from_slots(slots: &[u8]) -> (f32, f32, f32, f32) {
    let mut un = 0.0_f32;
    let mut vn = 0.0_f32;
    let mut rn = 0.0_f32;
    let mut size = 1.0_f32;
    for &slot in slots {
        let (us, vs, rs) = slot_coords(slot as usize);
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
        // Build: root → body → face(PosY) → uvr chain × MIN_SPHERE_SUB_DEPTH.
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(empty_children());
        let mut chain = leaf;
        for _ in 0..MIN_SPHERE_SUB_DEPTH {
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
        for _ in 0..MIN_SPHERE_SUB_DEPTH {
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
                // deep_m = MIN_SPHERE_SUB_DEPTH ≤ MAX_EVAL_M, so
                // eval_m == deep_m and deep_scale == 1.
                let expected = (1.0_f32 / 3.0).powi(MIN_SPHERE_SUB_DEPTH as i32);
                assert!(
                    (sub.frame_size - expected).abs() < 1e-6,
                    "frame_size {} ≠ {}", sub.frame_size, expected,
                );
                assert_eq!(sub.eval_m, MIN_SPHERE_SUB_DEPTH);
                assert!((sub.deep_scale - 1.0).abs() < 1e-6);
                assert!((sub.deep_frame_size() - expected).abs() < 1e-6);
            }
            k => panic!("expected SphereSub, got {k:?}"),
        }
    }

    #[test]
    fn deep_sub_frame_uses_shallow_eval() {
        // At `deep_m > MAX_EVAL_M` the eval corner clamps to
        // `MAX_EVAL_M` and `deep_scale = 3^-(deep_m - MAX_EVAL_M)`.
        // J's columns shrink by the same factor; the eval corner
        // itself stays f32-precise.
        use crate::world::anchor::SphereState;
        use crate::world::cubesphere::{Face, FACE_SLOTS};
        // Build a minimal tree with a face-root node so
        // `resolve_node` finds the face-root id.
        let mut lib = NodeLibrary::default();
        let leaf = lib.insert(empty_children());
        let face = lib.insert_with_kind(
            uniform_children(Child::Node(leaf)),
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
        let deep_m: u8 = 20;
        let mut uvr_path = Path::root();
        for _ in 0..deep_m {
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
        let desired = body_path.depth() + 1 + deep_m;
        let f = compute_render_frame(&lib, root, &camera, desired);
        let sub = match f.kind {
            ActiveFrameKind::SphereSub(s) => s,
            k => panic!("expected SphereSub, got {k:?}"),
        };
        assert_eq!(sub.eval_m, MAX_EVAL_M);
        // deep_scale = 1/3^(deep_m - MAX_EVAL_M).
        let delta = (deep_m - MAX_EVAL_M) as i32;
        let expected_scale = (1.0_f32 / 3.0).powi(delta);
        assert!(
            (sub.deep_scale - expected_scale).abs() < 1e-10,
            "deep_scale {} ≠ {}", sub.deep_scale, expected_scale,
        );
        // Eval corner is the sum over the first MAX_EVAL_M slots —
        // each (1,1,1), so (un, vn, rn) should all be ~0.5 ·
        // (1 - 3^-MAX_EVAL_M). frame_size = 1/3^MAX_EVAL_M.
        let expected_fs = (1.0_f32 / 3.0).powi(MAX_EVAL_M as i32);
        assert!(
            (sub.frame_size - expected_fs).abs() < 1e-9,
            "frame_size {} ≠ {}", sub.frame_size, expected_fs,
        );
        let expected_deep_fs = (1.0_f32 / 3.0).powi(deep_m as i32);
        assert!(
            (sub.deep_frame_size() - expected_deep_fs).abs()
                < expected_deep_fs * 1e-3,
            "deep_frame_size {} ≠ {}", sub.deep_frame_size(), expected_deep_fs,
        );
        // J's columns scale with deep cell size. At deep_m=20 a
        // column should have magnitude O(1/3^20) · body_size ≈ 8.6e-10.
        let col_u_mag = (sub.j[0][0].powi(2)
            + sub.j[0][1].powi(2)
            + sub.j[0][2].powi(2))
        .sqrt();
        assert!(
            col_u_mag < 1e-8 && col_u_mag > 1e-12,
            "col_u magnitude {} outside expected deep-scale range",
            col_u_mag,
        );
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
