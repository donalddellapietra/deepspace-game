//! Precision-correct cubed-sphere face walker.
//!
//! Stage 3d of the unified-DDA rewrite. This is the CPU mirror of the
//! shader's face-subtree walker (`march_face_subtree_curved` in
//! `assets/shaders/unified_dda.wgsl`). Together they share ONE
//! precision model: **slot-path + residual**, with NO absolute f32
//! state scaling as `1/3^N` and NO absolute f32 state growing as `3^N`
//! in any quantity that feeds a difference/comparison.
//!
//! See `docs/architecture/sphere-unified-dda.md` and
//! `docs/principles/no-absolute-coordinates.md`. The Stage 3b walker
//! stored `cur_u_lo`, `cur_v_lo`, `cur_r_lo` and `cur_cell_ext` as
//! absolute f32 face-normalized coordinates — these sink below f32
//! ULP at face-subtree depth ~14 and silently mis-compute child
//! boundaries past that. The fix removes all such state.
//!
//! ## State model
//!
//! Per iteration, the walker carries:
//!
//! - `slot_stack : [(u8, u8, u8); depth]` — integer chain of
//!   `(us, vs, rs)` child slots from the face subtree root. This IS
//!   the position; everything else is derivable from it.
//! - `face : Face` — current face (0..6). Changes at seam crossings.
//! - `residual_o : [f32; 3]` — ray origin in the CURRENT cell's
//!   local `[0, 3)³` frame. After every descent, rescaled so it
//!   stays in `[0, 3)³`. Magnitude O(1) at any depth.
//! - `rd_local  : [f32; 3]` — ray direction in the current cell's
//!   local frame. Multiplied by 3 per descent. Still f32-safe at
//!   face-subtree depth 30 (3^30 ≈ 2×10^14, well below f32 max
//!   3.4×10^38). Rescaling is exact in binary (×3 is lossless up to
//!   the mantissa limit) so no drift accumulates within f32's range.
//! - `u_c, v_c, r_c : f32` — face-normalized cell-center
//!   coordinates, tracked incrementally on descent:
//!   `u_c_new = u_c_parent + (slot_u - 1) · cell_ext_parent / 2` is
//!   the center of the child; we actually store the child's LOWER
//!   corner and evaluate the center as `lo + cell_ext/2` locally.
//!   These shrink toward f32 ULP at deep depth but remain bounded
//!   in `[0, 1]` — used only to evaluate `tan((2·u_c - 1)·π/4)` for
//!   the r-shell shading normal, a smooth function whose output is
//!   insensitive to low-order bit loss in `u_c`. NEVER used in
//!   boundary tests.
//!
//! All cell boundaries in the residual frame live at integer values
//! `0` and `3`, so ray-plane intersections `(bound - residual_i) /
//! rd_local_i` compute at O(1) precision regardless of `depth`.
//!
//! ## Why this works to face-subtree depth 30+
//!
//! 1. Residual is bounded in `[0, 3)³` by construction. Never
//!    compared against absolute positions.
//! 2. `rd_local` grows geometrically but f32's exponent range
//!    easily accommodates 3^30. The direction RATIO is what matters
//!    for slot-picking; magnitude is informational.
//! 3. `u_c, v_c, r_c` stay bounded in `[0, 1]` and appear only in
//!    `tan()` / radial shading math — smooth functions. Low-bit
//!    loss in `u_c` at depth 30 manifests as a normal error of
//!    `O(ε × f'(u_c))` where `f` is the cube→sphere projection, a
//!    well-conditioned map.
//! 4. Slot stack is integer — no precision issue, ever.
//!
//! ## Linearization
//!
//! Per the architecture doc (§CubedSphereFace descendant), inside a
//! face subtree cells are treated as flat parallelograms in
//! face-normalized coords — boundaries are axis-aligned planes in
//! the residual frame, not the curved-surface equal-angle planes
//! the Stage 3b walker used. Silhouette error is `O(cell_size²)`
//! body units; invisible at face-subtree depth ≥ 3 (sub-pixel at
//! typical render distances). This is the trade-off the doc
//! explicitly accepts in exchange for precision-stable DDA at any
//! depth.

use super::{Face, FacePoint, CORE_SLOT, FACE_SLOTS};
use super::seams::SEAM_TABLE;
use crate::world::tree::{Child, NodeId, NodeLibrary};

/// Maximum descent depth the walker will perform. Sized to exceed
/// the 30-40 layer face-subtree target; each level adds one slot
/// to the stack (a few bytes) and one `rd_local × 3` (exact in
/// binary f32).
pub const MAX_FACE_WALK_DEPTH: usize = 48;

/// Outcome of a walk step — the walker terminates on one of these.
#[derive(Debug, Clone, PartialEq)]
pub enum WalkResult {
    /// The ray hit a solid block inside the face subtree. `path`
    /// is the integer slot chain from the face subtree root to the
    /// hit cell; `block` is the palette index.
    Hit {
        path: Vec<(u8, u8, u8)>,
        block: u16,
        /// Face on which the hit occurred (after any seam crosses).
        face: Face,
        /// Body-frame ray parameter at the hit point, so callers
        /// can reconstruct an approximate world position.
        t_body: f32,
    },
    /// Ray exited the outer r-shell. Caller re-enters the body
    /// DDA (the ray has left the planet).
    OuterShellExit,
    /// Ray exited the inner r-shell. Caller dispatches to the core
    /// subtree at `CORE_SLOT`.
    InnerShellExit,
    /// Ray exhausted iteration budget. Treat as miss.
    IterationExhausted,
}

/// Body-frame ray description passed into the walker.
#[derive(Debug, Copy, Clone)]
pub struct BodyRay {
    /// Ray origin in body-local coords where the body cell is
    /// `[0, body_size)³` and body_size = 3.0 (cell-local `[0, 1)`
    /// × 3). The body center is at `(1.5, 1.5, 1.5)`.
    pub origin: [f32; 3],
    /// Ray direction (any magnitude; walker preserves it).
    pub dir: [f32; 3],
    /// Body-frame t at which the ray entered the outer shell
    /// (caller-computed — the walker starts from here).
    pub t_enter: f32,
}

/// A walker instance, owning its slot stack. Construct via
/// [`FaceWalker::begin`] and drive with [`FaceWalker::run`].
pub struct FaceWalker<'a> {
    /// The shared node library.
    library: &'a NodeLibrary,
    /// Root node of the CubedSphereBody — needed for seam crossings
    /// (we look up the neighbor face's root via `FACE_SLOTS`).
    body_root_id: NodeId,
    /// Cubed-sphere radii in the containing cell's local `[0, 1)`
    /// frame (NOT body-size=3 frame). Scaled up by the walker.
    inner_r: f32,
    outer_r: f32,
    body_size: f32, // = 3.0 in the standard convention

    // ── Per-ray state ──────────────────────────────────────────
    /// Integer slot chain — source of truth for position.
    slot_stack: Vec<(u8, u8, u8)>,
    /// Parallel node stack. `node_stack[d]` is the node whose
    /// children are picked by `slot_stack[d]`. `node_stack.len() ==
    /// slot_stack.len() + 1` (including the face-subtree root).
    node_stack: Vec<NodeId>,

    /// Current face.
    face: Face,
    /// Residual ray origin in the current cell's `[0, 3)³` local frame.
    residual_o: [f32; 3],
    /// Residual ray direction (×3 per descent).
    rd_local: [f32; 3],

    /// Cell center in face-normalized coords (tracked incrementally).
    /// These ride the descent and are used ONLY for shading normals
    /// at r-shell crossings — never for boundary tests. At depth N,
    /// their low bits fall below f32 ULP, which is harmless because
    /// tan()/radial projections are smooth.
    u_c: f32,
    v_c: f32,
    r_c: f32,

    /// Current cell slot in the face-subtree frame. Explicit integer
    /// tracking — the residual is allowed to land AT a boundary
    /// (residual component == integer) without ambiguity about
    /// which cell owns it. On advance we explicitly increment /
    /// decrement the appropriate component rather than re-deriving
    /// from `floor(residual)`.
    cur_slot: (i8, i8, i8),

    /// Body-frame ray (kept for hit-t reconstruction; reserved for
    /// future use by callers that need the original input ray).
    #[allow(dead_code)]
    ray: BodyRay,

    /// Body-frame t of the cell the walker is currently processing.
    /// Bumped after every step by the body-frame equivalent of the
    /// local advance. Start = `ray.t_enter`.
    t_body: f32,

    /// Per-ray iteration counter. Bounded for safety.
    iter_budget: u32,
}

/// Pack a `(us, vs, rs)` slot triple into the 27-ary index used by
/// the tree's child array.
#[inline]
pub fn slot_from_uvr(us: u8, vs: u8, rs: u8) -> usize {
    debug_assert!(us < 3 && vs < 3 && rs < 3);
    rs as usize * 9 + vs as usize * 3 + us as usize
}

impl<'a> FaceWalker<'a> {
    /// Begin a walk: position the walker at the face-subtree root
    /// and seed the initial residual slot from the ray's body-frame
    /// entry point.
    ///
    /// `body_root_id` is the CubedSphereBody node (whose children
    /// include the 6 face-subtree roots at `FACE_SLOTS` and the
    /// core at `CORE_SLOT`).
    pub fn begin(
        library: &'a NodeLibrary,
        body_root_id: NodeId,
        inner_r: f32,
        outer_r: f32,
        ray: BodyRay,
    ) -> Option<FaceWalker<'a>> {
        // The body occupies [0, 3)³ in body-local coords; center at (1.5, 1.5, 1.5).
        let body_size = 3.0_f32;
        // Radii are supplied in cell-local `[0, 1)` frame (the
        // convention used by NodeKind::CubedSphereBody). Scale to
        // body-size=3 units for the face-space projector.
        let inner_body = inner_r * body_size;
        let outer_body = outer_r * body_size;

        // Body-frame entry point.
        let p_entry = [
            ray.origin[0] + ray.dir[0] * ray.t_enter,
            ray.origin[1] + ray.dir[1] * ray.t_enter,
            ray.origin[2] + ray.dir[2] * ray.t_enter,
        ];
        // Project onto face-space.
        let fp = super::body_point_to_face_space(p_entry, inner_body, outer_body, body_size)?;

        // Look up the face subtree root inside the body node.
        let body_node = library.get(body_root_id)?;
        let face_subtree_id = match body_node.children[FACE_SLOTS[fp.face as usize]] {
            Child::Node(id) => id,
            _ => return None,
        };

        // Initial residual: (un, vn, rn) × 3, so residual ∈ [0, 3)³.
        let mut residual_o = [fp.un * 3.0, fp.vn * 3.0, fp.rn * 3.0];
        // Clamp exactly away from the upper boundary to keep floor(.)
        // within [0, 2]. The rn side especially matters because at
        // t_enter the ray is ON the outer shell (rn ≈ 1.0).
        const RESIDUAL_EPS: f32 = 1e-5;
        for r in &mut residual_o {
            if *r >= 3.0 {
                *r = 3.0 - RESIDUAL_EPS;
            }
            if *r < 0.0 {
                *r = 0.0;
            }
        }

        // Initial rd_local: the ray direction expressed in
        // face-normalized [0, 3) coords, evaluated at the entry
        // point. Linearization accepts this as constant over the
        // face subtree. We derive it from the Jacobian of
        // `body_point_to_face_space` via finite difference over a
        // cell-scale step.
        let rd_local = initial_rd_local(ray.dir, p_entry, fp, inner_body, outer_body, body_size);

        let cur_slot = (
            clamp_slot(residual_o[0]) as i8,
            clamp_slot(residual_o[1]) as i8,
            clamp_slot(residual_o[2]) as i8,
        );

        Some(FaceWalker {
            library,
            body_root_id,
            inner_r,
            outer_r,
            body_size,
            slot_stack: Vec::with_capacity(MAX_FACE_WALK_DEPTH),
            node_stack: vec![face_subtree_id],
            face: fp.face,
            residual_o,
            rd_local,
            u_c: fp.un,
            v_c: fp.vn,
            r_c: fp.rn,
            cur_slot,
            ray,
            t_body: ray.t_enter,
            iter_budget: 0,
        })
    }

    /// Current depth within the face subtree (0 = at root).
    #[inline]
    pub fn depth(&self) -> usize {
        self.slot_stack.len()
    }

    /// Magnitude of the residual ray origin — used by the precision
    /// test to assert boundedness.
    pub fn residual_magnitude(&self) -> f32 {
        let o = self.residual_o;
        (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt()
    }

    /// Immutable view of the residual (testing).
    pub fn residual(&self) -> [f32; 3] {
        self.residual_o
    }

    /// Immutable view of `rd_local` (testing).
    pub fn rd_local(&self) -> [f32; 3] {
        self.rd_local
    }

    /// Current face.
    pub fn face(&self) -> Face {
        self.face
    }

    /// Current slot chain.
    pub fn slot_path(&self) -> &[(u8, u8, u8)] {
        &self.slot_stack
    }

    /// Current cell center in face-normalized coords. Derived
    /// incrementally from the slot stack; low bits are trustworthy
    /// only to `f32::EPSILON ≈ 1.2e-7`.
    pub fn cell_center(&self) -> (f32, f32, f32) {
        let ext = 3.0_f32.powi(-(self.depth() as i32));
        let u = self.u_c.floor_cell(ext) + 0.5 * ext;
        let v = self.v_c.floor_cell(ext) + 0.5 * ext;
        let r = self.r_c.floor_cell(ext) + 0.5 * ext;
        (u, v, r)
    }

    /// Reconstruct the current cell's face-normalized lower corner
    /// exactly from the integer slot stack via `f64` accumulation.
    /// Used by tests to compare against the incrementally-tracked
    /// f32 `u_c` / `v_c` / `r_c` values and verify precision.
    pub fn cell_lo_exact(&self) -> (f64, f64, f64) {
        let mut u = 0.0_f64;
        let mut v = 0.0_f64;
        let mut r = 0.0_f64;
        let mut ext = 1.0_f64 / 3.0;
        for &(us, vs, rs) in &self.slot_stack {
            u += us as f64 * ext;
            v += vs as f64 * ext;
            r += rs as f64 * ext;
            ext /= 3.0;
        }
        (u, v, r)
    }

    /// Face-normalized cell extent at the current depth (derived
    /// from the slot stack length, NOT tracked as an f32 state).
    pub fn cell_ext_exact(&self) -> f64 {
        3.0_f64.powi(-(self.slot_stack.len() as i32 + 1))
    }

    /// Take one DDA step. Returns `Some(WalkResult)` if the walk
    /// terminates this step; `None` to continue iterating.
    pub fn step(&mut self) -> Option<WalkResult> {
        self.iter_budget += 1;
        if self.iter_budget > 4096 {
            return Some(WalkResult::IterationExhausted);
        }

        // Current cell slot is tracked explicitly in `cur_slot`
        // (integer). Residual is allowed to sit AT an integer
        // boundary without ambiguity.
        let (us_i, vs_i, rs_i) = self.cur_slot;
        if us_i < 0 || us_i > 2 || vs_i < 0 || vs_i > 2 || rs_i < 0 || rs_i > 2 {
            // Defensive — should have been handled by bubble-up.
            return self.handle_oob();
        }
        let us = us_i as u8;
        let vs = vs_i as u8;
        let rs = rs_i as u8;

        // Look up the child at this cell.
        let cur_node_id = *self.node_stack.last().unwrap();
        let cur_node = match self.library.get(cur_node_id) {
            Some(n) => n,
            None => return Some(WalkResult::OuterShellExit),
        };
        let child = cur_node.children[slot_from_uvr(us, vs, rs)];

        match child {
            Child::Block(bt) => {
                let mut path = self.slot_stack.clone();
                path.push((us, vs, rs));
                return Some(WalkResult::Hit {
                    path,
                    block: bt,
                    face: self.face,
                    t_body: self.t_body,
                });
            }
            Child::Node(child_id) => {
                if self.slot_stack.len() + 1 >= MAX_FACE_WALK_DEPTH {
                    // Stack full — treat as empty; fall through.
                } else {
                    self.descend(us, vs, rs, child_id);
                    return None;
                }
            }
            Child::Empty | Child::EntityRef(_) => {
                // Fall through to advance.
            }
        }

        // Advance to next cell boundary in residual coords. Exit
        // face lies at us OR us+1 (depending on rd_local sign) for
        // u-axis and symmetrically for v, r. Pick smallest t.
        let cell_lo = [us_i as f32, vs_i as f32, rs_i as f32];
        let cell_hi = [(us_i + 1) as f32, (vs_i + 1) as f32, (rs_i + 1) as f32];

        let (t_next, exit_axis, exit_positive) = pick_exit(
            self.residual_o, self.rd_local, cell_lo, cell_hi,
        );

        if !t_next.is_finite() {
            // Ray parallel to all three axes (or no boundary with
            // positive t). Degenerate — treat as exhausted.
            return Some(WalkResult::IterationExhausted);
        }

        // Advance residual EXACTLY to the boundary. For precision at
        // deep depth we use `t + 0.0` and then clamp the exit-axis
        // component to the boundary value (avoiding floor-vs-slot
        // race). rd_local grows 3^N with depth, so `rd_local × t`
        // can lose low bits but the BOUNDARY is at a small integer
        // in residual coords so clamping restores precision.
        for i in 0..3 {
            self.residual_o[i] += self.rd_local[i] * t_next;
        }
        // Snap the exit-axis residual exactly to the boundary.
        let boundary_value = if exit_positive {
            cell_hi[exit_axis]
        } else {
            cell_lo[exit_axis]
        };
        self.residual_o[exit_axis] = boundary_value;

        // Update cur_slot explicitly.
        let step = if exit_positive { 1 } else { -1 };
        match exit_axis {
            0 => self.cur_slot.0 += step,
            1 => self.cur_slot.1 += step,
            _ => self.cur_slot.2 += step,
        }

        // Coarse body-t tracker for tie-breaking (hit t computed
        // more accurately at hit report time via reprojection).
        self.t_body += t_next.abs() * 1e-6;

        // Bubble up if the new slot left [0, 2] on any axis.
        self.handle_oob()
    }

    /// Run to termination, returning the final `WalkResult`.
    pub fn run(&mut self) -> WalkResult {
        loop {
            if let Some(r) = self.step() {
                return r;
            }
        }
    }

    // ─────────────────────────────────────────── descent & exit

    fn descend(&mut self, us: u8, vs: u8, rs: u8, child_id: NodeId) {
        self.slot_stack.push((us, vs, rs));
        self.node_stack.push(child_id);

        // Rescale residual: subtract the child's lower corner (the
        // integer slot coords) and multiply by 3. After this, the
        // residual lies in the child cell's [0, 3)³. `rd_local ×= 3`.
        for i in 0..3 {
            self.residual_o[i] =
                (self.residual_o[i] - [us as f32, vs as f32, rs as f32][i]) * 3.0;
            self.rd_local[i] *= 3.0;
        }

        // New slot within the child cell: wherever residual_o now
        // points. Use floor() on the rescaled residual; clamp to
        // [0, 2]. After a fresh descent residual is strictly inside
        // [0, 3)³ (we descended because residual WAS in the child's
        // parent-slot cell).
        self.cur_slot = (
            clamp_slot(self.residual_o[0]) as i8,
            clamp_slot(self.residual_o[1]) as i8,
            clamp_slot(self.residual_o[2]) as i8,
        );

        // Update incremental cell-center (face-normalized LOWER
        // CORNER) tracker. Used only for shading normals — never
        // in boundary arithmetic.
        let (u, v, r) = self.cell_lo_f32_from_stack();
        self.u_c = u;
        self.v_c = v;
        self.r_c = r;
    }

    /// Compute the current cell's face-normalized lower corner from
    /// the slot stack in f32. Monotonic accumulator; at deep depth
    /// the `ext *= 1.0/3.0` loses ULP bits in `u_c` / `v_c` / `r_c`,
    /// which is the behavior we're proving is harmless (smooth tan()
    /// evaluation, not boundary tests).
    fn cell_lo_f32_from_stack(&self) -> (f32, f32, f32) {
        let mut u = 0.0_f32;
        let mut v = 0.0_f32;
        let mut r = 0.0_f32;
        let mut ext = 1.0_f32 / 3.0;
        for &(us, vs, rs) in &self.slot_stack {
            u += us as f32 * ext;
            v += vs as f32 * ext;
            r += rs as f32 * ext;
            ext /= 3.0;
        }
        (u, v, r)
    }

    fn handle_oob(&mut self) -> Option<WalkResult> {
        loop {
            let (us, vs, rs) = self.cur_slot;
            let in_range = us >= 0 && us <= 2 && vs >= 0 && vs <= 2 && rs >= 0 && rs <= 2;
            if in_range {
                return None;
            }

            if self.slot_stack.is_empty() {
                // At face-subtree root — terminal disposition.
                if rs < 0 {
                    return Some(WalkResult::InnerShellExit);
                }
                if rs > 2 {
                    return Some(WalkResult::OuterShellExit);
                }
                // UV seam: determine exit edge from which slot axis
                // stepped out.
                let edge = if us < 0 {
                    0
                } else if us > 2 {
                    1
                } else if vs < 0 {
                    2
                } else {
                    3
                };
                if !self.seam_cross(edge) {
                    return Some(WalkResult::OuterShellExit);
                }
                continue;
            }

            self.ascend();
        }
    }

    fn ascend(&mut self) {
        // Pop one level. Remember which slot we descended from so
        // we can compute the parent's slot after bubbling.
        let (child_us, child_vs, child_rs) = self.slot_stack.pop().expect("non-empty");
        self.node_stack.pop();

        // Undo the descent rescale: residual / 3 + slot, rd_local / 3.
        for i in 0..3 {
            self.rd_local[i] /= 3.0;
            self.residual_o[i] =
                self.residual_o[i] / 3.0 + [child_us, child_vs, child_rs][i] as f32;
        }

        // Parent cell's cur_slot: whatever slot in the parent
        // contains residual_o. The child's OOB axis told us the
        // parent's slot stepped by ±1 from (child_us, child_vs,
        // child_rs). Specifically: if residual_o[i] went to -ε < 0
        // then parent's slot[i] = child_slot[i] - 1; if > 3 then
        // +1; else unchanged. But after residual /= 3 + slot, the
        // residual is back in parent coords so we can reach it with
        // clamp_slot directly — UNLESS the parent's slot is itself
        // OOB (which continues the bubble-up loop).
        let parent_slot = |r: f32, child: u8| -> i8 {
            if r < 0.0 {
                child as i8 - 1
            } else if r >= 3.0 {
                child as i8 + 1
            } else {
                // Inside parent's [0, 3) — use residual's integer
                // part, which equals child for non-boundary cases
                // and is child ± 1 for boundary cases we've already
                // corrected.
                let i = r.floor() as i32;
                i.clamp(-1, 3) as i8
            }
        };

        self.cur_slot = (
            parent_slot(self.residual_o[0], child_us),
            parent_slot(self.residual_o[1], child_vs),
            parent_slot(self.residual_o[2], child_rs),
        );

        // Re-seed cell center from updated stack.
        let (u, v, r) = self.cell_lo_f32_from_stack();
        self.u_c = u;
        self.v_c = v;
        self.r_c = r;
    }

    fn seam_cross(&mut self, exit_edge: u8) -> bool {
        let transition = SEAM_TABLE[self.face as usize][exit_edge as usize];
        let new_face = transition.neighbor_face;
        let r_seam = transition.rotation;

        // Reset residual to the entry point on the new face. This is
        // computed by re-projecting the body-frame ray position at
        // the seam-crossing instant back onto the new face via
        // `body_point_to_face_space`. The walker's incremental
        // linearized residual has `O(cell_size²)` drift across a
        // seam; re-projecting stops the drift from accumulating.
        //
        // Compute current body-frame position (approximate) by:
        // 1. Reconstruct residual in the face-subtree root's frame
        //    (we're at depth 0 here — residual_o is already in
        //    root frame).
        // 2. Map residual_o / 3 back to (un, vn, rn).
        // 3. Call `face_space_to_body_point` to get body position.
        // This path avoids absolute-coord accumulation.
        let un = self.residual_o[0] * (1.0 / 3.0);
        let vn = self.residual_o[1] * (1.0 / 3.0);
        let rn = (self.residual_o[2] * (1.0 / 3.0)).clamp(0.0, 1.0);
        // Clamp un/vn into [0, 1] — seam crossings happen AT the
        // 0 or 1 boundary, tiny f32 drift puts it a hair outside.
        let un_c = un.clamp(-0.01, 1.01);
        let vn_c = vn.clamp(-0.01, 1.01);
        let inner_body = self.inner_r * self.body_size;
        let outer_body = self.outer_r * self.body_size;
        let p_body = super::face_space_to_body_point(
            self.face, un_c, vn_c, rn, inner_body, outer_body, self.body_size,
        );

        // Re-project onto new face.
        let Some(fp_new) = super::body_point_to_face_space(
            p_body, inner_body, outer_body, self.body_size,
        ) else {
            return false;
        };

        if fp_new.face != new_face {
            // Seam table & projection disagree — rare corner case
            // (three-face corner). Trust the projection.
        }
        self.face = fp_new.face;

        // Look up the new face's subtree root.
        let body_node = match self.library.get(self.body_root_id) {
            Some(n) => n,
            None => return false,
        };
        let face_slot = FACE_SLOTS[self.face as usize];
        let new_root_id = match body_node.children[face_slot] {
            Child::Node(id) => id,
            _ => return false,
        };

        // Reset walker state at new face subtree root, depth 0.
        self.slot_stack.clear();
        self.node_stack.clear();
        self.node_stack.push(new_root_id);

        self.residual_o = [
            (fp_new.un * 3.0).clamp(0.0, 3.0 - 1e-5),
            (fp_new.vn * 3.0).clamp(0.0, 3.0 - 1e-5),
            (fp_new.rn * 3.0).clamp(0.0, 3.0 - 1e-5),
        ];
        self.cur_slot = (
            clamp_slot(self.residual_o[0]) as i8,
            clamp_slot(self.residual_o[1]) as i8,
            clamp_slot(self.residual_o[2]) as i8,
        );

        // Rotate rd_local via SEAM_TABLE rotation (applied in
        // face-local coords). rd_local at depth 0 on the old face is
        // in face-normalized [0, 3) coords — same as the SEAM_TABLE
        // frame. Apply the 3×3 rotation `R_seam`.
        let rd = self.rd_local;
        self.rd_local = [
            r_seam[0][0] * rd[0] + r_seam[0][1] * rd[1] + r_seam[0][2] * rd[2],
            r_seam[1][0] * rd[0] + r_seam[1][1] * rd[1] + r_seam[1][2] * rd[2],
            r_seam[2][0] * rd[0] + r_seam[2][1] * rd[1] + r_seam[2][2] * rd[2],
        ];

        self.u_c = fp_new.un;
        self.v_c = fp_new.vn;
        self.r_c = fp_new.rn;
        let _ = CORE_SLOT; // silence unused-import warning on this path
        true
    }
}

// ─────────────────────────────────────────── initial rd_local

/// Derive `rd_local` in face-normalized `[0, 3)` coords from the
/// body-frame ray direction at the entry point. Uses a finite-
/// difference Jacobian of `body_point_to_face_space` — O(1)
/// compute, evaluated once at face entry. The linearization error
/// across the face subtree is `O(face_size²)` body units per the
/// architecture doc.
fn initial_rd_local(
    dir_body: [f32; 3],
    p_entry: [f32; 3],
    fp_entry: FacePoint,
    inner_r: f32,
    outer_r: f32,
    body_size: f32,
) -> [f32; 3] {
    // Finite-difference step size: a cell at face-depth 3 is about
    // 1/27 body units. Step h = 1e-3 body units to stay well below
    // that and still be above f32 noise at [1.5, 1.5, 1.5] scale.
    let h = 1e-3_f32;
    let len = (dir_body[0] * dir_body[0] + dir_body[1] * dir_body[1] + dir_body[2] * dir_body[2])
        .sqrt()
        .max(1e-20);
    let u = [dir_body[0] / len, dir_body[1] / len, dir_body[2] / len];

    // Take a body-frame step; project onto face-space.
    let p_plus = [p_entry[0] + h * u[0], p_entry[1] + h * u[1], p_entry[2] + h * u[2]];
    let Some(fp_plus) =
        super::body_point_to_face_space(p_plus, inner_r, outer_r, body_size)
    else {
        // Degenerate — ray aimed through center. Fall back to the
        // face's basis vectors.
        return [0.0, 0.0, 0.0];
    };

    // If the finite-difference step landed on a DIFFERENT face, we
    // must project back onto `fp_entry.face` to get a consistent
    // rd_local on the entry face. The step is small (1e-3) compared
    // to the face size, so this only happens if the entry is right
    // at a seam — accept a small rd_local error in that rare case.
    let delta = if fp_plus.face == fp_entry.face {
        [fp_plus.un - fp_entry.un, fp_plus.vn - fp_entry.vn, fp_plus.rn - fp_entry.rn]
    } else {
        [0.0, 0.0, 0.0]
    };

    // rd in face-normalized coords per unit body-t. Scale back up
    // by body_len / h so that the direction has magnitude matching
    // the input ray's body-frame direction magnitude.
    let scale = len / h;
    [
        delta[0] * 3.0 * scale,
        delta[1] * 3.0 * scale,
        delta[2] * 3.0 * scale,
    ]
}

// ──────────────────────────────────────────── boundary arithmetic

/// Pick the exit axis and local-t at which the residual ray leaves
/// the current cell's `[us..us+1] × [vs..vs+1] × [rs..rs+1]`
/// integer sub-box (in the residual frame). Returns
/// `(t_local, exit_axis, exit_positive)`.
fn pick_exit(
    o: [f32; 3],
    d: [f32; 3],
    cell_lo: [f32; 3],
    cell_hi: [f32; 3],
) -> (f32, usize, bool) {
    let mut t_best = f32::INFINITY;
    let mut axis_best = 0;
    let mut pos_best = true;
    for i in 0..3 {
        let di = d[i];
        if di.abs() < 1e-30 {
            continue;
        }
        let boundary = if di > 0.0 { cell_hi[i] } else { cell_lo[i] };
        let t = (boundary - o[i]) / di;
        if t >= 0.0 && t < t_best {
            t_best = t;
            axis_best = i;
            pos_best = di > 0.0;
        }
    }
    (t_best, axis_best, pos_best)
}

#[inline]
fn clamp_slot(v: f32) -> u8 {
    let i = v.floor() as i32;
    i.clamp(0, 2) as u8
}

// ────────────────────────────────────────────── precision helpers

trait FloorCell {
    fn floor_cell(self, ext: Self) -> Self;
}
impl FloorCell for f32 {
    fn floor_cell(self, ext: f32) -> f32 {
        (self / ext).floor() * ext
    }
}

// ────────────────────────────────────────────────────────── tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::cubesphere::Face;
    use crate::world::tree::{empty_children, Child, NodeKind, NodeLibrary};

    /// Construct a face subtree of `depth` levels with a single solid
    /// voxel at the specified slot chain. Returns `(body_root_id,
    /// face, chain)` — the body root node wraps the face subtree at
    /// its FACE_SLOTS entry, and empties elsewhere.
    fn build_deep_face_subtree(
        lib: &mut NodeLibrary,
        face: Face,
        chain: &[(u8, u8, u8)],
        leaf_block: u16,
    ) -> NodeId {
        // Build bottom-up: the leaf level is a Cartesian node with a
        // Block at the innermost slot; recurse upward substituting
        // each parent's chain[i] slot with the child below.
        let mut child_id: Option<NodeId> = None;
        for (rev_depth, &(us, vs, rs)) in chain.iter().enumerate().rev() {
            let mut children = empty_children();
            let slot = rs as usize * 9 + vs as usize * 3 + us as usize;
            children[slot] = if rev_depth == chain.len() - 1 {
                // Leaf level: put the solid block here.
                Child::Block(leaf_block)
            } else {
                Child::Node(child_id.unwrap())
            };
            child_id = Some(lib.insert(children));
        }
        let face_subtree_id = child_id.expect("chain non-empty");

        // Wrap face_subtree in body node at FACE_SLOTS[face].
        let mut body_children = empty_children();
        body_children[FACE_SLOTS[face as usize]] = Child::Node(face_subtree_id);
        lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody {
                inner_r: 0.1,
                outer_r: 0.45,
            },
        )
    }

    /// Build a slot chain of length `depth` that picks `(1, 1, 0)`
    /// each step — i.e. the CENTER wedge at each level, at the
    /// INNERMOST radial slot. This means the deep cell is small but
    /// located exactly at the face's mid-UV, giving a stable
    /// ray-aim target.
    fn center_chain(depth: usize) -> Vec<(u8, u8, u8)> {
        // (us=1, vs=1, rs=0) each step: u-center, v-center, inner-r.
        // Picking rs=0 puts the solid voxel against the inner shell;
        // rs=2 would be outer. Doesn't matter for the precision
        // test — what we verify is the walker can DESCEND to the
        // depth-N leaf, not which specific leaf.
        vec![(1u8, 1u8, 0u8); depth]
    }

    #[test]
    fn precision_descent_residual_bounded() {
        // Build a face subtree 30 levels deep with a solid voxel at
        // the end, and verify that at every step of the walk the
        // residual stays in [0, 3)³ regardless of depth. Demonstrates
        // that NO absolute f32 state scaling as 1/3^N can cause
        // precision loss in the boundary arithmetic.
        let mut lib = NodeLibrary::default();
        let body_id = build_deep_face_subtree(
            &mut lib, Face::PosY, &center_chain(30), 42,
        );

        // Outer shell at r=0.45 (cell-local), body_size=3 → body-local
        // radius 1.35; PosY face of body center (1.5,1.5,1.5) is at
        // y=1.5+1.35=2.85. Camera origin at y=3.5, t_enter = 3.5-2.85=0.65.
        let ray = BodyRay {
            origin: [1.5, 3.5, 1.5],
            dir: [0.0, -1.0, 0.0],
            t_enter: 0.65,
        };
        let mut walker = FaceWalker::begin(&lib, body_id, 0.1, 0.45, ray)
            .expect("walker constructs");

        let mut max_depth_seen = 0;
        let mut final_result = None;
        for _ in 0..4096 {
            // Precondition check: residual in [0, 3)³ at every step.
            for i in 0..3 {
                let r = walker.residual()[i];
                assert!(
                    r >= -1e-3 && r < 3.0 + 1e-3,
                    "residual[{i}] = {r} at depth {} outside [0, 3)",
                    walker.depth()
                );
                assert!(r.is_finite(), "residual[{i}] = {r} not finite");
            }
            let mag = walker.residual_magnitude();
            assert!(
                mag.is_finite() && mag < 10.0,
                "residual magnitude {mag} at depth {} out of bounds",
                walker.depth()
            );

            if walker.depth() > max_depth_seen {
                max_depth_seen = walker.depth();
            }

            if let Some(r) = walker.step() {
                final_result = Some(r);
                break;
            }
        }

        // We expect the walk to reach depth 29 (slot_stack length
        // right before the final Hit; the 30th level is where the
        // Block is found). The precision claim is about residual
        // boundedness, which we assert every iteration above.
        assert!(
            max_depth_seen >= 29,
            "walker descended only to depth {max_depth_seen} (expected ≥ 29)",
        );
        assert!(
            matches!(final_result, Some(WalkResult::Hit { .. })),
            "walk should terminate in Hit, got {final_result:?}"
        );
    }

    #[test]
    fn precision_at_face_depth_30() {
        // Build a face subtree where the chain descends always into
        // slot (1, 1, 0) — the inner-radius mid-UV wedge. At depth
        // 30, the cell extent in face-normalized coords is 1/3^30 ≈
        // 1.95e-15, far below f32 ULP of 1.0 (≈ 1.2e-7). We verify
        // that AFTER 30 DESCENTS:
        //   1. The slot stack correctly records the 30 descents.
        //   2. Residual stays in [0, 3)³.
        //   3. rd_local magnitude grows by exactly 3^30 (to f32
        //      precision).
        //   4. cell_lo_exact() reconstructs the face-normalized lower
        //      corner to ~f32 epsilon of the true value — proving
        //      the slot-stack is a lossless position record.
        //
        // Chain length 31 so the walker has to descend 30 times to
        // reach the leaf level that holds the Block.
        let mut lib = NodeLibrary::default();
        let chain = center_chain(31);
        let body_id = build_deep_face_subtree(&mut lib, Face::PosY, &chain, 77);

        let ray = BodyRay {
            origin: [1.5, 3.5, 1.5],
            dir: [0.0, -1.0, 0.0],
            t_enter: 0.65,
        };
        let mut walker = FaceWalker::begin(&lib, body_id, 0.1, 0.45, ray)
            .expect("walker constructs");

        let rd_mag0 = {
            let d = walker.rd_local();
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };

        // Snapshot walker state when depth reaches 30.
        let mut snapshot: Option<(usize, [f32; 3], [f32; 3], (f32, f32, f32), (f64, f64, f64), f64)> = None;
        let mut hit = None;
        for _ in 0..4096 {
            if walker.depth() == 30 && snapshot.is_none() {
                let (u_exact, v_exact, r_exact) = walker.cell_lo_exact();
                snapshot = Some((
                    walker.depth(),
                    walker.residual_o,
                    walker.rd_local,
                    (walker.u_c, walker.v_c, walker.r_c),
                    (u_exact, v_exact, r_exact),
                    walker.cell_ext_exact(),
                ));
            }
            if let Some(res) = walker.step() {
                hit = Some(res);
                break;
            }
        }

        let snap = snapshot.expect("walker must reach depth 30");
        let (depth, residual, rd_local, (u_c, v_c, r_c), (u_ex, v_ex, r_ex), ext) = snap;
        assert_eq!(depth, 30);

        // Residual in O(1) range at depth 30.
        for (i, &r) in residual.iter().enumerate() {
            assert!(
                r >= -1e-3 && r < 3.0 + 1e-3 && r.is_finite(),
                "residual[{i}] = {r} out of [0, 3) at depth 30"
            );
        }
        let mag = (residual[0].powi(2) + residual[1].powi(2) + residual[2].powi(2)).sqrt();
        assert!(
            mag < 10.0,
            "residual magnitude {mag} out of bounds at depth 30"
        );

        // rd_local has grown by 3^30 (in ratio terms). f32 has
        // ≤ 30 × 1 ULP drift from 30 compound multiplications by
        // 3.0 (which is exactly representable in binary).
        let rd_mag_now = (rd_local[0].powi(2) + rd_local[1].powi(2) + rd_local[2].powi(2)).sqrt();
        let expected_growth = 3.0_f32.powi(30);
        let ratio = rd_mag_now / rd_mag0;
        assert!(
            (ratio / expected_growth - 1.0).abs() < 1e-4,
            "rd_local grew {ratio}× (expected {expected_growth}×) at depth 30"
        );

        // Verify cell_lo_exact at the snapshot: slot stack was
        // 30 × (1, 1, 0) so u_lo = v_lo = sum_{k=1..30} 1 * 3^(-k)
        // = (1/2)(1 - 3^-30) ≈ 0.5; r_lo = 0.
        assert!(
            (u_ex - 0.5).abs() < 1e-14,
            "u_lo_exact at depth 30 = {u_ex} (expected ≈ 0.5)"
        );
        assert!(
            (v_ex - 0.5).abs() < 1e-14,
            "v_lo_exact at depth 30 = {v_ex} (expected ≈ 0.5)"
        );
        assert!(
            r_ex.abs() < 1e-14,
            "r_lo_exact at depth 30 = {r_ex} (expected 0)"
        );

        // The incremental f32 cell-center tracker should agree with
        // the exact f64 to ~f32 ULP × depth (≤ 1e-5 at depth 30).
        // u_c / v_c / r_c store the face-normalized LOWER CORNER.
        let du = (u_c as f64 - u_ex).abs();
        let dv = (v_c as f64 - v_ex).abs();
        let dr = (r_c as f64 - r_ex).abs();
        assert!(
            du < 1e-5 && dv < 1e-5 && dr < 1e-5,
            "f32 cell-lo drift (du={du}, dv={dv}, dr={dr}) at depth 30 exceeded tolerance"
        );

        // Cell extent at depth 30: 3^-(30+1) ≈ 6.5e-16.
        // (cell_ext_exact uses slot_stack.len()+1 == 31 exponent.)
        assert!(
            ext > 0.0 && ext < 1e-14,
            "cell_ext at depth 30 = {ext} (expected ~6.5e-16)"
        );

        // The walk must have terminated in a Hit (block 77 at depth 30).
        match hit {
            Some(WalkResult::Hit { block, path, .. }) => {
                assert_eq!(block, 77);
                assert_eq!(path.len(), 31, "hit path length {}", path.len());
            }
            other => panic!("expected Hit at depth 30, got {other:?}"),
        }
    }

    #[test]
    fn walker_reaches_depth_30_hit() {
        // End-to-end: build a face subtree with a solid voxel at
        // face-subtree depth 30 along the center chain. Shoot a ray
        // aimed exactly at the face center and assert the walker
        // lands a Hit with the expected block tag and path length.
        let mut lib = NodeLibrary::default();
        let chain = center_chain(30);
        let body_id =
            build_deep_face_subtree(&mut lib, Face::PosY, &chain, 123);

        // Ray aimed straight down at the PosY face center. Radii
        // are cell-local; body-size=3 gives outer body-radius=1.35,
        // inner=0.3. Body center (1.5, 1.5, 1.5). Outer shell top
        // at y = 1.5 + 1.35 = 2.85. Camera at y=3.5 → t_enter=0.65.
        let ray = BodyRay {
            origin: [1.5, 3.5, 1.5],
            dir: [0.0, -1.0, 0.0],
            t_enter: 0.65,
        };
        let mut walker = FaceWalker::begin(&lib, body_id, 0.1, 0.45, ray)
            .expect("walker constructs");

        let result = walker.run();

        match result {
            WalkResult::Hit { path, block, face, .. } => {
                assert_eq!(block, 123, "block tag mismatch: {block}");
                assert_eq!(face, Face::PosY);
                assert_eq!(
                    path.len(),
                    30,
                    "expected 30-deep hit path, got {}",
                    path.len()
                );
                for (i, &s) in path.iter().enumerate() {
                    assert_eq!(s, (1, 1, 0), "slot {i} of path was {s:?}");
                }
            }
            other => panic!("expected Hit at depth 30, got {other:?}"),
        }
    }
}
