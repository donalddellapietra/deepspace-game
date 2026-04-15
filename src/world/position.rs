//! Path-based coordinates.
//!
//! A `Position` locates a point in the tree without any absolute-XYZ
//! accumulator. `path[0..depth]` walks from root (index 0) to the
//! deepest resolved node (index `depth - 1`); `offset` is the
//! sub-slot fractional position in `[0, 1)³` inside that node.
//!
//! See `docs/experimental-architecture/refactor-decisions.md` §2.
//!
//! Step 1 of the migration: this module is purely additive — no
//! existing code calls it yet. `step_neighbor` here is the Cartesian
//! dispatch only; `NodeKind`-aware dispatch arrives in step 2.

use crate::world::tree::{
    slot_coords, slot_index, Child, NodeId, NodeKind, NodeLibrary,
    CHILDREN_PER_NODE, MAX_DEPTH,
};

/// Body-face-center slot indices, indexed by `cubesphere::Face as u8`.
/// Order: +X, −X, +Y, −Y, +Z, −Z. Same table the shader uses.
const FACE_CENTER_SLOTS: [u8; 6] = [14, 12, 16, 10, 22, 4];

fn slot_to_face(slot: u8) -> Option<u8> {
    FACE_CENTER_SLOTS.iter().position(|&s| s == slot).map(|i| i as u8)
}

// -------------------------------------------------------------- position

/// A point in the recursive tree expressed as (path, offset).
///
/// `path[0]` is the root's chosen child slot; `path[depth - 1]` is
/// the slot of the deepest resolved node. Slots beyond `depth` are
/// unused (kept at 0 for hygiene; never read).
///
/// `offset` is the sub-slot fractional position, each axis in
/// `[0, 1)`. For Cartesian nodes the axes are `(x, y, z)`; for
/// `CubedSphereFace` nodes (step 2+) they're `(u, v, r)`. The struct
/// is agnostic — the deepest node's `NodeKind` determines meaning.
///
/// `PartialEq` compares path, depth, and offset exactly. Use
/// [`Position::same_cell`] to compare only cell identity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Position {
    pub path: [u8; MAX_DEPTH],
    pub depth: u8,
    pub offset: [f32; 3],
}

impl Default for Position {
    fn default() -> Self {
        Self::root()
    }
}

impl Position {
    /// The root of the tree, offset `(0, 0, 0)`.
    pub const fn root() -> Self {
        Self { path: [0; MAX_DEPTH], depth: 0, offset: [0.0; 3] }
    }

    /// Construct from an explicit slot slice and sub-slot offset.
    ///
    /// Panics if `slots.len() > MAX_DEPTH`, any slot is ≥ 27, or any
    /// offset axis is outside `[0, 1)` or not finite. Callers should
    /// use this constructor rather than poking at the `path` array
    /// directly.
    pub fn at_slot_path(slots: &[u8], offset: [f32; 3]) -> Self {
        assert!(slots.len() <= MAX_DEPTH, "path exceeds MAX_DEPTH");
        for &s in slots {
            assert!((s as usize) < CHILDREN_PER_NODE, "slot {} out of range", s);
        }
        for (i, v) in offset.iter().enumerate() {
            assert!(v.is_finite(), "offset[{}] not finite", i);
            assert!(*v >= 0.0 && *v < 1.0, "offset[{}] = {} outside [0, 1)", i, v);
        }
        let mut path = [0u8; MAX_DEPTH];
        path[..slots.len()].copy_from_slice(slots);
        Self { path, depth: slots.len() as u8, offset }
    }

    /// The populated portion of the path.
    #[inline]
    pub fn slots(&self) -> &[u8] {
        &self.path[..self.depth as usize]
    }

    /// Cell identity: same path + depth, ignoring offset.
    ///
    /// Two positions "in the same cell" agree on `same_cell` but may
    /// differ in `==` because their sub-slot offsets differ. Use this
    /// whenever you mean "do these refer to the same node?" rather
    /// than "are these bit-identical locations?".
    pub fn same_cell(&self, other: &Self) -> bool {
        self.depth == other.depth && self.slots() == other.slots()
    }

    /// Descend one level: push the slot determined by offset overflow.
    ///
    /// `offset * 3`'s integer part becomes the new slot; the
    /// fractional part is the new offset in the child's frame. No-op
    /// and returns `false` if already at `MAX_DEPTH`.
    pub fn zoom_in(&mut self) -> bool {
        if self.depth as usize >= MAX_DEPTH {
            return false;
        }
        let mut coords = [0usize; 3];
        for axis in 0..3 {
            let v = self.offset[axis] * 3.0;
            let i = (v.floor() as i32).clamp(0, 2) as usize;
            coords[axis] = i;
            self.offset[axis] = v - i as f32;
            // Floating-point can leave offset at exactly 1.0 (e.g.
            // 0.333... * 3.0 rounds to 1.0). Clamp back into [0, 1).
            if self.offset[axis] >= 1.0 {
                self.offset[axis] = 0.0;
                coords[axis] = (coords[axis] + 1).min(2);
            } else if self.offset[axis] < 0.0 {
                self.offset[axis] = 0.0;
            }
        }
        self.path[self.depth as usize] =
            slot_index(coords[0], coords[1], coords[2]) as u8;
        self.depth += 1;
        true
    }

    /// Ascend one level: pop the deepest slot, remapping the offset
    /// into the parent's frame.
    ///
    /// Returns `false` if already at root — zoom-out past root clamps,
    /// per §2b of refactor-decisions.md.
    pub fn zoom_out(&mut self) -> bool {
        if self.depth == 0 {
            return false;
        }
        self.depth -= 1;
        let slot = self.path[self.depth as usize] as usize;
        let (sx, sy, sz) = slot_coords(slot);
        let s = [sx as f32, sy as f32, sz as f32];
        for axis in 0..3 {
            self.offset[axis] = (self.offset[axis] + s[axis]) / 3.0;
        }
        self.path[self.depth as usize] = 0;
        true
    }

    /// Reconstruct XYZ coordinates in the frame of an ancestor node.
    ///
    /// `ancestor_depth` is how many of the leading slots in `path` to
    /// treat as "above the render root" (i.e. skipped). The ancestor
    /// cell is mapped to the `[0, 3)³` box; the returned coordinates
    /// are the point's position inside that box.
    ///
    /// This is the path-native replacement for [`Self::world_pos`]:
    /// callers that previously read absolute XYZ in the tree root's
    /// frame now pass `ancestor_depth = 0` and get identical numbers,
    /// but the call site is explicit about which frame the XYZ lives
    /// in. Step 5 introduces it as the render-root accessor; step 6
    /// migrates raycast/edit call sites to use it.
    ///
    /// Panics if `ancestor_depth > self.depth`.
    pub fn pos_in_ancestor_frame(&self, ancestor_depth: u8) -> [f32; 3] {
        let d = self.depth as usize;
        let a = ancestor_depth as usize;
        assert!(a <= d, "ancestor_depth {} > depth {}", a, d);
        if d == a {
            // Identity frame: offset IS the position in the [0, 1)³
            // sub-cell; scale up to the ancestor's [0, 3)³ extent.
            return [self.offset[0] * 3.0, self.offset[1] * 3.0, self.offset[2] * 3.0];
        }
        // Reverse Horner in f64:
        //
        //   acc_d = slot_d + offset                  // in [0, 3)
        //   acc_k = slot_k + acc_{k+1} / 3           // in [0, 3)
        //   result = acc_{a+1}
        //
        // Every intermediate value is bounded in `[0, 3)`, so no
        // accumulator grows with depth — unlike forward Horner,
        // which blows through f32's mantissa at depth > 15 and f64's
        // at depth > 33. That's the only way to keep `pos_in_ancestor_frame`
        // precision-stable across the full `MAX_DEPTH = 63` range,
        // and it's what makes 40+ rendering layers possible at all
        // within the tree-root render frame.
        //
        // f64 throughout because each `/ 3.0` step loses ~1 ulp and
        // the accumulated rounding at MAX_DEPTH in f32 would still
        // be visible at shader scale. This is a local numerical
        // method at the path→XYZ boundary — nothing upstream ever
        // stores or accumulates these f64 values.
        let innermost = self.path[d - 1] as usize;
        let (sx, sy, sz) = slot_coords(innermost);
        let mut acc = [
            sx as f64 + self.offset[0] as f64,
            sy as f64 + self.offset[1] as f64,
            sz as f64 + self.offset[2] as f64,
        ];
        for k in (a + 1..d).rev() {
            let slot = self.path[k - 1] as usize;
            let (qx, qy, qz) = slot_coords(slot);
            acc[0] = qx as f64 + acc[0] / 3.0;
            acc[1] = qy as f64 + acc[1] / 3.0;
            acc[2] = qz as f64 + acc[2] / 3.0;
        }
        [acc[0] as f32, acc[1] as f32, acc[2] as f32]
    }

    /// Reconstruct absolute XYZ coordinates in the root cell's frame.
    ///
    /// The root cell has extent `[0, 3)` on each axis (its 27 children
    /// cover `[0, 1), [1, 2), [2, 3)`). This shim exists so old
    /// XYZ-expecting code paths can keep working during the migration;
    /// step 6 deletes all callers.
    pub fn world_pos(&self) -> [f32; 3] {
        self.pos_in_ancestor_frame(0)
    }

    /// Inverse shim: construct a Position at the given tree depth
    /// from absolute XYZ in the root cell's `[0, 3)³` frame. Clamps
    /// out-of-range inputs. Used by XYZ-accepting setters during the
    /// migration; step 6 deletes it.
    pub fn from_world_pos(pos: [f32; 3], depth: u8) -> Self {
        let d = depth as usize;
        assert!(d <= MAX_DEPTH, "depth exceeds MAX_DEPTH");
        let mut path = [0u8; MAX_DEPTH];
        let clamp_hi = 3.0 - f32::EPSILON * 4.0;
        let mut offset = [
            (pos[0] / 3.0).clamp(0.0, 1.0 - f32::EPSILON),
            (pos[1] / 3.0).clamp(0.0, 1.0 - f32::EPSILON),
            (pos[2] / 3.0).clamp(0.0, 1.0 - f32::EPSILON),
        ];
        // Re-clamp in case of NaN or tiny out-of-range drift.
        for axis in 0..3 {
            if pos[axis] >= clamp_hi {
                offset[axis] = 1.0 - f32::EPSILON;
            }
        }
        for k in 0..d {
            let mut coords = [0usize; 3];
            for axis in 0..3 {
                let v = offset[axis] * 3.0;
                let i = (v.floor() as i32).clamp(0, 2) as usize;
                coords[axis] = i;
                offset[axis] = v - i as f32;
                if offset[axis] >= 1.0 {
                    offset[axis] = 0.0;
                    coords[axis] = (coords[axis] + 1).min(2);
                } else if offset[axis] < 0.0 {
                    offset[axis] = 0.0;
                }
            }
            path[k] = slot_index(coords[0], coords[1], coords[2]) as u8;
        }
        Self { path, depth, offset }
    }

    /// NodeKind-aware path construction. Walks the actual tree from
    /// `tree_root`, dispatching subdivision on each node's `NodeKind`:
    /// at a `CubedSphereBody` the Cartesian offset is converted to
    /// `(face, un, vn, rn)` via sphere geometry and the path
    /// descends into the corresponding face-center child; everywhere
    /// else does standard base-3 subdivision (the arithmetic is
    /// identical for `Cartesian` and `CubedSphereFace` — only the
    /// axis interpretation differs).
    ///
    /// Returns a Position whose path correctly indexes the node the
    /// camera is inside at every level — usable as a render-frame
    /// anchor without the "walking through a face subtree as if it
    /// were Cartesian" garbage that vanilla `from_world_pos`
    /// produces.
    pub fn from_world_pos_in_tree(
        pos: [f32; 3],
        target_depth: u8,
        library: &NodeLibrary,
        tree_root: NodeId,
    ) -> Self {
        let d = target_depth as usize;
        assert!(d <= MAX_DEPTH);
        let mut path = [0u8; MAX_DEPTH];
        let clamp_hi = 1.0 - f32::EPSILON;
        let mut offset = [
            (pos[0] / 3.0).clamp(0.0, clamp_hi),
            (pos[1] / 3.0).clamp(0.0, clamp_hi),
            (pos[2] / 3.0).clamp(0.0, clamp_hi),
        ];
        let mut current_id = tree_root;
        for k in 0..d {
            let Some(node) = library.get(current_id) else { break };
            let slot = match node.kind {
                NodeKind::Cartesian | NodeKind::CubedSphereFace { .. } => {
                    subdivide_in_place(&mut offset)
                }
                NodeKind::CubedSphereBody { inner_r, outer_r } => {
                    subdivide_body(&mut offset, inner_r, outer_r)
                }
            };
            path[k] = slot;
            match node.children[slot as usize] {
                Child::Node(child) => current_id = child,
                _ => break,
            }
        }
        Self { path, depth: target_depth, offset }
    }

    /// NodeKind-aware reverse: reconstruct camera XYZ in the frame
    /// of an ancestor node, dispatching on each node's kind on the
    /// way down. At a body ancestor, the path's face-center slot
    /// plus accumulated `(un, vn, rn)` decode to body-local XYZ via
    /// sphere geometry. Returned XYZ is in the ancestor's `[0, 3)³`.
    pub fn pos_in_ancestor_frame_in_tree(
        &self,
        ancestor_depth: u8,
        library: &NodeLibrary,
        tree_root: NodeId,
    ) -> [f32; 3] {
        let d = self.depth as usize;
        let a = ancestor_depth as usize;
        assert!(a <= d);
        if d == a {
            return [
                self.offset[0] * 3.0,
                self.offset[1] * 3.0,
                self.offset[2] * 3.0,
            ];
        }
        // Walk down `path[0..d]` collecting kinds at every level.
        let mut kinds = [NodeKind::Cartesian; MAX_DEPTH];
        let mut current_id = tree_root;
        for k in 0..d {
            let Some(node) = library.get(current_id) else { break };
            kinds[k] = node.kind;
            match node.children[self.path[k] as usize] {
                Child::Node(child) => current_id = child,
                _ => break,
            }
        }
        // Reverse fold: at each level, decode (slot, child_offset)
        // back to a parent-cell offset. Body ancestors use sphere
        // math; everything else is base-3 inverse.
        let mut child_offset = self.offset;
        for k in (a..d).rev() {
            let slot = self.path[k];
            child_offset = match kinds[k] {
                NodeKind::Cartesian | NodeKind::CubedSphereFace { .. } => {
                    let (sx, sy, sz) = slot_coords(slot as usize);
                    [
                        (sx as f32 + child_offset[0]) / 3.0,
                        (sy as f32 + child_offset[1]) / 3.0,
                        (sz as f32 + child_offset[2]) / 3.0,
                    ]
                }
                NodeKind::CubedSphereBody { inner_r, outer_r } => {
                    unbody(slot, child_offset, inner_r, outer_r)
                }
            };
        }
        [child_offset[0] * 3.0, child_offset[1] * 3.0, child_offset[2] * 3.0]
    }

    /// Integrate a velocity step: `offset += delta`, then carry across
    /// cell boundaries via [`carry_axis`]. Clamps at the tree root if
    /// the carry bubbles past it.
    ///
    /// Large deltas (e.g. world-space step converted to offset at a
    /// deep anchoring depth — step 4's player physics can produce
    /// offsets well above 1.0) are handled in O(depth) by extracting
    /// the integer cell count once, not one cell at a time.
    pub fn add_offset(&mut self, delta: [f32; 3]) {
        for axis in 0..3 {
            self.offset[axis] += delta[axis];
            let whole = self.offset[axis].floor();
            self.offset[axis] -= whole;
            // After floor-subtract, offset ∈ [0, 1) in exact math.
            // A tiny positive remainder is still possible from float
            // roundoff; snap back inside the range.
            if self.offset[axis] >= 1.0 {
                self.offset[axis] = 1.0 - f32::EPSILON;
            } else if self.offset[axis] < 0.0 {
                self.offset[axis] = 0.0;
            }
            let cells = whole as i64;
            if cells != 0 {
                let leftover = carry_axis(self, axis, cells);
                if leftover != 0 {
                    // Carry walked off the root. Pin the offset to
                    // the boundary so world_pos stays inside [0, 3).
                    if leftover > 0 {
                        self.offset[axis] = 1.0 - f32::EPSILON;
                    } else {
                        self.offset[axis] = 0.0;
                    }
                }
            }
        }
    }
}

// -------------------------------------------------------- step_neighbor

/// Move one cell along `axis` by `dir` (`+1` or `-1`) with Cartesian
/// semantics only. Increment/decrement the slot coord, carry to
/// parent on wrap; returns `false` and leaves `pos` unchanged if the
/// step walks past the root.
///
/// Use this when the path is known to contain only Cartesian nodes —
/// the default across the engine today. Library-aware stepping (which
/// dispatches over `NodeKind::CubedSphereBody`/`CubedSphereFace` per
/// §2c of refactor-decisions.md) is [`step_neighbor_in`]; it delegates
/// to this function when the parent node is Cartesian.
pub fn step_neighbor(pos: &mut Position, axis: usize, dir: i32) -> bool {
    debug_assert!(axis < 3, "axis must be 0, 1, or 2");
    debug_assert!(dir == 1 || dir == -1, "dir must be ±1");
    carry_axis(pos, axis, dir as i64) == 0
}

/// NodeKind-aware step: walks up `pos.path` and, at each level,
/// consults `library` for the containing node's [`NodeKind`]. Cartesian
/// parents use the bulk base-3 carry ([`carry_axis`]); body / face
/// parents apply cubed-sphere-specific rewrites (see
/// [`crate::world::face_transitions`]).
///
/// Step 8 wires the dispatch; the non-Cartesian branches today are
/// scaffolded and will be fleshed out alongside step 9 when the tree
/// actually starts containing body/face nodes. Falls back to plain
/// Cartesian stepping if `library` has no entry for the containing
/// node (e.g. during bring-up).
pub fn step_neighbor_in(
    pos: &mut Position,
    library: &NodeLibrary,
    axis: usize,
    dir: i32,
) -> bool {
    debug_assert!(axis < 3);
    debug_assert!(dir == 1 || dir == -1);
    // Find the deepest parent node id by walking `path` from root.
    // For Cartesian-only scenes this is an O(depth) pass but cheap
    // because the library is a HashMap.
    let Some(parent_id) = resolve_parent_node(pos, library) else {
        // Missing parent — behave as Cartesian so tests and legacy
        // code paths keep working during bring-up.
        return step_neighbor(pos, axis, dir);
    };
    let Some(parent) = library.get(parent_id) else {
        return step_neighbor(pos, axis, dir);
    };
    match parent.kind {
        NodeKind::Cartesian => step_neighbor(pos, axis, dir),
        NodeKind::CubedSphereFace { .. } => {
            // TODO(step 9): apply u/v seam crossing via
            // face_transitions::seam_transition and the full axis
            // swap/flip rules. Radial (axis == 2) exits via
            // face_transitions::radial_exit. Scaffolded skeleton
            // below falls through to Cartesian — correct in the
            // degenerate case of stepping within the face without
            // crossing a seam.
            step_neighbor(pos, axis, dir)
        }
        NodeKind::CubedSphereBody { .. } => {
            // TODO(step 9): a camera inside a body's 27-child grid is
            // effectively one level above a face subtree; any step
            // beyond the current face slot bubbles into the body's
            // parent (Cartesian) for handling. Scaffolded skeleton
            // uses Cartesian carry, which is correct for body
            // interior-filler slots but wrong for face-adjacent
            // motion. Full rules land with step 9.
            step_neighbor(pos, axis, dir)
        }
    }
}

/// Walk the path down to depth-1 and return the node id that holds
/// `pos`'s deepest slot. `None` if the path descends through a Block
/// or Empty child (i.e. `pos` is deeper than the tree supports) or if
/// the root id isn't in the library yet.
fn resolve_parent_node(
    pos: &Position,
    library: &NodeLibrary,
) -> Option<crate::world::tree::NodeId> {
    use crate::world::tree::Child;
    if pos.depth == 0 {
        return None;
    }
    // Caller typically knows the root; we can't recover it from the
    // Position alone. Future refactor threads a render_root through
    // here; for step 8 we walk from the conventional root id 1 if
    // present, else skip. This helper only exists to keep the
    // dispatch surface honest — not to be relied on in hot paths.
    let root = {
        // Try library ids in order; pick the first one that has a
        // child matching the first slot. This is a best-effort probe
        // during bring-up.
        let first_slot = pos.path[0] as usize;
        let mut candidate: Option<crate::world::tree::NodeId> = None;
        for id in 1u64..=(library.len() as u64 + 16) {
            if let Some(n) = library.get(id) {
                if matches!(n.children[first_slot], Child::Node(_) | Child::Block(_) | Child::Empty) {
                    candidate = Some(id);
                    break;
                }
            }
        }
        candidate?
    };
    let mut current = root;
    for k in 0..(pos.depth as usize) {
        let node = library.get(current)?;
        match node.children[pos.path[k] as usize] {
            Child::Node(nid) if k + 1 < pos.depth as usize => current = nid,
            _ => {
                // If we're at the deepest slot, `current` IS the parent.
                return if k + 1 == pos.depth as usize { Some(current) } else { None };
            }
        }
    }
    Some(current)
}

/// Standard base-3 subdivision (`offset ∈ [0, 1)³` → slot + child
/// offset). Used by `from_world_pos_in_tree` for Cartesian and
/// CubedSphereFace nodes (subdivision math identical; axis meaning
/// is the kind's concern).
fn subdivide_in_place(offset: &mut [f32; 3]) -> u8 {
    let mut coords = [0usize; 3];
    for axis in 0..3 {
        let v = offset[axis] * 3.0;
        let i = (v.floor() as i32).clamp(0, 2) as usize;
        coords[axis] = i;
        offset[axis] = v - i as f32;
        if offset[axis] >= 1.0 {
            offset[axis] = 0.0;
            coords[axis] = (coords[axis] + 1).min(2);
        } else if offset[axis] < 0.0 {
            offset[axis] = 0.0;
        }
    }
    slot_index(coords[0], coords[1], coords[2]) as u8
}

/// CubedSphereBody subdivision: convert Cartesian offset in
/// `[0, 1)³` to `(face, un, vn, rn)` via sphere geometry, set
/// `offset = (un, vn, rn)`, return the body-level slot holding that
/// face subtree. Out-of-shell points fall back to the center slot
/// (interior filler) with Cartesian subdivision.
fn subdivide_body(offset: &mut [f32; 3], inner_r: f32, outer_r: f32) -> u8 {
    let local = [offset[0] - 0.5, offset[1] - 0.5, offset[2] - 0.5];
    let r2 = local[0].powi(2) + local[1].powi(2) + local[2].powi(2);
    let r = r2.sqrt();
    if r < inner_r || r >= outer_r {
        for axis in 0..3 {
            offset[axis] = (offset[axis] * 3.0 - 1.0).clamp(0.0, 1.0 - f32::EPSILON);
        }
        return slot_index(1, 1, 1) as u8;
    }
    let inv_r = 1.0 / r;
    let dir = [local[0] * inv_r, local[1] * inv_r, local[2] * inv_r];
    let ax = dir[0].abs(); let ay = dir[1].abs(); let az = dir[2].abs();
    let (face, cube_u, cube_v) = if ax >= ay && ax >= az {
        if dir[0] > 0.0 { (0u8, -dir[2] / ax, dir[1] / ax) }
        else            { (1u8,  dir[2] / ax, dir[1] / ax) }
    } else if ay >= az {
        if dir[1] > 0.0 { (2u8, dir[0] / ay, -dir[2] / ay) }
        else            { (3u8, dir[0] / ay,  dir[2] / ay) }
    } else if dir[2] > 0.0 { (4u8,  dir[0] / az, dir[1] / az) }
        else               { (5u8, -dir[0] / az, dir[1] / az) };
    let u_ea = cube_to_ea(cube_u);
    let v_ea = cube_to_ea(cube_v);
    let un = ((u_ea + 1.0) * 0.5).clamp(0.0, 1.0 - f32::EPSILON);
    let vn = ((v_ea + 1.0) * 0.5).clamp(0.0, 1.0 - f32::EPSILON);
    let rn = ((r - inner_r) / (outer_r - inner_r)).clamp(0.0, 1.0 - f32::EPSILON);
    offset[0] = un;
    offset[1] = vn;
    offset[2] = rn;
    FACE_CENTER_SLOTS[face as usize]
}

/// Reverse of [`subdivide_body`]: given a body slot + child's
/// `(un, vn, rn)` in `[0, 1)³`, return body-cell-local Cartesian XYZ
/// in `[0, 1)³`.
fn unbody(body_slot: u8, child: [f32; 3], inner_r: f32, outer_r: f32) -> [f32; 3] {
    match slot_to_face(body_slot) {
        Some(face) => {
            let u_ea = child[0] * 2.0 - 1.0;
            let v_ea = child[1] * 2.0 - 1.0;
            let cube_u = ea_to_cube(u_ea);
            let cube_v = ea_to_cube(v_ea);
            let (n_ax, u_ax, v_ax) = face_basis(face);
            let cube_pt = [
                n_ax[0] + cube_u * u_ax[0] + cube_v * v_ax[0],
                n_ax[1] + cube_u * u_ax[1] + cube_v * v_ax[1],
                n_ax[2] + cube_u * u_ax[2] + cube_v * v_ax[2],
            ];
            let mag = (cube_pt[0].powi(2) + cube_pt[1].powi(2) + cube_pt[2].powi(2)).sqrt();
            let dir = if mag > 1e-12 {
                [cube_pt[0] / mag, cube_pt[1] / mag, cube_pt[2] / mag]
            } else { n_ax };
            let r = inner_r + child[2] * (outer_r - inner_r);
            [
                (0.5 + dir[0] * r).clamp(0.0, 1.0 - f32::EPSILON),
                (0.5 + dir[1] * r).clamp(0.0, 1.0 - f32::EPSILON),
                (0.5 + dir[2] * r).clamp(0.0, 1.0 - f32::EPSILON),
            ]
        }
        None => {
            let (sx, sy, sz) = slot_coords(body_slot as usize);
            [
                (sx as f32 + child[0]) / 3.0,
                (sy as f32 + child[1]) / 3.0,
                (sz as f32 + child[2]) / 3.0,
            ]
        }
    }
}

#[inline] fn cube_to_ea(c: f32) -> f32 { c.atan() * (4.0 / std::f32::consts::PI) }
#[inline] fn ea_to_cube(e: f32) -> f32 { (e * std::f32::consts::FRAC_PI_4).tan() }

fn face_basis(face: u8) -> ([f32; 3], [f32; 3], [f32; 3]) {
    // (normal, u_axis, v_axis) — matches `cubesphere::Face::tangents`.
    match face {
        0 => ([ 1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        1 => ([-1.0, 0.0, 0.0], [0.0, 0.0,  1.0], [0.0, 1.0, 0.0]),
        2 => ([0.0,  1.0, 0.0], [1.0, 0.0,  0.0], [0.0, 0.0, -1.0]),
        3 => ([0.0, -1.0, 0.0], [1.0, 0.0,  0.0], [0.0, 0.0,  1.0]),
        4 => ([0.0, 0.0,  1.0], [1.0, 0.0,  0.0], [0.0, 1.0, 0.0]),
        _ => ([0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
    }
}

/// Shift `pos` by `amount` cells along `axis`, carrying through
/// base-3 wraps up the path. O(depth).
///
/// Returns any leftover cells that couldn't be resolved before the
/// carry ran past the root; callers clamp the offset when this
/// happens. On leftover != 0 the path is restored to its original
/// state so the caller sees an atomic "either-or" outcome.
fn carry_axis(pos: &mut Position, axis: usize, amount: i64) -> i64 {
    if amount == 0 {
        return 0;
    }
    let saved = pos.path;
    let mut carry = amount;
    let mut d = pos.depth as usize;
    while d > 0 && carry != 0 {
        let slot = pos.path[d - 1] as usize;
        let (sx, sy, sz) = slot_coords(slot);
        let mut coords = [sx as i64, sy as i64, sz as i64];
        let shifted = coords[axis] + carry;
        let new_coord = shifted.rem_euclid(3);
        let parent_carry = shifted.div_euclid(3);
        coords[axis] = new_coord;
        pos.path[d - 1] = slot_index(
            coords[0] as usize,
            coords[1] as usize,
            coords[2] as usize,
        ) as u8;
        carry = parent_carry;
        d -= 1;
    }
    if carry != 0 {
        pos.path = saved;
    }
    carry
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    fn pos_at(slots: &[u8], offset: [f32; 3]) -> Position {
        Position::at_slot_path(slots, offset)
    }

    #[test]
    fn root_defaults() {
        let p = Position::root();
        assert_eq!(p.depth, 0);
        assert_eq!(p.offset, [0.0; 3]);
        assert_eq!(p.slots(), &[] as &[u8]);
    }

    #[test]
    fn at_slot_path_round_trip() {
        let slots = [slot_index(1, 0, 2) as u8, slot_index(2, 2, 1) as u8];
        let p = pos_at(&slots, [0.25, 0.5, 0.75]);
        assert_eq!(p.slots(), &slots);
        assert_eq!(p.depth, 2);
        assert_eq!(p.offset, [0.25, 0.5, 0.75]);
    }

    #[test]
    #[should_panic]
    fn at_slot_path_rejects_bad_slot() {
        Position::at_slot_path(&[27], [0.0; 3]);
    }

    #[test]
    #[should_panic]
    fn at_slot_path_rejects_offset_at_one() {
        Position::at_slot_path(&[0], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn same_cell_ignores_offset() {
        let a = pos_at(&[1, 2], [0.1, 0.2, 0.3]);
        let b = pos_at(&[1, 2], [0.9, 0.8, 0.7]);
        assert!(a.same_cell(&b));
        assert_ne!(a, b);
    }

    #[test]
    fn same_cell_respects_path_and_depth() {
        let a = pos_at(&[1, 2], [0.0; 3]);
        let b = pos_at(&[1, 3], [0.0; 3]);
        let c = pos_at(&[1], [0.0; 3]);
        assert!(!a.same_cell(&b));
        assert!(!a.same_cell(&c));
    }

    #[test]
    fn zoom_in_then_out_is_identity() {
        let original = pos_at(&[5, 10], [0.2, 0.6, 0.4]);
        let mut p = original;
        assert!(p.zoom_in());
        assert_eq!(p.depth, 3);
        assert!(p.zoom_out());
        assert_eq!(p.depth, 2);
        // Path prefix matches; offset is close (modulo float roundoff).
        assert_eq!(p.slots(), original.slots());
        for axis in 0..3 {
            assert!(
                (p.offset[axis] - original.offset[axis]).abs() < 1e-5,
                "axis {} drifted: {} vs {}",
                axis,
                p.offset[axis],
                original.offset[axis]
            );
        }
    }

    #[test]
    fn zoom_in_derives_correct_slot() {
        // offset (0.7, 0.2, 0.5) * 3 = (2.1, 0.6, 1.5).
        // slot coords (2, 0, 1); new offset (0.1, 0.6, 0.5).
        let mut p = pos_at(&[], [0.7, 0.2, 0.5]);
        assert!(p.zoom_in());
        assert_eq!(p.slots(), &[slot_index(2, 0, 1) as u8]);
        assert!((p.offset[0] - 0.1).abs() < 1e-5);
        assert!((p.offset[1] - 0.6).abs() < 1e-5);
        assert!((p.offset[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn zoom_out_at_root_clamps() {
        let mut p = Position::root();
        assert!(!p.zoom_out());
        assert_eq!(p, Position::root());
    }

    #[test]
    fn step_neighbor_within_parent() {
        // Start at slot (1, 1, 1) inside one parent; step +x → (2, 1, 1).
        let mut p = pos_at(&[slot_index(1, 1, 1) as u8], [0.5; 3]);
        assert!(step_neighbor(&mut p, 0, 1));
        assert_eq!(p.slots(), &[slot_index(2, 1, 1) as u8]);
    }

    #[test]
    fn step_neighbor_carries_to_parent() {
        // Slot (2, 0, 0) inside slot (0, 0, 0); step +x wraps the
        // deeper slot to (0, 0, 0) and carries +x at the parent.
        let mut p = pos_at(
            &[slot_index(0, 0, 0) as u8, slot_index(2, 0, 0) as u8],
            [0.5; 3],
        );
        assert!(step_neighbor(&mut p, 0, 1));
        assert_eq!(
            p.slots(),
            &[slot_index(1, 0, 0) as u8, slot_index(0, 0, 0) as u8]
        );
    }

    #[test]
    fn step_neighbor_multi_level_carry() {
        // Root slot is middle-x; two descendants sit at the +x edge.
        // Step +x at the deepest level carries through both edges and
        // resolves at the root by incrementing its x from 1 to 2.
        let edge = slot_index(2, 0, 0) as u8;
        let mut p = pos_at(&[slot_index(1, 0, 0) as u8, edge, edge], [0.5; 3]);
        assert!(step_neighbor(&mut p, 0, 1));
        assert_eq!(
            p.slots(),
            &[
                slot_index(2, 0, 0) as u8,
                slot_index(0, 0, 0) as u8,
                slot_index(0, 0, 0) as u8,
            ]
        );
    }

    #[test]
    fn step_neighbor_past_root_fails_and_restores() {
        // Single-level position at the -x edge; step -x should hit
        // root and return false, leaving path unchanged.
        let before = pos_at(&[slot_index(0, 1, 1) as u8], [0.5; 3]);
        let mut p = before;
        assert!(!step_neighbor(&mut p, 0, -1));
        assert_eq!(p.path, before.path);
    }

    #[test]
    fn add_offset_no_carry() {
        let mut p = pos_at(&[slot_index(1, 1, 1) as u8], [0.1, 0.2, 0.3]);
        p.add_offset([0.1, 0.1, 0.1]);
        assert_eq!(p.slots(), &[slot_index(1, 1, 1) as u8]);
        for (i, expected) in [0.2, 0.3, 0.4].iter().enumerate() {
            assert!((p.offset[i] - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn add_offset_crosses_cell() {
        // Offset near +x edge; +0.3 overflows into next cell.
        let mut p = pos_at(&[slot_index(1, 1, 1) as u8], [0.8, 0.0, 0.0]);
        p.add_offset([0.3, 0.0, 0.0]);
        assert_eq!(p.slots(), &[slot_index(2, 1, 1) as u8]);
        assert!((p.offset[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn add_offset_crosses_multiple_cells() {
        // +2.5 on x crosses two cell boundaries: (0, *, *) → (1, *, *)
        // after +1.0, → (2, *, *) after another +1.0, lands at
        // offset 0.5 in the third cell.
        let mut p = pos_at(&[slot_index(0, 1, 1) as u8], [0.0, 0.0, 0.0]);
        p.add_offset([2.5, 0.0, 0.0]);
        assert_eq!(p.slots(), &[slot_index(2, 1, 1) as u8]);
        assert!((p.offset[0] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn add_offset_large_delta_preserves_world_pos() {
        // Huge offset delta — simulates step 4's world-to-offset
        // conversion at a deep anchoring depth. Round-tripping
        // through world_pos must still land at the same XYZ point
        // (within float precision).
        let start = Position::from_world_pos([1.5, 1.5, 1.5], 15);
        let mut p = start;
        let delta_world = [0.1, -0.05, 0.02];
        let depth = p.depth as i32;
        let inv_cell = 3.0f32.powi(depth - 1);
        p.add_offset([
            delta_world[0] * inv_cell,
            delta_world[1] * inv_cell,
            delta_world[2] * inv_cell,
        ]);
        let expected = [
            1.5 + delta_world[0],
            1.5 + delta_world[1],
            1.5 + delta_world[2],
        ];
        let got = p.world_pos();
        for axis in 0..3 {
            assert!(
                (got[axis] - expected[axis]).abs() < 1e-3,
                "axis {}: got {} expected {}",
                axis,
                got[axis],
                expected[axis]
            );
        }
    }

    #[test]
    fn add_offset_clamps_at_root_positive() {
        // At root-level +x edge; push past root in +x should clamp.
        let mut p = pos_at(&[slot_index(2, 1, 1) as u8], [0.9, 0.5, 0.5]);
        p.add_offset([0.5, 0.0, 0.0]);
        // Clamped: still at (2, 1, 1), offset[0] pinned just below 1.
        assert_eq!(p.slots(), &[slot_index(2, 1, 1) as u8]);
        assert!(p.offset[0] < 1.0);
        assert!(p.offset[0] > 0.99);
    }

    #[test]
    fn world_pos_depth_0_is_offset_times_three() {
        let p = Position { path: [0; MAX_DEPTH], depth: 0, offset: [0.25, 0.5, 0.75] };
        assert_eq!(p.world_pos(), [0.75, 1.5, 2.25]);
    }

    #[test]
    fn world_pos_depth_1_matches_cell_origin_plus_offset() {
        // Cell (2, 1, 0) at root has LL = (2, 1, 0); cell size 1.
        let p = pos_at(&[slot_index(2, 1, 0) as u8], [0.5; 3]);
        let w = p.world_pos();
        assert!((w[0] - 2.5).abs() < 1e-5);
        assert!((w[1] - 1.5).abs() < 1e-5);
        assert!((w[2] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn world_pos_depth_2_matches_nested_cell() {
        let p = pos_at(
            &[slot_index(1, 0, 0) as u8, slot_index(2, 0, 0) as u8],
            [0.5, 0.0, 0.0],
        );
        // Expected: 1 + 2/3 + 0.5 * 1/3 = 1.8333
        let w = p.world_pos();
        assert!((w[0] - (1.0 + 2.0 / 3.0 + 0.5 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn pos_in_ancestor_frame_matches_world_pos_at_depth_zero() {
        let p = pos_at(&[5, 10, 15], [0.3, 0.4, 0.5]);
        assert_eq!(p.pos_in_ancestor_frame(0), p.world_pos());
    }

    #[test]
    fn pos_in_ancestor_frame_rescales_for_deeper_ancestor() {
        // If the ancestor is the camera's parent (depth d - 1),
        // the point's position in the ancestor's [0, 3)³ frame is
        // slot_coord(last) + offset (each axis).
        let slot = slot_index(2, 0, 1) as u8;
        let p = pos_at(&[slot], [0.25, 0.5, 0.75]);
        let frame_pos = p.pos_in_ancestor_frame(0);
        // Ancestor at depth 1 (i.e. the node itself): offset in its
        // own [0, 3)³ frame is offset * 3 (since the node's children
        // each cover 1.0 — the node as a whole covers 3.0).
        let self_frame = p.pos_in_ancestor_frame(1);
        // offset * 3 = [0.75, 1.5, 2.25]
        for axis in 0..3 {
            assert!((self_frame[axis] - p.offset[axis] * 3.0).abs() < 1e-5);
        }
        // And the root-frame position is at slot_origin + child_frame/3.
        let (sx, sy, sz) = slot_coords(slot as usize);
        let expected = [
            sx as f32 + self_frame[0] / 3.0,
            sy as f32 + self_frame[1] / 3.0,
            sz as f32 + self_frame[2] / 3.0,
        ];
        for axis in 0..3 {
            assert!((frame_pos[axis] - expected[axis]).abs() < 1e-5);
        }
    }

    #[test]
    fn world_pos_round_trip_at_various_depths() {
        for depth in [1u8, 5, 10, 15, 20, 30, 40, 50, 60] {
            let original = [1.234, 0.567, 2.891];
            let p = Position::from_world_pos(original, depth);
            let back = p.world_pos();
            // Tolerance ≈ cell extent at the anchoring depth — any
            // round-trip through `from_world_pos` loses up to one
            // cell's precision when discretizing XYZ into slot
            // choices, but the reverse Horner in pos_in_ancestor_frame
            // shouldn't add any further drift.
            let tol = (3.0f32.powi(1 - depth as i32)).max(1e-6);
            for axis in 0..3 {
                assert!(
                    (back[axis] - original[axis]).abs() <= tol,
                    "depth {} axis {} drifted: {} vs {} (tol {})",
                    depth,
                    axis,
                    back[axis],
                    original[axis],
                    tol
                );
            }
        }
    }

    #[test]
    fn add_offset_clamps_at_root_negative() {
        let mut p = pos_at(&[slot_index(0, 1, 1) as u8], [0.1, 0.5, 0.5]);
        p.add_offset([-0.5, 0.0, 0.0]);
        assert_eq!(p.slots(), &[slot_index(0, 1, 1) as u8]);
        assert_eq!(p.offset[0], 0.0);
    }
}
