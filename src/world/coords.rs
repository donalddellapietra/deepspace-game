//! Anchor-based world coordinates.
//!
//! Every world position is `(anchor, offset)`:
//!   - `anchor: Path` — a sequence of slot indices descending the
//!     27-ary tree. Exact, symbolic, no f32 loss at any depth.
//!   - `offset: [f32; 3]` — a small local coordinate inside the
//!     anchor cell's `[0, 1)³` frame.
//!
//! Invariant: `offset[i] ∈ [0, 1)` for each axis. Primitives preserve
//! this across moves and zooms. f32 never accumulates across cells;
//! rendering always happens in a frame small enough for f32.
//!
//! For the full design see
//! `docs/experimental-architecture/anchor-refactor-decisions.md`.

use super::cubesphere::{body_face_center_slot, Face};
use super::face_transitions;
use super::tree::{slot_coords, slot_index, Child, MAX_DEPTH, NodeKind, NodeLibrary, NodeId};

// ---------------------------------------------------------------- Path

/// A sequence of slot indices (0..27) descending from the tree root.
///
/// `depth` is the number of live entries in `slots`. `depth == 0` is
/// the root cell (no descent). Each `push(slot)` descends one level.
///
/// Equality is a depth compare plus a memcmp over `slots[..depth]`, so
/// path compares are fast regardless of capacity.
#[derive(Clone, Copy)]
pub struct Path {
    slots: [u8; MAX_DEPTH],
    depth: u8,
}

impl Path {
    pub const fn root() -> Self {
        Self { slots: [0; MAX_DEPTH], depth: 0 }
    }

    #[inline]
    pub fn depth(&self) -> u8 { self.depth }

    #[inline]
    pub fn is_root(&self) -> bool { self.depth == 0 }

    #[inline]
    pub fn slots(&self) -> &[u8] {
        &self.slots[..self.depth as usize]
    }

    /// Descend one level. Returns false if already at `MAX_DEPTH`.
    pub fn push(&mut self, slot: u8) -> bool {
        debug_assert!((slot as usize) < 27, "slot {} out of range", slot);
        if (self.depth as usize) >= MAX_DEPTH {
            return false;
        }
        self.slots[self.depth as usize] = slot;
        self.depth += 1;
        true
    }

    /// Ascend one level. Returns the popped slot, or `None` at root.
    pub fn pop(&mut self) -> Option<u8> {
        if self.depth == 0 { return None; }
        self.depth -= 1;
        Some(self.slots[self.depth as usize])
    }

    /// The slot at the deepest level, or `None` at root.
    #[inline]
    pub fn last_slot(&self) -> Option<u8> {
        if self.depth == 0 { None } else { Some(self.slots[(self.depth - 1) as usize]) }
    }

    /// Overwrite the slot at `index` (must be `< depth()`).
    pub fn set_slot(&mut self, index: usize, slot: u8) {
        debug_assert!(index < self.depth as usize, "set_slot index out of range");
        debug_assert!((slot as usize) < 27, "slot {} out of range", slot);
        self.slots[index] = slot;
    }

    /// Truncate to at most `new_depth` levels (no-op if shorter).
    pub fn truncate(&mut self, new_depth: u8) {
        if new_depth < self.depth {
            self.depth = new_depth;
        }
    }

    /// Return a copy truncated to `new_depth`.
    pub fn ancestor(&self, new_depth: u8) -> Path {
        let mut p = *self;
        p.truncate(new_depth);
        p
    }

    /// Length of the common prefix between `self` and `other`.
    pub fn common_prefix_len(&self, other: &Path) -> u8 {
        let n = self.depth.min(other.depth) as usize;
        for i in 0..n {
            if self.slots[i] != other.slots[i] { return i as u8; }
        }
        n as u8
    }

    /// Move one cell along `axis` (0..3) in `dir` (-1 or +1) at the
    /// current depth. On overflow at this level, bubbles up to the
    /// parent and retries, then pushes the mirror slot on the far
    /// side. Returns `false` if clamped at the root (no further
    /// parent exists to bubble into).
    ///
    /// This is the pure Cartesian neighbor step. Sphere-aware stepping
    /// (face seams, radial r axis) will be layered on top of this by
    /// `WorldPos::add_local` once `NodeKind` is introduced.
    pub fn step_neighbor_cartesian(&mut self, axis: u8, dir: i8) -> bool {
        debug_assert!(axis < 3, "axis must be 0..3");
        debug_assert!(dir == 1 || dir == -1, "dir must be ±1");
        if self.depth == 0 { return false; }
        let slot_pos = (self.depth - 1) as usize;
        let s = self.slots[slot_pos] as usize;
        let (cx, cy, cz) = slot_coords(s);
        let mut coords = [cx, cy, cz];
        let c = coords[axis as usize] as i32 + dir as i32;
        if (0..3).contains(&c) {
            coords[axis as usize] = c as usize;
            self.slots[slot_pos] = slot_index(coords[0], coords[1], coords[2]) as u8;
            true
        } else {
            // Overflow; bubble up to parent.
            let saved = coords;
            self.depth -= 1;
            let ok = self.step_neighbor_cartesian(axis, dir);
            if !ok {
                self.depth += 1;
                return false;
            }
            // Re-enter on the far side along `axis`.
            let mut nc = saved;
            nc[axis as usize] = if c < 0 { 2 } else { 0 };
            self.slots[self.depth as usize] =
                slot_index(nc[0], nc[1], nc[2]) as u8;
            self.depth += 1;
            true
        }
    }
}

impl Default for Path {
    fn default() -> Self { Self::root() }
}

impl PartialEq for Path {
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth
            && self.slots[..self.depth as usize] == other.slots[..other.depth as usize]
    }
}

impl Eq for Path {}

impl std::hash::Hash for Path {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.depth.hash(state);
        self.slots[..self.depth as usize].hash(state);
    }
}

impl std::fmt::Debug for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Path[depth={}, slots={:?}]", self.depth, self.slots())
    }
}

// -------------------------------------------- world-frame conversions
//
// The shader, GPU packer, and editing code operate in an absolute
// world frame spanning `[0, ROOT_EXTENT)³`. These helpers project
// `WorldPos` into that frame and back so those systems can continue
// to consume plain `[f32; 3]`s. The `WorldPos` remains the
// authoritative identity of a point; the f32 form is a derived
// view used at system boundaries.

/// World extent spanned by the root cell in legacy coordinates.
/// Matches the shader and `gpu.rs` where "root node cells are 1.0
/// wide, node spans [0, 3)".
pub const ROOT_EXTENT: f32 = 3.0;

/// Reconstruct a legacy world `[f32; 3]` from a `WorldPos`.
///
/// Exact if `anchor.depth() == 0` (trivial), lossy only by the usual
/// f32 rounding at deeper anchors. The mapping is
/// `world_i = ROOT_EXTENT · (Σ slot_coord_at_k · 3^-(k+1) + offset · 3^-depth)`.
pub fn world_pos_to_f32(pos: &WorldPos) -> [f32; 3] {
    let slots = pos.anchor.slots();
    let mut out = [0.0f32; 3];
    for axis in 0..3usize {
        let mut frac = 0.0f32;
        let mut scale = 1.0f32 / 3.0;
        for &slot in slots {
            let (sx, sy, sz) = slot_coords(slot as usize);
            let s = [sx, sy, sz][axis] as f32;
            frac += s * scale;
            scale /= 3.0;
        }
        frac += pos.offset[axis] * scale * 3.0; // last `scale` was for the next (unused) level
        out[axis] = frac * ROOT_EXTENT;
    }
    out
}

/// Build a `WorldPos` at the given depth from a legacy world `[f32; 3]`.
///
/// Values outside `[0, ROOT_EXTENT)` are clamped. At `depth == 0` the
/// result is `(root, world/ROOT_EXTENT)`, exact within f32.
pub fn world_pos_from_f32(world: [f32; 3], depth: u8) -> WorldPos {
    let mut frac = [
        (world[0] / ROOT_EXTENT).clamp(0.0, below_one()),
        (world[1] / ROOT_EXTENT).clamp(0.0, below_one()),
        (world[2] / ROOT_EXTENT).clamp(0.0, below_one()),
    ];
    let mut anchor = Path::root();
    for _ in 0..depth {
        let sx = pick_slot(frac[0]);
        let sy = pick_slot(frac[1]);
        let sz = pick_slot(frac[2]);
        anchor.push(slot_index(sx, sy, sz) as u8);
        frac[0] = rescale_up_remainder(frac[0], sx);
        frac[1] = rescale_up_remainder(frac[1], sy);
        frac[2] = rescale_up_remainder(frac[2], sz);
    }
    WorldPos { anchor, offset: frac }
}

#[inline]
fn rescale_up_remainder(v: f32, slot: usize) -> f32 {
    let r = v * 3.0 - slot as f32;
    r.clamp(0.0, below_one())
}

// ----------------------------------------- NodeKind resolution for Path

/// Walk a path from `root` and return the `NodeKind` of each node
/// visited (including the root), up to (but not past) the path's
/// depth. The returned vec has `path.depth() + 1` entries when the
/// walk succeeds; fewer if the path overshoots the instantiated
/// portion of the tree.
pub fn resolve_kinds_along(
    lib: &NodeLibrary,
    root: NodeId,
    path: &Path,
) -> Vec<NodeKind> {
    let mut out = Vec::with_capacity(path.depth() as usize + 1);
    let Some(root_node) = lib.get(root) else { return out; };
    out.push(root_node.kind);
    let mut id = root;
    for &slot in path.slots() {
        let Some(node) = lib.get(id) else { break; };
        match node.children[slot as usize] {
            Child::Node(child_id) => {
                let Some(child) = lib.get(child_id) else { break; };
                out.push(child.kind);
                id = child_id;
            }
            _ => break,
        }
    }
    out
}

/// The kind of the deepest instantiated node reachable from `root`
/// along `path`. Returns `Cartesian` as a default when the walk
/// terminates at a `Block` or `Empty` child (those have no kind).
pub fn deepest_kind(lib: &NodeLibrary, root: NodeId, path: &Path) -> NodeKind {
    resolve_kinds_along(lib, root, path)
        .last()
        .copied()
        .unwrap_or(NodeKind::Cartesian)
}

// ------------------------------------------------------------ Transition

/// Semantic event emitted when the anchor crosses a coordinate-meaning
/// boundary. Coordinate math itself is handled inside the primitives;
/// these events let game-level code react (camera up-vector rotation,
/// orientation re-expression, UI hints).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    None,
    SphereEntry { body_path: Path },
    SphereExit  { body_path: Path },
    FaceEntry   { face: Face },
    FaceExit    { face: Face },
    CubeSeam    { from_face: Face, to_face: Face },
}

// ------------------------------------------------------------- WorldPos

/// A position anywhere in the world: an anchor path plus a local
/// offset inside the anchor cell's `[0, 1)³` frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorldPos {
    pub anchor: Path,
    pub offset: [f32; 3],
}

impl WorldPos {
    pub const fn root() -> Self {
        Self { anchor: Path::root(), offset: [0.0, 0.0, 0.0] }
    }

    pub fn new(anchor: Path, offset: [f32; 3]) -> Self {
        let mut p = Self { anchor, offset };
        p.normalize();
        p
    }

    /// Force `offset` into `[0, 1)` by clamping (no step). Callers
    /// should prefer `add_local` which steps across cell boundaries;
    /// this is a safety net after direct field assignment.
    fn normalize(&mut self) {
        for i in 0..3 {
            if !self.offset[i].is_finite() { self.offset[i] = 0.0; }
            if self.offset[i] < 0.0 { self.offset[i] = 0.0; }
            if self.offset[i] >= 1.0 { self.offset[i] = below_one(); }
        }
    }

    /// Add `delta` to `offset` and cross cell boundaries as needed.
    /// Dispatches on the anchor's deepest `NodeKind` via `lib`/`root`
    /// and consults the cube-seam table when a step would exit a
    /// face subtree's lateral bounds. Returns a `Transition`.
    ///
    /// Cartesian and CubedSphereBody overflow → vanilla Cartesian
    /// neighbor step. CubedSphereFace overflow on the `u`/`v` axes
    /// → `face_transitions::seam_neighbor` rewrites the path to
    /// land on the adjacent face with the right axis remap.
    pub fn add_local(
        &mut self,
        delta: [f32; 3],
        lib: &NodeLibrary,
        root: NodeId,
    ) -> Transition {
        let kind_before = deepest_kind(lib, root, &self.anchor);
        let depth_before = self.anchor.depth();
        for i in 0..3 {
            self.offset[i] += delta[i];
        }
        for axis in 0..3usize {
            while self.offset[axis] >= 1.0 {
                if !self.step_seam_aware(axis as u8, 1, lib, root) {
                    self.offset[axis] = below_one();
                    break;
                }
                self.offset[axis] -= 1.0;
            }
            while self.offset[axis] < 0.0 {
                if !self.step_seam_aware(axis as u8, -1, lib, root) {
                    self.offset[axis] = 0.0;
                    break;
                }
                self.offset[axis] += 1.0;
            }
        }
        if self.anchor.depth() == depth_before && matches!(kind_before, NodeKind::Cartesian | NodeKind::CubedSphereBody { .. }) {
            return Transition::None;
        }
        classify_transition(lib, root, kind_before, &self.anchor)
    }

    /// One-cell step that respects sphere-face seams. If the current
    /// anchor sits inside a `CubedSphereFace` subtree and the step
    /// would push the face's `(u, v)` index outside `[0, 3^N)`,
    /// applies `face_transitions::seam_neighbor` instead of letting
    /// the path bubble up into the body's empty corner cells.
    /// Otherwise falls through to the pure Cartesian step.
    fn step_seam_aware(
        &mut self,
        axis: u8,
        dir: i8,
        lib: &NodeLibrary,
        root: NodeId,
    ) -> bool {
        if axis < 2 {
            if let Some(fr) = find_face_root_in_path(lib, root, &self.anchor) {
                if let Some(stepped) = self.try_face_step(axis, dir, &fr) {
                    return stepped;
                }
            }
        }
        self.anchor.step_neighbor_cartesian(axis, dir)
    }

    /// Step within the face subtree rooted at `fr.face_root_depth`.
    /// Aggregates `(cu, cv, cr)` from the slots beneath the face
    /// root, applies `delta` along `axis`, and either rewrites the
    /// slots in place (within face) or rewrites them with a seam
    /// crossing (to the neighbor face's subtree). Returns `Some(true)`
    /// when the step succeeded, `Some(false)` when no seam exists in
    /// the requested direction (clamp), or `None` when the current
    /// path doesn't actually cross a face boundary (caller falls
    /// back to Cartesian step).
    fn try_face_step(&mut self, axis: u8, dir: i8, fr: &FaceRoot) -> Option<bool> {
        let n = (self.anchor.depth() as usize).saturating_sub(fr.face_root_depth as usize);
        if n == 0 { return None; } // anchor IS the face root; one cell IS the entire face
        let cells = 3u32.pow(n as u32);
        let (cu, cv, cr) = aggregate_face_coords(&self.anchor, fr.face_root_depth, n);
        let (mut nu, mut nv, nr) = (cu as i64, cv as i64, cr);
        match axis {
            0 => nu += dir as i64,
            1 => nv += dir as i64,
            _ => return None,
        }
        if nu >= 0 && nu < cells as i64 && nv >= 0 && nv < cells as i64 {
            // Stays inside the same face — let the normal Cartesian
            // step handle it (cheaper, no seam math needed).
            return None;
        }
        // Out of face bounds → seam crossing.
        let crossing = match face_transitions::seam_neighbor(fr.face, axis, dir) {
            Some(c) => c,
            None => return Some(false),
        };
        // Apply the AxisRemap at the COARSEST face cell — the cube
        // edge — and let the finer cells fall through linearly. For
        // the simplest correct behaviour with one-cell steps, take
        // the source `(cu, cv)` mod 3 for the coarsest digit, apply
        // remap to that, and place the new cell at the corresponding
        // edge of the destination face.
        let entry_u: u32;
        let entry_v: u32;
        match (axis, dir) {
            (0,  1) => { entry_u = 0;          entry_v = cv; } // crossed +u: enter at u'=0
            (0, -1) => { entry_u = cells - 1;  entry_v = cv; } // crossed -u: enter at u'=last
            (1,  1) => { entry_u = cu;         entry_v = 0;  }
            (1, -1) => { entry_u = cu;         entry_v = cells - 1; }
            _ => return Some(false),
        }
        // Map the source cell (cu, cv) at the COARSEST level (digit n-1)
        // to find which edge cell on the new face we enter at.
        // The seam table is per-edge; the perpendicular coordinate
        // (entry_u or entry_v above) is then remapped via the table's
        // axis-swap/flip rules.
        let perp = if axis == 0 { entry_v } else { entry_u };
        let remapped_perp = remap_face_coord(perp, cells, &crossing, axis, dir);
        let (new_cu, new_cv) = match axis {
            0 => (entry_u, remapped_perp),
            1 => (remapped_perp, entry_v),
            _ => unreachable!(),
        };
        // Rewrite slots: body face-center slot to the new face, then
        // base-3 decompose (new_cu, new_cv, nr) into the face-internal
        // slots.
        let body_slot_pos = (fr.face_root_depth - 1) as usize;
        self.anchor.set_slot(body_slot_pos, body_face_center_slot(crossing.to_face) as u8);
        write_face_coords(&mut self.anchor, fr.face_root_depth, n, new_cu, new_cv, nr);
        Some(true)
    }

    /// Backwards-compatible shim: equivalent to calling `add_local`
    /// with `root = 0`, which skips kind dispatch. Only use this from
    /// tests where the library root isn't meaningful.
    #[cfg(test)]
    pub fn add_local_cartesian(
        &mut self,
        delta: [f32; 3],
        _lib: &NodeLibrary,
    ) -> Transition {
        for i in 0..3 {
            self.offset[i] += delta[i];
        }
        for axis in 0..3usize {
            while self.offset[axis] >= 1.0 {
                if !self.anchor.step_neighbor_cartesian(axis as u8, 1) {
                    self.offset[axis] = below_one();
                    break;
                }
                self.offset[axis] -= 1.0;
            }
            while self.offset[axis] < 0.0 {
                if !self.anchor.step_neighbor_cartesian(axis as u8, -1) {
                    self.offset[axis] = 0.0;
                    break;
                }
                self.offset[axis] += 1.0;
            }
        }
        Transition::None
    }

    /// Zoom in one level, resolving the kind of the child we're
    /// descending into so transitions can fire. When kind info isn't
    /// available (no library root handy — e.g., tests), use the pure
    /// `zoom_in` below.
    pub fn zoom_in_in(
        &mut self,
        lib: &NodeLibrary,
        root: NodeId,
    ) -> Transition {
        let kind_before = deepest_kind(lib, root, &self.anchor);
        let t = self.zoom_in();
        if !matches!(t, Transition::None) { return t; }
        let kind_after = deepest_kind(lib, root, &self.anchor);
        classify_zoom_transition(kind_before, kind_after)
    }

    /// Zoom out one level, resolving kinds for transition dispatch.
    pub fn zoom_out_in(
        &mut self,
        lib: &NodeLibrary,
        root: NodeId,
    ) -> Transition {
        let kind_before = deepest_kind(lib, root, &self.anchor);
        let t = self.zoom_out();
        if !matches!(t, Transition::None) { return t; }
        let kind_after = deepest_kind(lib, root, &self.anchor);
        classify_zoom_transition(kind_before, kind_after)
    }

    /// Descend the anchor into the child slot currently containing the
    /// offset. Offset is rescaled so the world point is unchanged; the
    /// anchor just expresses it in a finer cell. No-op at `MAX_DEPTH`.
    pub fn zoom_in(&mut self) -> Transition {
        if (self.anchor.depth() as usize) >= MAX_DEPTH {
            return Transition::None;
        }
        let sx = pick_slot(self.offset[0]);
        let sy = pick_slot(self.offset[1]);
        let sz = pick_slot(self.offset[2]);
        self.offset[0] = rescale_down(self.offset[0], sx);
        self.offset[1] = rescale_down(self.offset[1], sy);
        self.offset[2] = rescale_down(self.offset[2], sz);
        self.anchor.push(slot_index(sx, sy, sz) as u8);
        Transition::None
    }

    /// Ascend the anchor to its parent. Offset is rescaled to remain
    /// the same world point. No-op at root.
    pub fn zoom_out(&mut self) -> Transition {
        let Some(popped) = self.anchor.pop() else {
            return Transition::None;
        };
        let (sx, sy, sz) = slot_coords(popped as usize);
        self.offset[0] = (self.offset[0] + sx as f32) / 3.0;
        self.offset[1] = (self.offset[1] + sy as f32) / 3.0;
        self.offset[2] = (self.offset[2] + sz as f32) / 3.0;
        Transition::None
    }

    /// Repeatedly `zoom_out` until `anchor.depth() <= target_depth`.
    pub fn zoom_out_to(&mut self, target_depth: u8) {
        while self.anchor.depth() > target_depth {
            self.zoom_out();
        }
    }
}

// ----------------------------------------- face-aware step helpers

/// Captures where in the path the deepest `CubedSphereFace` root sits,
/// plus the face it represents.
struct FaceRoot {
    face: Face,
    /// Depth at which the face root node sits. The slot at
    /// `face_root_depth - 1` in the path is the body's face-center
    /// slot for `face`.
    face_root_depth: u8,
}

fn find_face_root_in_path(lib: &NodeLibrary, root: NodeId, path: &Path) -> Option<FaceRoot> {
    let kinds = resolve_kinds_along(lib, root, path);
    for (i, k) in kinds.iter().enumerate().rev() {
        if let NodeKind::CubedSphereFace { face } = k {
            return Some(FaceRoot {
                face: face_from_index(*face),
                face_root_depth: i as u8,
            });
        }
    }
    None
}

fn aggregate_face_coords(path: &Path, face_root_depth: u8, n: usize) -> (u32, u32, u32) {
    let mut cu = 0u32;
    let mut cv = 0u32;
    let mut cr = 0u32;
    let slots = path.slots();
    for i in 0..n {
        let s = slots[face_root_depth as usize + i] as usize;
        let (u, v, r) = slot_coords(s);
        cu = cu * 3 + u as u32;
        cv = cv * 3 + v as u32;
        cr = cr * 3 + r as u32;
    }
    (cu, cv, cr)
}

fn write_face_coords(path: &mut Path, face_root_depth: u8, n: usize, cu: u32, cv: u32, cr: u32) {
    for i in 0..n {
        let level = n - 1 - i;
        let div = 3u32.pow(level as u32);
        let u = ((cu / div) % 3) as usize;
        let v = ((cv / div) % 3) as usize;
        let r = ((cr / div) % 3) as usize;
        path.set_slot(face_root_depth as usize + i, slot_index(u, v, r) as u8);
    }
}

/// Apply a seam's `AxisRemap` to the perpendicular face coordinate
/// at `cells = 3^n` resolution. The seam table operates on slot
/// indices `0..3`; finer-grained coordinates use the same forward /
/// reverse semantics extended to `[0, cells)`.
fn remap_face_coord(
    coord: u32,
    cells: u32,
    crossing: &face_transitions::SeamCrossing,
    axis: u8,
    _dir: i8,
) -> u32 {
    use face_transitions::AxisRemap;
    let dst_axis_remap = if axis == 0 { crossing.new_v } else { crossing.new_u };
    match dst_axis_remap {
        AxisRemap::UForward | AxisRemap::VForward => coord,
        AxisRemap::UReverse | AxisRemap::VReverse => cells - 1 - coord,
        AxisRemap::Edge0 => 0,
        AxisRemap::Edge2 => cells - 1,
    }
}

// ----------------------------------------------- transition classifiers

/// Classify a `zoom_in`/`zoom_out` that crossed a kind boundary.
/// Same kind before and after = no transition. Sphere body ↔ face
/// transitions fire when the body's face-center slot is entered or
/// left. Face ↔ face transitions (cube seams) do not arise from
/// zoom alone — they need a lateral step.
fn classify_zoom_transition(
    before: NodeKind,
    after: NodeKind,
) -> Transition {
    match (before, after) {
        (a, b) if a == b => Transition::None,
        (NodeKind::Cartesian, NodeKind::CubedSphereBody { .. })
            => Transition::SphereEntry { body_path: Path::root() },
        (NodeKind::CubedSphereBody { .. }, NodeKind::Cartesian)
            => Transition::SphereExit { body_path: Path::root() },
        (NodeKind::CubedSphereBody { .. }, NodeKind::CubedSphereFace { face })
            => Transition::FaceEntry { face: face_from_index(face) },
        (NodeKind::CubedSphereFace { face }, NodeKind::CubedSphereBody { .. })
            => Transition::FaceExit { face: face_from_index(face) },
        _ => Transition::None,
    }
}

/// Classify a post-`add_local` state where the anchor either changed
/// depth or crossed a kind boundary. The pre-move kind is known; the
/// post-move kind is re-resolved. Currently emits sphere-entry/exit
/// and stubs a `CubeSeam` placeholder when moving laterally across
/// the face roots; the 24-case face adjacency table is scaffolded
/// (see `face_transitions::seam_neighbor`) but not wired into
/// `step_neighbor` until the body-as-tree-node work in step 9.
fn classify_transition(
    lib: &NodeLibrary,
    root: NodeId,
    kind_before: NodeKind,
    anchor_after: &Path,
) -> Transition {
    let kind_after = deepest_kind(lib, root, anchor_after);
    match (kind_before, kind_after) {
        (a, b) if a == b => Transition::None,
        (NodeKind::Cartesian, NodeKind::CubedSphereBody { .. })
            => Transition::SphereEntry { body_path: *anchor_after },
        (NodeKind::CubedSphereBody { .. }, NodeKind::Cartesian)
            => Transition::SphereExit { body_path: *anchor_after },
        (NodeKind::CubedSphereBody { .. }, NodeKind::CubedSphereFace { face })
            => Transition::FaceEntry { face: face_from_index(face) },
        (NodeKind::CubedSphereFace { face }, NodeKind::CubedSphereBody { .. })
            => Transition::FaceExit { face: face_from_index(face) },
        (NodeKind::CubedSphereFace { face: from }, NodeKind::CubedSphereFace { face: to })
            => Transition::CubeSeam {
                from_face: face_from_index(from),
                to_face: face_from_index(to),
            },
        _ => Transition::None,
    }
}

#[inline]
fn face_from_index(i: u8) -> Face {
    match i {
        0 => Face::PosX, 1 => Face::NegX,
        2 => Face::PosY, 3 => Face::NegY,
        4 => Face::PosZ, 5 => Face::NegZ,
        _ => Face::PosX,
    }
}

// ---------------------------------------------------------------- helpers

/// Largest representable f32 strictly less than 1.0.
#[inline]
fn below_one() -> f32 {
    // 1.0 - 2^-24 is the next representable f32 below 1.0.
    // We use a slightly larger gap so math on it stays safely < 1.0.
    1.0 - f32::EPSILON
}

#[inline]
fn pick_slot(v: f32) -> usize {
    ((v * 3.0).floor() as i32).clamp(0, 2) as usize
}

#[inline]
fn rescale_down(v: f32, slot: usize) -> f32 {
    let r = v * 3.0 - slot as f32;
    if r < 0.0 { 0.0 } else if r >= 1.0 { below_one() } else { r }
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Path ----

    #[test]
    fn path_root_is_empty() {
        let p = Path::root();
        assert_eq!(p.depth(), 0);
        assert!(p.is_root());
        assert!(p.slots().is_empty());
    }

    #[test]
    fn path_push_pop() {
        let mut p = Path::root();
        assert!(p.push(5));
        assert!(p.push(12));
        assert_eq!(p.depth(), 2);
        assert_eq!(p.slots(), &[5, 12]);
        assert_eq!(p.last_slot(), Some(12));
        assert_eq!(p.pop(), Some(12));
        assert_eq!(p.pop(), Some(5));
        assert_eq!(p.pop(), None);
    }

    #[test]
    fn path_full_rejects_push() {
        let mut p = Path::root();
        for _ in 0..MAX_DEPTH {
            assert!(p.push(0));
        }
        assert!(!p.push(0), "push at MAX_DEPTH must fail");
    }

    #[test]
    fn path_equality_memcmp_semantics() {
        let mut a = Path::root();
        let mut b = Path::root();
        a.push(1); a.push(2); a.push(3);
        b.push(1); b.push(2); b.push(3);
        assert_eq!(a, b);
        b.pop();
        assert_ne!(a, b);
    }

    #[test]
    fn path_common_prefix() {
        let mut a = Path::root();
        let mut b = Path::root();
        for s in [1u8, 2, 3, 4] { a.push(s); }
        for s in [1u8, 2, 9] { b.push(s); }
        assert_eq!(a.common_prefix_len(&b), 2);
        assert_eq!(a.common_prefix_len(&a), 4);
    }

    #[test]
    fn step_neighbor_same_level() {
        let mut p = Path::root();
        // Start at slot (1,1,1) = 13 one level down.
        p.push(slot_index(1, 1, 1) as u8);
        assert!(p.step_neighbor_cartesian(0, 1)); // +x
        assert_eq!(p.last_slot(), Some(slot_index(2, 1, 1) as u8));
        assert!(p.step_neighbor_cartesian(1, -1)); // -y
        assert_eq!(p.last_slot(), Some(slot_index(2, 0, 1) as u8));
    }

    #[test]
    fn step_neighbor_bubbles_across_parent() {
        // Start at (x=2, y=1, z=1) within parent (x=0, y=0, z=0).
        // Stepping +x must bubble: parent becomes (1,0,0), child
        // reenters at (0, 1, 1).
        let mut p = Path::root();
        p.push(slot_index(0, 0, 0) as u8);
        p.push(slot_index(2, 1, 1) as u8);
        assert!(p.step_neighbor_cartesian(0, 1));
        assert_eq!(p.depth(), 2);
        assert_eq!(p.slots()[0], slot_index(1, 0, 0) as u8);
        assert_eq!(p.slots()[1], slot_index(0, 1, 1) as u8);
    }

    #[test]
    fn step_neighbor_clamps_at_root() {
        // No-op at root: step_neighbor returns false, path unchanged.
        let mut p = Path::root();
        assert!(!p.step_neighbor_cartesian(0, 1));
        assert_eq!(p.depth(), 0);
    }

    #[test]
    fn step_neighbor_clamps_at_edge_of_world() {
        // Single-level path at the far +x edge of root; any +x step
        // attempts to bubble past root and must fail gracefully,
        // leaving the path unchanged.
        let mut p = Path::root();
        p.push(slot_index(2, 0, 0) as u8);
        let before = p;
        assert!(!p.step_neighbor_cartesian(0, 1));
        assert_eq!(p, before);
    }

    // ---- WorldPos ----

    #[test]
    fn world_pos_default_at_root() {
        let p = WorldPos::root();
        assert_eq!(p.anchor.depth(), 0);
        assert_eq!(p.offset, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn add_local_small_delta() {
        let lib = NodeLibrary::default();
        let mut p = WorldPos::root();
        p.anchor.push(slot_index(1, 1, 1) as u8);
        let t = p.add_local_cartesian([0.1, 0.2, 0.3], &lib);
        assert_eq!(t, Transition::None);
        assert_eq!(p.anchor.depth(), 1);
        assert!((p.offset[0] - 0.1).abs() < 1e-6);
        assert!((p.offset[1] - 0.2).abs() < 1e-6);
        assert!((p.offset[2] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn add_local_crosses_cell_boundary() {
        let lib = NodeLibrary::default();
        let mut p = WorldPos::root();
        // Anchor at slot (1,1,1); offset near +x edge.
        p.anchor.push(slot_index(1, 1, 1) as u8);
        p.offset = [0.9, 0.5, 0.5];
        p.add_local_cartesian([0.2, 0.0, 0.0], &lib);
        // Stepped +x one cell: slot now (2,1,1), offset.x wraps to 0.1.
        assert_eq!(p.anchor.last_slot(), Some(slot_index(2, 1, 1) as u8));
        assert!((p.offset[0] - 0.1).abs() < 1e-5);
    }

    #[test]
    fn add_local_bubbles_up_and_back_down() {
        let lib = NodeLibrary::default();
        // Two-deep path at (0,0,0)→(2,1,1); +x crosses parent seam.
        let mut p = WorldPos::root();
        p.anchor.push(slot_index(0, 0, 0) as u8);
        p.anchor.push(slot_index(2, 1, 1) as u8);
        p.offset = [0.9, 0.5, 0.5];
        p.add_local_cartesian([0.2, 0.0, 0.0], &lib);
        assert_eq!(p.anchor.depth(), 2);
        assert_eq!(p.anchor.slots()[0], slot_index(1, 0, 0) as u8);
        assert_eq!(p.anchor.slots()[1], slot_index(0, 1, 1) as u8);
    }

    #[test]
    fn add_local_negative_crosses_back() {
        let lib = NodeLibrary::default();
        let mut p = WorldPos::root();
        p.anchor.push(slot_index(1, 1, 1) as u8);
        p.offset = [0.05, 0.5, 0.5];
        p.add_local_cartesian([-0.1, 0.0, 0.0], &lib);
        assert_eq!(p.anchor.last_slot(), Some(slot_index(0, 1, 1) as u8));
        assert!((p.offset[0] - 0.95).abs() < 1e-4);
    }

    #[test]
    fn classify_zoom_transition_body_to_face() {
        let before = NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 };
        let after = NodeKind::CubedSphereFace { face: Face::PosZ as u8 };
        match classify_zoom_transition(before, after) {
            Transition::FaceEntry { face } => assert_eq!(face, Face::PosZ),
            other => panic!("expected FaceEntry, got {:?}", other),
        }
    }

    #[test]
    fn classify_zoom_transition_face_to_body() {
        let before = NodeKind::CubedSphereFace { face: Face::NegY as u8 };
        let after = NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 };
        match classify_zoom_transition(before, after) {
            Transition::FaceExit { face } => assert_eq!(face, Face::NegY),
            other => panic!("expected FaceExit, got {:?}", other),
        }
    }

    #[test]
    fn add_local_seam_crosses_to_neighbor_face() {
        // Build a minimal tree: world root → body (CubedSphereBody)
        // → 6 face roots at face-center slots. Anchor inside +X face,
        // step +u → should land inside -Z face.
        use super::super::tree::{empty_children, Child};
        let mut lib = NodeLibrary::default();
        let mut face_roots = [0u64; 6];
        for &face in &Face::ALL {
            face_roots[face as usize] = lib.insert_with_kind(
                empty_children(),
                NodeKind::CubedSphereFace { face: face as u8 },
            );
        }
        let mut body_children = empty_children();
        for &face in &Face::ALL {
            body_children[body_face_center_slot(face)] =
                Child::Node(face_roots[face as usize]);
        }
        let body = lib.insert_with_kind(
            body_children,
            NodeKind::CubedSphereBody { inner_r: 0.1, outer_r: 0.4 },
        );
        let mut world_root_children = empty_children();
        world_root_children[slot_index(1, 2, 1)] = Child::Node(body);
        let world_root = lib.insert(world_root_children);

        // Anchor: world_root [16] → body [14 = +X face] → face_root [some cell at u=2 edge]
        let mut p = WorldPos::root();
        p.anchor.push(slot_index(1, 2, 1) as u8); // body
        p.anchor.push(body_face_center_slot(Face::PosX) as u8); // +X face_root
        p.anchor.push(slot_index(2, 1, 1) as u8); // u=2, v=1, r=1 inside face
        p.offset = [0.9, 0.5, 0.5];

        let t = p.add_local([0.2, 0.0, 0.0], &lib, world_root);

        // Stepping +u past u=2 on +X face should cross the seam to -Z.
        // The new face-center slot at body level should be -Z's slot.
        assert_eq!(p.anchor.depth(), 3);
        assert_eq!(
            p.anchor.slots()[1],
            body_face_center_slot(Face::NegZ) as u8,
            "seam should land on NegZ face after +u from PosX",
        );
        match t {
            Transition::CubeSeam { from_face, to_face } => {
                assert_eq!(from_face, Face::PosX);
                assert_eq!(to_face, Face::NegZ);
            }
            other => panic!("expected CubeSeam transition, got {:?}", other),
        }
    }

    #[test]
    fn add_local_kind_aware_stays_cartesian() {
        // When root is Cartesian and the path stays Cartesian,
        // add_local must still behave identically to the pure
        // Cartesian variant.
        use super::super::tree::{empty_children, Child};
        let mut lib = NodeLibrary::default();
        let child = lib.insert(empty_children());
        let mut root_children = empty_children();
        root_children[slot_index(1, 1, 1)] = Child::Node(child);
        let root = lib.insert(root_children);

        let mut p = WorldPos::root();
        p.anchor.push(slot_index(1, 1, 1) as u8);
        p.offset = [0.9, 0.5, 0.5];
        let t = p.add_local([0.2, 0.0, 0.0], &lib, root);
        assert_eq!(t, Transition::None);
        assert_eq!(p.anchor.last_slot(), Some(slot_index(2, 1, 1) as u8));
    }

    #[test]
    fn zoom_in_then_out_preserves_position() {
        let mut p = WorldPos::root();
        p.anchor.push(slot_index(1, 1, 1) as u8);
        p.offset = [0.37, 0.72, 0.45];
        let before = p;
        p.zoom_in();
        assert_eq!(p.anchor.depth(), before.anchor.depth() + 1);
        p.zoom_out();
        assert_eq!(p.anchor.depth(), before.anchor.depth());
        for i in 0..3 {
            assert!((p.offset[i] - before.offset[i]).abs() < 1e-5,
                "axis {i}: got {}, want {}", p.offset[i], before.offset[i]);
        }
    }

    #[test]
    fn zoom_in_picks_correct_slot() {
        let mut p = WorldPos::root();
        p.anchor.push(slot_index(0, 0, 0) as u8);
        p.offset = [0.8, 0.1, 0.5];
        p.zoom_in();
        // 0.8 * 3 = 2.4 → slot 2 on x; 0.1*3 = 0.3 → 0 on y; 0.5*3=1.5 → 1 on z.
        assert_eq!(p.anchor.last_slot(), Some(slot_index(2, 0, 1) as u8));
        assert!((p.offset[0] - 0.4).abs() < 1e-5);
        assert!((p.offset[1] - 0.3).abs() < 1e-5);
        assert!((p.offset[2] - 0.5).abs() < 1e-5);
    }

    // ---- legacy-world-coord bridge ----

    #[test]
    fn world_pos_to_f32_root_is_scaled_offset() {
        let p = WorldPos { anchor: Path::root(), offset: [0.5, 0.5, 0.5] };
        let xyz = world_pos_to_f32(&p);
        assert!((xyz[0] - 1.5).abs() < 1e-5);
        assert!((xyz[1] - 1.5).abs() < 1e-5);
        assert!((xyz[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn world_pos_from_f32_round_trips_at_depth_zero() {
        for &xyz in &[[1.5, 2.3, 1.5], [0.1, 0.2, 0.3], [2.9, 0.0, 1.0]] {
            let p = world_pos_from_f32(xyz, 0);
            let back = world_pos_to_f32(&p);
            for i in 0..3 {
                assert!((back[i] - xyz[i]).abs() < 1e-5, "axis {i}: {} vs {}", back[i], xyz[i]);
            }
        }
    }

    #[test]
    fn world_pos_from_f32_round_trips_at_deeper_anchors() {
        let xyz = [1.5, 2.3, 1.5];
        for depth in 0u8..=6 {
            let p = world_pos_from_f32(xyz, depth);
            assert_eq!(p.anchor.depth(), depth);
            let back = world_pos_to_f32(&p);
            for i in 0..3 {
                assert!(
                    (back[i] - xyz[i]).abs() < 1e-4,
                    "depth {depth} axis {i}: {} vs {}", back[i], xyz[i]
                );
            }
        }
    }

    #[test]
    fn world_pos_from_f32_clamps_out_of_range() {
        let p = world_pos_from_f32([-1.0, 5.0, 1.5], 2);
        let back = world_pos_to_f32(&p);
        // Negative clamped to 0, past-extent clamped to < ROOT_EXTENT.
        assert!(back[0] >= 0.0 && back[0] < ROOT_EXTENT);
        assert!(back[1] >= 0.0 && back[1] < ROOT_EXTENT);
        assert!((back[2] - 1.5).abs() < 1e-4);
    }

    #[test]
    fn zoom_out_at_root_is_noop() {
        let mut p = WorldPos::root();
        p.offset = [0.5, 0.5, 0.5];
        let t = p.zoom_out();
        assert_eq!(t, Transition::None);
        assert_eq!(p.anchor.depth(), 0);
        assert_eq!(p.offset, [0.5, 0.5, 0.5]);
    }
}
