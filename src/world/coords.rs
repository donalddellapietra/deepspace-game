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
use super::face_transitions::{face_basis, offset_remap_on_overflow_axis, seam_neighbor, FaceBasis, Seam};
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

/// Walk `path` from `root` and return `(face_root_depth, face)` for
/// the deepest `CubedSphereFace` ancestor of the current anchor, or
/// `None` if the anchor is not inside any face subtree.
///
/// `face_root_depth` is the depth at which the face root node lives
/// (i.e., the anchor's `slots()[face_root_depth-1]` is the body's
/// face-center slot choosing this face root). The anchor's own
/// cells are at `face_root_depth+1` or deeper when inside the face.
pub fn find_face_ancestor(
    lib: &NodeLibrary,
    root: NodeId,
    path: &Path,
) -> Option<(u8, Face)> {
    let kinds = resolve_kinds_along(lib, root, path);
    // kinds[0] = root node's kind, kinds[i] = kind reached by descending i slots.
    // Face root at depth i means kinds[i] is CubedSphereFace (after i descent slots).
    let mut deepest: Option<(u8, Face)> = None;
    for (i, k) in kinds.iter().enumerate() {
        if let NodeKind::CubedSphereFace { face } = *k {
            deepest = Some((i as u8, face_from_index(face)));
        }
    }
    deepest
}

/// Helper: given the face_anc from before a step and the new anchor
/// state, decide the new face_anc. Pure path inspection — if the
/// anchor is still at or below the face root level, the ancestor is
/// unchanged; if it popped above it, there's no face ancestor.
fn face_ancestor_from_anchor_kinds(
    prior: Option<(u8, Face)>,
    anchor: &Path,
) -> Option<(u8, Face)> {
    match prior {
        Some((fd, f)) if anchor.depth() > fd => Some((fd, f)),
        _ => None,
    }
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

    /// Add `delta` (interpreted in world `(x, y, z)`) to `offset` and
    /// cross cell boundaries as needed.
    ///
    /// Kind-aware: if the anchor sits inside a `CubedSphereFace`
    /// subtree, the world delta is first transformed into the face's
    /// local `(u, v, r)` basis (orthonormal approximation — exact at
    /// the face center), then cells are stepped in face-local axes.
    /// When a step overflows the face root on `u` or `v`, the seam
    /// table in [`super::face_transitions`] remaps the anchor onto
    /// the neighbor face instead of bubbling up through the body.
    ///
    /// For Cartesian / CubedSphereBody anchors the basis is identity
    /// and stepping bubbles normally through the 3-ary tree.
    ///
    /// Returns the last coordinate-meaning transition crossed. When
    /// no transition happens, returns `Transition::None`.
    pub fn add_local(
        &mut self,
        delta_world: [f32; 3],
        lib: &NodeLibrary,
        root: NodeId,
    ) -> Transition {
        let kind_before = deepest_kind(lib, root, &self.anchor);
        let mut face_anc = find_face_ancestor(lib, root, &self.anchor);
        let basis = match face_anc {
            Some((_, f)) => face_basis(f),
            None => FaceBasis::IDENTITY,
        };
        let delta_local = basis.world_to_local(delta_world);
        for i in 0..3 {
            self.offset[i] += delta_local[i];
        }

        // Cell-by-cell overflow resolution. Done iteratively rather
        // than axis-by-axis because a seam crossing on one axis can
        // flip another axis's offset, requiring a re-check. Capped
        // at a generous iteration count to avoid infinite loops if
        // something pathological happens.
        let mut last_seam: Option<(Face, Face)> = None;
        const MAX_ITERS: usize = 128;
        'outer: for _ in 0..MAX_ITERS {
            for axis in 0..3u8 {
                let a = axis as usize;
                if self.offset[a] >= 1.0 {
                    self.offset[a] -= 1.0;
                    if !self.try_step_cell(axis, 1, face_anc, &mut last_seam, &mut face_anc) {
                        self.offset[a] = below_one();
                    }
                    continue 'outer;
                } else if self.offset[a] < 0.0 {
                    self.offset[a] += 1.0;
                    if !self.try_step_cell(axis, -1, face_anc, &mut last_seam, &mut face_anc) {
                        self.offset[a] = 0.0;
                    }
                    continue 'outer;
                }
            }
            break;
        }

        if let Some((from, to)) = last_seam {
            return Transition::CubeSeam { from_face: from, to_face: to };
        }
        let kind_after = deepest_kind(lib, root, &self.anchor);
        if kind_before == kind_after {
            return Transition::None;
        }
        classify_transition(lib, root, kind_before, &self.anchor)
    }

    /// Step the anchor one cell in `axis` / `dir`. If the anchor sits
    /// at the direct-child level of a `CubedSphereFace` root and this
    /// step would overflow on `u` or `v`, apply a seam remap to the
    /// neighboring face. Otherwise fall through to the pure Cartesian
    /// neighbor step. Returns false only when clamped at the world
    /// root (nowhere left to step).
    fn try_step_cell(
        &mut self,
        axis: u8,
        dir: i8,
        face_anc: Option<(u8, Face)>,
        last_seam: &mut Option<(Face, Face)>,
        face_anc_out: &mut Option<(u8, Face)>,
    ) -> bool {
        if let Some((fd, face)) = face_anc {
            if axis < 2 && self.anchor.depth() == fd + 1 {
                // At the face-root's direct-child level. If this cell's
                // slot on the axis is already at the +/- boundary AND
                // we're stepping further out, it's a seam crossing.
                let leaf = self.anchor.last_slot().unwrap() as usize;
                let (cx, cy, cz) = slot_coords(leaf);
                let old_coords = [cx, cy, cz];
                let would_overflow = match dir {
                    1 => old_coords[axis as usize] == 2,
                    -1 => old_coords[axis as usize] == 0,
                    _ => false,
                };
                if would_overflow {
                    let seam = seam_neighbor(face, axis, dir);
                    self.apply_seam(axis, seam, old_coords);
                    *last_seam = Some((face, seam.to_face));
                    *face_anc_out = Some((fd, seam.to_face));
                    return true;
                }
            }
        }
        // Non-seam path.
        let moved = self.anchor.step_neighbor_cartesian(axis, dir);
        if moved {
            // Face membership may have changed (e.g., r-axis bubble
            // out of face into body). Re-resolve so subsequent seam
            // checks use the right basis.
            *face_anc_out = face_ancestor_from_anchor_kinds(face_anc, &self.anchor);
        }
        moved
    }

    /// Apply a seam crossing: rewrite the anchor and offset so the
    /// position moves from the old face subtree into `seam.to_face`.
    /// Assumes `self.offset[axis_overflow]` has already been wrapped
    /// into `[0, 1)` by the caller (the "just past old edge" value).
    fn apply_seam(&mut self, axis_overflow: u8, seam: Seam, old_coords: [usize; 3]) {
        let old_offset = self.offset;
        // Pop leaf + old face-root body-slot, land on body level.
        self.anchor.pop();
        self.anchor.pop();
        // Push new face-root body-slot.
        self.anchor.push(body_face_center_slot(seam.to_face) as u8);

        let mut new_slot = [0usize; 3];
        let mut new_offset = [0.0f32; 3];
        for i in 0..3 {
            let new_axis = seam.axis_map[i] as usize;
            if i == axis_overflow as usize {
                // Overflow axis: slot is the entering edge; offset is
                // the wrapped value, possibly mirrored.
                new_slot[new_axis] = if seam.entering_sign == -1 { 0 } else { 2 };
                new_offset[new_axis] =
                    offset_remap_on_overflow_axis(old_offset[i], seam.entering_sign);
            } else {
                new_slot[new_axis] = if seam.flip[i] { 2 - old_coords[i] } else { old_coords[i] };
                new_offset[new_axis] = if seam.flip[i] { 1.0 - old_offset[i] } else { old_offset[i] };
            }
        }
        self.anchor.push(slot_index(new_slot[0], new_slot[1], new_slot[2]) as u8);
        self.offset = new_offset;
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

    /// Seam integration: anchor sits at the +u edge slot of a
    /// CubedSphereFace root; delta on the face-local u axis (expressed
    /// in world coords via the face basis) crosses the seam onto the
    /// expected neighbor face.
    #[test]
    fn add_local_crosses_seam_from_posx_to_negz() {
        use super::super::spherical_worldgen::{build, demo_planet};
        let setup = demo_planet();
        let scene = build(&setup);
        let body_anchor = scene.body_anchor;
        let world = scene.world;

        // Build an anchor: body_anchor + body's +X face-center slot +
        // face-root child at us=2 (the +u edge slot).
        let mut anchor = body_anchor;
        anchor.push(body_face_center_slot(Face::PosX) as u8);
        anchor.push(slot_index(2, 1, 1) as u8); // +u edge, v=middle, r=middle
        let mut p = WorldPos { anchor, offset: [0.9, 0.5, 0.5] };

        // Push in +u direction (world tu(+X) = (0, 0, -1)), enough to
        // cross the edge. delta_world = tu * 0.2 in the face frame.
        let tu: [f32; 3] = [0.0, 0.0, -1.0];
        let step: f32 = 0.2;
        let delta = [tu[0] * step, tu[1] * step, tu[2] * step];
        let t = p.add_local(delta, &world.library, world.root);

        // Should have emitted a CubeSeam transition from +X to -Z.
        match t {
            Transition::CubeSeam { from_face: Face::PosX, to_face: Face::NegZ } => {}
            other => panic!("expected CubeSeam PosX→NegZ, got {:?}", other),
        }
        // Anchor now sits under the -Z face-center slot of the body.
        let slots = p.anchor.slots();
        let body_d = body_anchor.depth() as usize;
        assert_eq!(slots[body_d], body_face_center_slot(Face::NegZ) as u8);
        // Entering edge: us=0 on -Z (flip on axis 0 was true).
        let (cx, _cy, _cz) = slot_coords(slots[body_d + 1] as usize);
        assert_eq!(cx, 0, "entering us on -Z must be 0 (the u=-1 edge slot)");
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
