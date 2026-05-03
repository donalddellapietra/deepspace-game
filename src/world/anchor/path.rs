//! Symbolic path through the 27-ary tree, plus the kind-aware
//! neighbor-stepping primitive that wraps the slot when the parent
//! is a `WrappedPlane`.

use std::hash::{Hash, Hasher};

use crate::world::tree::{slot_coords, slot_index, Child, NodeId, NodeKind, NodeLibrary, MAX_DEPTH};

/// Symbolic path through the 27-ary tree. Exact at any depth; no
/// f32 precision loss.
#[derive(Clone, Copy)]
pub struct Path {
    slots: [u8; MAX_DEPTH],
    depth: u8,
}

impl Path {
    pub const fn root() -> Self {
        Self { slots: [0u8; MAX_DEPTH], depth: 0 }
    }

    #[inline]
    pub fn depth(&self) -> u8 {
        self.depth
    }

    #[inline]
    pub fn is_root(&self) -> bool {
        self.depth == 0
    }

    #[inline]
    pub fn slot(&self, i: usize) -> u8 {
        debug_assert!(i < self.depth as usize);
        self.slots[i]
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.slots[..self.depth as usize]
    }

    pub fn push(&mut self, slot: u8) {
        debug_assert!((slot as usize) < 27, "slot must be < 27");
        debug_assert!((self.depth as usize) < MAX_DEPTH, "path overflow");
        self.slots[self.depth as usize] = slot;
        self.depth += 1;
    }

    pub fn pop(&mut self) -> Option<u8> {
        if self.depth == 0 {
            None
        } else {
            self.depth -= 1;
            Some(self.slots[self.depth as usize])
        }
    }

    /// Truncate to `new_depth` levels. If `new_depth >= depth`, no-op.
    pub fn truncate(&mut self, new_depth: u8) {
        if new_depth < self.depth {
            self.depth = new_depth;
        }
    }

    pub fn with_truncated(mut self, new_depth: u8) -> Self {
        self.truncate(new_depth);
        self
    }

    /// Length of the longest common prefix between `self` and `other`.
    pub fn common_prefix_len(&self, other: &Self) -> u8 {
        let n = self.depth.min(other.depth) as usize;
        let mut i = 0;
        while i < n && self.slots[i] == other.slots[i] {
            i += 1;
        }
        i as u8
    }

    /// Step one cell along `axis` (0=x, 1=y, 2=z) by `direction` ±1
    /// in the Cartesian interpretation. Bubbles up through parent
    /// cells on overflow. Returns `true` if the step succeeded,
    /// `false` if the camera hit the world boundary (root can't
    /// step further) — in that case the path is unchanged.
    pub fn step_neighbor_cartesian(&mut self, axis: usize, direction: i32) -> bool {
        debug_assert!(axis < 3);
        debug_assert!(direction == 1 || direction == -1);
        if self.depth == 0 {
            return false;
        }
        let d = self.depth as usize - 1;
        let slot = self.slots[d] as usize;
        let (x, y, z) = slot_coords(slot);
        let mut coords = [x, y, z];
        let v = coords[axis] as i32 + direction;
        if (0..3).contains(&v) {
            coords[axis] = v as usize;
            self.slots[d] = slot_index(coords[0], coords[1], coords[2]) as u8;
            return true;
        }
        self.depth -= 1;
        if !self.step_neighbor_cartesian(axis, direction) {
            self.depth += 1;
            return false;
        }
        let wrapped = if direction < 0 { 2 } else { 0 };
        coords[axis] = wrapped;
        let new_slot = slot_index(coords[0], coords[1], coords[2]) as u8;
        self.slots[self.depth as usize] = new_slot;
        self.depth += 1;
        true
    }

    /// Kind-aware neighbor step. Walks `self` from `world_root` to
    /// determine the parent NodeKind at each level. On overflow:
    /// - `NodeKind::Cartesian` parent: bubble up exactly like
    ///   `step_neighbor_cartesian`.
    /// - `NodeKind::WrappedPlane { dims, slab_depth }` parent AND the
    ///   overflow axis is the wrap axis (X = axis 0): wrap the slot's
    ///   X coord in place (set to 0 on east overflow, 2 on west) so
    ///   the path stays inside the WrappedPlane subtree. Y / Z still
    ///   bubble — those axes exit the slab via the normal ribbon-pop
    ///   path, not the wrap.
    ///
    /// Returns `true` iff a wrap fired during this step (caller can
    /// surface a `Transition::WrappedPlaneWrap`).
    ///
    /// **Wrap correctness invariant:** the wrap formula assumes the
    /// slab fully fills the WrappedPlane node along the wrap axis,
    /// i.e., `dims[0] == 3^slab_depth`. With dims_x < 3^slab_depth
    /// the wrap would land mid-slab, which is geometrically wrong;
    /// callers / worldgen MUST size the slab to fully fill the wrap
    /// axis.
    /// Returns `(stepped_ok, wrap_occurred)`. When `stepped_ok` is
    /// false the path is unchanged (world boundary hit).
    /// Returns `(step_succeeded, wrap_occurred)`.
    pub fn step_neighbor_in_world(
        &mut self,
        library: &NodeLibrary,
        world_root: NodeId,
        axis: usize,
        direction: i32,
    ) -> (bool, bool) {
        debug_assert!(axis < 3);
        debug_assert!(direction == 1 || direction == -1);
        if self.depth == 0 {
            return (false, false);
        }
        let d = self.depth as usize - 1;
        let slot = self.slots[d] as usize;
        let (x, y, z) = slot_coords(slot);
        let mut coords = [x, y, z];
        let v = coords[axis] as i32 + direction;
        if (0..3).contains(&v) {
            coords[axis] = v as usize;
            self.slots[d] = slot_index(coords[0], coords[1], coords[2]) as u8;
            return (true, false);
        }
        if axis == 0 {
            if let Some(parent_kind) = node_kind_at_depth(library, world_root, &self.slots[..d]) {
                if matches!(parent_kind, NodeKind::WrappedPlane { .. }) {
                    let wrapped = if direction < 0 { 2 } else { 0 };
                    coords[axis] = wrapped;
                    self.slots[d] = slot_index(coords[0], coords[1], coords[2]) as u8;
                    return (true, true);
                }
            }
        }
        self.depth -= 1;
        let (ok, wrapped_inner) = self.step_neighbor_in_world(library, world_root, axis, direction);
        if !ok {
            self.depth += 1;
            return (false, false);
        }
        let wrapped = if direction < 0 { 2 } else { 0 };
        coords[axis] = wrapped;
        let new_slot = slot_index(coords[0], coords[1], coords[2]) as u8;
        self.slots[self.depth as usize] = new_slot;
        self.depth += 1;
        (true, wrapped_inner)
    }
}

/// Walk the tree from `world_root` along `slots`, returning the
/// `NodeKind` of the node reached (i.e., the node at tree depth
/// `slots.len()`). Returns `None` if the walk hits a non-Node child
/// before consuming all slots, or if a node id is missing from the
/// library.
pub(super) fn node_kind_at_depth(
    library: &NodeLibrary,
    world_root: NodeId,
    slots: &[u8],
) -> Option<NodeKind> {
    let mut cur = world_root;
    for &s in slots {
        let node = library.get(cur)?;
        match node.children[s as usize] {
            Child::Node(child) => cur = child,
            _ => return None,
        }
    }
    library.get(cur).map(|n| n.kind)
}

impl Default for Path {
    fn default() -> Self {
        Path::root()
    }
}

impl PartialEq for Path {
    fn eq(&self, other: &Self) -> bool {
        self.depth == other.depth && self.as_slice() == other.as_slice()
    }
}

impl Eq for Path {}

impl Hash for Path {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.depth.hash(state);
        self.as_slice().hash(state);
    }
}

impl std::fmt::Debug for Path {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Path(d={},[", self.depth)?;
        for (i, s) in self.as_slice().iter().enumerate() {
            if i > 0 { write!(f, ",")?; }
            write!(f, "{}", s)?;
        }
        write!(f, "])")
    }
}

/// Event fired when a coordinate primitive crosses a meaningful
/// boundary. Game-level handlers react (camera up rotation, UI,
/// etc.); the coordinate math itself is already complete by the
/// time a transition is reported.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transition {
    None,
    /// A motion step crossed the wrap axis of a `WrappedPlane`
    /// ancestor and the path was wrapped (modulo `dims[axis]`)
    /// rather than bubbled out of the slab subtree. `axis` is the
    /// world axis that wrapped (0 = x). Phase 2 only wraps on
    /// axis 0; Y / Z always bubble.
    WrappedPlaneWrap { axis: u8 },
}
