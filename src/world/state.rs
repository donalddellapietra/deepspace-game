//! Runtime world state: the content-addressed tree.

use super::tree::{NodeId, NodeLibrary};

pub struct WorldState {
    pub root: NodeId,
    pub library: NodeLibrary,
}

impl WorldState {
    /// Maximum depth of the tree from the root. Depth = number of
    /// Node→Node edges from root to the deepest terminal
    /// (Block/Empty) children.
    ///
    /// O(1): `Node::depth` is cached at insertion time in
    /// `NodeLibrary::insert_with_kind`. This used to do a full DFS
    /// over the library on every call; a 94k-node library took ~38
    /// ms, and `upload_tree` called it on every edit.
    pub fn tree_depth(&self) -> u32 {
        self.library.get(self.root).map(|n| n.depth).unwrap_or(0)
    }

    pub fn swap_root(&mut self, new_root: NodeId) {
        if new_root == self.root { return; }
        self.library.ref_inc(new_root);
        let old = self.root;
        self.root = new_root;
        self.library.ref_dec(old);
    }
}
