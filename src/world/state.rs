//! Runtime world state: the content-addressed tree.

use super::tree::*;

pub struct WorldState {
    pub root: NodeId,
    pub library: NodeLibrary,
}

impl WorldState {
    /// Compute the maximum depth of the tree from the root.
    /// Depth = number of Node→Node edges from root to the deepest
    /// terminal (Block/Empty) children.
    /// Memoized — each unique NodeId is visited at most once.
    pub fn tree_depth(&self) -> u32 {
        let mut cache = std::collections::HashMap::new();
        self.depth_of(self.root, &mut cache)
    }

    fn depth_of(&self, id: NodeId, cache: &mut std::collections::HashMap<NodeId, u32>) -> u32 {
        if let Some(&d) = cache.get(&id) { return d; }
        let Some(node) = self.library.get(id) else { return 0 };
        let mut max_child_depth = 0u32;
        for child in &node.children {
            if let Child::Node(child_id) = child {
                max_child_depth = max_child_depth.max(self.depth_of(*child_id, cache));
            }
        }
        let d = 1 + max_child_depth;
        cache.insert(id, d);
        d
    }

    pub fn swap_root(&mut self, new_root: NodeId) {
        if new_root == self.root { return; }
        self.library.ref_inc(new_root);
        let old = self.root;
        self.root = new_root;
        self.library.ref_dec(old);
    }
}
