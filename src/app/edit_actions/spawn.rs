//! Test-entity spawning. Bound to `N` (10 entities) and `M` (1000)
//! in debug keybinds. Used to smoke-test the entity render + edit
//! pipeline at small and larger entity counts.
//!
//! Spawns entities at the camera's anchor cell stepped forward in
//! -Z, each in its own cell so they don't overlap. Content is a
//! uniform-stone subtree — same NodeId reused across every spawn,
//! so 1000 entities → 1 shared packed subtree in the GPU tree
//! buffer (content-addressed dedup proving itself).

use crate::world::palette::block;
use crate::world::tree::{Child, NodeId};

use crate::app::App;

impl App {
    /// Spawn `n` stone-cube entities in front of the camera.
    /// Anchored at the camera's anchor depth; content is a single
    /// uniform-stone subtree shared across every spawn via library
    /// dedup.
    pub(in crate::app) fn spawn_test_entities(&mut self, n: u32) {
        if n == 0 {
            return;
        }
        let subtree_id = self.get_or_build_stone_cube_subtree();
        let cam_anchor = self.camera.position.anchor;

        // Start one cell forward of the camera, then march further.
        // Step forward (-Z) by 1 cell; subsequent entities step +X
        // along the row.
        let mut base = cam_anchor;
        base.step_neighbor_cartesian(2, -1);

        // For counts >9 we split into a simple grid: row of min(n, 9)
        // entities along +X, then next row +Y, wrapping +Z.
        let row_len = 9u32;
        let grid_len = 81u32;

        let before = self.entities.len();
        for i in 0..n {
            let row = (i / row_len) % row_len;
            let col = i % row_len;
            let stack = i / grid_len;
            let mut anchor = base;
            // Row positions extend beyond a single cell in X by
            // calling step_neighbor_cartesian repeatedly — cheap
            // because it's slot arithmetic.
            for _ in 0..col {
                anchor.step_neighbor_cartesian(0, 1);
            }
            for _ in 0..row {
                anchor.step_neighbor_cartesian(1, 1);
            }
            for _ in 0..stack {
                anchor.step_neighbor_cartesian(2, -1);
            }
            self.entities.spawn(&mut self.world.library, anchor, subtree_id);
        }
        let after = self.entities.len();
        log::info!(
            "spawned {} entities ({} -> {}) subtree_id={} cam_depth={}",
            n, before, after, subtree_id, cam_anchor.depth(),
        );
        // Trigger an upload so the new entities show up on the next frame.
        self.upload_tree();
    }

    /// Return the cached canonical stone-cube NodeId. Builds it on
    /// first call and stashes it so all spawns reuse the same ID —
    /// the library's content-addressing would dedup anyway, but a
    /// cached handle avoids re-walking the build every call.
    fn get_or_build_stone_cube_subtree(&mut self) -> NodeId {
        // Depth-3 uniform subtree of stone: 27³ leaf voxels, ~4
        // unique NodeIds total thanks to dedup. The root's 27
        // children share one NodeId; that one's 27 children share
        // one more; leaves are Child::Block(STONE).
        let subtree = self.world.library.build_uniform_subtree(block::STONE, 3);
        match subtree {
            Child::Node(id) => id,
            Child::Block(_) | Child::Empty => {
                // build_uniform_subtree with depth > 0 always returns a Node.
                panic!("build_uniform_subtree(depth=3) returned a non-node");
            }
        }
    }
}

