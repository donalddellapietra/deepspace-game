//! Test-entity spawning. Bound to `N` (10) and `M` (1000) stone
//! cubes in debug keybinds. Also the entry point for
//! `--spawn-entity PATH` init-time spawning used by the entity
//! visibility test suite.
//!
//! Spawns entities at the camera's anchor cell stepped forward in
//! -Z, each in its own cell so they don't overlap. Content for the
//! keybind path is a uniform-stone subtree — same NodeId reused
//! across every spawn, so 1000 entities → 1 shared packed subtree.
//! The `--spawn-entity` path loads a `.vox` file and uses its
//! built tree as the shared NodeId instead.

use std::path::Path as FsPath;

use crate::import;
use crate::world::anchor::WorldPos;
use crate::world::palette::block;
use crate::world::tree::{Child, NodeId};

use crate::app::App;

/// Per-index velocity pattern for test spawns. Deterministic so
/// rendering tests can reproduce motion exactly.
///
/// Uses sin/cos of the entity index to scatter directions across
/// the horizontal plane; y-component is small so they don't drift
/// up into the sky. Magnitude ~0.3 cells/sec gives ~0.15 cells of
/// motion in 30 frames at 60fps — visually obvious without entities
/// walking off-screen during a short test.
fn entity_velocity(i: u32) -> [f32; 3] {
    let phase = i as f32 * 0.173;
    [phase.sin() * 0.30, phase.cos() * 0.05, phase.cos() * 0.30]
}

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
        self.spawn_grid(subtree_id, n);
        // Trigger an upload so the new entities show up on the next frame.
        self.upload_tree();
    }

    /// Init-time entity spawn driven by `--spawn-entity PATH`.
    /// Loads the `.vox` file, builds it into a library subtree, then
    /// spawns `count` copies in a row in front of the camera, each in
    /// its own anchor cell so they don't overlap.
    ///
    /// Palette entries registered by the vox loader are picked up on
    /// the next palette upload — see `run_render_harness` which calls
    /// `update_palette` after App construction.
    pub fn spawn_vox_entity_at_init(&mut self, path: &FsPath, count: u32) {
        if count == 0 {
            return;
        }
        let model = match import::load(path, &mut self.palette) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "spawn_entity: load {} failed: {}",
                    path.display(), e,
                );
                return;
            }
        };
        eprintln!(
            "spawn_entity: loaded {} ({}x{}x{} voxels, count={})",
            path.display(), model.size_x, model.size_y, model.size_z, count,
        );
        let subtree_root = import::tree_builder::build_tree(
            &model, &mut self.world.library,
        );
        self.spawn_grid(subtree_root, count);
        eprintln!(
            "spawn_entity: spawned {} entities subtree_node={} base_depth={}",
            count, subtree_root, self.camera.position.anchor.depth(),
        );
        // Don't call upload_tree here — the first frame's
        // upload_tree_lod picks up the entities through the normal
        // path. Calling it before the renderer is ready is a no-op
        // anyway.
    }

    /// Shared grid layout: `n` entities starting one cell in front of
    /// the camera, filling +X then +Y then +Z in a 9x9x... pattern.
    /// Each entity gets a deterministic per-index velocity from
    /// `entity_velocity(i)` so they drift in varied directions.
    fn spawn_grid(&mut self, subtree_id: NodeId, n: u32) {
        let cam_anchor = self.camera.position.anchor;
        let mut base = cam_anchor;
        base.step_neighbor_cartesian(2, -1);

        let row_len = 9u32;
        let grid_len = 81u32;

        let before = self.entities.len();
        for i in 0..n {
            let row = (i / row_len) % row_len;
            let col = i % row_len;
            let stack = i / grid_len;
            let mut anchor = base;
            for _ in 0..col {
                anchor.step_neighbor_cartesian(0, 1);
            }
            for _ in 0..row {
                anchor.step_neighbor_cartesian(1, 1);
            }
            for _ in 0..stack {
                anchor.step_neighbor_cartesian(2, -1);
            }
            let pos = WorldPos::new_unchecked(anchor, [0.0, 0.0, 0.0]);
            let velocity = entity_velocity(i);
            self.entities
                .spawn(&mut self.world.library, pos, velocity, subtree_id);
        }
        let after = self.entities.len();
        log::info!(
            "spawned {} entities ({} -> {}) subtree_id={} cam_depth={}",
            n, before, after, subtree_id, cam_anchor.depth(),
        );
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
