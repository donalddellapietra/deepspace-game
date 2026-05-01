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
    /// Spawn `n` soldier entities in front of the camera. The soldier
    /// model is loaded from `assets/vox/soldier.vox` on first call
    /// and its subtree NodeId is cached on the App, so repeat N/M
    /// presses reuse the parsed model (library dedup makes all
    /// copies share a single voxel subtree anyway).
    ///
    /// Falls back to a uniform-stone-cube subtree if the .vox file
    /// can't be loaded — the old behavior, useful on a dev build
    /// without the asset.
    pub(in crate::app) fn spawn_test_entities(&mut self, n: u32) {
        if n == 0 {
            return;
        }
        let subtree_id = self
            .get_or_load_soldier_subtree()
            .unwrap_or_else(|| self.get_or_build_stone_cube_subtree());
        self.spawn_grid(subtree_id, n);
        // Trigger an upload so the new entities show up on the next frame.
        self.upload_tree();
    }

    /// Spawn `n` uniform-stone-cube entities. Bound to B (10) and V
    /// (1000) — the original behavior of N/M before the soldier
    /// model took over those keys. Useful for raster-perf stress
    /// tests where we want a single-color subtree (greedy meshing
    /// merges every face of a 27-voxel stone cube into 6 quads, so
    /// 10k cubes are much cheaper to render than 10k soldiers).
    pub(in crate::app) fn spawn_test_cubes(&mut self, n: u32) {
        if n == 0 {
            return;
        }
        let subtree_id = self.get_or_build_stone_cube_subtree();
        self.spawn_grid(subtree_id, n);
        self.upload_tree();
    }

    /// Load `assets/vox/soldier.vox` on first call and stash the
    /// resulting subtree NodeId on `self.cached_soldier_subtree` so
    /// subsequent spawns skip the file read and palette registration.
    /// Returns `None` if the file is missing / unparseable — callers
    /// should fall back to a uniform-block subtree.
    fn get_or_load_soldier_subtree(&mut self) -> Option<NodeId> {
        if let Some(id) = self.cached_soldier_subtree {
            return Some(id);
        }
        let path = FsPath::new("assets/vox/soldier.vox");
        let model = match import::load(path, &mut self.palette) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "spawn_test_entities: load {} failed: {} (falling back to stone cube)",
                    path.display(), e,
                );
                return None;
            }
        };
        let id = import::tree_builder::build_tree(&model, &mut self.world.library);
        eprintln!(
            "spawn_test_entities: cached soldier subtree node={} ({}x{}x{} voxels)",
            id, model.size_x, model.size_y, model.size_z,
        );
        // Palette gained colors; push them to the GPU so the new
        // entities render with the correct tints on the next frame.
        // Fractal-presets moved the palette to a storage buffer —
        // updates now take &mut self (they can grow the buffer
        // when new palette entries were registered).
        if let Some(renderer) = self.renderer.as_mut() {
            renderer.update_palette(&self.palette.to_gpu_palette());
        }
        self.cached_soldier_subtree = Some(id);
        Some(id)
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

    /// Shared grid layout: `n` entities in a horizontal grid in
    /// front of the camera. Rows/cols step along the two ground-
    /// plane axes (X and Z); stacks step further forward in -Z when
    /// the XZ plane is full.
    ///
    /// On worlds with a defined sea level (`entity_surface_y`), the
    /// grid is snapped to that Y — so every entity lands on the
    /// ground regardless of where the camera is. On fractal worlds
    /// we fall back to the camera's Y as before.
    fn spawn_grid(&mut self, subtree_id: NodeId, n: u32) {
        let cam_anchor = self.camera.position.anchor;
        let anchor_depth = cam_anchor.depth();

        // Pick a base anchor one cell in front of the camera (-Z).
        // Then override Y to sea level when we have one: build a
        // world-coords position from the base anchor's XZ and
        // sea_level Y, and reconstruct an anchor at the same depth.
        let mut base = cam_anchor;
        base.step_neighbor_in_world(&self.world.library, self.world.root, 2, -1);

        let base_anchor = if let Some(sea_y) = self.entity_surface_y {
            let base_pos = WorldPos::new_unchecked(base, [0.0, 0.0, 0.0]);
            let base_world = base_pos.in_frame(&crate::world::anchor::Path::root());
            let ground = WorldPos::from_frame_local(
                &crate::world::anchor::Path::root(),
                [base_world[0], sea_y, base_world[2]],
                anchor_depth,
            );
            ground.anchor
        } else {
            base
        };

        let row_len = 9u32;
        let grid_len = row_len * row_len;

        let before = self.entities.len();
        for i in 0..n {
            // Step along X (col) and Z (row) in the horizontal plane.
            // When the plane fills up, stack forward in -Z (stack
            // doesn't change Y, so entities stay at sea level when
            // we have one).
            let col = i % row_len;
            let row = (i / row_len) % row_len;
            let stack = i / grid_len;
            let mut anchor = base_anchor;
            for _ in 0..col {
                anchor.step_neighbor_in_world(
                    &self.world.library,
                    self.world.root,
                    0,
                    1,
                );
            }
            for _ in 0..row {
                anchor.step_neighbor_in_world(
                    &self.world.library,
                    self.world.root,
                    2,
                    1,
                );
            }
            for _ in 0..stack {
                anchor.step_neighbor_in_world(
                    &self.world.library,
                    self.world.root,
                    2,
                    -1,
                );
            }
            let pos = WorldPos::new_unchecked(anchor, [0.0, 0.0, 0.0]);
            let velocity = entity_velocity(i);
            self.entities
                .spawn(&mut self.world.library, pos, velocity, subtree_id);
        }
        let after = self.entities.len();
        log::info!(
            "spawned {} entities ({} -> {}) subtree_id={} cam_depth={} sea_y={:?}",
            n, before, after, subtree_id, cam_anchor.depth(), self.entity_surface_y,
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
            Child::Block(_) | Child::Empty | Child::EntityRef(_) => {
                // build_uniform_subtree with depth > 0 always returns a Node.
                panic!("build_uniform_subtree(depth=3) returned a non-node");
            }
        }
    }
}
