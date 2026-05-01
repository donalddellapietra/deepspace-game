//! Entities: voxel subtrees placed at arbitrary cells in the world
//! tree, kept outside the tree in a flat buffer.
//!
//! Entities share voxel content with each other and with the world
//! via the content-addressed `NodeLibrary`. A crowd of 10k identical
//! NPCs stores 1 NodeId in the library plus 10k `Entity` records —
//! per-instance state is just a position + velocity + a pointer
//! into the library.
//!
//! Editing is clone-on-write: breaking a voxel inside an entity
//! calls `propagate_edit_on_library` against the entity's current
//! root, producing a new NodeId that gets stashed in
//! `override_root`. The base `subtree_root` stays shared.
//!
//! Motion: each entity carries a `velocity` in anchor-cell units
//! per second. `tick(library, dt)` advances `pos.add_local(vel*dt)`
//! per entity. Movement across cell boundaries is transparent
//! (WorldPos renormalizes), so anchor cells update automatically.
//!
//! Rendering: the entity buffer is rebuilt every frame (positions
//! change each tick). `march_entities` in WGSL tests rays against
//! each entity's AABB, transforms into local subtree space, and
//! reuses the existing `march_cartesian` to walk voxels.

use crate::world::anchor::WorldPos;
use crate::world::tree::{NodeId, NodeLibrary};

/// A single entity in the world.
///
/// `pos` is a full `WorldPos` (anchor cell + sub-cell offset),
/// so sub-cell motion shifts the entity's bbox continuously each
/// frame without jumping. `velocity` is in anchor-cell units per
/// second — 1.0 means "cross one anchor cell per second".
pub struct Entity {
    /// World position of the entity's bbox_min corner. Sub-cell
    /// offset carries the fine motion; anchor re-normalizes
    /// automatically when offset crosses a cell boundary.
    pub pos: WorldPos,
    /// Per-axis velocity in anchor-cell units per second.
    pub velocity: [f32; 3],
    /// Canonical voxel content. Shared across entities via library
    /// dedup. NodeLibrary refcount is incremented on spawn.
    pub subtree_root: NodeId,
    /// Per-entity clone-on-write override. `None` = use
    /// `subtree_root`; `Some` = this entity has been edited and
    /// owns its own NodeId chain.
    pub override_root: Option<NodeId>,
    /// Cached BFS idx into the GPU tree buffer for `active_root()`.
    /// Written by the upload path; invalid between a mutation and
    /// the next upload. 0 = "not yet packed" sentinel.
    pub bfs_idx: u32,
}

impl Entity {
    /// NodeId actually used for rendering/editing — the override if
    /// one exists, otherwise the shared subtree root.
    pub fn active_root(&self) -> NodeId {
        self.override_root.unwrap_or(self.subtree_root)
    }
}

/// Flat storage for all entities. Separate from the world tree;
/// lives on the App alongside `world: WorldState`.
pub struct EntityStore {
    pub entities: Vec<Entity>,
}

impl Default for EntityStore {
    fn default() -> Self { Self::new() }
}

impl EntityStore {
    pub fn new() -> Self { Self { entities: Vec::new() } }

    pub fn len(&self) -> usize { self.entities.len() }
    pub fn is_empty(&self) -> bool { self.entities.is_empty() }

    /// Spawn a new entity at `pos` with initial `velocity`, backed
    /// by `subtree_root`. Ref-counts the subtree in the library so
    /// it's not evicted while an entity references it.
    pub fn spawn(
        &mut self,
        library: &mut NodeLibrary,
        pos: WorldPos,
        velocity: [f32; 3],
        subtree_root: NodeId,
    ) -> u32 {
        library.ref_inc(subtree_root);
        let idx = self.entities.len() as u32;
        self.entities.push(Entity {
            pos,
            velocity,
            subtree_root,
            override_root: None,
            bfs_idx: 0,
        });
        idx
    }

    /// Remove all entities. Decrements library refs for every
    /// subtree/override so unused content can be reclaimed.
    pub fn clear(&mut self, library: &mut NodeLibrary) {
        for e in self.entities.drain(..) {
            library.ref_dec(e.subtree_root);
            if let Some(o) = e.override_root {
                library.ref_dec(o);
            }
        }
    }

    /// Replace entity `idx`'s override_root with `new_root`, updating
    /// library refcounts. Old override (if any) gets decremented;
    /// new one gets incremented.
    pub fn set_override(
        &mut self,
        library: &mut NodeLibrary,
        idx: u32,
        new_root: NodeId,
    ) {
        let e = &mut self.entities[idx as usize];
        library.ref_inc(new_root);
        if let Some(old) = e.override_root {
            library.ref_dec(old);
        }
        e.override_root = Some(new_root);
    }

    /// Advance every entity by `velocity * dt` in its anchor cell's
    /// local frame. `WorldPos::add_local` renormalizes so crossing a
    /// cell boundary steps the anchor and wraps the offset.
    ///
    /// When `surface_y` is `Some(y)` (flat worlds with a single
    /// ground plane), the Y component of velocity is dropped so
    /// entities can't drift off the surface over time. Spawn places
    /// them on sea level; this keeps them there without a per-tick
    /// snap (which would walk the anchor path twice per entity).
    /// `None` (fractal worlds) applies full 3-axis velocity and
    /// entities fly freely.
    ///
    /// CPU cost is O(N). At 10k entities × 60fps = 600k `add_local`
    /// calls/sec — trivial; rendering, not motion, is the bottleneck.
    pub fn tick(
        &mut self,
        library: &NodeLibrary,
        dt: f32,
        surface_y: Option<f32>,
    ) {
        let zero_y = surface_y.is_some();
        for e in &mut self.entities {
            let vy = if zero_y { 0.0 } else { e.velocity[1] };
            let delta = [
                e.velocity[0] * dt,
                vy * dt,
                e.velocity[2] * dt,
            ];
            e.pos.add_local(delta, library);
        }
    }
}
