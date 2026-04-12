//! Uniform-layer tree-walk renderer.
//!
//! Every frame, walk the content-addressed tree from the root down to
//! `CameraZoom.layer + 2` (the target layer — see
//! `view::target_layer_for`) and emit one Bevy entity per surviving
//! node at that layer. Entities are reused across frames via
//! `RenderState`, and meshes are baked lazily into a `NodeId`-keyed
//! cache. See `docs/architecture/rendering.md` for the full design.
//!
//! Key invariants:
//!
//! * One Bevy unit equals one leaf voxel *in the current
//!   [`WorldAnchor`] frame*. A leaf entity has scale `1.0` and its
//!   baked mesh is `bake_volume(NODE_VOXELS_PER_AXIS)`. A layer-K
//!   node has scale `5 ^ (MAX_LAYER - K)`.
//! * Node origins are computed as `leaf-coord - anchor.leaf_coord`,
//!   so rendered entities live in a small Bevy coordinate range
//!   around the player regardless of absolute world position. This
//!   is the "no big numbers in Bevy space" guarantee from
//!   `docs/architecture/coordinates.md`.
//! * The renderer skips frustum culling. It emits every node whose
//!   AABB intersects a sphere of `RADIUS_VIEW_CELLS` view-cells
//!   around the camera.

use bevy::ecs::hierarchy::ChildOf;
use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::block::Palette;
use crate::model::{mesher::bake_volume, BakedSubMesh};

use super::state::WorldState;
use super::tree::{
    slot_coords, slot_index, voxel_idx, NodeId, VoxelGrid,
    BRANCH_FACTOR, CHILDREN_PER_NODE, EMPTY_NODE, EMPTY_VOXEL, MAX_LAYER,
    NODE_VOXELS_PER_AXIS,
};
use super::view::{
    cell_size_at_layer, extent_for_layer, scale_for_layer, target_layer_for,
    WorldAnchor,
};

// ------------------------------------------------------- markers

/// Marker attached to the *parent* of each rendered node, carrying
/// the `NodeId` the entity is representing. Save-mode tinting looks
/// up entities by this component rather than keying into the
/// private `RenderState.entities` map.
#[derive(Component)]
pub struct WorldRenderedNode(pub NodeId);

/// Marker attached to each *child* sub-mesh entity, remembering its
/// canonical voxel index so callers that temporarily swap the
/// `MeshMaterial3d` (save-mode tinting) can restore the original
/// material without re-querying the library.
#[derive(Component)]
pub struct SubMeshBlock(pub u8);

// --------------------------------------------------------------- camera zoom

#[derive(Resource)]
pub struct CameraZoom {
    /// Which tree layer the camera renders. Clamped to
    /// `MIN_ZOOM..=MAX_ZOOM`.
    pub layer: u8,
}

pub const MIN_ZOOM: u8 = 2;
pub const MAX_ZOOM: u8 = MAX_LAYER;

impl Default for CameraZoom {
    fn default() -> Self {
        // Start at the leaf layer: that's where one Bevy unit equals one
        // voxel and the world is readable without any up-level scaling.
        Self { layer: MAX_LAYER }
    }
}

impl CameraZoom {
    pub fn zoom_in(&mut self) -> bool {
        if self.layer < MAX_ZOOM {
            self.layer += 1;
            true
        } else {
            false
        }
    }
    pub fn zoom_out(&mut self) -> bool {
        if self.layer > MIN_ZOOM {
            self.layer -= 1;
            true
        } else {
            false
        }
    }
}

// ---------------------------------------------------------------- constants

/// How far a rendered node's centre may be from the camera, measured
/// in **cells at the current view layer**. Used as the v1 replacement
/// for frustum culling. At walk time this is multiplied by
/// `cell_size_at_layer(view_layer)` so that the render distance scales
/// with zoom: you see the same number of cells out to the horizon
/// whether you're at the leaves or zoomed all the way out, matching
/// the 2D prototype's "viewport counts cells, not pixels" behaviour.
///
/// At view layer `L`, one visible cell is one target-layer node
/// (target = `(L + 2).min(MAX_LAYER)`), so N cells of radius emit at
/// most roughly `(2N)^3` target-layer nodes — keep modest.
pub const RADIUS_VIEW_CELLS: f32 = 32.0;

// ----------------------------------------------------------------- state

/// Per-frame timing data exposed to the diagnostics HUD.
#[derive(Resource, Default)]
pub struct RenderTimings {
    pub walk_us: u64,
    pub reconcile_us: u64,
    pub visit_count: usize,
    pub group_count: usize,
    pub collision_us: u64,
    pub triangles: usize,
}

/// A map from tree path (of length `depth`) to the Bevy entity
/// currently rendering that path. `NodeId` tracks what the entity's
/// mesh is a function of, so edits naturally invalidate it.
#[derive(Resource, Default)]
pub struct RenderState {
    /// Cached per-`NodeId` baked sub-meshes. An entry survives across
    /// frames and zoom layers — the mesh is a function of the node's
    /// voxel grid, not of which layer it was emitted at. v1 never
    /// evicts these.
    meshes: HashMap<NodeId, Vec<BakedSubMesh>>,
    /// Live entities, keyed by "path prefix" (a `SmallPath`).
    /// The `Vec3` is the last-written Bevy translation so we can skip
    /// redundant `Transform` inserts when the anchor hasn't moved.
    entities: HashMap<SmallPath, (Entity, NodeId, Vec3)>,
    /// Zoom layer the `entities` set was built for. If it changes,
    /// everything gets despawned and rebuilt.
    last_zoom_layer: u8,
    /// Whether we have done at least one render pass.
    initialised: bool,
    /// Reusable DFS stack for `walk()`. Stashed here so we don't
    /// reallocate a `Vec` every frame. Cleared at the start of each
    /// `walk()` call.
    walk_stack: Vec<WalkFrame>,
    /// Reusable buffer for the target-layer visits collected by
    /// `walk()`. Same motivation as `walk_stack`.
    visits: Vec<Visit>,
}

/// A compact identifier for a node's position in the tree during a
/// single-frame walk: `depth` significant slot indices from the root.
/// Used as the `RenderState.entities` key, so reuse survives across
/// frames as long as the camera keeps looking at the same spot.
#[derive(Clone, Copy, Eq, PartialEq, Hash)]
struct SmallPath {
    depth: u8,
    slots: [u8; MAX_LAYER as usize],
}

impl SmallPath {
    fn empty() -> Self {
        Self {
            depth: 0,
            slots: [0; MAX_LAYER as usize],
        }
    }

    fn push(&self, slot: u8) -> Self {
        let mut out = *self;
        out.slots[out.depth as usize] = slot;
        out.depth += 1;
        out
    }
}

// ----------------------------------------------------------- mesh caching

fn get_or_bake_mesh<'a>(
    render_state: &'a mut RenderState,
    world: &WorldState,
    node_id: NodeId,
    meshes: &mut Assets<Mesh>,
) -> &'a [BakedSubMesh] {
    render_state.get_or_bake(world, node_id, meshes)
}

impl RenderState {
    /// Fetch the baked sub-meshes for `node_id`, baking and caching
    /// on a miss. Internal helper for the renderer's reconciler.
    fn get_or_bake<'a>(
        &'a mut self,
        world: &WorldState,
        node_id: NodeId,
        meshes: &mut Assets<Mesh>,
    ) -> &'a [BakedSubMesh] {
        if !self.meshes.contains_key(&node_id) {
            let node = world
                .library
                .get(node_id)
                .expect("render: node missing from library");
            let baked = if let Some(children) = &node.children {
                // Non-leaf: bake 125³ mesh from children's voxel grids.
                let child_voxels: Vec<Option<VoxelGrid>> = children
                    .iter()
                    .map(|&id| {
                        if id == EMPTY_NODE {
                            None
                        } else {
                            Some(
                                world
                                    .library
                                    .get(id)
                                    .expect("render: child missing from library")
                                    .voxels
                                    .clone(),
                            )
                        }
                    })
                    .collect();
                let size = (BRANCH_FACTOR * NODE_VOXELS_PER_AXIS) as i32;
                bake_volume(
                    size,
                    move |x, y, z| {
                        if x < 0
                            || y < 0
                            || z < 0
                            || x >= size
                            || y >= size
                            || z >= size
                        {
                            return None;
                        }
                        let xu = x as usize;
                        let yu = y as usize;
                        let zu = z as usize;
                        let slot = slot_index(
                            xu / NODE_VOXELS_PER_AXIS,
                            yu / NODE_VOXELS_PER_AXIS,
                            zu / NODE_VOXELS_PER_AXIS,
                        );
                        let voxels = child_voxels[slot].as_ref()?;
                        let v = voxels[voxel_idx(
                            xu % NODE_VOXELS_PER_AXIS,
                            yu % NODE_VOXELS_PER_AXIS,
                            zu % NODE_VOXELS_PER_AXIS,
                        )];
                        if v == EMPTY_VOXEL { None } else { Some(v) }
                    },
                    meshes,
                )
            } else {
                // Leaf: bake 25³ from own voxels.
                let voxels = node.voxels.clone();
                bake_volume(
                    NODE_VOXELS_PER_AXIS as i32,
                    move |x, y, z| {
                        if x < 0
                            || y < 0
                            || z < 0
                            || x >= NODE_VOXELS_PER_AXIS as i32
                            || y >= NODE_VOXELS_PER_AXIS as i32
                            || z >= NODE_VOXELS_PER_AXIS as i32
                        {
                            return None;
                        }
                        let v = voxels[voxel_idx(
                            x as usize,
                            y as usize,
                            z as usize,
                        )];
                        if v == EMPTY_VOXEL { None } else { Some(v) }
                    },
                    meshes,
                )
            };
            self.meshes.insert(node_id, baked);
        }
        self.meshes
            .get(&node_id)
            .expect("just inserted")
            .as_slice()
    }
}

// ------------------------------------------------------------- tree walk

/// One "visit" the tree walk wants the reconciler to spawn/update.
/// `origin` is anchor-relative — already the final Bevy `Transform`
/// translation the renderer will give the entity.
struct Visit {
    path: SmallPath,
    node_id: NodeId,
    origin: Vec3,
    scale: f32,
}

/// One frame on the `walk()` DFS stack. Extracted into a named
/// struct so `Vec<WalkFrame>` is a nameable type on `RenderState`.
struct WalkFrame {
    node_id: NodeId,
    path: SmallPath,
    origin_leaves: [i64; 3],
    depth: u8,
}

/// Accumulate each node's absolute leaf-space origin as the walker
/// descends, then convert to a Bevy `Vec3` relative to the camera
/// anchor only when the node passes the cull / emit test. Tracking
/// the origin in `i64` is what keeps `f32` precision small even
/// when the player is billions of leaves deep inside the root — the
/// subtraction `(node_coord - anchor_coord)` stays exact in integer
/// space, and the cast to `f32` only ever happens on the small
/// difference.
fn walk(
    world: &WorldState,
    emit_layer: u8,
    target_layer: u8,
    camera_pos: Vec3,
    radius_bevy: f32,
    anchor: &WorldAnchor,
    stack: &mut Vec<WalkFrame>,
    out: &mut Vec<Visit>,
) {
    stack.clear();
    out.clear();
    if world.root == EMPTY_NODE {
        return;
    }
    // Precomputed child extents in leaves, indexed by the child's
    // layer number (layer 1 = root's direct children, layer MAX = leaves).
    // Root leaf-extent is `25 * 5^MAX_LAYER`; each descent divides by 5.
    let mut child_extent_leaves: [i64; MAX_LAYER as usize + 1] =
        [0; MAX_LAYER as usize + 1];
    {
        let mut ext: i64 = super::state::world_extent_voxels();
        child_extent_leaves[0] = ext;
        for layer in 1..=(MAX_LAYER as usize) {
            ext /= 5;
            child_extent_leaves[layer] = ext;
        }
    }

    stack.push(WalkFrame {
        node_id: world.root,
        path: SmallPath::empty(),
        origin_leaves: [0; 3],
        depth: 0,
    });

    let radius_sq = radius_bevy * radius_bevy;

    while let Some(frame) = stack.pop() {
        let WalkFrame { node_id, path, origin_leaves, depth } = frame;
        // Bevy-space origin of this node: delta from the anchor, in
        // leaves, cast to `f32`. For nodes near the player the delta
        // is small and `f32` is essentially exact.
        let origin_bevy = Vec3::new(
            (origin_leaves[0] - anchor.leaf_coord[0]) as f32,
            (origin_leaves[1] - anchor.leaf_coord[1]) as f32,
            (origin_leaves[2] - anchor.leaf_coord[2]) as f32,
        );
        let extent = extent_for_layer(depth);
        let aabb_min = origin_bevy;
        let aabb_max = origin_bevy + Vec3::splat(extent);

        // Per-axis "distance from camera to AABB": 0 if inside,
        // otherwise the gap to the nearest face. L2 norm of the three
        // gaps is the minimum distance from the camera point to the
        // AABB. Much more accurate than a centre-based sphere test
        // when the AABB is very large and contains the camera.
        let dx = (aabb_min.x - camera_pos.x)
            .max(0.0)
            .max(camera_pos.x - aabb_max.x);
        let dy = (aabb_min.y - camera_pos.y)
            .max(0.0)
            .max(camera_pos.y - aabb_max.y);
        let dz = (aabb_min.z - camera_pos.z)
            .max(0.0)
            .max(camera_pos.z - aabb_max.z);
        let min_dist_sq = dx * dx + dy * dy + dz * dz;
        if min_dist_sq > radius_sq {
            continue;
        }

        // Reached the emit layer → emit.
        if depth == emit_layer {
            out.push(Visit {
                path,
                node_id,
                origin: origin_bevy,
                scale: scale_for_layer(target_layer),
            });
            continue;
        }

        // Descend into children. If this node is already a leaf
        // (no children) we can't go deeper — emit it at its actual
        // layer instead. This shouldn't happen in a fully-materialised
        // grassland world but is safe.
        let Some(node) = world.library.get(node_id) else { continue };
        let Some(children) = node.children.as_ref() else {
            out.push(Visit {
                path,
                node_id,
                origin: origin_bevy,
                scale: scale_for_layer(depth),
            });
            continue;
        };

        let child_extent_i64 = child_extent_leaves[(depth + 1) as usize];
        for slot in 0..CHILDREN_PER_NODE {
            let child_id = children[slot];
            if child_id == EMPTY_NODE {
                continue;
            }
            let (sx, sy, sz) = slot_coords(slot);
            let child_origin_leaves = [
                origin_leaves[0] + (sx as i64) * child_extent_i64,
                origin_leaves[1] + (sy as i64) * child_extent_i64,
                origin_leaves[2] + (sz as i64) * child_extent_i64,
            ];
            let child_path = path.push(slot as u8);
            stack.push(WalkFrame {
                node_id: child_id,
                path: child_path,
                origin_leaves: child_origin_leaves,
                depth: depth + 1,
            });
        }
    }
}

// ----------------------------------------------------------------- system

/// Bevy system: walk the tree, reconcile `RenderState` entities.
pub fn render_world(
    mut commands: Commands,
    world: Res<WorldState>,
    zoom: Res<CameraZoom>,
    anchor: Res<WorldAnchor>,
    camera_q: Query<&Transform, With<Camera3d>>,
    palette: Option<Res<Palette>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut render_state: ResMut<RenderState>,
    mut timings: ResMut<RenderTimings>,
) {
    let Some(palette) = palette else {
        return;
    };
    let Ok(camera_tf) = camera_q.single() else {
        return;
    };
    let camera_pos = camera_tf.translation;

    let target_layer = target_layer_for(zoom.layer);
    let emit_layer = target_layer.saturating_sub(1);

    // Render radius in Bevy units = N cells × Bevy-units-per-cell at
    // the current view layer. This is what the 2D prototype does
    // implicitly (its viewport is measured in cells); without it,
    // zooming out collapses the visible world to a dot because the
    // per-cell Bevy size grows by 5× per layer.
    let radius_bevy = RADIUS_VIEW_CELLS * cell_size_at_layer(zoom.layer);

    // If emit layer changed, or we're on our first pass, drop everything.
    if !render_state.initialised || render_state.last_zoom_layer != emit_layer
    {
        for (_, (entity, _, _)) in render_state.entities.drain() {
            if let Ok(mut ec) = commands.get_entity(entity) {
                ec.despawn();
            }
        }
        render_state.last_zoom_layer = emit_layer;
        render_state.initialised = true;
    }

    // Walk the tree → collect visible positions. `mem::take` the
    // reusable buffers off `render_state` so the walk and reconcile
    // loop can hold `&mut render_state` freely (notably
    // `get_or_bake_mesh`). We put them back at the end; the buffers
    // are cleared internally by `walk()` and drained during
    // reconciliation, so they retain their allocated capacity.
    let walk_start = std::time::Instant::now();
    let mut walk_stack = std::mem::take(&mut render_state.walk_stack);
    let mut visits = std::mem::take(&mut render_state.visits);
    walk(
        &world,
        emit_layer,
        target_layer,
        camera_pos,
        radius_bevy,
        &anchor,
        &mut walk_stack,
        &mut visits,
    );
    timings.walk_us = walk_start.elapsed().as_micros() as u64;
    timings.visit_count = visits.len();

    // Count triangles: for each visit, sum the index counts of its
    // baked sub-meshes. This is approximate (counts all visits, not
    // just those surviving frustum culling) but cheap.
    {
        let mut tri_count = 0usize;
        for v in visits.iter() {
            if let Some(baked) = render_state.meshes.get(&v.node_id) {
                for sub in baked {
                    if let Some(mesh) = meshes.get(&sub.mesh) {
                        tri_count += mesh.count_vertices() / 3;
                    }
                }
            }
        }
        timings.triangles = tri_count;
    }

    // Reconcile: what's alive now, what changed, what's gone.
    let reconcile_start = std::time::Instant::now();
    let mut alive: HashMap<SmallPath, (Entity, NodeId, Vec3)> =
        HashMap::with_capacity(visits.len());

    for visit in visits.drain(..) {
        let new_node_id = visit.node_id;
        let existing = render_state.entities.remove(&visit.path);

        match existing {
            Some((entity, existing_id, last_origin))
                if existing_id == new_node_id =>
            {
                // Reuse. Only touch the Transform when the origin
                // actually changed (anchor shifted). Skipping this
                // avoids ~2k Bevy command dispatches per frame in
                // the common stationary-camera case.
                if last_origin != visit.origin {
                    if let Ok(mut ec) = commands.get_entity(entity) {
                        ec.insert(
                            Transform::from_translation(visit.origin)
                                .with_scale(Vec3::splat(visit.scale)),
                        );
                    }
                }
                alive.insert(visit.path, (entity, existing_id, visit.origin));
            }
            other => {
                if let Some((old_entity, _, _)) = other {
                    if let Ok(mut ec) = commands.get_entity(old_entity) {
                        ec.despawn();
                    }
                }
                // Spawn a new root entity for this path, parent one
                // child per sub-mesh.
                let baked = get_or_bake_mesh(
                    &mut render_state,
                    &world,
                    new_node_id,
                    &mut meshes,
                )
                .to_vec();

                let parent = commands
                    .spawn((
                        WorldRenderedNode(new_node_id),
                        Transform::from_translation(visit.origin)
                            .with_scale(Vec3::splat(visit.scale)),
                        Visibility::Visible,
                    ))
                    .id();

                for sub in &baked {
                    let Some(mat) = palette.material(sub.voxel) else {
                        continue;
                    };
                    commands.spawn((
                        Mesh3d(sub.mesh.clone()),
                        MeshMaterial3d(mat),
                        SubMeshBlock(sub.voxel),
                        Transform::default(),
                        Visibility::Inherited,
                        ChildOf(parent),
                    ));
                }

                alive.insert(visit.path, (parent, new_node_id, visit.origin));
            }
        }
    }

    // Everything left in the old map was NOT visited this frame →
    // despawn.
    for (_, (entity, _, _)) in render_state.entities.drain() {
        if let Ok(mut ec) = commands.get_entity(entity) {
            ec.despawn();
        }
    }
    render_state.entities = alive;

    timings.reconcile_us = reconcile_start.elapsed().as_micros() as u64;

    render_state.walk_stack = walk_stack;
    render_state.visits = visits;
}

// ------------------------------------------------------------------ tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_path_push() {
        let p = SmallPath::empty().push(7).push(12);
        assert_eq!(p.depth, 2);
        assert_eq!(p.slots[0], 7);
        assert_eq!(p.slots[1], 12);
    }

    fn anchor_origin() -> WorldAnchor {
        WorldAnchor { leaf_coord: [0, 0, 0] }
    }

    #[test]
    fn walk_grassland_at_leaves_emits_at_least_one_visit() {
        let world = WorldState::new_grassland();
        let mut stack = Vec::new();
        let mut visits = Vec::new();
        let target = target_layer_for(MAX_LAYER);
        let emit = target.saturating_sub(1);
        walk(
            &world,
            emit,
            target,
            Vec3::ZERO,
            RADIUS_VIEW_CELLS * cell_size_at_layer(MAX_LAYER),
            &anchor_origin(),
            &mut stack,
            &mut visits,
        );
        assert!(
            !visits.is_empty(),
            "grassland walk at leaf layer should emit at least one visit"
        );
    }

    #[test]
    fn walk_radius_limits_emit_count() {
        let world = WorldState::new_grassland();
        let mut stack = Vec::new();
        let mut visits = Vec::new();
        let target = target_layer_for(MAX_LAYER);
        let emit = target.saturating_sub(1);
        walk(
            &world,
            emit,
            target,
            Vec3::ZERO,
            RADIUS_VIEW_CELLS * cell_size_at_layer(MAX_LAYER),
            &anchor_origin(),
            &mut stack,
            &mut visits,
        );
        assert!(
            visits.len() < 200_000,
            "walk emitted {} visits; expected far fewer",
            visits.len()
        );
    }

    #[test]
    fn walk_radius_scales_with_view_layer() {
        // At every view layer the walk should emit a non-zero count
        // of emit-layer nodes.
        let world = WorldState::new_grassland();
        let anchor = anchor_origin();
        let mut counts = Vec::new();
        let mut stack = Vec::new();
        for view_layer in (MIN_ZOOM..=MAX_ZOOM).rev() {
            let target = target_layer_for(view_layer);
            let emit = target.saturating_sub(1);
            let radius = RADIUS_VIEW_CELLS * cell_size_at_layer(view_layer);
            let mut visits = Vec::new();
            walk(
                &world,
                emit,
                target,
                Vec3::ZERO,
                radius,
                &anchor,
                &mut stack,
                &mut visits,
            );
            counts.push((view_layer, visits.len()));
        }
        for &(view_layer, n) in &counts {
            assert!(
                n > 0,
                "view layer {view_layer}: walk emitted 0 visits — radius too small at this zoom?"
            );
        }
    }
}
