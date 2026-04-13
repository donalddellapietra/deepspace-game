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
use crate::model::mesher::{
    bake_volume, bake_child_faces, merge_child_faces,
    flatten_children, ChildClass, ChildFaces,
};
use crate::model::BakedSubMesh;

use super::state::WorldState;
use super::tree::{
    slot_coords, slot_index, voxel_idx, NodeId,
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
    pub render_total_us: u64,
    pub walk_us: u64,
    pub bake_us: u64,
    pub reconcile_us: u64,
    pub visit_count: usize,
    pub group_count: usize,
    pub collision_us: u64,
}

#[derive(Resource, Default)]
pub struct RenderState {
    /// Cached baked data keyed by `NodeId`. Stores intermediate
    /// products (flat grid, per-child faces) alongside the merged
    /// GPU meshes so edits can incrementally re-bake only the dirty
    /// children instead of the full 125³ grid.
    baked: HashMap<NodeId, BakedNode>,
    /// Per-path tracking: remembers which `NodeId` each path last
    /// displayed. Used to find the old `BakedNode` for incremental
    /// diff when an edit changes the NodeId at a path.
    path_node: HashMap<SmallPath, NodeId>,
    /// Live entities, keyed by "path prefix" (a `SmallPath`).
    entities: HashMap<SmallPath, (Entity, NodeId, Vec3)>,
    /// Zoom layer the `entities` set was built for. If it changes,
    /// everything gets despawned and rebuilt.
    last_zoom_layer: u8,
    /// Whether we have done at least one render pass.
    initialised: bool,
    /// Set to true to force a full entity rebuild on the next frame.
    pub force_rebuild: bool,
    /// Overlay (NPC) entity tracking and mesh cache.
    pub overlay: super::overlay::OverlayState,
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

/// Cached bake data for one emit-level node.
struct BakedNode {
    child_ids: [NodeId; CHILDREN_PER_NODE],
    child_class: Vec<ChildClass>,
    flat_grid: Vec<u8>,
    child_faces: Vec<ChildFaces>,
    merged: Vec<BakedSubMesh>,
}

fn classify_child(world: &WorldState, child_id: NodeId) -> ChildClass {
    if child_id == EMPTY_NODE { return ChildClass::Empty; }
    let child = world.library.get(child_id).expect("render: child missing");
    let first = child.voxels[0];
    if child.voxels.iter().all(|&v| v == first) {
        ChildClass::Uniform(first)
    } else {
        ChildClass::Mixed
    }
}

fn is_interior_uniform(slot: usize, v: u8, child_class: &[ChildClass]) -> bool {
    let bf = BRANCH_FACTOR;
    let (sx, sy, sz) = slot_coords(slot);
    let neighbors: [(usize, usize, usize); 6] = [
        (sx.wrapping_sub(1), sy, sz), (sx + 1, sy, sz),
        (sx, sy.wrapping_sub(1), sz), (sx, sy + 1, sz),
        (sx, sy, sz.wrapping_sub(1)), (sx, sy, sz + 1),
    ];
    neighbors.iter().all(|&(nx, ny, nz)| {
        if nx >= bf || ny >= bf || nz >= bf { return false; }
        child_class[slot_index(nx, ny, nz)] == ChildClass::Uniform(v)
    })
}

fn make_get(flat: &[u8]) -> impl Fn(i32, i32, i32) -> Option<u8> + '_ {
    let size = (BRANCH_FACTOR * NODE_VOXELS_PER_AXIS) as i32;
    let sz = size as usize;
    move |x: i32, y: i32, z: i32| -> Option<u8> {
        if x < 0 || y < 0 || z < 0 || x >= size || y >= size || z >= size {
            return None;
        }
        let v = flat[(z as usize * sz + y as usize) * sz + x as usize];
        if v == EMPTY_VOXEL { None } else { Some(v) }
    }
}

fn bake_all_children(
    flat: &[u8],
    child_ids: &[NodeId; CHILDREN_PER_NODE],
    child_class: &[ChildClass],
) -> Vec<ChildFaces> {
    let get = make_get(flat);
    (0..CHILDREN_PER_NODE).map(|slot| {
        if child_ids[slot] == EMPTY_NODE { return Default::default(); }
        if let ChildClass::Uniform(v) = child_class[slot] {
            if v == EMPTY_VOXEL || is_interior_uniform(slot, v, child_class) {
                return Default::default();
            }
        }
        bake_child_faces(&get, slot, NODE_VOXELS_PER_AXIS as i32, BRANCH_FACTOR)
    }).collect()
}

fn mark_dirty(dirty: &mut [bool; CHILDREN_PER_NODE], slot: usize) {
    dirty[slot] = true;
    let bf = BRANCH_FACTOR;
    let (sx, sy, sz) = slot_coords(slot);
    for (dx, dy, dz) in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)] {
        let (nx, ny, nz) = (sx as isize+dx, sy as isize+dy, sz as isize+dz);
        if nx >= 0 && nx < bf as isize
            && ny >= 0 && ny < bf as isize
            && nz >= 0 && nz < bf as isize
        {
            dirty[slot_index(nx as usize, ny as usize, nz as usize)] = true;
        }
    }
}

fn patch_flat_region(flat: &mut [u8], voxels: Option<&[u8]>, class: ChildClass, slot: usize) {
    let cs = NODE_VOXELS_PER_AXIS;
    let size = BRANCH_FACTOR * cs;
    let (sx, sy, sz) = slot_coords(slot);
    let (bx, by, bz) = (sx * cs, sy * cs, sz * cs);
    match class {
        ChildClass::Empty => {
            for z in 0..cs { for y in 0..cs {
                let s = (bz+z)*size*size + (by+y)*size + bx;
                flat[s..s+cs].fill(EMPTY_VOXEL);
            }}
        }
        ChildClass::Uniform(v) => {
            for z in 0..cs { for y in 0..cs {
                let s = (bz+z)*size*size + (by+y)*size + bx;
                flat[s..s+cs].fill(v);
            }}
        }
        ChildClass::Mixed => {
            if let Some(vox) = voxels {
                for z in 0..cs { for y in 0..cs {
                    let d = (bz+z)*size*size + (by+y)*size + bx;
                    let s = z*cs*cs + y*cs;
                    flat[d..d+cs].copy_from_slice(&vox[s..s+cs]);
                }}
            }
        }
    }
}

impl BakedNode {
    /// Full build from scratch.
    fn new_cold(world: &WorldState, node_id: NodeId, meshes: &mut Assets<Mesh>) -> Self {
        let node = world.library.get(node_id).expect("render: node missing");
        let children = node.children.as_ref().expect("render: expected non-leaf");
        let child_ids: [NodeId; CHILDREN_PER_NODE] = **children;

        let child_class: Vec<ChildClass> = (0..CHILDREN_PER_NODE)
            .map(|slot| classify_child(world, child_ids[slot]))
            .collect();

        let children_voxels: Vec<Option<&[u8]>> = (0..CHILDREN_PER_NODE)
            .map(|slot| {
                if child_ids[slot] == EMPTY_NODE { None }
                else { Some(world.library.get(child_ids[slot])
                    .expect("render: child missing").voxels.as_ref().as_slice()) }
            })
            .collect();

        let flat_grid = flatten_children(
            &children_voxels, &child_class,
            BRANCH_FACTOR, NODE_VOXELS_PER_AXIS, EMPTY_VOXEL,
        );
        let child_faces = bake_all_children(&flat_grid, &child_ids, &child_class);
        let merged = merge_child_faces(&child_faces, meshes);

        BakedNode { child_ids, child_class, flat_grid, child_faces, merged }
    }

    /// Incremental build: clone old data, diff children, patch only dirty slots.
    fn new_incremental(
        old: &BakedNode, world: &WorldState, node_id: NodeId, meshes: &mut Assets<Mesh>,
    ) -> Self {
        let node = world.library.get(node_id).expect("render: node missing");
        let children = node.children.as_ref().expect("render: expected non-leaf");
        let new_child_ids: [NodeId; CHILDREN_PER_NODE] = **children;

        let mut dirty = [false; CHILDREN_PER_NODE];
        let mut any_changed = false;
        for slot in 0..CHILDREN_PER_NODE {
            if new_child_ids[slot] != old.child_ids[slot] {
                mark_dirty(&mut dirty, slot);
                any_changed = true;
            }
        }

        if !any_changed {
            let merged = merge_child_faces(&old.child_faces, meshes);
            return BakedNode {
                child_ids: old.child_ids, child_class: old.child_class.clone(),
                flat_grid: old.flat_grid.clone(), child_faces: old.child_faces.clone(),
                merged,
            };
        }

        let mut child_class = old.child_class.clone();
        let mut flat_grid = old.flat_grid.clone();
        let mut child_faces = old.child_faces.clone();

        for slot in 0..CHILDREN_PER_NODE {
            if new_child_ids[slot] != old.child_ids[slot] {
                child_class[slot] = classify_child(world, new_child_ids[slot]);
                let voxels = if new_child_ids[slot] == EMPTY_NODE { None }
                    else { Some(world.library.get(new_child_ids[slot])
                        .expect("render: child missing").voxels.as_ref().as_slice()) };
                patch_flat_region(&mut flat_grid, voxels, child_class[slot], slot);
            }
        }

        {
            let get = make_get(&flat_grid);
            for slot in 0..CHILDREN_PER_NODE {
                if !dirty[slot] { continue; }
                child_faces[slot] = if new_child_ids[slot] == EMPTY_NODE {
                    Default::default()
                } else if let ChildClass::Uniform(v) = child_class[slot] {
                    if v == EMPTY_VOXEL || is_interior_uniform(slot, v, &child_class) {
                        Default::default()
                    } else {
                        bake_child_faces(&get, slot, NODE_VOXELS_PER_AXIS as i32, BRANCH_FACTOR)
                    }
                } else {
                    bake_child_faces(&get, slot, NODE_VOXELS_PER_AXIS as i32, BRANCH_FACTOR)
                };
            }
        }

        let merged = merge_child_faces(&child_faces, meshes);
        BakedNode { child_ids: new_child_ids, child_class, flat_grid, child_faces, merged }
    }
}

fn bake_leaf(world: &WorldState, node_id: NodeId, meshes: &mut Assets<Mesh>) -> Vec<BakedSubMesh> {
    let voxels = world.library.get(node_id).expect("render: leaf missing").voxels.clone();
    bake_volume(
        NODE_VOXELS_PER_AXIS as i32,
        move |x, y, z| {
            if x < 0 || y < 0 || z < 0
                || x >= NODE_VOXELS_PER_AXIS as i32
                || y >= NODE_VOXELS_PER_AXIS as i32
                || z >= NODE_VOXELS_PER_AXIS as i32
            { return None; }
            let v = voxels[voxel_idx(x as usize, y as usize, z as usize)];
            if v == EMPTY_VOXEL { None } else { Some(v) }
        },
        meshes,
    )
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
        let n = anchor.norm;
        let origin_bevy = Vec3::new(
            (origin_leaves[0] - anchor.leaf_coord[0]) as f32 / n,
            (origin_leaves[1] - anchor.leaf_coord[1]) as f32 / n,
            (origin_leaves[2] - anchor.leaf_coord[2]) as f32 / n,
        );
        let extent = extent_for_layer(depth) / n;
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
                scale: 1.0,
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
                scale: 1.0,
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
    overlay_list: Res<super::overlay::OverlayList>,
) {
    let Some(palette) = palette else {
        return;
    };
    let Ok(camera_tf) = camera_q.single() else {
        return;
    };
    let camera_pos = camera_tf.translation;
    let render_total_start = bevy::platform::time::Instant::now();

    let target_layer = target_layer_for(zoom.layer);
    let emit_layer = target_layer.saturating_sub(1);

    // Render radius in Bevy units = N cells × Bevy-units-per-cell at
    // the current view layer. This is what the 2D prototype does
    // implicitly (its viewport is measured in cells); without it,
    // zooming out collapses the visible world to a dot because the
    // per-cell Bevy size grows by 5× per layer.
    let radius_bevy = RADIUS_VIEW_CELLS * anchor.cell_bevy(zoom.layer);

    // If emit layer changed, first pass, or forced rebuild, drop everything.
    if !render_state.initialised
        || render_state.last_zoom_layer != emit_layer
        || render_state.force_rebuild
    {
        render_state.force_rebuild = false;
        for (_, (entity, _, _)) in render_state.entities.drain() {
            if let Ok(mut ec) = commands.get_entity(entity) {
                ec.despawn();
            }
        }
        super::overlay::clear_overlay_entities(&mut commands, &mut render_state.overlay);
        render_state.baked.clear();
        render_state.path_node.clear();
        render_state.last_zoom_layer = emit_layer;
        render_state.initialised = true;
    }

    // Walk the tree → collect visible positions.
    timings.bake_us = 0;
    let walk_start = bevy::platform::time::Instant::now();
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

    // Pre-bake: ensure every visited NodeId has a BakedNode.
    // On edit, use the old BakedNode for incremental diff.
    {
        let bake_start = bevy::platform::time::Instant::now();
        for v in visits.iter() {
            let node = world.library.get(v.node_id)
                .expect("render: node missing from library");

            if node.children.is_some() {
                if !render_state.baked.contains_key(&v.node_id) {
                    let baked = if let Some(&old_nid) = render_state.path_node.get(&v.path) {
                        if let Some(old_bake) = render_state.baked.get(&old_nid) {
                            BakedNode::new_incremental(old_bake, &world, v.node_id, &mut meshes)
                        } else {
                            BakedNode::new_cold(&world, v.node_id, &mut meshes)
                        }
                    } else {
                        BakedNode::new_cold(&world, v.node_id, &mut meshes)
                    };
                    render_state.baked.insert(v.node_id, baked);
                }
                render_state.path_node.insert(v.path, v.node_id);
            } else if !render_state.baked.contains_key(&v.node_id) {
                let merged = bake_leaf(&world, v.node_id, &mut meshes);
                render_state.baked.insert(v.node_id, BakedNode {
                    child_ids: [EMPTY_NODE; CHILDREN_PER_NODE],
                    child_class: Vec::new(),
                    flat_grid: Vec::new(),
                    child_faces: Vec::new(),
                    merged,
                });
            }
        }
        timings.bake_us = bake_start.elapsed().as_micros() as u64;
    }

    // Reconcile: what's alive now, what changed, what's gone.
    let reconcile_start = bevy::platform::time::Instant::now();
    let mut alive: HashMap<SmallPath, (Entity, NodeId, Vec3)> =
        HashMap::with_capacity(visits.len());

    for visit in visits.drain(..) {
        let new_node_id = visit.node_id;
        let existing = render_state.entities.remove(&visit.path);

        match existing {
            Some((entity, existing_id, last_origin))
                if existing_id == new_node_id =>
            {
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

                let baked = render_state.baked.get(&new_node_id)
                    .map(|b| &b.merged[..])
                    .unwrap_or(&[]);

                let parent = commands
                    .spawn((
                        WorldRenderedNode(new_node_id),
                        Transform::from_translation(visit.origin)
                            .with_scale(Vec3::splat(visit.scale)),
                        Visibility::Visible,
                    ))
                    .id();

                for sub in baked {
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

    for (_, (entity, _, _)) in render_state.entities.drain() {
        if let Ok(mut ec) = commands.get_entity(entity) {
            ec.despawn();
        }
    }
    render_state.entities = alive;

    // Overlay reconcile (NPCs and other overlay subtrees).
    super::overlay::reconcile_overlays(
        &mut commands,
        &world,
        &palette,
        &mut meshes,
        &overlay_list,
        &mut render_state.overlay,
    );

    timings.reconcile_us = reconcile_start.elapsed().as_micros() as u64;
    timings.render_total_us = render_total_start.elapsed().as_micros() as u64;

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
        WorldAnchor { leaf_coord: [0, 0, 0], norm: 1.0 }
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
