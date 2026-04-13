//! Uniform-layer tree-walk renderer.
//!
//! Every frame, walk the content-addressed tree from the root down to
//! `CameraZoom.layer + DETAIL_DEPTH` (the target layer — see
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
use std::collections::HashMap as StdHashMap;
use crate::model::mesher::{
    bake_volume, bake_child_faces, merge_child_faces, merge_child_faces_raw,
    flatten_children, compose_children_meshes, build_sub_meshes,
    ChildClass, ChildFaces, FaceData,
};
use crate::model::BakedSubMesh;

use super::state::WorldState;
use super::tree::{
    slot_coords, slot_index, voxel_idx, NodeId,
    BRANCH_FACTOR, CHILDREN_PER_NODE, DETAIL_DEPTH, EMPTY_NODE, EMPTY_VOXEL,
    MAX_LAYER, NODE_VOXELS_PER_AXIS,
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
    /// Pre-baked mesh data for composition children, keyed by `NodeId`.
    /// Each entry holds per-voxel-type `FaceData` (pre-GPU vertex
    /// buffers). These are shared across parent entities via
    /// content-addressed dedup.
    pre_baked: StdHashMap<NodeId, StdHashMap<u8, FaceData>>,
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

// ---------------------------------------------------------- composition

/// Pre-bake a child node into raw `FaceData` per voxel type. Same
/// pipeline as `BakedNode::new_cold` but stops before GPU upload:
/// flatten children into 125³, greedy mesh, merge to `FaceData`.
fn pre_bake_child(world: &WorldState, node_id: NodeId) -> StdHashMap<u8, FaceData> {
    let node = world.library.get(node_id).expect("pre_bake: node missing");
    let Some(children) = node.children.as_ref() else {
        // Leaf node: bake its 25³ grid to FaceData directly.
        let voxels = node.voxels.clone();
        let faces = crate::model::mesher::bake_faces_raw(
            NODE_VOXELS_PER_AXIS as i32,
            &move |x, y, z| {
                if x < 0 || y < 0 || z < 0
                    || x >= NODE_VOXELS_PER_AXIS as i32
                    || y >= NODE_VOXELS_PER_AXIS as i32
                    || z >= NODE_VOXELS_PER_AXIS as i32
                { return None; }
                let v = voxels[voxel_idx(x as usize, y as usize, z as usize)];
                if v == EMPTY_VOXEL { None } else { Some(v) }
            },
        );
        return faces;
    };
    let child_ids: [NodeId; CHILDREN_PER_NODE] = **children;
    let child_class: Vec<ChildClass> = (0..CHILDREN_PER_NODE)
        .map(|slot| classify_child(world, child_ids[slot]))
        .collect();
    let children_voxels: Vec<Option<&[u8]>> = (0..CHILDREN_PER_NODE)
        .map(|slot| {
            if child_ids[slot] == EMPTY_NODE { None }
            else { Some(world.library.get(child_ids[slot])
                .expect("pre_bake: child missing").voxels.as_ref().as_slice()) }
        })
        .collect();
    let flat_grid = flatten_children(
        &children_voxels, &child_class,
        BRANCH_FACTOR, NODE_VOXELS_PER_AXIS, EMPTY_VOXEL,
    );
    let child_faces = bake_all_children(&flat_grid, &child_ids, &child_class);
    merge_child_faces_raw(&child_faces)
}

/// Compose a node's mesh by concatenating its 125 pre-baked
/// children's meshes with position offsets. Each child's mesh
/// covers [0, 125); the composed result covers [0, 625).
fn compose_node(
    world: &WorldState,
    node_id: NodeId,
    pre_baked: &mut StdHashMap<NodeId, StdHashMap<u8, FaceData>>,
    meshes: &mut Assets<Mesh>,
) -> Vec<BakedSubMesh> {
    let node = world.library.get(node_id).expect("compose: node missing");
    let children = node.children.as_ref().expect("compose: expected non-leaf");
    let child_ids: [NodeId; CHILDREN_PER_NODE] = **children;

    // Fast path: if ALL children share the same NodeId, the entity is
    // a uniform "tower" (e.g., all-solid underground). Instead of
    // composing 98 boundary children with 588 quads, emit a simple
    // box mesh: 6 quads for the 6 outer faces of the whole entity.
    // This reduces underground geometry by ~100× while keeping the
    // surface faces visible.
    let first_non_empty = child_ids.iter().find(|&&id| id != EMPTY_NODE);
    if let Some(&first_id) = first_non_empty {
        if child_ids.iter().all(|&id| id == first_id || id == EMPTY_NODE)
            && child_ids.iter().all(|&id| id == first_id)
        {
            // All 125 children are identical → uniform tower.
            // Bake as a simple volume from the node's own 25³ downsample.
            let voxels = node.voxels.clone();
            let grid_size = (BRANCH_FACTOR * NODE_VOXELS_PER_AXIS) as i32;
            let first_v = voxels[0];
            // If truly uniform (all same voxel), emit a single box.
            if voxels.iter().all(|&v| v == first_v) && first_v != EMPTY_VOXEL {
                return bake_volume(
                    grid_size,
                    |x, y, z| {
                        if x < 0 || y < 0 || z < 0
                            || x >= grid_size || y >= grid_size || z >= grid_size
                        { None } else { Some(first_v) }
                    },
                    meshes,
                );
            }
        }
    }

    // Classify children at the parent's level (using their 25³
    // downsampled grids). Interior-uniform children — those completely
    // surrounded by same-material siblings — are skipped because
    // their outer faces are fully occluded.
    let child_class: Vec<ChildClass> = (0..CHILDREN_PER_NODE)
        .map(|slot| classify_child(world, child_ids[slot]))
        .collect();

    // Pre-bake only children that will contribute visible faces.
    for slot in 0..CHILDREN_PER_NODE {
        let child_id = child_ids[slot];
        if child_id == EMPTY_NODE { continue; }
        match child_class[slot] {
            ChildClass::Empty => continue,
            ChildClass::Uniform(v) if v == EMPTY_VOXEL => continue,
            ChildClass::Uniform(v) if is_interior_uniform(slot, v, &child_class) => continue,
            _ => {}
        }
        if pre_baked.contains_key(&child_id) { continue; }
        let faces = pre_bake_child(world, child_id);
        pre_baked.insert(child_id, faces);
    }

    // Collect references for composition, skipping interior children.
    let children_faces: Vec<Option<&StdHashMap<u8, FaceData>>> = (0..CHILDREN_PER_NODE)
        .map(|slot| {
            let child_id = child_ids[slot];
            if child_id == EMPTY_NODE { return None; }
            match child_class[slot] {
                ChildClass::Empty => None,
                ChildClass::Uniform(v) if v == EMPTY_VOXEL => None,
                ChildClass::Uniform(v) if is_interior_uniform(slot, v, &child_class) => None,
                _ => pre_baked.get(&child_id),
            }
        })
        .collect();

    let child_mesh_size = BRANCH_FACTOR * NODE_VOXELS_PER_AXIS;
    compose_children_meshes(
        &children_faces,
        CHILDREN_PER_NODE,
        BRANCH_FACTOR,
        child_mesh_size,
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
    entity_scale: f32,
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
        let n = anchor.norm;
        let origin_bevy = Vec3::new(
            (origin_leaves[0] - anchor.leaf_coord[0]) as f32 / n,
            (origin_leaves[1] - anchor.leaf_coord[1]) as f32 / n,
            (origin_leaves[2] - anchor.leaf_coord[2]) as f32 / n,
        );
        let extent = extent_for_layer(depth) / n;
        let aabb_min = origin_bevy;
        let aabb_max = origin_bevy + Vec3::splat(extent);

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

        // Reached emit layer → emit (skip uniform-empty nodes).
        if depth == emit_layer {
            if world.library.get(node_id).map_or(false, |n| n.uniform_empty) {
                continue;
            }
            out.push(Visit {
                path,
                node_id,
                origin: origin_bevy,
                scale: entity_scale,
            });
            continue;
        }

        // Descend into children. If this node is already a leaf
        // (no children) we can't go deeper — emit it at its actual
        // layer instead.
        let Some(node) = world.library.get(node_id) else { continue };
        let Some(children) = node.children.as_ref() else {
            out.push(Visit {
                path,
                node_id,
                origin: origin_bevy,
                scale: entity_scale,
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
    // Composition: emit at target-2 (entity composes pre-baked child
    // meshes for full DETAIL_DEPTH detail). Falls back to target-1
    // when the tree isn't deep enough for composition.
    let use_composition = zoom.layer + DETAIL_DEPTH <= MAX_LAYER;
    let emit_layer = if use_composition {
        target_layer.saturating_sub(2)
    } else {
        target_layer.saturating_sub(1)
    };

    let radius_bevy = RADIUS_VIEW_CELLS * anchor.cell_bevy(zoom.layer);

    // Entity scale: mesh coordinates are in target-layer voxels, but
    // norm is based on (zoom+1). Scale converts mesh units to Bevy units.
    let entity_scale = scale_for_layer(target_layer) / anchor.norm;

    // If zoom changed, first pass, or forced rebuild, drop everything.
    if !render_state.initialised
        || render_state.last_zoom_layer != zoom.layer
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
        render_state.last_zoom_layer = zoom.layer;
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
        entity_scale,
        camera_pos,
        radius_bevy,
        &anchor,
        &mut walk_stack,
        &mut visits,
    );
    timings.walk_us = walk_start.elapsed().as_micros() as u64;
    timings.visit_count = visits.len();

    // Pre-bake: ensure every visited NodeId has a BakedNode.
    {
        let bake_start = bevy::platform::time::Instant::now();
        for v in visits.iter() {
            if render_state.baked.contains_key(&v.node_id) {
                render_state.path_node.insert(v.path, v.node_id);
                continue;
            }

            let node = world.library.get(v.node_id)
                .expect("render: node missing from library");

            if node.children.is_none() {
                // Leaf node.
                let merged = bake_leaf(&world, v.node_id, &mut meshes);
                render_state.baked.insert(v.node_id, BakedNode {
                    child_ids: [EMPTY_NODE; CHILDREN_PER_NODE],
                    child_class: Vec::new(),
                    flat_grid: Vec::new(),
                    child_faces: Vec::new(),
                    merged,
                });
            } else if use_composition {
                // Composition: compose from pre-baked children.
                let merged = compose_node(
                    &world, v.node_id,
                    &mut render_state.pre_baked, &mut meshes,
                );
                render_state.baked.insert(v.node_id, BakedNode {
                    child_ids: [EMPTY_NODE; CHILDREN_PER_NODE],
                    child_class: Vec::new(),
                    flat_grid: Vec::new(),
                    child_faces: Vec::new(),
                    merged,
                });
            } else {
                // Fallback: single-level flatten.
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
