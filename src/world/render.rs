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
    /// Cached baked data keyed by `(NodeId, [bool; 6])`. The bool
    /// array records which border slabs were available at bake time,
    /// so nodes at different positions (with different neighbor
    /// availability) get separate meshes. Underground, all interior
    /// nodes share `[true; 6]` → one mesh.
    baked: HashMap<(NodeId, [bool; 6]), BakedNode>,
    /// Per-path tracking: remembers the full cache key each path
    /// last used, for incremental diff on edit.
    path_key: HashMap<SmallPath, (NodeId, [bool; 6])>,
    /// Live entities, keyed by "path prefix" (a `SmallPath`).
    entities: HashMap<SmallPath, (Entity, NodeId, Vec3)>,
    /// Zoom layer the `entities` set was built for. If it changes,
    /// everything gets despawned and rebuilt.
    last_zoom_layer: u8,
    /// Last zoom.layer value, for debug logging on change.
    last_view_layer: u8,
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

/// 6 border slabs from neighboring emit-level nodes. Each slab is a
/// 1-voxel-thick 125×125 plane from the neighbor's face adjacent to
/// ours, stored as a flat `[u8; 125*125]`. `None` = no neighbor.
/// Order: -x, +x, -y, +y, -z, +z.
type BorderSlabs = [Option<Box<[u8]>>; 6];

/// Extract a 1-voxel-thick slab from a neighbor node's boundary.
/// `axis` (0=x, 1=y, 2=z) and `at_max` (false=min face, true=max
/// face) specify which face of the neighbor to sample.
fn extract_border_slab(
    world: &WorldState,
    neighbor_id: NodeId,
    axis: usize,
    at_max: bool,
) -> Option<Box<[u8]>> {
    if neighbor_id == EMPTY_NODE { return None; }
    let node = world.library.get(neighbor_id)?;
    let children = node.children.as_ref()?;

    let cs = NODE_VOXELS_PER_AXIS;
    let size = BRANCH_FACTOR * cs;
    // The coordinate on `axis` to sample.
    let fixed = if at_max { size - 1 } else { 0 };
    let mut slab = vec![EMPTY_VOXEL; size * size].into_boxed_slice();

    // Which child column contains `fixed` on this axis?
    let child_idx_on_axis = fixed / cs;
    let local_on_axis = fixed % cs;

    // Iterate the 5×5 children on the other two axes.
    let bf = BRANCH_FACTOR;
    for ca in 0..bf {
        for cb in 0..bf {
            let (cx, cy, cz) = match axis {
                0 => (child_idx_on_axis, ca, cb),
                1 => (ca, child_idx_on_axis, cb),
                _ => (ca, cb, child_idx_on_axis),
            };
            let child_slot = slot_index(cx, cy, cz);
            let child_id = children[child_slot];
            if child_id == EMPTY_NODE { continue; }
            let child_node = world.library.get(child_id)?;
            let voxels = child_node.voxels.as_ref().as_slice();

            // Copy the relevant slice from this child's 25³ grid.
            let base_a = ca * cs;
            let base_b = cb * cs;
            for da in 0..cs {
                for db in 0..cs {
                    let (lx, ly, lz) = match axis {
                        0 => (local_on_axis, da, db),
                        1 => (da, local_on_axis, db),
                        _ => (da, db, local_on_axis),
                    };
                    let v = voxels[lz * cs * cs + ly * cs + lx];
                    // Map (axis=fixed, a=base_a+da, b=base_b+db) to
                    // the 2D slab indexed by the two non-axis coords.
                    let (sa, sb) = match axis {
                        0 => (base_a + da, base_b + db), // slab indexed by (y, z)
                        1 => (base_a + da, base_b + db), // slab indexed by (x, z)
                        _ => (base_a + da, base_b + db), // slab indexed by (x, y)
                    };
                    slab[sb * size + sa] = v;
                }
            }
        }
    }
    Some(slab)
}

/// Cheaply check which of the 6 neighbors exist (without extracting voxels).
fn compute_border_existence(
    world: &WorldState,
    path: &SmallPath,
    emit_depth: u8,
) -> [bool; 6] {
    if emit_depth == 0 { return [false; 6]; }
    let mut parent_id = world.root;
    for i in 0..(emit_depth as usize - 1) {
        let Some(node) = world.library.get(parent_id) else { return [false; 6] };
        let Some(ch) = node.children.as_ref() else { return [false; 6] };
        parent_id = ch[path.slots[i] as usize];
        if parent_id == EMPTY_NODE { return [false; 6] }
    }
    let Some(parent) = world.library.get(parent_id) else { return [false; 6] };
    let Some(parent_ch) = parent.children.as_ref() else { return [false; 6] };

    let slot = path.slots[emit_depth as usize - 1] as usize;
    let (sx, sy, sz) = slot_coords(slot);
    let bf = BRANCH_FACTOR;

    let mut ancestor_ids: Vec<NodeId> = Vec::new();
    // Rebuild ancestor path for grandparent lookup if needed.
    if emit_depth >= 2 {
        ancestor_ids.push(world.root);
        for i in 0..(emit_depth as usize - 1) {
            let n = world.library.get(*ancestor_ids.last().unwrap()).unwrap();
            ancestor_ids.push(n.children.as_ref().unwrap()[path.slots[i] as usize]);
        }
    }

    let has_neighbor = |dx: isize, dy: isize, dz: isize| -> bool {
        let (nx, ny, nz) = (sx as isize + dx, sy as isize + dy, sz as isize + dz);
        if nx >= 0 && ny >= 0 && nz >= 0
            && (nx as usize) < bf && (ny as usize) < bf && (nz as usize) < bf
        {
            let ns = slot_index(nx as usize, ny as usize, nz as usize);
            return parent_ch[ns] != EMPTY_NODE;
        }
        if emit_depth < 2 { return false; }
        let grandparent_id = ancestor_ids[emit_depth as usize - 2];
        let Some(gp) = world.library.get(grandparent_id) else { return false };
        let Some(gp_ch) = gp.children.as_ref() else { return false };
        let parent_slot = path.slots[emit_depth as usize - 2] as usize;
        let (psx, psy, psz) = slot_coords(parent_slot);
        let (gpx, gpy, gpz) = (psx as isize + dx, psy as isize + dy, psz as isize + dz);
        if gpx < 0 || gpy < 0 || gpz < 0
            || gpx as usize >= bf || gpy as usize >= bf || gpz as usize >= bf
        { return false; }
        let adj_parent_id = gp_ch[slot_index(gpx as usize, gpy as usize, gpz as usize)];
        if adj_parent_id == EMPTY_NODE { return false; }
        let Some(adj) = world.library.get(adj_parent_id) else { return false };
        let Some(adj_ch) = adj.children.as_ref() else { return false };
        let cnx = if nx < 0 { bf - 1 } else if nx as usize >= bf { 0 } else { nx as usize };
        let cny = if ny < 0 { bf - 1 } else if ny as usize >= bf { 0 } else { ny as usize };
        let cnz = if nz < 0 { bf - 1 } else if nz as usize >= bf { 0 } else { nz as usize };
        adj_ch[slot_index(cnx, cny, cnz)] != EMPTY_NODE
    };

    [
        has_neighbor(-1, 0, 0), has_neighbor(1, 0, 0),
        has_neighbor(0, -1, 0), has_neighbor(0, 1, 0),
        has_neighbor(0, 0, -1), has_neighbor(0, 0, 1),
    ]
}

/// Compute border slabs by looking up neighboring emit-level nodes.
/// First tries siblings within the same parent. When the neighbor is
/// outside the parent, walks up to the grandparent to find the
/// adjacent parent and samples the correct child from there.
fn compute_border_slabs(
    world: &WorldState,
    path: &SmallPath,
    emit_depth: u8,
) -> BorderSlabs {
    if emit_depth == 0 { return Default::default(); }

    // Walk to the parent (and optionally grandparent) of the emit node.
    let mut ancestor_ids: Vec<NodeId> = Vec::with_capacity(emit_depth as usize);
    ancestor_ids.push(world.root);
    for i in 0..(emit_depth as usize - 1) {
        let cur = *ancestor_ids.last().unwrap();
        let Some(node) = world.library.get(cur) else { return Default::default() };
        let Some(ch) = node.children.as_ref() else { return Default::default() };
        let next = ch[path.slots[i] as usize];
        if next == EMPTY_NODE { return Default::default() }
        ancestor_ids.push(next);
    }
    // ancestor_ids[emit_depth - 1] = parent of emit node
    let parent_id = ancestor_ids[emit_depth as usize - 1];
    let Some(parent) = world.library.get(parent_id) else { return Default::default() };
    let Some(parent_ch) = parent.children.as_ref() else { return Default::default() };

    let slot = path.slots[emit_depth as usize - 1] as usize;
    let (sx, sy, sz) = slot_coords(slot);
    let bf = BRANCH_FACTOR;

    // Resolve the emit-level neighbor in direction (dx, dy, dz).
    // Returns the NodeId of the neighboring emit-level node.
    let resolve_neighbor = |dx: isize, dy: isize, dz: isize| -> Option<NodeId> {
        let (nx, ny, nz) = (sx as isize + dx, sy as isize + dy, sz as isize + dz);
        if nx >= 0 && ny >= 0 && nz >= 0
            && (nx as usize) < bf && (ny as usize) < bf && (nz as usize) < bf
        {
            // Sibling within the same parent.
            let ns = slot_index(nx as usize, ny as usize, nz as usize);
            let nid = parent_ch[ns];
            return if nid == EMPTY_NODE { None } else { Some(nid) };
        }
        // Cross-parent: walk up to grandparent.
        if emit_depth < 2 { return None; }
        let grandparent_id = ancestor_ids[emit_depth as usize - 2];
        let grandparent = world.library.get(grandparent_id)?;
        let grandparent_ch = grandparent.children.as_ref()?;

        let parent_slot = path.slots[emit_depth as usize - 2] as usize;
        let (psx, psy, psz) = slot_coords(parent_slot);
        let (gpx, gpy, gpz) = (psx as isize + dx, psy as isize + dy, psz as isize + dz);
        if gpx < 0 || gpy < 0 || gpz < 0
            || gpx as usize >= bf || gpy as usize >= bf || gpz as usize >= bf
        { return None; }
        let adj_parent_id = grandparent_ch[slot_index(gpx as usize, gpy as usize, gpz as usize)];
        if adj_parent_id == EMPTY_NODE { return None; }
        let adj_parent = world.library.get(adj_parent_id)?;
        let adj_ch = adj_parent.children.as_ref()?;
        // The child in the adjacent parent at the opposite edge.
        let cnx = if nx < 0 { bf - 1 } else if nx as usize >= bf { 0 } else { nx as usize };
        let cny = if ny < 0 { bf - 1 } else if ny as usize >= bf { 0 } else { ny as usize };
        let cnz = if nz < 0 { bf - 1 } else if nz as usize >= bf { 0 } else { nz as usize };
        let child_id = adj_ch[slot_index(cnx, cny, cnz)];
        if child_id == EMPTY_NODE { None } else { Some(child_id) }
    };

    let slab = |dx: isize, dy: isize, dz: isize, axis: usize, at_max: bool| -> Option<Box<[u8]>> {
        let nid = resolve_neighbor(dx, dy, dz)?;
        extract_border_slab(world, nid, axis, at_max)
    };

    [
        slab(-1, 0, 0, 0, true),  // -x neighbor's max-x face
        slab( 1, 0, 0, 0, false), // +x neighbor's min-x face
        slab( 0,-1, 0, 1, true),  // -y neighbor's max-y face
        slab( 0, 1, 0, 1, false), // +y neighbor's min-y face
        slab( 0, 0,-1, 2, true),  // -z neighbor's max-z face
        slab( 0, 0, 1, 2, false), // +z neighbor's min-z face
    ]
}

fn make_get<'a>(flat: &'a [u8], borders: &'a BorderSlabs) -> impl Fn(i32, i32, i32) -> Option<u8> + 'a {
    let size = (BRANCH_FACTOR * NODE_VOXELS_PER_AXIS) as i32;
    let sz = size as usize;
    move |x: i32, y: i32, z: i32| -> Option<u8> {
        if x >= 0 && y >= 0 && z >= 0 && x < size && y < size && z < size {
            let v = flat[(z as usize * sz + y as usize) * sz + x as usize];
            return if v == EMPTY_VOXEL { None } else { Some(v) };
        }
        // Out of bounds — look up the border slab.
        // Determine which direction is out of bounds and map to the
        // 2D slab coordinate (the two in-bounds axes).
        let (dir, a, b) = if x < 0 {
            (0usize, y, z)
        } else if x >= size {
            (1, y, z)
        } else if y < 0 {
            (2, x, z)
        } else if y >= size {
            (3, x, z)
        } else if z < 0 {
            (4, x, y)
        } else {
            (5, x, y)
        };
        if a < 0 || b < 0 || a >= size || b >= size { return None; }
        let slab = borders[dir].as_ref()?;
        let v = slab[b as usize * sz + a as usize];
        if v == EMPTY_VOXEL { None } else { Some(v) }
    }
}

fn bake_all_children(
    flat: &[u8],
    child_ids: &[NodeId; CHILDREN_PER_NODE],
    child_class: &[ChildClass],
    borders: &BorderSlabs,
) -> Vec<ChildFaces> {
    let get = make_get(flat, borders);
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
    fn new_cold(world: &WorldState, node_id: NodeId, borders: &BorderSlabs, meshes: &mut Assets<Mesh>) -> Self {
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
        let child_faces = bake_all_children(&flat_grid, &child_ids, &child_class, borders);
        let merged = merge_child_faces(&child_faces, meshes);

        BakedNode { child_ids, child_class, flat_grid, child_faces, merged }
    }

    /// Incremental build: clone old data, diff children, patch only dirty slots.
    fn new_incremental(
        old: &BakedNode, world: &WorldState, node_id: NodeId, borders: &BorderSlabs, meshes: &mut Assets<Mesh>,
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
            let get = make_get(&flat_grid, borders);
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

fn tile_transform(origin: Vec3, scale: f32) -> Transform {
    Transform::from_translation(origin)
        .with_scale(Vec3::splat(scale))
}

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
    let render_total_start = std::time::Instant::now();

    let target_layer = target_layer_for(zoom.layer);
    let emit_layer = target_layer.saturating_sub(1);

    // Render radius in Bevy units = N cells × Bevy-units-per-cell at
    // the current view layer. This is what the 2D prototype does
    // implicitly (its viewport is measured in cells); without it,
    // zooming out collapses the visible world to a dot because the
    // per-cell Bevy size grows by 5× per layer.
    let radius_bevy = RADIUS_VIEW_CELLS * cell_size_at_layer(zoom.layer);

    // Log on any view-layer change (even if emit_layer stays the same).
    let view_layer_changed = !render_state.initialised || render_state.last_view_layer != zoom.layer;
    if view_layer_changed {
        render_state.last_view_layer = zoom.layer;
        info!(
            "=== VIEW LAYER CHANGE === zoom.layer={} target_layer={} emit_layer={} scale={} radius_bevy={}",
            zoom.layer, target_layer, emit_layer, scale_for_layer(target_layer), radius_bevy,
        );
    }

    // If emit layer changed, or we're on our first pass, drop everything.
    let layer_changed = !render_state.initialised || render_state.last_zoom_layer != emit_layer;
    if layer_changed
    {
        for (_, (entity, _, _)) in render_state.entities.drain() {
            if let Ok(mut ec) = commands.get_entity(entity) {
                ec.despawn();
            }
        }
        render_state.baked.clear();
        render_state.path_key.clear();
        render_state.last_zoom_layer = emit_layer;
        render_state.initialised = true;
    }

    // Walk the tree → collect visible positions.
    timings.bake_us = 0;
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

    // Debug: on view-layer change, extensive tile diagnostics.
    if view_layer_changed && !visits.is_empty() {
        let mesh_extent = (BRANCH_FACTOR * NODE_VOXELS_PER_AXIS) as f32; // 125.0
        let scale = visits[0].scale;
        let tile_size = mesh_extent * scale;

        info!(
            "=== TILE DEBUG === mesh_extent={} scale={} tile_world_size={} seam_bloat=vertex-level",
            mesh_extent, scale, tile_size,
        );

        // Analyze tile origin spacing for gaps.
        let mut gap_count = 0u32;
        let mut max_gap = 0.0f32;
        let mut perfect_count = 0u32;
        let check_limit = visits.len().min(200);
        for i in 0..check_limit {
            let a = &visits[i];
            for j in (i+1)..check_limit {
                let b = &visits[j];
                let dx = (b.origin.x - a.origin.x).abs();
                let dy = (b.origin.y - a.origin.y).abs();
                let dz = (b.origin.z - a.origin.z).abs();

                for (axis_name, d_axis, d_other1, d_other2) in [
                    ("X", dx, dy, dz),
                    ("Y", dy, dx, dz),
                    ("Z", dz, dx, dy),
                ] {
                    if (d_axis - tile_size).abs() < 1.0 && d_other1 < 1.0 && d_other2 < 1.0 {
                        let gap = d_axis - tile_size;
                        if gap.abs() > 0.0001 {
                            gap_count += 1;
                            max_gap = max_gap.max(gap.abs());
                            if gap_count <= 3 {
                                info!(
                                    "  {}-gap: a={:?} b={:?} dist={} expected={} gap={:.6}",
                                    axis_name, a.origin, b.origin, d_axis, tile_size, gap,
                                );
                            }
                        } else {
                            perfect_count += 1;
                        }
                    }
                }
            }
        }
        info!(
            "  tile spacing: {} perfect, {} gapped (>0.0001), max_gap={:.6} (checked {} tiles)",
            perfect_count, gap_count, max_gap, check_limit,
        );

        // Count unique node_ids to understand mesh reuse.
        let mut unique_ids: Vec<NodeId> = visits.iter().map(|v| v.node_id).collect();
        unique_ids.sort();
        unique_ids.dedup();
        info!(
            "  mesh reuse: {} unique node_ids across {} visits",
            unique_ids.len(), visits.len(),
        );

        // Check if origins are exact integers (f32 precision loss from i64 cast).
        let mut non_integer = 0u32;
        for v in visits.iter().take(50) {
            if v.origin.x != v.origin.x.round()
                || v.origin.y != v.origin.y.round()
                || v.origin.z != v.origin.z.round()
            {
                non_integer += 1;
                if non_integer <= 3 {
                    info!(
                        "  non-integer origin: {:?} (frac: {:.6}, {:.6}, {:.6})",
                        v.origin,
                        v.origin.x - v.origin.x.round(),
                        v.origin.y - v.origin.y.round(),
                        v.origin.z - v.origin.z.round(),
                    );
                }
            }
        }
        if non_integer > 0 {
            warn!("  {} of first 50 origins have non-integer coords (f32 precision loss!)", non_integer);
        } else {
            info!("  all first 50 origins are exact integers (no f32 precision loss)");
        }

        // Origin range — how far from camera are tiles placed.
        let (mut min_o, mut max_o) = (visits[0].origin, visits[0].origin);
        for v in visits.iter() {
            min_o = min_o.min(v.origin);
            max_o = max_o.max(v.origin);
        }
        info!(
            "  origin range: min={:?} max={:?} span={:?}",
            min_o, max_o, max_o - min_o,
        );

        // Check if origins are exact multiples of tile_size.
        let mut non_aligned = 0u32;
        for v in visits.iter().take(50) {
            let rx = v.origin.x % tile_size;
            let ry = v.origin.y % tile_size;
            let rz = v.origin.z % tile_size;
            if rx.abs() > 0.01 && (rx - tile_size).abs() > 0.01
                || ry.abs() > 0.01 && (ry - tile_size).abs() > 0.01
                || rz.abs() > 0.01 && (rz - tile_size).abs() > 0.01
            {
                non_aligned += 1;
                if non_aligned <= 3 {
                    info!(
                        "  non-aligned origin: {:?} mod tile_size=({:.2}, {:.2}, {:.2})",
                        v.origin, rx, ry, rz,
                    );
                }
            }
        }
        if non_aligned > 0 {
            warn!("  {} of first 50 origins are NOT aligned to tile_size grid!", non_aligned);
        } else {
            info!("  all first 50 origins are aligned to tile_size={} grid", tile_size);
        }
    }

    // Pre-bake: ensure every visited NodeId has a BakedNode.
    // On edit, use the old BakedNode for incremental diff.
    {
        let bake_start = std::time::Instant::now();
        for (vi, v) in visits.iter().enumerate() {
            let node = world.library.get(v.node_id)
                .expect("render: node missing from library");

            if view_layer_changed && vi < 10 {
                let has_children = node.children.is_some();
                let child_depth_info = if let Some(children) = node.children.as_ref() {
                    let mut leaves = 0u32;
                    let mut branches = 0u32;
                    let mut empty = 0u32;
                    for slot in 0..CHILDREN_PER_NODE {
                        let cid = children[slot];
                        if cid == EMPTY_NODE {
                            empty += 1;
                        } else if let Some(child_node) = world.library.get(cid) {
                            if child_node.children.is_some() {
                                branches += 1;
                            } else {
                                leaves += 1;
                            }
                        }
                    }
                    format!("children: {} leaves, {} branches, {} empty", leaves, branches, empty)
                } else {
                    "LEAF (no children)".to_string()
                };
                info!(
                    "  visit[{}] node_id={} has_children={} origin={:?} scale={} path_depth={} | {}",
                    vi, v.node_id, has_children, v.origin, v.scale,
                    v.path.depth, child_depth_info,
                );
            }

            if node.children.is_some() {
                let border_exists = compute_border_existence(&world, &v.path, emit_layer);
                let key = (v.node_id, border_exists);
                if !render_state.baked.contains_key(&key) {
                    let borders = compute_border_slabs(&world, &v.path, emit_layer);
                    if view_layer_changed {
                        let slab_sizes: Vec<usize> = borders.iter()
                            .map(|b| b.as_ref().map_or(0, |s| s.len()))
                            .collect();
                        info!(
                            "  BAKE node_id={} border_exists={:?} slab_sizes={:?} path_depth={}",
                            v.node_id, border_exists, slab_sizes, v.path.depth,
                        );
                    }
                    let baked = if let Some(&old_key) = render_state.path_key.get(&v.path) {
                        if let Some(old_bake) = render_state.baked.get(&old_key) {
                            BakedNode::new_incremental(old_bake, &world, v.node_id, &borders, &mut meshes)
                        } else {
                            BakedNode::new_cold(&world, v.node_id, &borders, &mut meshes)
                        }
                    } else {
                        BakedNode::new_cold(&world, v.node_id, &borders, &mut meshes)
                    };
                    render_state.baked.insert(key, baked);
                }
                render_state.path_key.insert(v.path, key);
            } else {
                let key = (v.node_id, [false; 6]);
                if !render_state.baked.contains_key(&key) {
                    let merged = bake_leaf(&world, v.node_id, &mut meshes);
                    render_state.baked.insert(key, BakedNode {
                        child_ids: [EMPTY_NODE; CHILDREN_PER_NODE],
                        child_class: Vec::new(),
                        flat_grid: Vec::new(),
                        child_faces: Vec::new(),
                        merged,
                    });
                }
            }
        }
        if view_layer_changed {
            info!("  total visits: {} (showing first 10)", visits.len());
            info!("  bake cache entries: {}", render_state.baked.len());

            // Log mesh AABB from child_faces positions (avoids
            // private VertexAttributeValues access).
            let expected = (BRANCH_FACTOR * NODE_VOXELS_PER_AXIS) as f32;
            for (key, baked) in render_state.baked.iter() {
                let (nid, borders) = key;
                let mut min = [f32::MAX; 3];
                let mut max = [f32::MIN; 3];
                let mut total_verts = 0usize;
                for cf in &baked.child_faces {
                    for (_voxel, fd) in cf.iter() {
                        for pos in &fd.positions {
                            total_verts += 1;
                            for a in 0..3 {
                                min[a] = min[a].min(pos[a]);
                                max[a] = max[a].max(pos[a]);
                            }
                        }
                    }
                }
                if total_verts > 0 {
                    let extent = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
                    let x_ok = min[0] <= 0.01 && max[0] >= expected - 0.01;
                    let z_ok = min[2] <= 0.01 && max[2] >= expected - 0.01;
                    info!(
                        "  MESH AABB node_id={} borders={:?} verts={} submeshes={} \
                         min=[{:.2},{:.2},{:.2}] max=[{:.2},{:.2},{:.2}] \
                         extent=[{:.2},{:.2},{:.2}] x_full={} z_full={}",
                        nid, borders, total_verts, baked.merged.len(),
                        min[0], min[1], min[2], max[0], max[1], max[2],
                        extent[0], extent[1], extent[2], x_ok, z_ok,
                    );
                } else {
                    info!(
                        "  MESH AABB node_id={} borders={:?} NO VERTICES (empty mesh)",
                        nid, borders,
                    );
                }
            }

            // Log unique NodeIds and how many visits share them.
            let mut nid_counts: HashMap<NodeId, u32> = HashMap::new();
            for v in visits.iter() {
                *nid_counts.entry(v.node_id).or_default() += 1;
            }
            info!(
                "  unique NodeIds: {} (visits sharing: {:?})",
                nid_counts.len(),
                nid_counts.values().collect::<Vec<_>>(),
            );

            // Check flat_grid boundary consistency between adjacent
            // X-axis tiles (first pair found).
            let tile_size = expected;
            let scale = if visits.is_empty() { 1.0 } else { visits[0].scale };
            let tile_world = tile_size * scale;
            'adj_check: for i in 0..visits.len().min(100) {
                for j in (i+1)..visits.len().min(100) {
                    let a = &visits[i];
                    let b = &visits[j];
                    let dx = (b.origin.x - a.origin.x).abs();
                    let dy = (b.origin.y - a.origin.y).abs();
                    let dz = (b.origin.z - a.origin.z).abs();
                    if (dx - tile_world).abs() < 0.1 && dy < 0.1 && dz < 0.1 {
                        // Found X-adjacent pair. Check flat_grid boundary.
                        let (left, right) = if b.origin.x > a.origin.x { (a, b) } else { (b, a) };
                        let lk = render_state.path_key.get(&left.path);
                        let rk = render_state.path_key.get(&right.path);
                        if let (Some(lk), Some(rk)) = (lk, rk) {
                            let lb = render_state.baked.get(lk);
                            let rb = render_state.baked.get(rk);
                            if let (Some(lb), Some(rb)) = (lb, rb) {
                                let gs = BRANCH_FACTOR * NODE_VOXELS_PER_AXIS;
                                // Left tile's +X face vs right tile's -X face.
                                let mut mismatch = 0u32;
                                let mut left_solid = 0u32;
                                let mut right_solid = 0u32;
                                if !lb.flat_grid.is_empty() && !rb.flat_grid.is_empty() {
                                    for z in 0..gs {
                                        for y in 0..gs {
                                            let li = (z * gs + y) * gs + (gs - 1); // x = gs-1
                                            let ri = (z * gs + y) * gs + 0;        // x = 0
                                            let lv = lb.flat_grid[li];
                                            let rv = rb.flat_grid[ri];
                                            if lv != EMPTY_VOXEL { left_solid += 1; }
                                            if rv != EMPTY_VOXEL { right_solid += 1; }
                                            if lv != rv { mismatch += 1; }
                                        }
                                    }
                                    info!(
                                        "  X-ADJ BOUNDARY left_origin={:?} right_origin={:?} \
                                         left_key={:?} right_key={:?} same_key={} \
                                         left_+x_solid={} right_-x_solid={} mismatch={}",
                                        left.origin, right.origin,
                                        lk, rk, lk == rk,
                                        left_solid, right_solid, mismatch,
                                    );
                                } else {
                                    info!(
                                        "  X-ADJ BOUNDARY flat_grid empty! left_len={} right_len={}",
                                        lb.flat_grid.len(), rb.flat_grid.len(),
                                    );
                                }
                                // Count faces at left tile's +X boundary (x=124).
                                let mut left_boundary_faces = 0u32;
                                for (_voxel, fd) in lb.child_faces.iter().flat_map(|cf| cf.iter()) {
                                    for pos in &fd.positions {
                                        if pos[0] >= (gs as f32 - 0.01) {
                                            left_boundary_faces += 1;
                                            break; // count per-face, not per-vertex
                                        }
                                    }
                                }
                                let mut right_boundary_faces = 0u32;
                                for (_voxel, fd) in rb.child_faces.iter().flat_map(|cf| cf.iter()) {
                                    for pos in &fd.positions {
                                        if pos[0] <= 0.01 {
                                            right_boundary_faces += 1;
                                            break;
                                        }
                                    }
                                }
                                info!(
                                    "  X-ADJ FACES left_+x_boundary_faces={} right_-x_boundary_faces={}",
                                    left_boundary_faces, right_boundary_faces,
                                );
                            }
                        }
                        break 'adj_check;
                    }
                }
            }
        }
        timings.bake_us = bake_start.elapsed().as_micros() as u64;
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
                if last_origin != visit.origin {
                    if let Ok(mut ec) = commands.get_entity(entity) {
                        ec.insert(
                            tile_transform(visit.origin, visit.scale),
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

                let baked = render_state.path_key.get(&visit.path)
                    .and_then(|key| render_state.baked.get(key))
                    .map(|b| &b.merged[..])
                    .unwrap_or(&[]);

                let parent = commands
                    .spawn((
                        WorldRenderedNode(new_node_id),
                        tile_transform(visit.origin, visit.scale),
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
