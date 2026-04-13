//! Mesh cache: baked mesh data keyed by NodeId.
//!
//! Stores the intermediate products (flat grid, per-child faces) alongside
//! the final GPU meshes so edits can incrementally re-bake only the dirty
//! children instead of rebuilding the full 125³ grid.
//!
//! Also provides `prebake_node_raw` for the offline gen_world tool.

use bevy::platform::collections::HashMap;
use bevy::prelude::*;

use crate::model::mesher::{
    bake_volume, bake_volume_raw, bake_child_faces, merge_child_faces,
    merge_child_faces_raw, flatten_children, ChildClass, ChildFaces,
    FaceData,
};
use crate::model::BakedSubMesh;

use super::state::WorldState;
use super::tree::{
    slot_coords, slot_index, voxel_idx, NodeId,
    BRANCH_FACTOR, CHILDREN_PER_NODE, EMPTY_NODE, EMPTY_VOXEL,
    NODE_VOXELS_PER_AXIS,
};

// ----------------------------------------------------------- BakedNode

/// Cached bake data for one node.
pub struct BakedNode {
    pub child_ids: [NodeId; CHILDREN_PER_NODE],
    pub child_class: Vec<ChildClass>,
    pub flat_grid: Vec<u8>,
    pub child_faces: Vec<ChildFaces>,
    pub merged: Vec<BakedSubMesh>,
}

impl BakedNode {
    /// Create a BakedNode from prebaked mesh data (no intermediate data).
    /// These nodes cannot be used for incremental baking — edits will
    /// trigger a cold bake instead.
    pub fn from_prebaked(merged: Vec<BakedSubMesh>) -> Self {
        Self {
            child_ids: [EMPTY_NODE; CHILDREN_PER_NODE],
            child_class: Vec::new(),
            flat_grid: Vec::new(),
            child_faces: Vec::new(),
            merged,
        }
    }

    /// Whether this node has intermediate data for incremental baking.
    pub fn has_intermediate_data(&self) -> bool {
        !self.child_class.is_empty()
    }

    /// Full build from scratch.
    pub fn new_cold(world: &WorldState, node_id: NodeId, meshes: &mut Assets<Mesh>) -> Self {
        let node = world.library.get(node_id).expect("mesh_cache: node missing");
        let children = node.children.as_ref().expect("mesh_cache: expected non-leaf");
        let child_ids: [NodeId; CHILDREN_PER_NODE] = **children;

        let child_class: Vec<ChildClass> = (0..CHILDREN_PER_NODE)
            .map(|slot| classify_child(world, child_ids[slot]))
            .collect();

        let children_voxels: Vec<Option<&[u8]>> = (0..CHILDREN_PER_NODE)
            .map(|slot| {
                if child_ids[slot] == EMPTY_NODE { None }
                else { Some(world.library.get(child_ids[slot])
                    .expect("mesh_cache: child missing").voxels.as_ref().as_slice()) }
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
    pub fn new_incremental(
        old: &BakedNode, world: &WorldState, node_id: NodeId, meshes: &mut Assets<Mesh>,
    ) -> Self {
        let node = world.library.get(node_id).expect("mesh_cache: node missing");
        let children = node.children.as_ref().expect("mesh_cache: expected non-leaf");
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
                        .expect("mesh_cache: child missing").voxels.as_ref().as_slice()) };
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

/// Bake a leaf node into GPU meshes.
pub fn bake_leaf(world: &WorldState, node_id: NodeId, meshes: &mut Assets<Mesh>) -> Vec<BakedSubMesh> {
    let voxels = world.library.get(node_id).expect("mesh_cache: leaf missing").voxels.clone();
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

// -------------------------------------------------------- offline prebake

/// Bake a node into raw mesh data, without Bevy. Used by gen_world.
pub fn prebake_node_raw(world: &WorldState, node_id: NodeId) -> Vec<(u8, FaceData)> {
    let node = world.library.get(node_id).expect("prebake: node missing");
    if let Some(children) = node.children.as_ref() {
        let child_ids: [NodeId; CHILDREN_PER_NODE] = **children;
        let child_class: Vec<ChildClass> = (0..CHILDREN_PER_NODE)
            .map(|slot| classify_child(world, child_ids[slot]))
            .collect();
        let children_voxels: Vec<Option<&[u8]>> = (0..CHILDREN_PER_NODE)
            .map(|slot| {
                if child_ids[slot] == EMPTY_NODE { None }
                else { Some(world.library.get(child_ids[slot])
                    .expect("prebake: child missing").voxels.as_ref().as_slice()) }
            })
            .collect();
        let flat_grid = flatten_children(
            &children_voxels, &child_class,
            BRANCH_FACTOR, NODE_VOXELS_PER_AXIS, EMPTY_VOXEL,
        );
        let child_faces = bake_all_children(&flat_grid, &child_ids, &child_class);
        merge_child_faces_raw(&child_faces)
    } else {
        let voxels = &node.voxels;
        bake_volume_raw(
            NODE_VOXELS_PER_AXIS as i32,
            |x, y, z| {
                if x < 0 || y < 0 || z < 0
                    || x >= NODE_VOXELS_PER_AXIS as i32
                    || y >= NODE_VOXELS_PER_AXIS as i32
                    || z >= NODE_VOXELS_PER_AXIS as i32
                { return None; }
                let v = voxels[voxel_idx(x as usize, y as usize, z as usize)];
                if v == EMPTY_VOXEL { None } else { Some(v) }
            },
        )
    }
}

// -------------------------------------------------------- mesh store

#[cfg(not(target_arch = "wasm32"))]
use crate::world::mesh_stream::MeshStreamer;

/// Manages the mesh bake cache, async streaming, and parent-fallback.
///
/// All mesh lookups go through here. The lifecycle:
/// 1. On first frame, start the async I/O thread (loads meshes.idx).
/// 2. Each frame, drain completed async responses into the bake cache.
/// 3. When a node needs a mesh: if baked → use it. If in the async
///    index → request it (async, non-blocking). If neither → cold bake
///    (budget-limited).
/// 4. If a node isn't baked yet, the renderer shows its parent's mesh
///    (parent-fallback, handled by the renderer, not here).
pub struct MeshStore {
    /// Baked meshes keyed by NodeId.
    baked: HashMap<NodeId, BakedNode>,
    /// Path→NodeId tracking for incremental baking.
    path_node: HashMap<super::walk::SmallPath, NodeId>,
    /// Monolithic prebaked mesh map loaded at startup.
    prebaked: Option<super::serial::PrebakedMeshes>,
    /// Native: async mesh streamer (I/O thread + index).
    #[cfg(not(target_arch = "wasm32"))]
    streamer: Option<MeshStreamer>,
    loaded: bool,
    diag_frame: u32,
}

impl Default for MeshStore {
    fn default() -> Self {
        Self {
            baked: HashMap::new(),
            path_node: HashMap::new(),
            prebaked: None,
            #[cfg(not(target_arch = "wasm32"))]
            streamer: None,
            loaded: false,
            diag_frame: 0,
        }
    }
}

impl MeshStore {
    /// Load prebaked meshes on first call.
    /// Native: starts the async I/O thread with indexed format.
    /// WASM: loads the monolithic meshes.bin into memory.
    pub fn ensure_loaded(&mut self, world_hash: u64) {
        if self.loaded { return; }
        self.loaded = true;

        let path = std::path::Path::new("assets/meshes_mono.bin");
        match super::serial::read_prebaked_file(path, world_hash) {
            Ok(prebaked) => {
                eprintln!("loaded prebaked meshes: {} entries", prebaked.len());
                self.prebaked = Some(prebaked);
            }
            Err(e) => {
                eprintln!("no prebaked meshes ({}), will cold bake", e);
            }
        }
    }

    /// Drain completed async mesh loads into the bake cache.
    /// Call once per frame, before the bake pass.
    pub fn receive_async_meshes(&mut self, meshes: &mut Assets<Mesh>) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let Some(streamer) = self.streamer.as_mut() else { return };
            for resp in streamer.drain_responses() {
                if self.baked.contains_key(&resp.node_id) { continue; }
                let merged: Vec<BakedSubMesh> = resp.entry
                    .into_iter()
                    .map(|(voxel, data)| BakedSubMesh {
                        mesh: meshes.add(data.build()),
                        voxel,
                    })
                    .collect();
                self.baked.insert(resp.node_id, BakedNode::from_prebaked(merged));
            }
        }
    }

    pub fn is_baked(&self, node_id: NodeId) -> bool {
        self.baked.contains_key(&node_id)
    }

    pub fn get_merged(&self, node_id: NodeId) -> Option<&[BakedSubMesh]> {
        self.baked.get(&node_id).map(|b| b.merged.as_slice())
    }

    pub fn get_path_node(&self, path: &super::walk::SmallPath) -> Option<NodeId> {
        self.path_node.get(path).copied()
    }

    pub fn set_path_node(&mut self, path: super::walk::SmallPath, node_id: NodeId) {
        self.path_node.insert(path, node_id);
    }

    /// Ensure a node has a baked mesh. If it's in the async index,
    /// request it (non-blocking). Falls back to cold bake if not in
    /// the index. Returns true if the node is baked NOW (available
    /// this frame). Returns false if pending async load — the renderer
    /// should show the parent's mesh as fallback.
    pub fn ensure_baked(
        &mut self,
        world: &WorldState,
        node_id: NodeId,
        path: &super::walk::SmallPath,
        meshes: &mut Assets<Mesh>,
        cold_bakes: &mut usize,
        max_cold_bakes: usize,
        distance_priority: u32,
    ) -> bool {
        if self.baked.contains_key(&node_id) {
            return true;
        }

        self.diag_frame += 1;
        if self.diag_frame <= 500 {
            eprintln!("ENSURE_BAKED[{}]: node {} not in cache, attempting load", self.diag_frame, node_id);
        }

        // Monolithic prebaked loader — no budget cap, loads instantly.
        let prebaked_entry = self.prebaked.as_mut()
            .and_then(|p| p.remove(&node_id));
        if let Some(entry) = prebaked_entry {
            let n = entry.len();
            let merged: Vec<BakedSubMesh> = entry
                .into_iter()
                .map(|(voxel, data)| BakedSubMesh {
                    mesh: meshes.add(data.build()),
                    voxel,
                })
                .collect();
            if self.diag_frame <= 500 {
                eprintln!("ENSURE_BAKED[{}]: node {} loaded OK ({} submeshes)", self.diag_frame, node_id, n);
            }
            self.baked.insert(node_id, BakedNode::from_prebaked(merged));
            return true;
        }

        if self.diag_frame <= 500 {
            eprintln!("ENSURE_BAKED[{}]: node {} prebaked=None, falling to cold bake", self.diag_frame, node_id);
        }

        // No prebaked entry — cold/incremental bake (budget-limited).
        let node = world.library.get(node_id).expect("mesh_cache: node missing");
        if node.children.is_some() {
            if *cold_bakes >= max_cold_bakes {
                if self.diag_frame <= 500 {
                    eprintln!("ENSURE_BAKED[{}]: node {} BUDGET EXCEEDED ({}/{})", self.diag_frame, node_id, cold_bakes, max_cold_bakes);
                }
                return false;
            }
            let baked = if let Some(old_nid) = self.get_path_node(path) {
                if let Some(old_bake) = self.baked.get(&old_nid) {
                    if old_bake.has_intermediate_data() {
                        BakedNode::new_incremental(old_bake, world, node_id, meshes)
                    } else {
                        *cold_bakes += 1;
                        BakedNode::new_cold(world, node_id, meshes)
                    }
                } else {
                    *cold_bakes += 1;
                    BakedNode::new_cold(world, node_id, meshes)
                }
            } else {
                *cold_bakes += 1;
                BakedNode::new_cold(world, node_id, meshes)
            };
            if self.diag_frame <= 500 {
                eprintln!("ENSURE_BAKED[{}]: node {} cold-baked non-leaf ({} submeshes)", self.diag_frame, node_id, baked.merged.len());
            }
            self.baked.insert(node_id, baked);
        } else {
            if *cold_bakes >= max_cold_bakes { return false; }
            *cold_bakes += 1;
            let merged = bake_leaf(world, node_id, meshes);
            if self.diag_frame <= 500 {
                eprintln!("ENSURE_BAKED[{}]: node {} cold-baked leaf ({} submeshes)", self.diag_frame, node_id, merged.len());
            }
            self.baked.insert(node_id, BakedNode::from_prebaked(merged));
        }
        true
    }

    /// Request async prefetch for a node (non-blocking). Used for
    /// nodes not yet visible but likely to be needed soon.
    /// No-op on WASM (all meshes are in memory already).
    pub fn prefetch(&mut self, _node_id: NodeId, _priority: u32) {
        #[cfg(not(target_arch = "wasm32"))]
        {
            if self.baked.contains_key(&_node_id) { return; }
            if let Some(streamer) = self.streamer.as_mut() {
                streamer.request(_node_id, _priority);
            }
        }
    }

    /// Clear entity tracking (on zoom change). Keeps baked mesh data.
    pub fn clear_paths(&mut self) {
        self.path_node.clear();
    }

    /// Clear everything (on zoom change). Forces fresh bake of all nodes.
    pub fn clear_all(&mut self) {
        self.baked.clear();
        self.path_node.clear();
    }
}

// -------------------------------------------------------- internal helpers

fn classify_child(world: &WorldState, child_id: NodeId) -> ChildClass {
    if child_id == EMPTY_NODE { return ChildClass::Empty; }
    let child = world.library.get(child_id).expect("mesh_cache: child missing");
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
