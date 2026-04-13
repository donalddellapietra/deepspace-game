//! World serialization: canned library + save/load with overrides.
//!
//! The serialization is split into two formats:
//!
//! 1. **Canned world** (`world.bin`) — the full node library generated
//!    offline by `gen-world`. Loaded at startup. Immutable at runtime.
//!
//! 2. **Save file** (`save.bin`) — player-created nodes (edits) plus
//!    the current root. References a canned world by version hash.
//!    On load: deserialize canned world, insert override nodes, set root.
//!
//! Both formats use bincode for binary encoding and lz4 for compression.
//! Hash indices and refcounts are **not** serialized — they are rebuilt
//! on load from the node data. This keeps the format minimal and
//! forward-compatible (the rebuild logic can change without breaking
//! saved files).
//!
//! ## Canned runs compatibility
//!
//! The canned world format stores a flat list of `(NodeId, SerialNode)`
//! entries. A canned run (simulation snapshots over time) would store
//! multiple root NodeIds indexing into the same library — the format
//! supports this naturally since the library is just a flat node list
//! and roots are just NodeId values. The `CannedWorld.roots` field is
//! a `Vec` to anticipate this, though currently only one root is used.

use std::io::{self, Write as _};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::tree::{
    Children, Node, NodeId, NodeLibrary, VoxelGrid, CHILDREN_PER_NODE,
    NODE_VOXELS,
};

// --------------------------------------------------------- wire types

/// A node as stored on disk. No hash indices, no refcount.
#[derive(Serialize, Deserialize)]
struct SerialNode {
    voxels: Vec<u8>,
    children: Option<Vec<u64>>,
}

/// The canned world file: a complete node library + roots.
#[derive(Serialize, Deserialize)]
struct CannedWorld {
    /// Format version. Bump when the layout changes.
    version: u32,
    /// The root NodeId(s). Index 0 is the primary root.
    /// Vec to support canned runs (multiple timestep roots) in future.
    roots: Vec<u64>,
    /// Next available NodeId (so runtime edits start above this).
    next_id: u64,
    /// All nodes, in insertion order.
    nodes: Vec<(u64, SerialNode)>,
}

/// A save file: player-created nodes + the current root.
#[derive(Serialize, Deserialize)]
struct SaveFile {
    /// Format version.
    version: u32,
    /// Hash of the canned world this save was created against.
    /// Used to detect base-world mismatches on load.
    base_world_hash: u64,
    /// The canned world's next_id at load time. Nodes with id >= this
    /// are player-created overrides.
    canned_node_count: u64,
    /// The current root NodeId (may differ from canned root due to edits).
    root: u64,
    /// Player-created nodes (id >= canned_node_count).
    overrides: Vec<(u64, SerialNode)>,
}

const CANNED_VERSION: u32 = 1;
const SAVE_VERSION: u32 = 1;

// ---------------------------------------------- serialization helpers

fn node_to_serial(node: &Node) -> SerialNode {
    SerialNode {
        voxels: node.voxels.to_vec(),
        children: node.children.as_ref().map(|c| c.to_vec()),
    }
}

fn serial_to_node(s: SerialNode) -> Node {
    let voxels: VoxelGrid = s
        .voxels
        .into_boxed_slice()
        .try_into()
        .unwrap_or_else(|_| panic!("voxel grid must be {} bytes", NODE_VOXELS));
    let children: Option<Children> = s.children.map(|c| {
        let boxed: Box<[u64]> = c.into_boxed_slice();
        boxed
            .try_into()
            .unwrap_or_else(|_| panic!("children must be {} entries", CHILDREN_PER_NODE))
    });
    Node {
        voxels,
        children,
        ref_count: 0, // rebuilt on load
    }
}

// ------------------------------------------------ compress / decompress

fn compress(data: &[u8]) -> Vec<u8> {
    lz4_flex::compress_prepend_size(data)
}

fn decompress(data: &[u8]) -> io::Result<Vec<u8>> {
    lz4_flex::decompress_size_prepended(data)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

// --------------------------------------------------- canned world I/O

/// Serialize a `WorldState` to the canned world format.
pub fn serialize_world(
    root: NodeId,
    library: &NodeLibrary,
) -> io::Result<Vec<u8>> {
    let nodes: Vec<(u64, SerialNode)> = library
        .iter_nodes()
        .map(|(&id, node)| (id, node_to_serial(node)))
        .collect();
    let canned = CannedWorld {
        version: CANNED_VERSION,
        roots: vec![root],
        next_id: library.next_id(),
        nodes,
    };
    let raw = bincode::serialize(&canned)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(compress(&raw))
}

/// Write a canned world file to disk.
pub fn write_world_file(
    path: &Path,
    root: NodeId,
    library: &NodeLibrary,
) -> io::Result<()> {
    let data = serialize_world(root, library)?;
    let mut f = std::fs::File::create(path)?;
    f.write_all(&data)?;
    Ok(())
}

/// Deserialize a canned world from bytes. Returns `(root, library, next_id)`.
/// The `next_id` is stored so the caller can track which NodeIds are
/// canned vs player-created.
pub fn deserialize_world(data: &[u8]) -> io::Result<(NodeId, NodeLibrary, u64)> {
    let raw = decompress(data)?;
    let canned: CannedWorld = bincode::deserialize(&raw)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if canned.version != CANNED_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported canned world version: {}", canned.version),
        ));
    }
    let root = *canned.roots.first().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidData, "canned world has no root")
    })?;

    let mut library = NodeLibrary::with_next_id(canned.next_id);
    for (id, snode) in canned.nodes {
        let node = serial_to_node(snode);
        library.insert_raw(id, node);
    }
    library.rebuild_hash_indices();
    library.rebuild_refcounts();
    // The root needs an external ref (same as swap_root does).
    library.ref_inc(root);

    Ok((root, library, canned.next_id))
}

/// Read a canned world file from disk.
pub fn read_world_file(path: &Path) -> io::Result<(NodeId, NodeLibrary, u64)> {
    let data = std::fs::read(path)?;
    deserialize_world(&data)
}

// --------------------------------------------------------- save file I/O

/// Compute a simple hash of the canned world for mismatch detection.
/// Uses the root + node count + next_id.
pub fn canned_world_hash(root: NodeId, next_id: u64, node_count: usize) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    let mut h = DefaultHasher::new();
    h.write_u64(root);
    h.write_u64(next_id);
    h.write_usize(node_count);
    h.finish()
}

/// Serialize a save file: extract player-created nodes (id >= canned_node_count)
/// and the current root.
pub fn serialize_save(
    root: NodeId,
    library: &NodeLibrary,
    canned_node_count: u64,
    base_world_hash: u64,
) -> io::Result<Vec<u8>> {
    let overrides: Vec<(u64, SerialNode)> = library
        .iter_nodes()
        .filter(|&(&id, _)| id >= canned_node_count)
        .map(|(&id, node)| (id, node_to_serial(node)))
        .collect();
    let save = SaveFile {
        version: SAVE_VERSION,
        base_world_hash,
        canned_node_count,
        root,
        overrides,
    };
    let raw = bincode::serialize(&save)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(compress(&raw))
}

/// Write a save file to disk.
pub fn write_save_file(
    path: &Path,
    root: NodeId,
    library: &NodeLibrary,
    canned_node_count: u64,
    base_world_hash: u64,
) -> io::Result<()> {
    let data = serialize_save(root, library, canned_node_count, base_world_hash)?;
    let mut f = std::fs::File::create(path)?;
    f.write_all(&data)?;
    Ok(())
}

/// Load a save file on top of an already-loaded canned world.
/// Returns the new root. Inserts override nodes into the library
/// and rebuilds indices/refcounts.
pub fn load_save(
    data: &[u8],
    library: &mut NodeLibrary,
    expected_base_hash: u64,
) -> io::Result<NodeId> {
    let raw = decompress(data)?;
    let save: SaveFile = bincode::deserialize(&raw)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if save.version != SAVE_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported save version: {}", save.version),
        ));
    }
    if save.base_world_hash != expected_base_hash {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "save file was created against a different base world",
        ));
    }

    // Insert override nodes. Their children may reference canned nodes
    // (already in the library) or other override nodes.
    for (id, snode) in save.overrides {
        let node = serial_to_node(snode);
        library.insert_raw(id, node);
    }
    // Advance next_id past all override nodes.
    library.ensure_next_id_above(save.root);
    // Full rebuild since overrides change the graph.
    library.rebuild_hash_indices();
    library.rebuild_refcounts();
    library.ref_inc(save.root);

    Ok(save.root)
}

/// Read a save file from disk and apply it to the library.
pub fn read_save_file(
    path: &Path,
    library: &mut NodeLibrary,
    expected_base_hash: u64,
) -> io::Result<NodeId> {
    let data = std::fs::read(path)?;
    load_save(&data, library, expected_base_hash)
}

// -------------------------------------------------------- prebaked meshes

use crate::model::mesher::FaceData;

/// Prebaked mesh data for one node: list of (voxel_type, mesh_geometry).
pub type PrebakedEntry = Vec<(u8, FaceData)>;

/// All prebaked meshes, keyed by NodeId.
pub type PrebakedMeshes = std::collections::HashMap<NodeId, PrebakedEntry>;

#[derive(Serialize, Deserialize)]
struct PrebakedFile {
    version: u32,
    entries: Vec<(u64, PrebakedEntry)>,
}

const PREBAKED_VERSION: u32 = 1;

pub fn serialize_prebaked(meshes: &PrebakedMeshes) -> io::Result<Vec<u8>> {
    let entries: Vec<(u64, PrebakedEntry)> = meshes
        .iter()
        .map(|(&id, entry)| (id, entry.clone()))
        .collect();
    let file = PrebakedFile {
        version: PREBAKED_VERSION,
        entries,
    };
    let raw = bincode::serialize(&file)
        .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
    Ok(compress(&raw))
}

pub fn write_prebaked_file(path: &Path, meshes: &PrebakedMeshes) -> io::Result<()> {
    let data = serialize_prebaked(meshes)?;
    let mut f = std::fs::File::create(path)?;
    f.write_all(&data)?;
    Ok(())
}

pub fn deserialize_prebaked(data: &[u8]) -> io::Result<PrebakedMeshes> {
    let raw = decompress(data)?;
    let file: PrebakedFile = bincode::deserialize(&raw)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    if file.version != PREBAKED_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported prebaked version: {}", file.version),
        ));
    }
    Ok(file.entries.into_iter().collect())
}

pub fn read_prebaked_file(path: &Path) -> io::Result<PrebakedMeshes> {
    let data = std::fs::read(path)?;
    deserialize_prebaked(&data)
}

// ----------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::state::WorldState;

    #[test]
    fn round_trip_grassland() {
        let world = WorldState::new_grassland();
        let data = serialize_world(world.root, &world.library).unwrap();
        let (root, library, _) = deserialize_world(&data).unwrap();
        assert_eq!(root, world.root);
        assert_eq!(library.len(), world.library.len());
    }

    #[test]
    fn round_trip_sphere() {
        let world = WorldState::new_sphere();
        let data = serialize_world(world.root, &world.library).unwrap();
        let (root, library, _) = deserialize_world(&data).unwrap();
        assert_eq!(root, world.root);
        assert_eq!(library.len(), world.library.len());
    }

    #[test]
    fn round_trip_preserves_voxels() {
        let world = WorldState::new_grassland();
        let data = serialize_world(world.root, &world.library).unwrap();
        let (root, library, _) = deserialize_world(&data).unwrap();
        // Walk the zero path and check leaf content matches.
        let mut orig_id = world.root;
        let mut load_id = root;
        for _ in 0..crate::world::tree::MAX_LAYER {
            let orig = world.library.get(orig_id).unwrap();
            let loaded = library.get(load_id).unwrap();
            assert_eq!(orig.voxels.as_ref(), loaded.voxels.as_ref());
            orig_id = orig.children.as_ref().unwrap()[0];
            load_id = loaded.children.as_ref().unwrap()[0];
        }
        // Leaf voxels.
        let orig_leaf = world.library.get(orig_id).unwrap();
        let load_leaf = library.get(load_id).unwrap();
        assert_eq!(orig_leaf.voxels.as_ref(), load_leaf.voxels.as_ref());
    }

    #[test]
    fn save_with_no_edits_is_tiny() {
        let world = WorldState::new_grassland();
        let next_id = world.library.next_id();
        let base_hash = canned_world_hash(world.root, next_id, world.library.len());
        let data = serialize_save(
            world.root, &world.library, next_id, base_hash,
        ).unwrap();
        // No overrides → the save file should be very small.
        assert!(data.len() < 200, "save with no edits was {} bytes", data.len());
    }

    #[test]
    fn save_load_round_trip_with_edit() {
        use crate::world::edit::{edit_at_layer_pos, get_voxel};
        use crate::world::position::LayerPos;
        use crate::world::tree::{voxel_from_block, MAX_LAYER};
        use crate::world::view::position_from_leaf_coord;
        use crate::block::BlockType;

        // Build world and record canned state.
        let mut world = WorldState::new_grassland();
        let canned_next_id = world.library.next_id();
        let canned_root = world.root;
        let canned_node_count = world.library.len();
        let base_hash = canned_world_hash(canned_root, canned_next_id, canned_node_count);

        // Make an edit at leaf layer.
        let lp = LayerPos::from_path_and_cell(
            [0; MAX_LAYER as usize],
            [0, 0, 0],
            MAX_LAYER,
        );
        let stone = voxel_from_block(Some(BlockType::Stone));
        edit_at_layer_pos(&mut world, &lp, stone);
        assert_ne!(world.root, canned_root);

        // Serialize save.
        let save_data = serialize_save(
            world.root, &world.library, canned_next_id, base_hash,
        ).unwrap();

        // Simulate fresh load: only canned nodes.
        let fresh_world = WorldState::new_grassland();
        let (fresh_root, mut fresh_lib, _) =
            deserialize_world(
                &serialize_world(fresh_world.root, &fresh_world.library).unwrap(),
            ).unwrap();
        let fresh_hash = canned_world_hash(fresh_root, fresh_lib.next_id(), fresh_lib.len());

        let loaded_root = load_save(&save_data, &mut fresh_lib, fresh_hash).unwrap();

        // Verify the edit is present.
        let loaded_world = WorldState {
            root: loaded_root,
            library: fresh_lib,
            canned_node_count: 0,
            canned_world_hash: 0,
        };
        let pos = position_from_leaf_coord([0, 0, 0]).unwrap();
        let v = get_voxel(&loaded_world, &pos);
        assert_eq!(v, stone, "edit should survive save/load round trip");
    }

    #[test]
    fn compressed_grassland_is_small() {
        let world = WorldState::new_grassland();
        let data = serialize_world(world.root, &world.library).unwrap();
        // 25 nodes, mostly uniform content → compresses well.
        assert!(
            data.len() < 50_000,
            "compressed grassland was {} bytes",
            data.len()
        );
    }
}
