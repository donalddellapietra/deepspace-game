//! BFS packing of the world tree into the interleaved sparse GPU
//! layout.
//!
//! A single storage buffer `tree: Vec<u32>` carries headers and
//! children interleaved per node. Each node occupies
//! `2 + 2*popcount(occupancy)` contiguous u32s:
//!
//! ```text
//! tree[base + 0]                     = occupancy mask (27 bits)
//! tree[base + 1]                     = first_child_offset
//! tree[first_child_offset + rank*2]     = packed (tag|block_type|pad)
//! tree[first_child_offset + rank*2 + 1] = child.node_index (BFS idx,
//!                                         valid when tag==2)
//! ```
//!
//! `first_child_offset` is an ABSOLUTE u32 offset into this same
//! `tree` buffer. In BFS layout, every node's children are packed
//! immediately after its 2-u32 header, so
//! `first_child_offset == base + 2` — the header + first child
//! entry share a 64-byte cache line.
//!
//! Two parallel side-buffers:
//!
//! - `node_kinds: Vec<GpuNodeKind>` — indexed by BFS position.
//!   Carries `NodeKind` discriminant + per-kind data (sphere body
//!   radii, cube face index). Touched only on descent / ribbon pop.
//! - `node_offsets: Vec<u32>` — indexed by BFS position. Maps a
//!   node's BFS index to its header offset in `tree[]`. Also touched
//!   only on descent / ribbon pop, so the DDA inner loop only hits
//!   `tree[]`.
//!
//! Pack functions:
//!
//! - `pack_tree`: full BFS, no LOD. Used by tests and sanity
//!   debugging.
//! - `pack_tree_lod` / `pack_tree_lod_selective`: path-local LOD.
//!   Cartesian subtrees that subtend less than `LOD_THRESHOLD`
//!   pixels in the current node's local frame get flattened into a
//!   single Block leaf (their representative block type). Sphere
//!   bodies and face cells are exempt from flattening.

use std::collections::{HashMap, HashSet};

use crate::world::anchor::{Path, WorldPos};
use crate::world::tree::{
    slot_coords, Child, NodeId, NodeKind, NodeLibrary,
    CHILDREN_PER_NODE, UNIFORM_EMPTY, UNIFORM_MIXED,
};

use super::types::{GpuChild, GpuNodeKind};

// ---------------------------------------------------------------- bricks
//
// A brick is a flat 27³ voxel grid that replaces a 3-level Cartesian
// subtree in the GPU buffer. The shader's outer DDA dispatches on
// `node_kinds[child_idx].kind == 3u` and walks a flat in-brick DDA
// instead of recursing into the sparse tree (see brick.wgsl). The
// inner DDA is dependency-chain-shorter than the recursive one — no
// popcount / rank / per-cell storage chasing.
//
// Pack-time decision: a Cartesian subtree becomes a brick when (a) its
// depth-from-leaves is exactly 3 (so it fits 27³ exactly), (b) every
// descendant is also Cartesian (no sphere bodies/faces — those have
// dedicated dispatch), and (c) the flattened brick is > 5% non-empty
// (sparse subtrees stay sparse for storage).

/// Brick edge length in cells. 3³ = three tree levels collapsed.
pub(crate) const BRICK_DIM: usize = 27;
/// Total cells per brick (27³).
pub(crate) const BRICK_VOXELS: usize = BRICK_DIM * BRICK_DIM * BRICK_DIM;
/// u32 words per brick when packing 4 cells per word, ceil-divided.
pub(crate) const BRICK_U32S: usize = (BRICK_VOXELS + 3) / 4;
/// Minimum non-empty fraction needed to brick a subtree.
///
/// **PHASE 1 RESULT: bricks regress every workload measured. Set to
/// 2.0 (impossible) to disable brick emission while keeping the
/// dispatch path in the shader for future re-enabling. See
/// `docs/testing/perf-brickmap-null-result.md` for the full writeup.**
///
/// Brief: any threshold low enough to emit bricks emits sparse ones
/// that lose to the recursive DDA's empty-representative bypass; any
/// threshold high enough to avoid sparse bricks emits zero, leaving
/// only the dispatch tax (~+1.2 ms steady-state on soldier even with
/// zero bricks). The recursive DDA on this branch has been heavily
/// optimized (AABB cull, empty-repr bypass, scalar header cache,
/// branchless min-axis); the per-cell win brickmaps capture against
/// naive sparse octrees has already been captured here, so the
/// brick's per-cell efficiency advantage has nothing left to give.
const BRICK_DENSITY_THRESHOLD: f32 = 2.0;

/// One slot in the BFS-ordered output. Sparse nodes carry a real
/// `NodeId` (their children get walked); bricks carry the offset of
/// their flattened content in `brick_data[]` and have no children to
/// walk (a stub 2-u32 header is emitted in `tree[]` for layout
/// consistency, never read by the shader).
#[derive(Copy, Clone)]
enum OrderedEntry {
    Node(NodeId),
    Brick { offset: u32 },
}

/// Memoized depth-from-leaves: max levels from `node_id` to its
/// deepest Block/Empty leaf. A node whose children are all
/// Block/Empty has depth 1 (one level above the leaves).
fn subtree_depth_from_leaves(
    library: &NodeLibrary,
    node_id: NodeId,
    cache: &mut HashMap<NodeId, u8>,
) -> u8 {
    if let Some(&d) = cache.get(&node_id) {
        return d;
    }
    let Some(node) = library.get(node_id) else {
        return 0;
    };
    let mut max_child = 0u8;
    for child in &node.children {
        if let Child::Node(child_id) = child {
            let d = subtree_depth_from_leaves(library, *child_id, cache);
            if d > max_child {
                max_child = d;
            }
        }
    }
    let d = max_child.saturating_add(1);
    cache.insert(node_id, d);
    d
}

/// Memoized: does `node_id`'s subtree (this node + every descendant)
/// use only `NodeKind::Cartesian`? Bricks can only replace pure
/// Cartesian subtrees — sphere bodies/faces have dedicated dispatch.
fn subtree_all_cartesian(
    library: &NodeLibrary,
    node_id: NodeId,
    cache: &mut HashMap<NodeId, bool>,
) -> bool {
    if let Some(&b) = cache.get(&node_id) {
        return b;
    }
    let Some(node) = library.get(node_id) else {
        return true;
    };
    if !matches!(node.kind, NodeKind::Cartesian) {
        cache.insert(node_id, false);
        return false;
    }
    for child in &node.children {
        if let Child::Node(child_id) = child {
            if !subtree_all_cartesian(library, *child_id, cache) {
                cache.insert(node_id, false);
                return false;
            }
        }
    }
    cache.insert(node_id, true);
    true
}

/// Recursively expand a Cartesian subtree into a flat 27³ palette-
/// index buffer. Caller guarantees `subtree_depth_from_leaves <= 3`,
/// so `span` divides evenly down to per-cell granularity. Block
/// children fill their span³ region with the block's palette index;
/// Empty children leave the region as 0 (the empty sentinel).
///
/// `base` is the (x,y,z) corner of this node's region in brick cells.
/// `span` is the edge length in brick cells (27 at top, 9 / 3 / 1
/// at successive levels).
fn flatten_brick_recursive(
    library: &NodeLibrary,
    node_id: NodeId,
    base: (usize, usize, usize),
    span: usize,
    out: &mut [u8],
) {
    let Some(node) = library.get(node_id) else {
        return;
    };
    let child_span = span / 3;
    for slot in 0..CHILDREN_PER_NODE {
        let (sx, sy, sz) = slot_coords(slot);
        let bx = base.0 + sx * child_span;
        let by = base.1 + sy * child_span;
        let bz = base.2 + sz * child_span;
        match node.children[slot] {
            Child::Empty => {} // already 0 = empty sentinel
            Child::Block(bt) => {
                fill_brick_region(out, bx, by, bz, child_span, bt);
            }
            Child::Node(child_id) => {
                if child_span >= 1 {
                    flatten_brick_recursive(
                        library, child_id, (bx, by, bz), child_span, out,
                    );
                }
            }
        }
    }
}

fn fill_brick_region(out: &mut [u8], bx: usize, by: usize, bz: usize, span: usize, bt: u8) {
    for z in 0..span {
        for y in 0..span {
            for x in 0..span {
                let i = (bz + z) * (BRICK_DIM * BRICK_DIM)
                    + (by + y) * BRICK_DIM
                    + (bx + x);
                out[i] = bt;
            }
        }
    }
}

/// Pack a flattened brick (19683 u8s) into 4921 u32s, little-endian
/// byte order: bits 0..7 = cell+0, 8..15 = cell+1, etc. Matches the
/// shader's `brick_cell_value` decoder.
fn pack_brick_into_u32s(brick: &[u8], out: &mut Vec<u32>) {
    debug_assert_eq!(brick.len(), BRICK_VOXELS);
    out.reserve(BRICK_U32S);
    for chunk_start in (0..BRICK_VOXELS).step_by(4) {
        let mut word = 0u32;
        for i in 0..4 {
            let idx = chunk_start + i;
            if idx < BRICK_VOXELS {
                word |= (brick[idx] as u32) << (i * 8);
            }
        }
        out.push(word);
    }
}

/// Try to convert a Cartesian subtree into a brick. Returns the
/// `brick_data` u32-offset on success, or `None` if the subtree
/// fails any eligibility gate (depth ≠ 3, contains non-Cartesian
/// nodes, or density ≤ 5%). Bricks the subtree exactly once per
/// `NodeId` — repeat calls reuse the cached offset (matches sparse-
/// tree dedup semantics).
fn try_emit_brick(
    library: &NodeLibrary,
    node_id: NodeId,
    depth_cache: &mut HashMap<NodeId, u8>,
    cart_cache: &mut HashMap<NodeId, bool>,
    brick_offset_cache: &mut HashMap<NodeId, u32>,
    brick_data: &mut Vec<u32>,
) -> Option<u32> {
    if let Some(&offset) = brick_offset_cache.get(&node_id) {
        return Some(offset);
    }
    let depth = subtree_depth_from_leaves(library, node_id, depth_cache);
    if depth != 3 {
        return None;
    }
    if !subtree_all_cartesian(library, node_id, cart_cache) {
        return None;
    }
    let mut bytes = vec![0u8; BRICK_VOXELS];
    flatten_brick_recursive(library, node_id, (0, 0, 0), BRICK_DIM, &mut bytes);
    let nonempty = bytes.iter().filter(|&&v| v != 0).count();
    if (nonempty as f32) / (BRICK_VOXELS as f32) <= BRICK_DENSITY_THRESHOLD {
        return None;
    }
    let offset = brick_data.len() as u32;
    pack_brick_into_u32s(&bytes, brick_data);
    brick_offset_cache.insert(node_id, offset);
    Some(offset)
}

/// Result of packing: (tree, node_kinds, node_offsets, brick_data, root_bfs_index).
///
/// - `tree`: single interleaved u32 buffer holding headers +
///   children inline. See module docs.
/// - `node_kinds`: per-BFS-node kind metadata. Bricks have kind=3
///   with `face` carrying the brick's u32-offset into `brick_data`.
/// - `node_offsets`: BFS index → tree[] u32-offset of that node's
///   header.
/// - `brick_data`: packed brick voxel storage, 4 cells per u32. May
///   be empty when no subtree is brick-eligible. The shader binds
///   this at binding 8.
/// - `root_bfs_index`: BFS index of the root node. The renderer
///   converts this to a tree-offset via `node_offsets[root]`.
pub type PackedTree = (Vec<u32>, Vec<GpuNodeKind>, Vec<u32>, Vec<u32>, u32);

/// Pack the visible portion of the tree into the interleaved GPU
/// buffer. Returns `(tree, node_kinds, node_offsets, brick_data, root_bfs_idx)`.
pub fn pack_tree(
    library: &NodeLibrary,
    root: NodeId,
) -> PackedTree {
    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut ordered: Vec<OrderedEntry> = Vec::new();
    let mut depth_cache: HashMap<NodeId, u8> = HashMap::new();
    let mut cart_cache: HashMap<NodeId, bool> = HashMap::new();
    let mut brick_offset_cache: HashMap<NodeId, u32> = HashMap::new();
    let mut brick_data: Vec<u32> = Vec::new();

    visit_for_pack(
        library, root,
        &mut visited, &mut ordered,
        &mut depth_cache, &mut cart_cache,
        &mut brick_offset_cache, &mut brick_data,
    );

    // Phase 2: compute each entry's occupancy + header offset.
    let n_nodes = ordered.len();
    let mut occupancies: Vec<u32> = Vec::with_capacity(n_nodes);
    for entry in &ordered {
        match entry {
            OrderedEntry::Node(nid) => {
                let node = library.get(*nid).expect("node in ordered list must exist");
                let mut occ: u32 = 0;
                for (slot, child) in node.children.iter().enumerate() {
                    if !matches!(child, Child::Empty) {
                        occ |= 1u32 << slot;
                    }
                }
                occupancies.push(occ);
            }
            // Bricks have no children — stub header with occupancy=0
            // (shader dispatches on kind==3 before reading the header).
            OrderedEntry::Brick { .. } => occupancies.push(0),
        }
    }
    let mut node_offsets: Vec<u32> = Vec::with_capacity(n_nodes);
    let mut running: u32 = 0;
    for &occ in &occupancies {
        node_offsets.push(running);
        running = running + 2 + 2 * occ.count_ones();
    }
    let total_u32s = running as usize;

    // Phase 3: emit interleaved tree[].
    let mut tree: Vec<u32> = Vec::with_capacity(total_u32s);
    let mut kinds: Vec<GpuNodeKind> = Vec::with_capacity(n_nodes);
    for (i, entry) in ordered.iter().enumerate() {
        let header_off = node_offsets[i];
        let first_child_off = header_off + 2;
        debug_assert_eq!(tree.len() as u32, header_off);
        match entry {
            OrderedEntry::Brick { offset } => {
                kinds.push(GpuNodeKind::brick(*offset));
                tree.push(0); // occupancy = 0 (no children)
                tree.push(first_child_off);
            }
            OrderedEntry::Node(nid) => {
                let node = library.get(*nid).expect("node in ordered list must exist");
                kinds.push(GpuNodeKind::from_node_kind(node.kind));
                let occupancy = occupancies[i];
                tree.push(occupancy);
                tree.push(first_child_off);
                for (slot, child) in node.children.iter().enumerate() {
                    let gc = match child {
                        Child::Empty => continue,
                        Child::Block(bt) => GpuChild {
                            tag: 1, block_type: *bt, _pad: 0, node_index: 0,
                        },
                        Child::Node(child_id) => {
                            let child_bfs = *visited.get(child_id)
                                .expect("child must be visited");
                            let child_aabb = brick_or_node_child_aabb(
                                &ordered, &occupancies, child_bfs,
                            );
                            let repr = library
                                .get(*child_id)
                                .map(|n| n.representative_block)
                                .unwrap_or(0);
                            GpuChild {
                                tag: 2,
                                block_type: repr,
                                _pad: child_aabb,
                                node_index: child_bfs,
                            }
                        }
                    };
                    debug_assert_ne!(occupancy & (1u32 << slot), 0);
                    tree.push(pack_child_first(gc));
                    tree.push(gc.node_index);
                }
            }
        }
    }
    debug_assert_eq!(tree.len(), total_u32s);

    let root_idx = *visited.get(&root).unwrap();
    (tree, kinds, node_offsets, brick_data, root_idx)
}

/// AABB hint for a child. Bricks span their full parent cell (the
/// flatten lays out cells densely), so `_pad=0` (the shader treats
/// 0 as "no AABB" and uses the full [0, 3)³ box). Sparse nodes use
/// the existing slot-granular content AABB.
fn brick_or_node_child_aabb(
    ordered: &[OrderedEntry],
    occupancies: &[u32],
    child_bfs: u32,
) -> u16 {
    match ordered[child_bfs as usize] {
        OrderedEntry::Brick { .. } => 0,
        OrderedEntry::Node(_) => content_aabb(occupancies[child_bfs as usize]),
    }
}

/// Flat BFS visit shared by `pack_tree` and `pack_tree_lod_selective`'s
/// no-LOD-flatten code path. Walks every reachable node, deciding for
/// each whether to make it a sparse `Node` entry (and enqueue its
/// children) or a flat `Brick` entry (no children). Pushes phantom
/// kind entries to `ordered` and packed brick voxels to `brick_data`.
///
/// Used directly by `pack_tree`. The LOD-aware packer does its own
/// BFS (the LOD/preserve logic is interleaved per-slot) but reuses
/// the same brick decision via `try_emit_brick`.
fn visit_for_pack(
    library: &NodeLibrary,
    root: NodeId,
    visited: &mut HashMap<NodeId, u32>,
    ordered: &mut Vec<OrderedEntry>,
    depth_cache: &mut HashMap<NodeId, u8>,
    cart_cache: &mut HashMap<NodeId, bool>,
    brick_offset_cache: &mut HashMap<NodeId, u32>,
    brick_data: &mut Vec<u32>,
) {
    visited.insert(root, 0);
    ordered.push(OrderedEntry::Node(root));
    let mut head = 0usize;
    while head < ordered.len() {
        let entry = ordered[head];
        head += 1;
        let nid = match entry {
            OrderedEntry::Node(nid) => nid,
            OrderedEntry::Brick { .. } => continue,
        };
        let Some(node) = library.get(nid) else { continue };
        for child in &node.children {
            let Child::Node(child_id) = child else { continue };
            if visited.contains_key(child_id) {
                continue;
            }
            // Try brick first. If eligible, emit phantom brick entry
            // (no children to enqueue). Otherwise, sparse Node entry.
            if let Some(brick_offset) = try_emit_brick(
                library, *child_id,
                depth_cache, cart_cache,
                brick_offset_cache, brick_data,
            ) {
                let idx = ordered.len() as u32;
                visited.insert(*child_id, idx);
                ordered.push(OrderedEntry::Brick { offset: brick_offset });
            } else {
                let idx = ordered.len() as u32;
                visited.insert(*child_id, idx);
                ordered.push(OrderedEntry::Node(*child_id));
            }
        }
    }
}

/// Encode a child's first u32: tag | (block_type << 8) | (_pad << 16).
fn pack_child_first(c: GpuChild) -> u32 {
    (c.tag as u32) | ((c.block_type as u32) << 8) | ((c._pad as u32) << 16)
}

/// Compute a tight axis-aligned bounding box (slot-granular) of the
/// occupied slots in a 3×3×3 node. Returns a 12-bit packed value:
///
/// ```text
/// bits  0-1: min_x (0..=2, inclusive)
/// bits  2-3: min_y
/// bits  4-5: min_z
/// bits  6-7: max_x (0..=2, inclusive — shader treats as max+1 exclusive)
/// bits  8-9: max_y
/// bits 10-11: max_z
/// ```
///
/// Used by the ray-march shader to cull descents: on a non-empty
/// child, ray-box-test against the child's content AABB before
/// committing to the child DDA. If the ray misses the AABB, skip
/// the descent entirely. See `assets/shaders/march.wgsl`.
///
/// For an empty occupancy mask (shouldn't happen for a referenced
/// child), returns `0` — the shader treats an all-zero AABB as a
/// degenerate "no content" box that never hits, so a miss falls
/// through to the usual DDA path.
pub(crate) fn content_aabb(occupancy: u32) -> u16 {
    if occupancy == 0 {
        return 0;
    }
    let mut min_x = 3u32;
    let mut max_x = 0u32;
    let mut min_y = 3u32;
    let mut max_y = 0u32;
    let mut min_z = 3u32;
    let mut max_z = 0u32;
    for slot in 0..27u32 {
        if (occupancy >> slot) & 1 == 0 {
            continue;
        }
        let x = slot % 3;
        let y = (slot / 3) % 3;
        let z = slot / 9;
        if x < min_x { min_x = x; }
        if x > max_x { max_x = x; }
        if y < min_y { min_y = y; }
        if y > max_y { max_y = y; }
        if z < min_z { min_z = z; }
        if z > max_z { max_z = z; }
    }
    ((min_x << 0) | (min_y << 2) | (min_z << 4)
        | (max_x << 6) | (max_y << 8) | (max_z << 10)) as u16
}

fn child_screen_pixels(
    camera: &WorldPos,
    node_path: &Path,
    slot: usize,
    screen_height: f32,
    fov: f32,
) -> f32 {
    let camera_local = camera.in_frame(node_path);
    let (cx, cy, cz) = slot_coords(slot);
    let child_center = [cx as f32 + 0.5, cy as f32 + 0.5, cz as f32 + 0.5];
    let dx = child_center[0] - camera_local[0];
    let dy = child_center[1] - camera_local[1];
    let dz = child_center[2] - camera_local[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt().max(0.001);
    let half_fov_recip = screen_height / (2.0 * (fov * 0.5).tan());
    half_fov_recip / dist
}

/// LOD-aware tree packing: only uploads nodes large enough to see.
pub fn pack_tree_lod(
    library: &NodeLibrary,
    root: NodeId,
    camera: &WorldPos,
    screen_height: f32,
    fov: f32,
) -> PackedTree {
    pack_tree_lod_selective(library, root, camera, screen_height, fov, &[], &[])
}

/// Like `pack_tree_lod`, but with one or more `preserve_paths`.
pub fn pack_tree_lod_preserving(
    library: &NodeLibrary,
    root: NodeId,
    camera: &WorldPos,
    screen_height: f32,
    fov: f32,
    preserve_paths: &[&[u8]],
) -> PackedTree {
    pack_tree_lod_selective(library, root, camera, screen_height, fov, preserve_paths, &[])
}

/// Like `pack_tree_lod_preserving`, but also supports bounded preserve regions.
pub fn pack_tree_lod_selective(
    library: &NodeLibrary,
    root: NodeId,
    camera: &WorldPos,
    screen_height: f32,
    fov: f32,
    preserve_paths: &[&[u8]],
    preserve_regions: &[(Path, u8)],
) -> PackedTree {
    const LOD_THRESHOLD: f32 = 0.5;

    let mut preserve_pairs: HashSet<(NodeId, u8)> = HashSet::new();
    for preserve_path in preserve_paths {
        let mut current = root;
        for &slot in *preserve_path {
            preserve_pairs.insert((current, slot));
            let Some(node) = library.get(current) else { break };
            match node.children[slot as usize] {
                Child::Node(child_id) => current = child_id,
                _ => break,
            }
        }
    }

    let preserve_regions: Vec<(Path, u8)> = preserve_regions.to_vec();

    fn in_preserve_region(path: &Path, preserve_regions: &[(Path, u8)]) -> bool {
        preserve_regions.iter().any(|(region_root, extra_depth)| {
            let required_depth = region_root.depth().saturating_add(*extra_depth);
            path.depth() <= required_depth
                && path.common_prefix_len(region_root) == region_root.depth()
        })
    }

    struct QueueEntry {
        node_id: NodeId,
        path: Path,
        ordered_idx: u32,
    }

    /// Per-slot pack result for a single packed node. `None` =
    /// empty (will be absent from the interleaved tree); `Some(child)`
    /// with `child.tag in {1, 2}` = non-empty entry.
    type SlotOverride = [Option<GpuChild>; CHILDREN_PER_NODE];

    let mut visited: HashMap<NodeId, u32> = HashMap::new();
    let mut queue: Vec<QueueEntry> = Vec::new();
    let mut ordered: Vec<OrderedEntry> = Vec::new();
    let mut per_node: Vec<SlotOverride> = Vec::new();
    let mut depth_cache: HashMap<NodeId, u8> = HashMap::new();
    let mut cart_cache: HashMap<NodeId, bool> = HashMap::new();
    let mut brick_offset_cache: HashMap<NodeId, u32> = HashMap::new();
    let mut brick_data: Vec<u32> = Vec::new();

    visited.insert(root, 0);
    ordered.push(OrderedEntry::Node(root));
    per_node.push([None; CHILDREN_PER_NODE]);
    queue.push(QueueEntry { node_id: root, path: Path::root(), ordered_idx: 0 });

    let mut head = 0usize;
    while head < queue.len() {
        let entry = &queue[head];
        let node_id = entry.node_id;
        let node_path = entry.path;
        let ordered_idx = entry.ordered_idx as usize;
        head += 1;

        let Some(node) = library.get(node_id) else { continue };
        let lod_active = matches!(node.kind, NodeKind::Cartesian);

        for (slot, child) in node.children.iter().enumerate() {
            let stored = match child {
                Child::Empty => None,
                Child::Block(bt) => Some(GpuChild {
                    tag: 1, block_type: *bt, _pad: 0, node_index: 0,
                }),
                Child::Node(_) => None,
            };
            if let Some(c) = stored {
                per_node[ordered_idx][slot] = Some(c);
            }

            let Child::Node(child_id) = child else { continue };
            let Some(child_node) = library.get(*child_id) else { continue };

            let mut child_path = node_path;
            child_path.push(slot as u8);
            let on_preserve = preserve_pairs.contains(&(node_id, slot as u8))
                || in_preserve_region(&child_path, &preserve_regions);
            let child_is_cartesian = matches!(child_node.kind, NodeKind::Cartesian);

            if !on_preserve && lod_active && child_is_cartesian
                && child_node.uniform_type != UNIFORM_MIXED
            {
                per_node[ordered_idx][slot] = if child_node.uniform_type == UNIFORM_EMPTY {
                    None
                } else {
                    Some(GpuChild {
                        tag: 1,
                        block_type: child_node.uniform_type,
                        _pad: 0,
                        node_index: 0,
                    })
                };
                continue;
            }

            if !on_preserve && lod_active && child_is_cartesian {
                let screen_pixels =
                    child_screen_pixels(camera, &node_path, slot, screen_height, fov);
                if screen_pixels < LOD_THRESHOLD {
                    per_node[ordered_idx][slot] = if child_node.representative_block < 255 {
                        Some(GpuChild {
                            tag: 1,
                            block_type: child_node.representative_block,
                            _pad: 0,
                            node_index: 0,
                        })
                    } else {
                        None
                    };
                    continue;
                }
            }

            // Brick decision. Only fires for first visits of a NodeId
            // (subsequent references reuse the visited entry, which
            // may already be a brick or sparse). The brick check
            // happens AFTER LOD flattening — LOD-flattened slots
            // already `continue`d; we only consider bricks for slots
            // that survived as sparse descents.
            let repr = child_node.representative_block;
            if !visited.contains_key(child_id) {
                if let Some(brick_offset) = try_emit_brick(
                    library, *child_id,
                    &mut depth_cache, &mut cart_cache,
                    &mut brick_offset_cache, &mut brick_data,
                ) {
                    let idx = ordered.len() as u32;
                    visited.insert(*child_id, idx);
                    ordered.push(OrderedEntry::Brick { offset: brick_offset });
                    per_node.push([None; CHILDREN_PER_NODE]);
                    per_node[ordered_idx][slot] = Some(GpuChild {
                        tag: 2, block_type: repr, _pad: 0, node_index: idx,
                    });
                    continue;
                }
                let idx = ordered.len() as u32;
                visited.insert(*child_id, idx);
                ordered.push(OrderedEntry::Node(*child_id));
                per_node.push([None; CHILDREN_PER_NODE]);
                queue.push(QueueEntry {
                    node_id: *child_id,
                    path: child_path,
                    ordered_idx: idx,
                });
            }
            let child_idx = *visited.get(child_id).expect("child just visited");
            per_node[ordered_idx][slot] = Some(GpuChild {
                tag: 2, block_type: repr, _pad: 0, node_index: child_idx,
            });
        }
    }

    // Phase 2: compute per-node header offsets. Bricks contribute
    // a stub 2-u32 header (occupancy=0, no children).
    let n_nodes = ordered.len();
    let mut occupancies: Vec<u32> = Vec::with_capacity(n_nodes);
    for (i, slots) in per_node.iter().enumerate() {
        match &ordered[i] {
            OrderedEntry::Brick { .. } => occupancies.push(0),
            OrderedEntry::Node(_) => {
                let mut occ: u32 = 0;
                for (slot, entry) in slots.iter().enumerate() {
                    if entry.is_some() {
                        occ |= 1u32 << slot;
                    }
                }
                occupancies.push(occ);
            }
        }
    }
    let mut node_offsets: Vec<u32> = Vec::with_capacity(n_nodes);
    let mut running: u32 = 0;
    for &occ in &occupancies {
        node_offsets.push(running);
        running = running + 2 + 2 * occ.count_ones();
    }
    let total_u32s = running as usize;

    // Phase 3: emit interleaved tree[]. Bricks emit a stub header +
    // a Brick kind entry; the shader dispatches on kind==3 before
    // ever reading the brick's tree[] header.
    let mut tree: Vec<u32> = Vec::with_capacity(total_u32s);
    let mut kinds: Vec<GpuNodeKind> = Vec::with_capacity(n_nodes);
    for (i, ord_entry) in ordered.iter().enumerate() {
        let header_off = node_offsets[i];
        let first_child_off = header_off + 2;
        debug_assert_eq!(tree.len() as u32, header_off);
        match ord_entry {
            OrderedEntry::Brick { offset } => {
                kinds.push(GpuNodeKind::brick(*offset));
                tree.push(0); // occupancy = 0
                tree.push(first_child_off);
            }
            OrderedEntry::Node(node_id) => {
                let node = library.get(*node_id).expect("node in ordered list must exist");
                kinds.push(GpuNodeKind::from_node_kind(node.kind));
                let occupancy = occupancies[i];
                tree.push(occupancy);
                tree.push(first_child_off);
                for slot in 0..CHILDREN_PER_NODE {
                    if let Some(mut entry) = per_node[i][slot] {
                        // For tag=2 node children, stash the child's content
                        // AABB in `_pad` so the shader can ray-box-cull the
                        // descent before committing. Brick children skip
                        // the AABB (they're dense across the full cell);
                        // tag=1 block leaves have no subtree.
                        if entry.tag == 2 {
                            entry._pad = brick_or_node_child_aabb(
                                &ordered, &occupancies, entry.node_index,
                            );
                        }
                        tree.push(pack_child_first(entry));
                        tree.push(entry.node_index);
                    }
                }
            }
        }
    }
    debug_assert_eq!(tree.len(), total_u32s);

    let root_idx = *visited.get(&root).unwrap();
    (tree, kinds, node_offsets, brick_data, root_idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::anchor::WorldPos;
    use crate::world::bootstrap::{menger_world, plain_test_world, plain_world};
    use crate::world::tree::{empty_children, uniform_children, CENTER_SLOT};

    /// Read the child at (bfs_idx, slot) from a packed tree. Returns
    /// a synthesized `tag=0` GpuChild when the slot is empty.
    pub(super) fn sparse_child(
        tree: &[u32],
        node_offsets: &[u32],
        bfs_idx: u32,
        slot: u8,
    ) -> GpuChild {
        let header_off = node_offsets[bfs_idx as usize] as usize;
        let occupancy = tree[header_off];
        let first_child = tree[header_off + 1] as usize;
        let bit = 1u32 << slot;
        if occupancy & bit == 0 {
            return GpuChild { tag: 0, block_type: 0, _pad: 0, node_index: 0 };
        }
        let rank = (occupancy & (bit - 1)).count_ones() as usize;
        let off = first_child + rank * 2;
        let packed = tree[off];
        let tag = (packed & 0xFF) as u8;
        let block_type = ((packed >> 8) & 0xFF) as u8;
        let _pad = ((packed >> 16) & 0xFFFF) as u16;
        let node_index = tree[off + 1];
        GpuChild { tag, block_type, _pad, node_index }
    }

    #[test]
    fn pack_test_world() {
        let world = plain_test_world();
        let (tree, kinds, node_offsets, _brick_data, root_idx) = pack_tree(&world.library, world.root);
        assert_eq!(root_idx, 0);
        assert_eq!(kinds.len(), world.library.len());
        assert_eq!(node_offsets.len(), world.library.len());
        assert!(!tree.is_empty());
        // Root at offset 0.
        assert_eq!(node_offsets[0], 0);
        for kind in &kinds {
            assert_eq!(kind.kind, 0);
        }
    }

    fn planet_world() -> crate::world::state::WorldState {
        let mut lib = NodeLibrary::default();
        let leaf_air = lib.insert(empty_children());
        let mut root_children = uniform_children(Child::Node(leaf_air));
        let body_id = lib.insert_with_kind(
            empty_children(),
            NodeKind::CubedSphereBody { inner_r: 0.12, outer_r: 0.45 },
        );
        root_children[CENTER_SLOT] = Child::Node(body_id);
        let root = lib.insert(root_children);
        lib.ref_inc(root);
        crate::world::state::WorldState { root, library: lib }
    }

    fn camera_at(xyz: [f32; 3]) -> WorldPos {
        // Root-frame-local coords at shallow depth, then deepened.
        WorldPos::from_frame_local(&crate::world::anchor::Path::root(), xyz, 2)
            .deepened_to(10)
    }

    #[test]
    fn pack_includes_body_kind_and_radii() {
        let world = planet_world();
        let camera = camera_at([1.5, 2.0, 1.5]);
        let (_tree, kinds, _offsets, _brick_data, _root_idx) = pack_tree_lod(
            &world.library, world.root, &camera, 1080.0, 1.2,
        );
        let body = kinds.iter().find(|kind| kind.kind == 1).expect("body kind in buffer");
        assert!((body.inner_r - 0.12).abs() < 1e-6);
        assert!((body.outer_r - 0.45).abs() < 1e-6);
    }

    #[test]
    fn pack_lod_flattens_far_uniform_cartesian() {
        let world = planet_world();
        let camera = camera_at([1.5, 2.0, 1.5]);
        let (tree, _kinds, offsets, _brick_data, _root_idx) = pack_tree_lod(
            &world.library, world.root, &camera, 1080.0, 1.2,
        );
        // Slot 0 (uniform-empty) has no bit in the root's occupancy.
        assert_eq!(sparse_child(&tree, &offsets, 0, 0).tag, 0);
        // Slot 13 (body) is present as a Node.
        let body_entry = sparse_child(&tree, &offsets, 0, 13);
        assert_eq!(body_entry.tag, 2);
        assert!(body_entry.node_index > 0);
    }

    #[test]
    fn pack_planet_body_present() {
        let world = planet_world();
        let camera = camera_at([1.5, 2.0, 1.5]);
        let (_tree, kinds, _offsets, _brick_data, _root_idx) = pack_tree_lod(
            &world.library, world.root, &camera, 1080.0, 1.2,
        );
        assert!(kinds.iter().any(|kind| kind.kind == 1), "body kind present");
    }

    #[test]
    fn preserve_path_prevents_uniform_collapse() {
        let mut lib = NodeLibrary::default();
        let air = lib.insert(empty_children());
        let root = lib.insert(uniform_children(Child::Node(air)));
        lib.ref_inc(root);
        let camera = camera_at([1.5, 2.0, 1.5]);

        let (tree, _, offsets, _brick_data, _) = pack_tree_lod(
            &lib, root, &camera, 1080.0, 1.2,
        );
        assert_eq!(sparse_child(&tree, &offsets, 0, 16).tag, 0);

        let (tree2, _, offsets2, _brick_data, _) = pack_tree_lod_preserving(
            &lib, root, &camera, 1080.0, 1.2,
            &[&[16u8]],
        );
        let preserved = sparse_child(&tree2, &offsets2, 0, 16);
        assert_eq!(preserved.tag, 2);
        assert!(preserved.node_index > 0);
    }

    #[test]
    fn preserve_path_chain_lets_ribbon_descend_to_depth_n() {
        use super::super::ribbon::build_ribbon;

        let mut lib = NodeLibrary::default();
        let mut node = lib.insert(empty_children());
        for _ in 1..10 {
            node = lib.insert(uniform_children(Child::Node(node)));
        }
        let root = node;
        lib.ref_inc(root);
        let camera = camera_at([1.5, 1.5, 1.5]);
        let path = [13u8; 9];

        let (tree, _kinds, offsets, _brick_data, _root_idx) = pack_tree_lod_preserving(
            &lib, root, &camera, 1080.0, 1.2,
            &[&path],
        );
        let ribbon = build_ribbon(&tree, &offsets, &path);
        assert_eq!(ribbon.reached_slots.len(), 9);
        assert_eq!(ribbon.ribbon.len(), 9);
    }

    #[test]
    fn preserve_path_only_affects_chain_slots() {
        let mut lib = NodeLibrary::default();
        let air = lib.insert(empty_children());
        let root = lib.insert(uniform_children(Child::Node(air)));
        lib.ref_inc(root);
        let camera = camera_at([1.5, 2.0, 1.5]);

        let (tree, _, offsets, _brick_data, _) = pack_tree_lod_preserving(
            &lib, root, &camera, 1080.0, 1.2,
            &[&[16u8]],
        );
        assert_eq!(sparse_child(&tree, &offsets, 0, 16).tag, 2);
        for sibling in [0u8, 5, 13, 26] {
            assert_eq!(
                sparse_child(&tree, &offsets, 0, sibling).tag, 0,
                "sibling slot {sibling} should be flattened",
            );
        }
    }

    #[test]
    fn pack_lod_keeps_near_subtrees_full() {
        let mut lib = NodeLibrary::default();
        let air = lib.insert(empty_children());
        let mut mixed = empty_children();
        mixed[0] = Child::Block(crate::world::palette::block::STONE);
        let mixed_node = lib.insert(mixed);
        let mut root_children = uniform_children(Child::Node(air));
        root_children[CENTER_SLOT] = Child::Node(mixed_node);
        let root = lib.insert(root_children);
        lib.ref_inc(root);

        let camera = camera_at([1.5, 1.5, 1.6]);
        let (tree, _kinds, offsets, _brick_data, _root_idx) = pack_tree_lod(
            &lib, root, &camera, 1080.0, 1.2,
        );
        assert_eq!(sparse_child(&tree, &offsets, 0, CENTER_SLOT as u8).tag, 2);
    }

    /// Baseline-capture test: builds a Menger sponge at depth 5 and
    /// packs it without LOD. Asserts the interleaved tree buffer stays
    /// below a hardcoded cap so silent pack-size drift (e.g. the 3.7%
    /// regression on fully-occupied nodes under sparse) trips CI.
    #[test]
    fn menger_pack_size_regression() {
        let world = menger_world(5);
        let (tree, _kinds, _offsets, _brick_data, _root) = pack_tree(&world.library, world.root);
        let u32s = tree.len();
        eprintln!(
            "menger depth=5 pack size: {} u32s ({} bytes)",
            u32s, u32s * 4
        );
        // Measured baseline (sparse interleaved layout): 210 u32s.
        // Threshold is ~1.5x the measured baseline so normal churn
        // doesn't trip it but material regressions (>50%) do.
        const MENGER_D5_MAX_U32S: usize = 320;
        assert!(
            u32s < MENGER_D5_MAX_U32S,
            "menger depth=5 pack size regressed: {} u32s (> {} u32s)",
            u32s, MENGER_D5_MAX_U32S,
        );
    }

    /// Baseline-capture test for the plain-world preset.
    #[test]
    fn plain_pack_size_regression() {
        let world = plain_world(5);
        let (tree, _kinds, _offsets, _brick_data, _root) = pack_tree(&world.library, world.root);
        let u32s = tree.len();
        eprintln!(
            "plain layers=5 pack size: {} u32s ({} bytes)",
            u32s, u32s * 4
        );
        // Measured baseline (sparse interleaved layout): 898 u32s.
        const PLAIN_L5_MAX_U32S: usize = 1400;
        assert!(
            u32s < PLAIN_L5_MAX_U32S,
            "plain layers=5 pack size regressed: {} u32s (> {} u32s)",
            u32s, PLAIN_L5_MAX_U32S,
        );
    }

    /// Verify that breaking a block at various depths actually changes
    /// the packed GPU tree data. This catches dedup bugs where the
    /// library returns the same node after a break, producing an
    /// identical packed buffer.
    #[test]
    fn break_at_every_depth_changes_packed_data() {
        use crate::world::anchor::Path;
        use crate::world::bootstrap;
        use crate::world::edit;
        use crate::world::raycast;

        for spawn_depth in [4u8, 8, 11, 15, 20, 25, 30, 33, 38] {
            let boot = bootstrap::bootstrap_world(
                bootstrap::WorldPreset::PlainTest,
                Some(40),
            );
            let mut world = boot.world;
            let pos = bootstrap::plain_surface_spawn(spawn_depth);
            bootstrap::carve_air_pocket(&mut world, &pos.anchor, 40);

            let camera = pos;
            let frame_depth = spawn_depth.saturating_sub(3);
            let mut frame_path = camera.anchor;
            frame_path.truncate(frame_depth);

            let preserve_paths: Vec<&[u8]> = vec![frame_path.as_slice()];
            let frame_path_owned = frame_path;
            let preserve_regions = vec![(frame_path_owned, 3u8)];

            let (tree_before, _, offsets_before, _brick_data, _) = pack_tree_lod_selective(
                &world.library,
                world.root,
                &camera,
                720.0,
                1.2,
                &preserve_paths,
                &preserve_regions,
            );

            // CPU raycast to find a block to break.
            let ray_origin = camera.in_frame(&Path::root());
            let ray_dir = [0.0f32, -0.4, -0.9];
            let hit = raycast::cpu_raycast(
                &world.library,
                world.root,
                ray_origin,
                ray_dir,
                spawn_depth as u32,
            );
            let Some(hit) = hit else {
                panic!("no raycast hit at spawn_depth={spawn_depth}");
            };
            assert!(
                edit::break_block(&mut world, &hit),
                "break_block returned false at spawn_depth={spawn_depth}",
            );

            let (tree_after, _, offsets_after, _brick_data, _) = pack_tree_lod_selective(
                &world.library,
                world.root,
                &camera,
                720.0,
                1.2,
                &preserve_paths,
                &preserve_regions,
            );

            assert!(
                tree_before != tree_after || offsets_before != offsets_after,
                "packed GPU data unchanged after break at spawn_depth={spawn_depth} \
                 (library dedup may be collapsing the edit)",
            );
        }
    }
}
