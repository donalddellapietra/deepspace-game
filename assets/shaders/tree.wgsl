#include "bindings.wgsl"

// Sparse-tree access helpers for the interleaved layout. Every
// descent goes through these so the occupancy-mask + popcount
// convention lives in one place.
//
// Storage layout (see bindings.wgsl for the full spec):
//
// - `tree[header_off + 0]` = occupancy bitmask (low 27 bits).
// - `tree[header_off + 1]` = first_child_offset in tree[] u32 units.
// - `tree[first_child + rank*2 + 0]` = packed tag|block_type|pad.
// - `tree[first_child + rank*2 + 1]` = BFS node_index (if tag==2).
// - `header_off = node_offsets[bfs_idx]` — cold, only on descent.
//
// Every helper here takes a BFS node index (`node_idx`). They do the
// `node_offsets[node_idx]` indirection internally. The hot-path DDA
// in march.wgsl bypasses these helpers and reads directly from the
// cached per-depth header for zero per-cell buffer chasing.

fn header_offset(node_idx: u32) -> u32 {
    return node_offsets[node_idx];
}

fn child_empty(node_idx: u32, slot: u32) -> bool {
    let occupancy = tree[header_offset(node_idx)];
    return (occupancy & (1u << slot)) == 0u;
}

fn child_rank(occupancy: u32, slot: u32) -> u32 {
    return countOneBits(occupancy & ((1u << slot) - 1u));
}

fn child_packed(node_idx: u32, slot: u32) -> u32 {
    let h = header_offset(node_idx);
    let occupancy = tree[h];
    let bit = 1u << slot;
    if ((occupancy & bit) == 0u) {
        // Empty sentinel: tag=0 falls out of `packed & 0xFFu`.
        return 0u;
    }
    let first_child = tree[h + 1u];
    let rank = child_rank(occupancy, slot);
    return tree[first_child + rank * 2u];
}

fn child_node_index(node_idx: u32, slot: u32) -> u32 {
    let h = header_offset(node_idx);
    let occupancy = tree[h];
    let first_child = tree[h + 1u];
    let rank = child_rank(occupancy, slot);
    return tree[first_child + rank * 2u + 1u];
}

fn child_tag(packed: u32) -> u32 { return packed & 0xFFu; }
fn child_block_type(packed: u32) -> u32 { return (packed >> 8u) & 0xFFu; }

// Unpack the RGB565 LOD color the CPU packer wrote into the `_pad`
// bits of a tag=2 child entry. Used at LOD-terminal instead of a
// palette lookup so imported voxel models degrade to their averaged
// subtree color, not the single dominant palette slot.
fn child_lod_rgb(packed: u32) -> vec3<f32> {
    let p = (packed >> 16u) & 0xFFFFu;
    let r5 = (p >> 11u) & 0x1Fu;
    let g6 = (p >> 5u)  & 0x3Fu;
    let b5 =  p         & 0x1Fu;
    // 5/6-bit → unit-interval float with standard high-bit replication
    // so 0x1F maps to 1.0 exactly. Matches the CPU `rgb565(rgb)`
    // encoding (`u8 >> 3 / >> 2`) in `src/world/gpu/pack.rs`.
    let r = f32((r5 << 3u) | (r5 >> 2u)) / 255.0;
    let g = f32((g6 << 2u) | (g6 >> 4u)) / 255.0;
    let b = f32((b5 << 3u) | (b5 >> 2u)) / 255.0;
    return vec3<f32>(r, g, b);
}

fn slot_from_xyz(x: i32, y: i32, z: i32) -> u32 {
    return u32(z * 9 + y * 3 + x);
}
