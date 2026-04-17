#include "bindings.wgsl"

// Sparse-tree access helpers. Every descent goes through these so
// the occupancy-mask + popcount convention lives in one place.
//
// Storage layout:
// - `nodes[n]` is a NodeHeader: occupancy bitmask (low 27 bits) +
//   first_child offset into the compact `tree` child array.
// - `tree` holds 2 u32s per non-empty child (packed = tag|block_type|pad,
//   node_index). Empty slots are absent; read them via `child_empty`.

fn child_empty(node_idx: u32, slot: u32) -> bool {
    let occupancy = nodes[node_idx].occupancy;
    return (occupancy & (1u << slot)) == 0u;
}

fn child_rank(occupancy: u32, slot: u32) -> u32 {
    return countOneBits(occupancy & ((1u << slot) - 1u));
}

fn child_packed(node_idx: u32, slot: u32) -> u32 {
    let header = nodes[node_idx];
    let bit = 1u << slot;
    if ((header.occupancy & bit) == 0u) {
        // Empty sentinel: tag=0 falls out of `packed & 0xFFu`.
        return 0u;
    }
    let rank = child_rank(header.occupancy, slot);
    return tree[(header.first_child + rank) * 2u];
}

fn child_node_index(node_idx: u32, slot: u32) -> u32 {
    let header = nodes[node_idx];
    let rank = child_rank(header.occupancy, slot);
    return tree[(header.first_child + rank) * 2u + 1u];
}

fn child_tag(packed: u32) -> u32 { return packed & 0xFFu; }
fn child_block_type(packed: u32) -> u32 { return (packed >> 8u) & 0xFFu; }

fn slot_from_xyz(x: i32, y: i32, z: i32) -> u32 {
    return u32(z * 9 + y * 3 + x);
}
