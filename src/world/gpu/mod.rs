//! GPU data packing: convert tree nodes into the interleaved sparse
//! GPU buffer for the ray march shader.
//!
//! The packer produces four parallel pieces of data:
//!
//! - `tree: Vec<u32>` — interleaved header-and-children layout.
//!   Each packed node occupies `2 + 2*popcount(occupancy)` u32s:
//!   the 2-u32 header (occupancy mask + first_child offset) followed
//!   by `popcount(occupancy)` inline child entries (each 2 u32s:
//!   packed tag/block_type/pad + BFS node index). Empty slots never
//!   appear — they're encoded by a clear occupancy bit.
//! - `node_kinds: Vec<GpuNodeKind>` — indexed by BFS position.
//!   Per-node NodeKind discriminant + per-kind data (sphere body
//!   radii, cube face index).
//! - `node_offsets: Vec<u32>` — indexed by BFS position. Maps BFS
//!   position to the header's u32-offset in `tree[]`. Touched only
//!   on descent and pop-up (cold path).
//! - `parent_info: Vec<GpuParentInfo>` — indexed by BFS position.
//!   Each entry records the node's parent BFS index, the slot it
//!   occupies in its parent, and a `siblings_all_empty` flag. The
//!   shader walks this on pop-up; no side ribbon is built per frame.
//!
//! The key property of the interleaved layout is that a node's
//! header and its first non-empty child entry share a 64-byte cache
//! line, so the popcount→child chain hits L1 on the second load.
//! The per-cell DDA hot path reads only `tree[]`.
//!
//! Submodules:
//!
//! - `types`: GPU-layout types (`GpuChild`, `GpuNodeKind`,
//!   `GpuParentInfo`, etc.).
//! - `pack`: BFS packing (`pack_tree`, `pack_tree_lod`,
//!   `walk_to_frame_root`).

mod pack;
mod types;

pub use pack::{
    pack_tree, pack_tree_lod, pack_tree_lod_preserving, pack_tree_lod_selective,
    walk_to_frame_root,
};
pub use types::{GpuCamera, GpuChild, GpuNodeKind, GpuPalette, GpuParentInfo};
