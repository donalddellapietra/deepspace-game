//! GPU data packing: convert tree nodes into sparse GPU buffers for
//! the ray march shader.
//!
//! The packer produces four parallel pieces of data:
//!
//! - `nodes: Vec<NodeHeader>` — one 8-byte header per packed node.
//!   Each header carries a 27-bit occupancy bitmask (bit `s` set iff
//!   slot `s` is non-empty) plus the offset into `children` where
//!   that node's non-empty entries begin. Bits 27..31 are reserved.
//! - `children: Vec<GpuChild>` — compact array of non-empty children,
//!   packed in slot-ascending order per node. Empty slots never
//!   appear — they're encoded by a clear occupancy bit.
//! - `node_kinds: Vec<GpuNodeKind>` — one entry per packed node,
//!   carrying its `NodeKind` discriminant + per-kind data (sphere
//!   body radii, cube face index).
//! - `ribbon: Vec<GpuRibbonEntry>` — pop-ordered ancestors from
//!   frame's direct parent up to the absolute world root. See
//!   `ribbon.rs`.
//!
//! Submodules:
//!
//! - `types`: GPU-layout types (`NodeHeader`, `GpuChild`, etc.).
//! - `pack`: BFS packing (`pack_tree`, `pack_tree_lod`).
//! - `ribbon`: ancestor ribbon (`build_ribbon`, `GpuRibbonEntry`).

mod pack;
mod ribbon;
mod types;

pub use pack::{pack_tree, pack_tree_lod, pack_tree_lod_preserving, pack_tree_lod_selective};
pub use ribbon::{build_ribbon, GpuRibbonEntry};
pub use types::{GpuCamera, GpuChild, GpuNodeKind, GpuPalette, NodeHeader};
