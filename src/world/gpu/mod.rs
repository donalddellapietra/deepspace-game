//! GPU data packing: convert tree nodes into flat buffers for the
//! ray march shader.
//!
//! The packer produces three parallel pieces of data:
//!
//! - `tree: Vec<GpuChild>` — 27 children per node, BFS-ordered. Each
//!   child has a tag (Empty / Block / Node) and either a block_type
//!   or a buffer-local node index.
//! - `node_kinds: Vec<GpuNodeKind>` — one entry per packed node,
//!   carrying its `NodeKind` discriminant + per-kind data (sphere
//!   body radii, cube face index). The shader looks this up when
//!   it walks into a Node child to decide whether to descend with
//!   the standard Cartesian DDA or switch to the cubed-sphere DDA.
//! - `ribbon: Vec<GpuRibbonEntry>` — pop-ordered ancestors from
//!   frame's direct parent up to the absolute world root. The
//!   shader uses this to "exit upward" when rays leave the frame's
//!   `[0, 3)³` bubble: scale the ray, switch to ancestor's buffer
//!   index, continue DDA. See `ribbon.rs`.
//!
//! Submodules:
//!
//! - `types`: GPU-layout types (`GpuChild`, `GpuNodeKind`, etc.).
//! - `pack`: BFS packing (`pack_tree`, `pack_tree_lod`).
//! - `ribbon`: ancestor ribbon (`build_ribbon`, `GpuRibbonEntry`).

mod pack;
mod ribbon;
mod types;

pub use pack::{pack_tree, pack_tree_lod, pack_tree_lod_preserving};
pub use ribbon::{build_ribbon, GpuRibbonEntry};
pub use types::{GpuCamera, GpuChild, GpuNodeKind, GpuPalette, GPU_NODE_SIZE};
