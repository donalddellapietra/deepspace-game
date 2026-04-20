# Phase 1 — Perf infrastructure

**Goal.** Establish the scene-scale streaming machinery that all later
lighting phases depend on. No visual change after this phase — pure
infra.

**Dependencies.** None. This is the foundation.

**Deliverables.**
- Chunk pool with NodeId-keyed indirection
- Per-frame visibility buffer (visible nodes + voxels)
- Indirect dispatch args buffer
- `.vxs` on-disk format with zlib compression
- `tools/scene_voxelize` writes `.vxs`; runtime loads and streams

## Why this must come first

Lighting passes (shadow, AO, GI, specular) naively run per-pixel:
1920×1080 ≈ 2M threads/frame. With a visibility buffer the same
passes dispatch over visible geometry — ~50k voxels for a typical
camera. That's a 40× multiplier, and it's the difference between
"lighting is affordable" and "lighting is not".

The chunk pool is what makes the visibility buffer small enough to
be a dense list. Without it, "voxel ID" is an ambiguous concept
(what's a voxel in a content-addressed tree?) and visibility
tracking degenerates into screen-space rehashing.

## Architecture

### Chunk pool

A fixed-size GPU buffer of chunk slots, each holding one node's
children payload. CPU-side residency table maps `NodeId → slot_idx`
or `NOT_RESIDENT`.

```rust
// src/world/gpu/chunk_pool.rs (new)
pub struct ChunkPool {
    pub slots: Vec<ChunkSlot>,          // capacity = N (runtime-derived)
    pub residency: HashMap<NodeId, u32>, // NodeId → slot_idx
    pub lru: VecDeque<u32>,             // eviction order
    pub gpu_buffer: wgpu::Buffer,       // storage buffer, N × CHUNK_SIZE
}

#[repr(C)]
pub struct ChunkSlot {
    pub children: [GpuChild; 27],       // 27 × 8 B = 216 B
    pub node_kind: GpuNodeKind,         // 16 B (union tag + sphere radii)
    pub uniform_type: u32,
    pub representative_block: u32,
    // pad to 256 B for alignment
}
```

Node indirection in WGSL:

```wgsl
// Before (Phase 0):
let child = tree[parent.node_index * 27u + slot];

// After (Phase 1):
let slot_idx = pool_residency[node_id % POOL_SIZE];  // GPU hashmap probe
let child = pool[slot_idx].children[slot];
```

The residency table on GPU is a flat open-addressing hashmap keyed
by `NodeId`. Linear probing with 75% load factor. Probe length is
bounded (<4 in practice for well-chosen pool size).

### Pool sizing

`POOL_SIZE` is derived at runtime from:
```
POOL_SIZE = max_visible_nodes × safety_factor
         = (frame_volume_nodes + ribbon_length) × 1.5
```
For a `K=3` frame (27³ = 20k cells at leaf) plus `MAX_DEPTH = 63`
ribbon, ~30k slots is enough. Each slot is 256 B, so ~8 MB. Scales
with scene complexity, not scene size — content-addressed dedup
means two identical subtrees share one slot.

### Visibility buffer

Two buffers:

```rust
// Atomic counter + hashmap from NodeId to slot-local voxel coord
pub struct VisibilityBuffer {
    pub hit_count: wgpu::Buffer,        // u32 atomic
    pub hit_map: wgpu::Buffer,          // [HitEntry; MAX_HITS] open-addr
    pub hit_list: wgpu::Buffer,         // dense compacted list (secondary)
}

#[repr(C)]
pub struct HitEntry {
    pub node_id: u64,            // content-addressed key
    pub voxel_slot: u32,         // 0..27 within the node
    pub surface_data: u32,       // packed: hit face + material bits
}
```

Primary raymarch shader appends each first-hit voxel to `hit_map`
via `atomicCAS` on the hash slot. Dedup is automatic — rays hitting
the same voxel from different pixels increment nothing and re-find
the existing entry.

A small compaction compute pass then writes `hit_list[0..hit_count]`
for dense iteration in secondary passes.

Sizing: `MAX_HITS = screen_pixels / 8` (typical depth complexity ~1,
but hits cluster). For 1080p that's ~260k entries × 16 B = 4 MB.

### Indirect dispatch

```wgsl
// assets/shaders/indirect_args.wgsl (new)
@compute @workgroup_size(1)
fn build_args() {
    let count = atomicLoad(&visibility.hit_count);
    indirect_args.workgroup_count_x = (count + 63u) / 64u;
    indirect_args.workgroup_count_y = 1u;
    indirect_args.workgroup_count_z = 1u;
}
```

Every secondary pass does:
```rust
pass.dispatch_workgroups_indirect(&indirect_args_buffer, 0);
```

### `.vxs` file format

```
┌─────────────────────────────────────────────┐
│ Header (JSON, length-prefixed)              │
│   { version, layer_count, root_node_id,     │
│     palette_entries, chunk_count, ... }     │
├─────────────────────────────────────────────┤
│ Palette      (N × 32 B PBR material records)│
├─────────────────────────────────────────────┤
│ Chunk index  ([NodeId; chunk_count])        │
├─────────────────────────────────────────────┤
│ Chunks       (chunk_count × 256 B)          │
└─────────────────────────────────────────────┘
         (entire blob is zlib-deflated)
```

- Header JSON includes per-layer metadata: layer depth, path
  prefix, bounding box, child layer references.
- Palette entries are 32 B PBR records (see Phase 2).
- Chunk index is sorted by NodeId for binary search; the loader
  can mmap the file and seek to any chunk in `O(log chunks)`.
- Content-addressed dedup at write time: if two chunks hash
  identically, the index stores the same offset twice.

One `.vxs` per layer-preset. The content pipeline doc
(`docs/design/content-pipeline.md`) gets an extension describing the
write path. Loader lives in `src/world/vxs/`.

### Runtime streaming loop

```rust
// src/world/gpu/streamer.rs (new)
impl Streamer {
    fn frame(&mut self, frame: &ActiveFrame, camera: &Camera) {
        // 1. Read previous frame's hit_count/hit_list (one-frame lag).
        let hits = self.read_visibility_async();

        // 2. Mark residency LRU: each hit NodeId gets "touched".
        for entry in hits {
            if let Some(slot) = self.pool.residency.get(&entry.node_id) {
                self.pool.touch(*slot);
            }
        }

        // 3. Upload any new NodeIds on frame path / ribbon / prefetch.
        let required = collect_required_nodes(frame);
        for node_id in required {
            if !self.pool.residency.contains_key(&node_id) {
                let slot = self.pool.evict_lru();
                let chunk = self.vxs_loader.load_chunk(node_id);
                self.pool.upload(slot, node_id, chunk);
            }
        }

        // 4. Zero visibility buffers for the next frame.
        self.clear_visibility();
    }
}
```

One-frame lag is fine — content is consistent at rest, and edits
create new NodeIds anyway (CoW) so no invalidation is needed.

## Shaders touched

- **New:** `assets/shaders/indirect_args.wgsl` (tiny), `visibility.wgsl` (helpers for atomic hashmap insert)
- **New:** `assets/shaders/compact.wgsl` (hit_map → hit_list compaction)
- **Modified:** `march.wgsl` — child fetch goes through pool indirection; primary march appends to visibility buffer on first hit
- **Modified:** `tree.wgsl` — `get_child()` helper takes a NodeId + slot instead of a flat index
- **Modified:** `bindings.wgsl` — new bind group for pool + visibility + indirect args

## Rust code touched

- **New:** `src/world/gpu/chunk_pool.rs` — pool, residency, LRU
- **New:** `src/world/gpu/streamer.rs` — per-frame streaming loop
- **New:** `src/world/vxs/` — `.vxs` read/write, zlib
- **Modified:** `src/world/gpu/pack.rs` — replaced by streamer; BFS pack deleted
- **Modified:** `src/world/gpu/ribbon.rs` — ribbon entries now carry NodeIds, not flat indices
- **Modified:** `src/renderer/init.rs` — set up pool + visibility + indirect buffers
- **Modified:** `src/renderer/draw.rs` — dispatch indirect args compute before any secondary pass
- **Modified:** `tools/scene_voxelize/generate/src/` — write `.vxs` instead of in-memory output

## Recursive architecture integration

- **NodeId-keyed pool**: a repeated subtree across layers (e.g. a
  forest of identical trees) occupies one slot. The pool capacity
  scales with *unique* subtrees, not total volume. This is the same
  property streaming.md relies on at the CDN level.
- **Per-layer .vxs**: layer descent = load that layer's `.vxs` index
  into the loader; chunks stream in as rays hit them. Ascent =
  retained in the pool until LRU-evicted.
- **Content-addressed edits**: an edit creates 63 new NodeIds (one
  per ancestor to root). On next frame, their rays miss the pool
  and trigger loads from the in-memory `NodeLibrary` (edits are
  local; no `.vxs` roundtrip).

## Layer-uniformity check

Every piece works identically at every layer:
- Pool: same buffer, same residency table at all depths
- Visibility: same hash, same compaction
- Indirect: same dispatch path
- `.vxs`: same format per layer, distinguished only by header metadata

No "leaf layer" specialization anywhere. Sphere nodes
(`CubedSphereBody`, `CubedSphereFace`) are the one allowed exception
per the architecture rule.

## Acceptance criteria

- `cargo test` green. Existing harness tests still pass (same scenes
  render visually identically).
- A new perf test: primary raymarch time on a canonical scene is
  within 10% of Phase 0 baseline. (Pool indirection adds one buffer
  read per node descent; this should be lost in memory latency.)
- Visibility buffer correctness test: hit count matches a reference
  CPU implementation for a fixed 32×32 test viewport.
- `.vxs` roundtrip test: generate → load → render produces identical
  output to in-memory generation.
- Pool eviction stress test: 10× oversubscribed scene still renders
  correctly, just with slower streaming.

## Perf target

| Metric | Target |
|---|---|
| Primary raymarch (1080p) | ≤5.5 ms (same as baseline) |
| Pool indirection overhead | ≤0.5 ms |
| Visibility buffer write | ≤0.3 ms |
| Indirect args compute | ≤0.05 ms |
| **Phase 1 frame total** | **≤6.5 ms** |

## Risks & open questions

- **WebGPU atomic hashmap performance.** WGSL atomics on storage
  buffers are slower than compute-native APIs. Mitigation: dedup on
  insert with linear probing, cap probe length at 8, let rare
  overflows fall through (cost is correctness-neutral, just wastes
  one hit entry).
- **One-frame lag on streaming.** A fast turn could show untextured
  holes for one frame. Mitigation: prefetch a fat frustum margin;
  degrade missing chunks to parent representative_block (same as
  existing LOD fallback).
- **Pool thrashing on layer descent.** Entering a new layer causes
  a burst of loads. Mitigation: LRU warms over multiple frames;
  descent is visibly smooth in the baseline.
- **`.vxs` file size for large layers.** A 4.5M-voxel Sponza-class
  scene compresses to ~5 MB. Our fractal content could be
  substantially larger. Mitigation: per-layer files, load the
  in-frustum layer only.

## Scope estimate

~2000 LoC net (1200 Rust, 400 WGSL, 400 tests). 1–2 week solo
effort. Largest single phase in the roadmap.
