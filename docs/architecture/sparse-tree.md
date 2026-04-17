# Sparse GPU Tree Layout

**Status**: design, not yet implemented. This doc specifies the target layout for the migration from the current 216-bytes-per-node dense format to an occupancy-masked sparse format.

## Motivation

Today every packed tree node occupies 216 contiguous bytes: 27 × `GpuChild` (8 B each), one slot per child, empty slots fully materialized. For a typical frame in the "deep zoom over empty sky" workload, most slots in most nodes are `tag=0` — we pay full storage for absence.

Measured impact of that absence:
- Ray-march empties dominate after INV8: `avg_empty=18.7` per ray in the regression scenario.
- Every empty-slot lookup is an 8-byte memory load for zero information.
- The packed buffer is ~5× larger than it needs to be for sparse scenes (trees, detail, open space between objects).

The sparse layout stores only non-empty children. Empty cells become a single bit in an occupancy mask — no memory load beyond the mask.

## Target Layout

Two GPU storage buffers replace today's single `tree: array<u32>`:

```
nodes:    array<NodeHeader>     // fixed 8 bytes per packed node
children: array<u32>            // compact child data, 2 u32s per non-empty child
```

### `NodeHeader` (8 bytes)

```rust
#[repr(C)]
pub struct NodeHeader {
    pub occupancy: u32,    // 27-bit mask; bit s = 1 iff slot s is non-empty
    pub first_child: u32,  // offset into `children` buffer (in GpuChild units)
}
```

Only 27 bits of `occupancy` are used for slots (slot 0..26). Bits 27-31 are reserved for future per-node flags — `representative_empty`, `uniform_solid`, and similar extensions from the INV7/INV8 family. The free-bits-in-free-padding discipline from today's `GpuChild._pad` carries over.

### Compact child array

Each entry is the same 8-byte `GpuChild` as today:

```rust
#[repr(C)]
pub struct GpuChild {
    pub tag: u8,          // 1 = Block, 2 = Node  (tag=0 no longer appears)
    pub block_type: u8,   // block ID for tag=1; representative block for tag=2
    pub _pad: u16,        // reserved for per-child flags
    pub node_index: u32,  // index into `nodes` buffer for tag=2
}
```

`tag=0` never appears in the child array — emptiness is encoded by the absence of a bit in the header's `occupancy` mask. The `tag=1` and `tag=2` semantics are unchanged.

### Slot lookup

To read slot `s` at node index `n`:

```rust
let h = nodes[n];
let bit = 1u32 << s;
if h.occupancy & bit == 0 {
    // empty — no memory access beyond the header
    return EMPTY_SENTINEL;
}
let rank = (h.occupancy & (bit - 1)).count_ones();  // popcount
let child = children[h.first_child + rank];
```

On WGSL: `countOneBits(mask)` is the popcount intrinsic and compiles to a single hardware instruction on every current GPU. Tested on Apple Silicon (Metal), NVIDIA (Vulkan), and WebGPU via Dawn.

## Why This Is the Optimal Solution, Not a Stopgap

The alternatives considered and rejected:

1. **Keep dense + add empty-slot bitmask in `_pad`.** A 16-bit mask doesn't cover 27 slots. A 32-bit mask fits but requires growing `GpuChild` to 12 bytes (cache-line-unfriendly) or splitting across fields (ugly). Either way it adds metadata without removing the underlying waste: we still store 27 × 8 B per node when most slots are empty.

2. **Octree (2³ = 8 children).** Breaks the game's identity. Every layer, zoom, anchor, path, and UI element is built on 3-fold subdivision. This is not a rendering trade; it's foundational.

3. **BVH (bounding-volume hierarchy).** A different geometric paradigm — nodes bound arbitrary content rather than sub-dividing uniformly. Loses the scale-invariant ternary structure the rest of the engine relies on. Also an order-of-magnitude bigger refactor.

4. **Distance fields.** Complementary, not competing. Adds empty-run skipping on top of any layout. Worth doing later *on top of* the sparse representation; does not obviate this work.

The sparse ESVO-style layout gets all of:
- Constant-factor win on every empty cell check (bit test vs 8-byte load).
- Asymptotic memory savings for sparse content (forests, cities, open space).
- Room for future per-node flags in the 5 free occupancy bits.
- Unchanged invariants: ternary subdivision, scale-invariance, ribbon structure, dedup, LOD.

## Migration Scope

### Files that must change

**Core layout (required rewrite)**
| file | role | LOC impact |
|---|---|---|
| `src/world/gpu/types.rs` | Define `NodeHeader`, keep `GpuChild` minus the empty variant. | ~30 lines added |
| `src/world/gpu/pack.rs` | Emit sparse layout: build occupancy mask per node, flatten child arrays, compute `first_child` offsets. Preserve the INV7 `siblings_all_empty` semantics (trivially derivable from `occupancy.count_ones() <= 1`). Preserve INV8 empty-representative collapse (the packer already emits tag=0 for effectively-empty subtrees; in sparse layout those slots just don't appear in the occupancy mask). | ~150 lines changed / added |
| `src/world/gpu/ribbon.rs` | `build_ribbon` changes the walker: instead of `tree[current * 27 + slot]`, read `nodes[current]`, check the mask, popcount to find the child. `siblings_all_empty` becomes `nodes[node].occupancy.count_ones() == 1` (the one non-empty slot is the popped one). Tests rewrite their fixtures via new sparse-builder helpers. | ~100 lines changed; ~150 lines of test fixture rewrites |
| `assets/shaders/tree.wgsl` | Rewrite `child_packed` and `child_node_index` to use the sparse lookup. Add `child_empty(node_idx, slot)` as an explicit predicate backed by the occupancy mask so the shader can short-circuit before a memory load. | ~30 lines |
| `assets/shaders/bindings.wgsl` | New storage binding for the `nodes` header array. Existing `tree` binding repurposed to hold the compact child array. | ~5 lines |
| `src/renderer/buffers.rs` | Two uploads instead of one: `update_tree(headers, children, kinds, root)`. Signature change, no logic change. | ~40 lines |
| `src/renderer/init.rs` | Two buffers + bind group layout entry. | ~25 lines |
| `src/renderer/mod.rs` | One extra `wgpu::Buffer` field. | ~5 lines |

**Callers (mechanical signature updates)**
| file | change |
|---|---|
| `src/app/event_loop.rs` | `tree_data.len() / 27` → `headers.len()` at L139, L199. |
| `src/app/edit_actions/upload.rs` | Same: receive `(headers, children, kinds)` from pack, pass to `update_tree`. L48, L82, L93, L103, L122. |
| `src/app/test_runner/runner.rs` | Same at L183, L187. |
| `src/app/mod.rs` | Stat field comment update (L184). |

**Unaffected (verified)**
- `src/world/raycast/*` — walks the CPU `NodeLibrary`, not the GPU buffer. Zero changes.
- `src/app/edit_actions/` — edits mutate the library; the rebuild goes through pack. Zero changes.
- `tests/e2e_layer_descent*` — parses harness output, does not inspect the packed buffer. Zero changes.
- `tests/render_perf.rs` — same.
- `assets/shaders/march.wgsl`, `face_walk.wgsl` — all tree access goes through `tree.wgsl` helpers. Zero call-site changes; helper bodies are the only shader logic change.

### Total footprint

~560 LOC changed or added across 8 files, plus ~150 LOC of test-fixture rewrites in 2 test modules. Binary size of the packed buffer drops by ~4-7× for typical sparse frames.

## Shader Performance Model

The sparse inner loop per tag check:

| op | dense (today) | sparse | delta |
|---|---|---|---|
| empty cell | 8 B load + tag branch | `countOneBits(mask & ...)` (1 instr) + branch | **much faster** |
| non-empty cell | 8 B load + tag branch | 8 B header load + popcount + offset add + 8 B child load + tag branch | +1 instr + 1 extra load |

The empty case is the common case in our workload — `avg_empty=18.7` vs `avg_descend=4` and `avg_lod_terminal=2.6` per ray at the regression point. Making empties near-free is a direct win.

The non-empty case adds one cache miss if the header and child happen to be on different cache lines. A small optimization: **read the entire node header once per descent** (store it in a register), then all 27 potential slot lookups within that descent reuse the cached mask and offset without re-loading the header. This is natural in the DDA inner loop where we stay at the same node until we descend or pop.

Projected per-ray iteration count in the current regression scenario (depth=13 post-zoom, plain world):

| | before INV7 | after INV8 | sparse (projected) |
|---|---|---|---|
| `avg_steps` | 90.3 | 34.7 | **18-22** |
| `avg_empty` | 11.9 | 18.7 | ~6-8 |
| `avg_descend` | 17.3 | 4.0 | unchanged |

The estimate: empty-check overhead drops from full memory access to bitmask test, cutting `avg_empty` by ~2-3×. Other categories unchanged (the sparse layout doesn't change ray geometry or descent logic).

## Migration Strategy: Single Atomic Diff

The project rule is no stopgaps. Incrementalizing this change introduces a shim layer — a "dense-compat accessor" that reads sparse data and returns dense tuples — and that shim becomes the bug source while it exists. The CLAUDE.md memory `feedback_dont_force_incremental_green.md` explicitly warns against this pattern.

The migration happens as one diff:

1. Replace `GpuChild` dense buffer with `NodeHeader + children` in `types.rs`, `pack.rs`, `ribbon.rs`.
2. Rewrite shader accessors in `tree.wgsl`, add new storage binding in `bindings.wgsl`.
3. Update renderer buffer management in `buffers.rs`, `init.rs`, `mod.rs`.
4. Fix caller signatures (`event_loop.rs`, `edit_actions/upload.rs`, `test_runner/runner.rs`).
5. Rewrite test fixtures in `ribbon.rs` + `pack.rs` via new sparse-builder helpers.
6. Run: unit tests → harness → screenshot → live perf run.

The working tree is broken between steps 1 and 6. That is expected and correct for a layout migration. The commit is green or isn't — no intermediate "dense-sparse bridge" state.

## Test Strategy

**Unit-level correctness**: the existing 24 `gpu::*` tests plus the sparse-builder helper tests must pass. Fixture rewrites preserve the semantic assertions; only the construction changes (e.g., `two_node_tree(slot)` emits a header with `occupancy = 1 << slot` and a one-element children array).

**Visual correctness**: pre-migration and post-migration screenshots at these scenarios, byte-identical expected:
- `--plain-world --spawn-depth 8 --spawn-pitch -1.0` (standard baseline)
- `--sphere-world --spawn-depth 8` (sphere dispatch unchanged)
- `--plain-world --spawn-depth 3 --spawn-pitch -1.0` + `zoom_in:10` script (the INV8 regression scenario)

**Performance verification**: `--shader-stats --live-sample-every 120` run at the INV8 regression scenario. Target `avg_steps ≤ 22` to confirm the empty-check speedup.

**No regression**: the empty-shell ribbon fast-exit (INV7) and the empty-representative fast-path (INV8) must still fire. In sparse:
- `siblings_all_empty` becomes `nodes[node].occupancy.count_ones() == 1`.
- `representative_block == 255` detection on tag=2 is unchanged (it's a property of the child, not the layout).

## Open Questions

None intended to block the migration. Things to validate during implementation:

1. **Packed-children offset stability under edits.** When a block is placed or broken, the tree is repacked from scratch each frame. No incremental offset maintenance — sparse is rebuilt with the tree.
2. **WGSL popcount performance on Metal.** Published numbers say `countOneBits` is a single-cycle intrinsic. The first implementation step is an isolated micro-benchmark shader to confirm on our target hardware before committing to the full migration.
3. **Header-prefetch ergonomics.** The optimization "cache header in a register during descent" is worth implementing from day one — the DDA inner loop already has a "current node" state, extending it to carry the header alongside the node index is trivial.

## Expected Timeline

With the scope above, ~3–5 days of focused work split as:

- Day 1: layout types + pack.rs rewrite. Unit tests green.
- Day 2: shader accessors + bindings. Harness renders correctly (screenshot byte-match).
- Day 3: renderer buffer plumbing + caller updates. Live game runs.
- Day 4: test fixture rewrites, perf measurement, screenshot diffs, commit.
- Day 5: buffer (nothing is ever a 3-day project).

## References

- `INVESTIGATING PERFORMANCE 7` (commit `805fc0a`) — ribbon `siblings_all_empty` flag; semantic preserved in sparse via occupancy popcount.
- `INVESTIGATING PERFORMANCE 8` (commit `b38ed96`) — shader empty-representative fast-path; unchanged in sparse.
- `docs/testing/perf-lod-diagnosis.md` — full perf arc culminating in the measurements that motivated this work.
- `docs/architecture/tree.md` — NodeLibrary CPU-side tree, unaffected by this migration.
