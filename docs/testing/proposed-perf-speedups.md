# Proposed Perf Speedups

Two candidate optimizations, sized by impact. Both target the Soldier scenario at retina resolution (currently 40 FPS at 2560×1440 inside the body at zoom level 4).

Measured baseline (harness, spawn inside soldier, anchor_depth=5):

```
total           = 23.2 ms     → 43 FPS
├── gpu_pass    =  2.85 ms    (per-pass timestamp — understates real shader work on TBDR)
└── submitted_done = 21.6 ms  (full GPU time until done callback)

avg_steps=32, avg_empty=15, avg_descend=12, avg_lod_terminal=4
```

---

## Speedup A: Render at logical resolution + compositor upscale

### What

Render the ray-marched scene to a framebuffer at the window's **logical size** (1280×720 on a 1280×720 retina window) instead of its **physical size** (2560×1440 = the 2× retina-scaled backing store). The display compositor upscales the result to fill physical pixels; UI overlay (hotbar, debug panel) still renders at full retina resolution on a separate layer.

### What problem it solves

4× fewer pixels to shade and write to main memory. The entire per-pixel cost — fragment shader work, storage-buffer loads, tile resolve — drops proportionally.

### Expected impact

Measured resolution sweep on the Soldier scenario:

| render resolution | total | FPS |
|---|---|---|
| 1280×720  | 8.9 ms  | **113** |
| 1920×1080 | 14.3 ms | 70  |
| 2560×1440 | 22.9 ms | 43  |

**~2.5× speedup** (43 → 113 FPS). Linear in pixel count.

### Implementation options

**Option 1 (simplest)**: tell winit/wgpu the surface is logical-sized. Change how the surface is configured so physical resolution = logical. macOS compositor handles upscaling.

- Code: ~10-20 LOC touch in `src/renderer/init.rs` and the resize handler.
- Risk: macOS may refuse a mismatched surface; may need Option 2 as fallback.

**Option 2 (controlled)**: render ray-march to an offscreen color texture at logical size, then a trivial fullscreen blit-pass that samples the offscreen into the physical-size swapchain.

- Code: ~50-80 LOC: new texture allocation, simple blit pipeline (fullscreen quad + texture sample), second render pass.
- Risk: minimal. Well-understood pattern.

### Quality impact

Voxel ray-marched content has no fine detail to lose:

- Voxels are inherently blocky; sub-pixel edges are already snapped.
- No PBR textures, no fine normal maps, no small text.
- The UI overlay is a separate compositing layer → stays crisp at full retina.
- Bilinear upscale slightly softens voxel cell boundaries. Matches the "Minecraft-style" aesthetic, often seen as a feature.

### When to use

Always, for retina displays. The FPS win is too large, the quality drop too small.

---

## Speedup B: Per-node empty-run metadata (INV9)

### What

At pack time, for each packed node, precompute **how many consecutive empty cells exist along each axis** starting from each row position. Store as ~2-4 extra bytes per node, packed into the header's currently-unused bits.

In the shader, when a DDA iteration lands on an empty cell, read the run-length for the current row in the dominant ray-axis, and advance that many cells in a single iteration instead of N separate iterations.

### What problem it solves

Ray traversal through empty regions dominates the shader work: `avg_empty=15` per ray. Each empty cell currently pays a full outer-loop iteration (OOB check, stats bump, header read, occupancy bit test, DDA advance). They all produce the same result — "advance to next cell." Batching them into single run-jumps cuts the iteration count.

### Expected impact

Projected shader-only effect:

| metric | today | with INV9 | gain |
|---|---|---|---|
| avg_empty | 15 | ~4-5 | ~3× fewer iterations |
| avg_steps | 32 | ~20 | ~1.6× fewer total |
| shader time | part of 2.85 ms | ~1.7 ms | ~1.5-1.8× shader speedup |

On the Soldier scenario: shader is a fraction of the 21.6 ms `submitted_done`. INV9 attacks the shader portion only. If shader is ~10 ms of the 21.6 ms total (best guess), INV9 saves ~4-5 ms → ~18 ms total → 56 FPS.

**Combined with Speedup A**: at 1280×720, total=8.9 ms; with INV9's shader savings ~2.5 ms at that resolution → ~6 ms total → **~165 FPS**. Nearly 4× the baseline.

### Implementation scope

- **`src/world/gpu/pack.rs`** (~30 LOC): for each packed node, per-axis bit-scan of the `occupancy` mask to compute 9 run-length values (3 axes × 3 starting rows). Pack into spare bits in the node header (occupancy uses 27 of 32 bits; 5 spare; `first_child` uses ~24 of 32 bits for realistic trees; 8 spare. Total ~13 spare bits across the 8-byte header — tight but enough for 9 × 2-bit run lengths = 18 bits if we widen header to 12 bytes, or 9 × 1.5 bits if we get creative).
- **`assets/shaders/tree.wgsl`** (~10 LOC): extract run length for `(axis, row)` from the header.
- **`assets/shaders/march.wgsl`** (~20 LOC): on empty-cell branch, compute run length in DDA-dominant axis and advance by that many cells in a single iteration.
- **`assets/shaders/face_walk.wgsl`** (~10 LOC): same pattern for sphere/face DDA.

Total: ~70 LOC, one commit.

### Why it pairs well with sparse

The sparse interleaved layout already has occupancy in the header. INV9 adds metadata derived from that same mask — no new memory reads. The shader's hot path stays tight: header is cached per-depth (scalar cache from `bf7ff20`); run-length extraction is register-only arithmetic.

Dense layout doesn't have the occupancy mask, so dense-INV9 would require adding one, regressing some of dense's compactness. Sparse-INV9 is "free" — you're already reading the occupancy.

### When to use

Always, after (A). Stacks multiplicatively with the resolution win. Biggest impact on empty-heavy scenes (deep zoom over sky, open spaces, sparse voxel sculptures like the Soldier).

---

## Combined target

The compound path from the current 43 FPS:

| optimization | FPS |
|---|---|
| baseline | 43 |
| + Speedup A (resolution) | ~113 |
| + Speedup B (INV9 empty-run) | ~165 |

**~4× combined on the exact Soldier-at-zoom-4 scenario.** With Speedup C (temporal reuse for stationary camera) on top, stationary views hit effectively infinite FPS by reusing frames.

Neither speedup changes the observable behavior of the game. Both are additive with no user-visible quality cost on voxel content.

## Sequencing

Do A first. It's the bigger win, it's measurable immediately, and it attacks the memory-bandwidth side of the problem (which INV9 doesn't touch). Once A is in, INV9 adds the shader-side multiplier.
