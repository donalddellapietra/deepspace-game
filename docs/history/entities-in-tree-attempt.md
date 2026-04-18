# Attempt: Entities as `Child::EntityRef` Nodes in the Tree

Status: **rolled back**. The branch tip is the hash-grid architecture
(`0aa7469`). This doc captures what I tried and why it didn't render.

## Context

After the hash-grid entity renderer shipped and was profiled, the
10k-soldier frame time sat at ~23 ms. Logging showed:

```
avg_bin_visits=31.63   avg_aabb_tests=997.09
avg_subpixel_skips=0   avg_subtree_marches=10.13
```

So the bulk of the GPU cost was iterating ~1000 entity AABBs per ray in
the dense spawn cluster. The hash grid helped vs. linear (which would
have been 10000 tests/ray) but was still redundant work: **the world
DDA already solves spatial culling beautifully for voxels — why not
reuse it for entities?**

## The design

Put entities *into the world tree* via a new `Child` variant:

```rust
pub enum Child {
    Empty,
    Block(u8),
    Node(NodeId),
    EntityRef(u32),   // u32 = index into entities GPU buffer
}
```

Per frame:

1. Build a fresh "scene root" by overlaying entity `EntityRef(idx)`
   cells onto the persistent terrain root. The overlay uses a trie of
   entity anchor paths and walks terrain + trie together (bottom-up,
   O(entity coverage × depth)).
2. Pack scene_root via the existing `CachedTree` — content-addressed
   dedup reuses terrain subtrees; only the ephemeral ancestor chain
   gets newly emitted.
3. `ref_inc` scene_root, `ref_dec` previous frame's scene_root so
   ephemeral ancestors cascade-evict.
4. Upload per-frame `entities[]` transform buffer (bbox_min/max +
   subtree_bfs + representative_block) indexed by the `u32` in tag=3
   cells.

The shader's `march_cartesian` gains a `tag == 3` branch: on
encountering an EntityRef child, hand off to `march_entity_ref` which
does the AABB test + ray transform + descent into
`march_entity_subtree` (a separate walker that DOES NOT branch on
`tag==3` so WGSL's no-recursion rule is satisfied).

No separate pass in `main.wgsl`. Just `march()` — one unified DDA.

## What got implemented

- `src/world/scene.rs` — trie-batched scene builder
- `src/world/tree.rs` — `Child::EntityRef(u32)` variant + representative/
  uniform-type handling for the new variant
- `src/world/gpu/pack.rs` — emits `tag=3` GPU child for EntityRef
- `assets/shaders/entities.wgsl` — `march_entity_ref` + standalone
  `march_entity_subtree` (a copy of `march_cartesian`'s DDA without
  sphere/ribbon/tag=3)
- `assets/shaders/march.wgsl` — tag=3 dispatch in the tag-1-or-else
  branch
- `src/app/edit_actions/upload.rs` — per-frame scene build +
  ref_inc/dec dance for the ephemeral scene root, ensure_root every
  entity subtree BEFORE building scene_root (so their BFS indices are
  valid when the shader dereferences `entities[idx].subtree_bfs`)
- Deleted `src/world/entity_bins.rs` and `update_entity_bins` from the
  renderer; shrank storage-buffer count back to 6

All CPU-side diagnostics confirmed the data flow was correct:

```
scene: trie insert entity 0 path=[13, 22, 13, 4, 13, 13] depth=6
scene: install EntityRef(0) at slot 13 of terrain node 174
scene_build terrain_root=179 scene_root=1159 entities=1
pack: emit tag=3 entity_idx=0
update_entities count=1 first=GpuEntity {
    bbox_min: [1.444, 1.444, 0.445],
    representative_block: 237,
    bbox_max: [1.556, 1.556, 0.556],
    subtree_bfs: 973,
}
```

Packed node count grew from 29 (terrain only) to 1008 (terrain +
soldier subtree + scene ancestors). The pack cache correctly deduped
on subsequent frames.

## The symptom

**The entity never appeared on screen.** Sky and terrain rendered
normally; the viewport showed no trace of the soldier — and no trace
of the debug color I substituted into the tag=3 branch of
`march_cartesian`. That's the diagnostic that matters: **the shader's
DDA never reaches the tag=3 cell**, even with:

- `--lod-base-depth 8` (giving a descent budget far beyond what's
  needed — entity is at relative depth 2 from the frame root)
- AABB culling in the tag=2 descent disabled
- Camera parked literally on top of the entity's bbox (AABB hit
  confirmed numerically from the logged bbox values)

## What I think is happening

Something between the frame root and the EntityRef cell is marking
the path as "empty" before the DDA gets there. Candidates, from most
to least likely:

1. **`representative_block` chain is still 255 somewhere up the
   ancestor path.** I patched `tree.rs` to count EntityRef as block
   type 253 (sentinel) when computing a node's `representative_block`,
   specifically so the shader's "empty-representative fast path"
   (`child_bt == 255u → DDA advance, never descend`) wouldn't fire.
   Green never appeared afterward, but I didn't confirm the change
   propagated upward through every ancestor level. If *any*
   intermediate scene-ancestor has `representative_block == 255`, the
   shader skips descent into it and the tag=3 cell is unreachable.

2. **`uniform_type` / uniform-flatten in pack.rs collapses an ancestor
   into a single `Child::Block`.** I added `Child::EntityRef(_) =>
   UNIFORM_MIXED` to the uniform_type computation, which should prevent
   this. But I didn't verify that every library-ancestor hit in
   `build_child_entry`'s `uniform_type != UNIFORM_MIXED` arm stays
   mixed all the way up to scene_root. If the terrain cell containing
   the entity ancestor happens to be a uniform-empty subtree before
   the overlay, and my `uniform_type` propagation doesn't account for
   the new EntityRef child correctly, pack could still flatten it.

3. **`content_aabb` of a scene ancestor doesn't cover the EntityRef
   slot.** I don't think this is it — occupancy bits are set
   correctly for non-Empty children including EntityRef, and
   `content_aabb(occ)` derives the AABB from those bits — but I
   didn't trace the bit layout on GPU for the actual scene run.

4. **Frame root resolution** picks a `frame_root_idx` (via
   `build_ribbon`) that lands OUTSIDE the scene tree. I read the code
   and it walks scene_root's BFS via the render_path, so this
   shouldn't happen — the render_path slots `[13, 22, 13]` correspond
   to scene's merged ancestor chain — but the packed frame_root_idx
   vs scene-tree-root-BFS invariant would be worth dumping on CPU
   side.

The fact that the debug green (forced in the tag=3 branch of
`march_cartesian` unconditionally on hit) *didn't render anywhere*
rules out "entity subtree doesn't descend correctly" — the problem is
upstream, in reaching the tag=3 cell at all.

## What to try next session

The most productive next move isn't "try fix #1" or "try fix #2" —
it's **instrument and confirm** before another code change:

1. After scene build, print the full ancestor chain from scene_root
   to the EntityRef cell: each node's `representative_block`,
   `uniform_type`, `children[anchor_slot]`, and the BFS idx the pack
   cache gave it. If any ancestor has `representative_block == 255`
   OR `uniform_type != UNIFORM_MIXED`, that level is where the
   shader's fast path kicks in.
2. Shader-side: write a single atomic counter each time the DDA hits
   a `tag == 3` check (separate from the ENABLE_STATS plumbing, just a
   focused probe). If it stays 0, we know the tag-3 branch is
   literally unreached and the issue is upstream culling. If it
   fires, the bug is in `march_entity_ref` itself.
3. The deepest unknown: after `cache.update_root(scene_root)`, dump
   the first 100 u32s of the pack buffer starting at the scene_root
   BFS and check each node's header + children tags. That tells us
   definitively what the GPU sees.

## Lessons

- **The unified-tree idea is architecturally correct.** The world DDA
  *is* the right spatial index; entity positions *should* just be
  tree cells. Motion stays smooth because the sub-cell offset lives
  in the per-entity transform buffer, NOT in the tree itself — the
  tree just says "entity N is somewhere in this cell", the GPU buffer
  says "at this offset + bbox within it."
- **Adding a new `Child` variant touches more invariants than I
  expected.** Not the obvious ones (match arms — the compiler found
  those), but the *cached computed fields* on `Node`:
  `representative_block`, `uniform_type`, `depth`. Each of those
  has an implicit contract the shader relies on, and adding a new
  Child type means re-deriving what the contract should be for *that
  variant* at *every ancestor level all the way to root*. That's
  subtle.
- **Given the above, the hash-grid architecture pays its way.** It's
  ~8 lines of bin-lookup code in one shader file + one CPU module.
  Simpler to reason about even if "less elegant" from a unified-DDA
  perspective. The elegance tax of EntityRef is real: one bug in
  how the library-level invariants interact with the pack-level fast
  paths, and entities go invisible.

If next session pins down the exact ancestor-level where the fast
path fires, the fix is likely one line — the approach isn't wrong,
the instrumentation was just insufficient to finish in one pass.

## How I tested during the attempt (and what went wrong)

The pattern I fell into, ranked roughly chronologically:

1. Made the refactor in one shot (scene.rs + pack change + shader
   change + upload rewiring + deletion of hash-grid plumbing).
2. Ran `cargo test --test render_entities`. All 7 tests failed.
3. Reached for `eprintln!` CPU-side. Added prints to `scene.rs` and
   `pack.rs` build_child_entry. Those confirmed the TREE content was
   correct: EntityRef emitted, tag=3 serialized into the pack
   buffer, entity GPU buffer populated with the right bbox.
4. Could not observe the shader. Reached for "forced debug colors"
   inside `march_cartesian`'s tag=3 branch — a solid green return
   when tag==3. Screen stayed terrain+sky. Ran the same single-frame
   screenshot command over and over with different shader tweaks:

   ```
   timeout 10 target/aarch64-apple-darwin/debug/deepspace-game \
     --render-harness --disable-overlay --disable-highlight \
     --plain-world --plain-layers 40 \
     --spawn-depth 6 --spawn-xyz 1.5 1.5 1.8 \
     --spawn-yaw 0 --spawn-pitch 0 \
     --spawn-entity assets/vox/soldier.vox --spawn-entity-count 1 \
     --harness-width 320 --harness-height 180 \
     --screenshot tmp/single.png \
     --exit-after-frames 3 --timeout-secs 6
   ```

5. Started disabling protections one at a time — empty-rep fast
   path, AABB cull, bumped `--lod-base-depth 8` — each run a fresh
   cargo build + screenshot + eyeball.

The failure of that loop:

- **Multiple hypotheses in flight at once.** I was simultaneously
  suspecting the representative_block fast path, the AABB cull, and
  the depth budget. I disabled them one at a time but kept each
  disable in place while testing the next, which means by the end I
  couldn't tell which of my "fixes" had actually mattered (spoiler:
  none, since the entity still wasn't visible).
- **CPU-side prints, GPU-side opacity.** I could see that the scene
  tree had the EntityRef in the right slot. I could NOT see what
  the shader's DDA actually did when it got to that cell's
  parent, or whether it got there at all. The eyeball-the-screenshot
  loop is a single-bit observation: "did green appear?" No green
  tells you nothing about WHY.
- **No dedicated probe.** I had `ENABLE_STATS` counters for world
  DDA branches, but adding a tag=3-specific atomic counter would
  have been 3 lines of WGSL. I never did it. Every subsequent run
  was a re-eyeball instead of a measurement.

## The correct iterative testing loop

For any refactor that changes rendering output, build the test
before the refactor, not after:

1. **Write a test that demonstrates the current behavior as a
   golden.** Specifically, a test that exercises the SINGLE
   path you're about to change. For entity rendering: a one-
   entity-visible test at a fixed depth, hit_fraction > 0, and a
   color-match check at the known entity center.
2. **Run the test — it passes under the current architecture.**
   Commit the test. This is the reference point.
3. **Make the refactor in the SMALLEST coherent slice that could
   keep the test passing.** For entity-in-tree: *first* add the
   Child variant + pack emission, but route it through the OLD
   hash-grid reader (shader still walks the bin grid, just needs
   to tolerate tag=3 showing up there or via a feature flag).
   Keep the test passing at every step.
4. **Delete the old path only after the new path passes.**

I did #1 only after the refactor, in the form of the existing
motion tests. They were too coarse — they just checked that the
frame differed from a baseline. At 0 entity visibility the frame
didn't differ, and the test failed, but with no signal about WHERE
the path broke.

## The correct debugging loop

Once a visual regression like "entity not rendering" shows up, the
loop is **observe, hypothesize, instrument, change** — in that
order, one hypothesis at a time:

1. **Observe.** What is the actual data? Not what I think it should
   be — what the CPU log, the GPU atomic counter, the pack buffer
   dump actually say. First pass, each layer:
   - Scene builder: `eprintln!` the resulting scene_root's children
     at each anchor-path slot. Confirmed the EntityRef is there.
   - Pack: `eprintln!` the `packed` u32 for each tag=3 emit.
     Confirmed tag byte is 3 and node_index is the entity idx.
   - GPU entity buffer: `eprintln!` the first entry. Confirmed.
   - **Shader: add a single `atomic<u32>` counter that increments
     each time the DDA loads a non-zero occupancy slot at scene
     depth 5 (the level of our EntityRef), and separately each
     time `tag == 3u` fires.** This is the observation I never
     made.
2. **Hypothesize — SINGLE hypothesis.** Write down, on paper or in
   the chat: "I think X is happening because Y." State what
   observation would confirm or falsify X.
3. **Instrument, not change.** If the hypothesis is "the
   representative_block chain is 255 at level N," the instrument
   is: dump all 5 ancestor levels' `representative_block`. Don't
   patch anything until you've seen the dump.
4. **Change only after data.** When the data identifies the layer
   where the invariant breaks, the fix is usually one line. If the
   fix is more than 10 lines, you've probably misread the data —
   go back to step 1.

Specific tools I should have reached for earlier:

- **`cat >> pack.rs` with a CPU-side `pack_dump(scene_root)`
  function** that walks the pack buffer from scene_root's BFS and
  prints every `(bfs_idx, header_offset, occupancy, [children
  tag/block_type/node_index])`. Call it once in upload_tree_lod
  before the GPU write. Takes 50 lines, gives a ground-truth view
  of what the shader will see.
- **A single `atomic<u32>` `tag_3_hits` counter** written in
  `march_cartesian` with no `ENABLE_STATS` gate (just an unconditional
  atomicAdd when tag == 3u). Takes 1 line. If it stays 0 over a
  frame, the bug is upstream; if non-zero, the bug is in the
  descent path.
- **Pre-build script that compares the composed shader string
  before and after the refactor**: `shader_compose::compose("main.wgsl")`
  → dump to a file, diff against the prior version. Would have
  instantly shown whether the tag=3 branch was literally present
  in the pipeline.
- **A "dry-run" CPU mirror of `march_entity_ref`** that walks the
  same pack buffer via `cpu_raycast`-style iteration and returns
  whether the entity should be hit. Running it on the test camera
  would have told me: "yes the ray should hit; the shader disagrees
  → the bug is shader-side."

## Why I didn't reach for these at the time

Context budget. Every debug iteration was "edit file + cargo build
+ launch binary + eyeball screenshot." At ~30s per round and 10+
rounds, that's the session. The time to add proper instrumentation
felt expensive compared to "try one more shader tweak," and I kept
betting that the next tweak would show me green. It didn't, and I
ran out of budget before ever measuring what the shader actually
saw.

The general rule this enforces: **if you've done three failing
eyeball-iterations in a row, the cost of adding the proper probe
is ALREADY less than the cost of four more eyeball-iterations.**
Skip ahead to step 1 of the correct debugging loop.
