# Terrain Generation: Problem & Approaches

## Goal

Add terrain (hills, mountains, biomes) that is visible and correct at every tree layer — not just at leaf zoom. Hills should be a layer-10 feature, mountains a layer-8 feature, etc.

## Constraint: The Tree Must Stay Fully Materialized

The renderer, collision, editing, and interaction systems all walk `node.children` from root to leaf and `.expect()` at every level. Any path that gets accessed must have a complete chain of nodes from root to leaf. We cannot change this — it's the core invariant.

## Approach 1: Lazy Root Rebuild (FAILED)

**What:** Generate nodes on-demand during the render walk. Replace the root every frame with a freshly-generated tree. Use `ResMut<WorldState>` in the render system.

**Why it failed:**
- `swap_root` every frame invalidated the renderer's `SmallPath -> (Entity, NodeId)` cache, causing entity churn (despawn/respawn everything every frame)
- The generated tree wasn't fully materialized to leaf depth — surface nodes at the generation depth got air placeholders, so the renderer saw empty voxels
- "Piss green field" at every layer, no block interaction worked

**Lesson:** Never rebuild the root per-frame. The renderer's entity cache depends on stable NodeIds across frames.

## Approach 2: Leaf-Only Generation via install_subtree (PARTIALLY WORKED)

**What:** Start with a flat grassland (fully materialized). Each frame, walk the tree to leaf level (MAX_LAYER=12), find pristine grass/air leaves near the camera, generate terrain-content leaves, and splice them in using `install_subtree` — the same code path as block editing. No changes to renderer.

**What worked:**
- All tests passed (84/84), including the critical `terrain_ground_solid_at_every_view_layer` test
- `install_subtree` correctly propagated changes up through incremental downsamples
- Pristine detection (comparing NodeId to grass_leaf_id/air_leaf_id) worked
- The tree remained fully materialized at all times

**What was wrong:**
- Terrain only appeared at leaf zoom (layer 12). At other zoom levels, the tiny patch of generated leaves was invisible against the vast ungenerated grassland.
- The noise wavelengths were initially absurdly large (mountain wavelength 50M, world is 6B) — terrain was invisible. After fixing to smaller wavelengths, terrain was visible at leaf zoom but still only affected a few leaves per frame (budget of 32).
- The player spawned underground because the terrain height differed from the grassland's flat GROUND_Y_VOXELS.
- Fundamental problem: generating at leaf level and relying on downsample propagation means you need to generate ALL leaves in the visible area before terrain is visible at zoomed-out layers. At zoom layer 10, the visible area contains millions of leaves. Budget of 32/frame can't keep up.

**Lesson:** Don't generate at leaf level. Generate at the layer the camera is viewing.

## Approach 3: Subtree Generation at Emit Layer (CURRENT)

**What:** Same `install_subtree` approach, but generate complete subtrees at the renderer's emit layer instead of individual leaves. When viewing at layer 10 (emit_layer=11), generate a full layer-11 subtree: recursively build from leaves up, using classification (AllAir/AllSolid/Surface) to shortcut uniform regions with cached tower nodes. Only surface-crossing leaves get noise-generated content.

**Key insight:** A layer-11 subtree contains only 5^1 = 5 leaves per axis (125 total). Most are AllAir or AllSolid — maybe ~25 are surface-crossing. Very fast to generate. A layer-9 subtree contains 5^3 = 125 leaves per axis, but AllAir/AllSolid shortcutting means only the thin surface shell (~hundreds) gets generated.

**Implementation (stashed):**
- `TerrainConfig` with 5 noise octaves at different scales (mountains, ridges, hills, detail, micro)
- `classify_node()` uses height bounds + safety margins to determine AllAir/AllSolid/Surface
- `build_terrain_subtree()` recursively builds from target layer to leaves, shortcutting with cached air/solid towers
- `terrain_generation_system` walks tree to emit_layer, finds pristine subtrees, generates and installs them
- `is_pristine_subtree()` recursively checks if a subtree is composed entirely of the 2 original grassland leaf patterns
- `generate_terrain_in_area()` test helper for non-ECS testing
- All 9 terrain tests pass; all 84 total tests pass (after fixing air_tower not to add extra library entries in new_grassland)

**Status:** Tests pass. Not yet tested visually. Need to also fix player spawn position to account for terrain height.

## Noise Parameters (current)

| Octave   | Wavelength (leaves) | Amplitude (leaves) | Visible at layer |
|----------|--------------------:|--------------------:|:-----------------|
| Mountain | 100,000             | 800                 | ~8               |
| Ridge    | 10,000              | 200                 | ~9-10            |
| Hill     | 1,000               | 40                  | ~10-11           |
| Detail   | 200                 | 8                   | ~11-12           |
| Micro    | 40                  | 2                   | ~12              |

Total max deviation from GROUND_Y_VOXELS: ±1,050 leaves.

## Key Files

- `src/world/terrain.rs` — terrain config, noise, subtree generation, Bevy system
- `src/world/state.rs` — WorldState gains `terrain`, `grass_leaf_id`, `air_leaf_id`, `air_tower`, `solid_towers`
- `src/world/mod.rs` — module registration, system ordering (terrain before render)
- `src/world/edit.rs` — `install_subtree` (unchanged, used as-is)

## Open Questions

1. Player spawn position: needs to query `terrain_height` and adjust Y. Previous attempt put player underground.
2. Is `is_pristine_subtree()` (recursive NodeId check) fast enough for per-frame use? The grassland tree has only ~25 unique NodeIds, so the recursion terminates quickly for pristine subtrees (they're all the same few patterns). Edited subtrees would fail fast on the first non-matching child.
3. Budget tuning: how many subtrees per frame? At emit_layer=11, each subtree is small. At emit_layer=9, each is larger. Should budget scale with subtree cost?
4. Biome temperature/moisture wavelengths (200k/150k) might be too large or too small. Need visual testing.
