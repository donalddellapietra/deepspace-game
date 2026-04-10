# Deep Space — Development Progress

## Project Overview

Deep Space is a hierarchical voxel game built in Bevy 0.18 (Rust). The core concept: voxels exist at multiple layers of abstraction. At Layer 0, you place individual blocks. At Layer 1, each "block" is a 5x5x5 model made of Layer 0 blocks. You can drill in (zoom into a cell to edit its blocks) and drill out (zoom back to the parent layer). The system is designed to scale to N layers.

---

## Phase 1: Engine Selection

### Evaluated
- **Godot 4.6** — Started here with a GDQuest TPS controller template. Abandoned because AI-driven development requires text-editable files; Godot's GUI-first workflow and binary import cache made CLI iteration difficult.
- **Three.js** — Considered for its pure-code workflow. Good for AI dev but lacks built-in physics, scene management, and desktop export.
- **Bevy 0.18** — Selected. Pure Rust, ECS architecture, everything is code. No GUI dependency. Fast incremental builds with dynamic linking.

### Key Decision
Bevy was chosen because: (1) ECS is the right pattern for voxel simulation, (2) Rust provides C++-level performance with memory safety, (3) everything is text files editable by AI, (4) the ecosystem has mature voxel examples to learn from.

---

## Phase 2: Initial Prototype (Heightmap Terrain)

Built a basic 3D world with:
- Perlin noise heightmap terrain (chunk-based, infinite)
- First-person camera with mouse look
- WASD movement with gravity and jumping
- Directional + ambient lighting

### Problems
- Heightmap terrain was not voxel-based — couldn't place/remove individual blocks
- No concept of layers or models
- Camera/player controls needed significant tuning

### What We Learned
- Bevy 0.18 API differences from documentation (e.g., `CursorOptions` as a component, `BorderColor::all()`, `GlobalAmbientLight`)
- Face winding order must be CCW for front-face rendering — got this wrong initially, causing "hollow" blocks with invisible faces
- Dev profile must optimize dependencies (`opt-level = 3` for deps) or Bevy runs at ~5 FPS

---

## Phase 3: Voxel Data Model

Replaced the heightmap with a proper voxel system:
- 10 block types with per-type materials (color, roughness, metallic, alpha)
- `VoxelGrid`: 5x5x5 array of `CellSlot` (Empty, Block, or Child)
- `VoxelWorld`: sparse HashMap of top-layer cells
- Face-culled mesh baking: internal faces between adjacent solid blocks are removed
- Per-block-type sub-meshes so each material renders correctly (metal is shiny, glass is transparent, etc.)

### Key Rewrite: Vertex Colors → Per-Type Materials
Initially used vertex colors with a single `StandardMaterial`. This made all blocks share the same PBR properties — metal looked the same as dirt. Fixed by splitting the baked mesh into one sub-mesh per block type present, each with its own `StandardMaterial`.

### Key Rewrite: Instance vs Template
Initially, all ground cells shared the same `ModelId` reference. Editing one cell mutated the template, changing ALL cells. Fixed by giving each cell its OWN copy of the block data. The `ModelRegistry` holds templates; cells hold instances.

---

## Phase 4: Layer System

### Architecture Iterations

**Attempt 1: Binary GameLayer state**
Two states: `World` and `Editing`. Hardcoded transitions. Didn't scale beyond 2 layers.

**Attempt 2: ActiveLayer with nav_stack**
Replaced the binary state with a navigation stack. Each entry records which cell was drilled into and the return position. Supports N layers. `is_top_layer()` = stack is empty.

### The Scaling Problem
Every function that dealt with neighbors had `if nav_stack.len() == 1` special cases for the top-layer HashMap vs inner grids. These broke at depth 2+. 

**Solution**: Generic `get_sibling()` and `get_sibling_slot()` methods on `VoxelWorld` that work at any depth. At depth 1, the parent is the HashMap. At depth 2+, the parent is the grid at `nav_stack[..len-1]`.

### The "Ant" Insight
The user's key insight: when you drill into a cell, you should feel like you're SHRINKING — becoming an ant inside the cell. The world around you should feel massive.

**Implementation**: At every layer, the coordinate convention is the same: 1 block = 1 unit. When you drill in, the cell's 5x5x5 blocks become the new world at 1-unit scale. The surrounding parent/grandparent cells are rendered as baked meshes at the appropriate offset and scale via `render_ancestors`.

---

## Phase 5: Collision System

### Iteration 1: Floor Detection Functions
`floor_top_layer` and `floor_inner` scanned block columns to find the highest solid surface below the player. Plagued by:
- Tolerance hacks (`STEP_UP_TOLERANCE = 0.5`) that caused auto-step-up behavior the user didn't want
- Gravity overshoot: large velocity could push the player below a surface, and the tolerance-gated detection would miss it
- Pre/post gravity ordering tricks that introduced new edge cases

### Iteration 2: Swept AABB (Push-Out)
Moved to a standard game collision approach: apply movement, check for AABB overlap with solid blocks, push the player out. Failed because:
- The "best push" selection logic picked the wrong face when multiple blocks overlapped (e.g., a parent Block filling a 5-unit region)
- Iteration order (bottom-to-top vs top-to-bottom) affected which face was chosen

### Iteration 3: AABB Clipping (Final, Correct)
Standard Minecraft/Quake algorithm. For each axis:
1. Collect all solid blocks near the player's path
2. For each block, compute the maximum safe movement distance (clip the delta)
3. Apply the clipped movement

**Key properties**:
- Movement is clamped BEFORE application — the player never penetrates a block
- No push-out logic, no face selection, no iteration order issues
- Resolves Y first (gravity), then X, then Z — allows wall sliding
- `clip_axis` is a pure function: given player AABB, movement delta, and one block, return the clipped delta

### The Ancestor Chain Bug
`block_solid` initially only checked one level up (parent siblings). When the player walked off the edge of the entire grandparent cell (25 units in current space), the ground from the top-layer HashMap was invisible to the collision system.

**Fix**: `block_solid` now walks the ENTIRE nav_stack. If a coordinate maps outside the parent grid, it transforms to the grandparent's coordinate system and checks there, continuing up to the top-layer HashMap. Same recursive ancestor logic as `render_ancestors`.

---

## Phase 6: Rendering

### render_ancestors
When drilled into a cell, the current cell's blocks render as individual cubes. But the surrounding world must also be visible — otherwise you're in a tiny room with blue sky.

`render_ancestors` walks up the nav_stack. At each ancestor level:
- `cumulative_scale` tracks how many current-blocks one ancestor slot spans (MODEL_SIZE^level)
- `cumulative_offset` aligns the ancestor's coordinate origin with current block-space
- Baked meshes are rendered at `offset + coord * scale` with `mesh_scale = scale / MODEL_SIZE`

**Critical bug found**: `cumulative_scale *= MODEL_SIZE` was initially at the END of the loop body instead of the BEGINNING. This meant ancestor cells were rendered at the wrong scale — tiny meshes overlapping with current blocks instead of surrounding them.

### Face Winding
The original face vertex definitions had incorrect winding order. Verified each face with cross products: `(V1-V0) × (V2-V0)` must match the declared normal. Fixed all 6 faces.

---

## Phase 7: Inventory & Hotbar

### Hotbar
10 slots, each holding either a `Block(BlockType)` or `SavedModel(usize)`. Number keys 1-0 switch the active slot. Right-click places whatever is in the active slot.

### Inventory Panel
Press E to open a full-screen overlay showing all block types and saved models. Clicking an item swaps it into the active hotbar slot. The game is fully paused while the inventory is open — no cursor grab, no movement, no camera rotation, no block interaction.

### The Cursor Grab Bug
The cursor grab system (`manage_cursor`) grabbed the cursor on ANY left click, including clicks on inventory UI buttons. This caused the inventory to close immediately when clicking a saved model. Fixed by checking `inv.open` in `manage_cursor` — no cursor management while inventory is open.

### Save System
Press P while inside a grid to save the current cell's block pattern as a named model template. The template appears in the inventory and can be placed into any hotbar slot for later use.

---

## Architecture (Current)

```
src/
  main.rs              — App setup, plugins, lighting
  block/
    mod.rs             — BlockType enum (10 types), properties
    materials.rs       — Per-type StandardMaterial initialization
  model/
    mod.rs             — VoxelModel, ModelRegistry, BakedSubMesh
    mesher.rs          — Face-culled mesh baking (per-block-type sub-meshes)
  layer/
    mod.rs             — ActiveLayer, NavEntry, nav_stack
  world/
    mod.rs             — VoxelWorld, VoxelGrid, CellSlot, rendering, render_ancestors
    collision.rs       — AABB clipping collision, block_solid (ancestor-walking)
  editor/
    mod.rs             — Hotbar, HotbarItem enum
    tools.rs           — drill_down/up, place/remove, save_as_template
  interaction/
    mod.rs             — DDA raycast (top-layer + grid), block highlight gizmos
  inventory.rs         — Inventory panel UI, click-to-swap-into-hotbar
  camera.rs            — First-person camera, cursor management
  player.rs            — Movement, gravity, AABB collision integration
  ui/mod.rs            — Hotbar display, mode indicator
  diagnostics.rs       — Debug logging (player pos, depth, entity count)
  tests.rs             — 22 automated tests (collision, navigation, raycast, rendering math)
```

---

## Test Coverage

22 tests covering:
- `block_solid` at top layer, depth 1, depth 2, with parent Block siblings
- Gravity landing at top layer and depth 1
- No step-up (must jump to climb ledges)
- Jump onto elevated platform
- Ceiling collision
- Depth 2 no-void-fall (parent block provides floor)
- Depth 2 fall off grandparent edge (ancestor chain detection)
- `get_grid` navigation at all depths
- `render_ancestor_transforms` math verification
- Model registry save/load, independence of copies

---

## Key Lessons

1. **Don't use tolerances for physics.** Every epsilon/tolerance introduced a new edge case. The AABB clipping approach is correct by construction — no tolerances needed.

2. **Ancestor-walking must be universal.** Any function that looks up world state (collision, rendering, raycasting) must walk the full nav_stack, not just check one level up. The `block_solid` function is the canonical example.

3. **Instance vs template.** Shared references to model data cause mutation bugs. Each placed cell must own its block data.

4. **Separate UI state from game state.** The inventory must fully gate all game input (cursor grab, movement, camera, block interaction) to prevent click-through.

5. **Per-block-type materials, not vertex colors.** Vertex colors force shared PBR parameters. Per-type materials allow metal to be shiny, glass to be transparent, dirt to be matte.

6. **Scale at the Transform level, not with constants everywhere.** `BLOCK_SCALE` threaded through every calculation was fragile. The correct approach: at each layer, blocks are 1 unit. Ancestor meshes get `Transform.scale` to fit.
