# Wrapped-Cartesian Planet — Implementation Plan

Companion to `HIGH_LEVEL_PLAN.md`. This document is the file:line-cited
playbook for the work. It assumes Phase 0.2 (legacy cubed-sphere
deletion) and Phase 0.3 (testing-infra port) land first; nothing here
plans around `NodeKind::CubedSphere*`, `ActiveFrameKind::Sphere`, or
`assets/shaders/sphere.wgsl`.

References:
- Architecture: `/Users/donalddellapietra/Downloads/wrapped-cartesian-planet-architecture.md`
- High-level: `docs/design/wrapped-planet/HIGH_LEVEL_PLAN.md`
- Old-attempt autopsy: `/Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/sphere-attempt-2-2-3-2/docs/spheres/new-architecture.md`
- Precision lessons: `/Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/sphere-attempt-2-2-3-2/docs/history/sphere-precision-confusion.md`

## Architectural anchors (do not violate)

1. **Same DDA at every depth.** The shader's `march_cartesian`
   (`assets/shaders/march.wgsl:271`) walks `[0,3)³` local frames; ribbon
   pop at `assets/shaders/march.wgsl:914-987` rescales the ray, never
   the world. The wrapped planet must behave the same way: any
   wrap/curvature edits must keep `march_cartesian` operating on
   `local_entry ∈ [0, 3)` and the walker advancing at unit cells.
2. **No absolute-XYZ math at deep depth.** The cubed-sphere autopsy at
   `sphere-precision-confusion.md:42-95` is the cautionary tale: the
   moment any code path starts accumulating cell-bound math at
   `1/3^N` world units, f32 dies. Wrap/curvature edits must read
   coords from `cur_node_origin + cell * cur_cell_size` only inside
   the current frame, where `cur_cell_size` is O(1).
3. **Storage is flat Cartesian.** The architecture proposal's
   centerpiece is that the planet voxels live in normal Cartesian
   nodes. The planet's "wrap" and "polar bound" are *frame-level*
   metadata at one specific depth, not properties of every node.

---

## Phase 1 — Hardcoded Flat Slab

End state: a `WrappedPlanet` node sits at a known depth in the tree,
with banned cells outside the planet's bounding box. Camera inside →
flat ground. Camera outside → finite slab visible.

### 1.1 New `NodeKind::WrappedPlanet`

**Decision: new variant, NOT a `Cartesian` extension.**

| Option | Pros | Cons |
|---|---|---|
| `NodeKind::Cartesian` + `wrap_mask: bvec3` + `bounds: Bounds` | Falls back to existing dispatch | Every Cartesian path now reads metadata it ignores; defeats content-addressing dedup (every Cartesian node hashes uniquely); `aabb`/`raycast` lookups become "is this *the* planet kind?" branches anyway |
| New `NodeKind::WrappedPlanet { width, height, depth }` | Single dispatch site mirrors existing `CubedSphereBody` pattern; opt-in dedup; explicit at the Rust+GPU boundary | New variant adds a tag everywhere `NodeKind` is matched |

The current `NodeKind` (`src/world/tree.rs:84-94`) already pays the
"new variant per geometry kind" cost, and the dispatcher pattern in
`assets/shaders/march.wgsl:558-580` already special-cases sphere body
nodes. Adding one more kind matches the established pattern; bolting
metadata onto Cartesian erodes the only invariant the shader's hot
path relies on (a "Cartesian" node is uniformly 27 children and
nothing else).

#### Rust signature

Add to `src/world/tree.rs:84` (the `pub enum NodeKind {` block, after
`CubedSphereFace`):

```rust
/// Root of a wrapped Cartesian planet. Children are Cartesian (XYZ
/// slot order), but the renderer treats this node specially:
///   - Cells outside the active region (banned cells) render no-hit.
///   - Rays leaving along ±X re-enter from the opposite face.
///   - Rays leaving along ±Y or ±Z exit to the parent shell.
///
/// `width`, `height`, `depth` are top-level cell counts of the
/// active region inside this node's [0, 3)³ frame, expressed in
/// units of the FIRST descent level beneath this node (i.e. each
/// sub-cell at depth N+1 spans 1/3 of the wrapped frame). They are
/// *not* tree depths.
WrappedPlanet {
    /// Active-region width in top-of-planet child cells. The X axis
    /// wraps modulo `width`; the wrapped origin is `[0, width)`.
    width: u16,
    /// Active-region height (Y axis). Bounded; rows outside
    /// `[0, height)` are banned.
    height: u16,
    /// Active-region depth (Z axis). Bounded; rows outside
    /// `[0, depth)` are banned.
    depth: u16,
},
```

Implement `Hash`/`Eq` extensions in the existing `impl Hash for NodeKind`
block at `src/world/tree.rs:104-118`:

```rust
NodeKind::WrappedPlanet { width, height, depth } => {
    width.hash(state);
    height.hash(state);
    depth.hash(state);
}
```

#### GPU mirror

Edit `src/world/gpu/types.rs`:

- The `GpuNodeKind` struct (`types.rs:69-76`) is currently
  `{ kind: u32, face: u32, inner_r: f32, outer_r: f32 }` (16 B).
  Reuse this layout: `kind = 3` for `WrappedPlanet`. The `face` /
  `inner_r` / `outer_r` slots get repurposed as
  `{ width: u32, height: u32, depth_bits_as_u32: u32 }` — pack
  `width` into `face`, `height` into `inner_r.to_bits()` (treat as
  u32), `depth` into `outer_r.to_bits()`. One-time deserialization
  on the GPU reads them back as bitcasts. This keeps the struct at
  16 B without a buffer-layout migration.

  Alternative: widen `GpuNodeKind` to 32 B with explicit fields. The
  bitcast version is cheaper if no other callsite needs the reuse.

- Update `from_node_kind` (`types.rs:79-90`):

  ```rust
  NodeKind::WrappedPlanet { width, height, depth } => Self {
      kind: 3,
      face: width as u32,
      inner_r: f32::from_bits(height as u32),
      outer_r: f32::from_bits(depth as u32),
  },
  ```

  Mirror in `assets/shaders/bindings.wgsl:90-95`:

  ```wgsl
  struct NodeKindGpu {
      kind: u32,        // 0=Cartesian, 1=Body, 2=Face, 3=WrappedPlanet
      face_or_width: u32,
      inner_r_or_height_bits: f32,
      outer_r_or_depth_bits: f32,
  }
  ```

  Helper accessors (`extractWidth(k)` etc.) live in `bindings.wgsl`
  next to the struct.

#### Test

Unit test in `src/world/tree.rs::tests` (next to `dedup_respects_node_kind`
at `tree.rs:448`):

```rust
#[test]
fn wrapped_planet_kind_distinct_from_cartesian() {
    let mut lib = NodeLibrary::default();
    let children = uniform_children(Child::Block(block::STONE));
    let a = lib.insert(children);
    let b = lib.insert_with_kind(
        children,
        NodeKind::WrappedPlanet { width: 20, height: 10, depth: 2 },
    );
    assert_ne!(a, b);
    let c = lib.insert_with_kind(
        children,
        NodeKind::WrappedPlanet { width: 20, height: 10, depth: 2 },
    );
    assert_eq!(b, c, "identical WrappedPlanet should dedup");
}
```

Plus a GPU type round-trip:

```rust
#[test]
fn from_node_kind_wrapped_planet_packs_dims() {
    let k = GpuNodeKind::from_node_kind(NodeKind::WrappedPlanet {
        width: 20, height: 10, depth: 2,
    });
    assert_eq!(k.kind, 3);
    assert_eq!(k.face, 20);
    assert_eq!(k.inner_r.to_bits(), 10);
    assert_eq!(k.outer_r.to_bits(), 2);
}
```

#### Risks

- WGSL strict aliasing: bitcasting `u32 ↔ f32` in a uniform buffer
  needs `bitcast<u32>(...)` calls, not implicit reinterpret. Verify
  the shader pipeline tolerates it (it does — `palette: array<vec4<f32>>`
  already mixes types in the same buffer).
- `NodeKind` is part of the dedup hash. If the same `(width, height,
  depth)` is hashed differently between sessions, the library breaks.
  Use the explicit field hash above; do not derive.

---

### 1.2 Hardcoded preset that drops one planet into the world

#### Geometry decision

The architecture proposal calls for a 2:1 width:height ratio. The
high-level plan suggests `20 × 10 × 2` cells (X × Y × Z) at depth 22
with anchor depth 22. Verify the depth math:

- `MAX_DEPTH = 63` (`src/world/tree.rs:18`).
- `RENDER_ANCHOR_DEPTH = MAX_DEPTH = 63` (`src/app/mod.rs:52`). The
  camera anchor walks down to 63 via `deepened_to`; the active frame
  is whatever node the path can reach.
- "Planet at depth 22" means: the `WrappedPlanet` node sits as a
  child of a depth-21 ancestor, whose 27 slots split the world. Its
  internal cell grid is `20×10×2` top-of-planet child cells, each at
  depth 23 in the tree. That's well below `MAX_DEPTH`.
- Camera inside the planet has anchor depth 22 + however many sub-cell
  zoom levels the player descends — same architecture as the
  fractals (which routinely sit at 30+).

A `width=20` grid is unusual: `[0, 20)` of "child cells" inside a
node whose own children are 0..2 (3 per axis). The rendered active
region therefore spans 20/3 = 6.67× the parent's [0,3) cell. **This
is the surprise** — the planet's `width=20` doesn't fit in one
WrappedPlanet node's 3-cell axis. Two options:

1. **WrappedPlanet at depth 22 with internal cells at depth 23.**
   The active region spans `[0, 20)` X-cells, where each "X-cell" is
   1/3 of the parent. So `width=20` = the planet's X extent in
   parent-frame units = 20/3 ≈ 6.67. The parent at depth 22 has only
   3 X-children — the planet must span multiple sibling nodes.
   Rejected: forces multi-node coordination; can't dedup; the wrap
   must operate across siblings. Defeats the design.
2. **WrappedPlanet at a depth where its 3³ children naturally tile
   the active region.** If the active region is `width × height ×
   depth = 20 × 10 × 2`, then it fits inside an enclosing volume of
   `27 × 9 × 3` (so that `width / 27 = 0.74`, `height / 9 = 1.11`, …
   no, doesn't work cleanly either).

**Resolution: the dimensions are in planet-internal units (sub-cells),
not parent slot units.** Reformulate:

- The `WrappedPlanet` node occupies one cell at its parent's depth
  (e.g. depth 22). Its `[0,3)³` local frame is the *bounding box*
  of the planet.
- The active region is `[0, width) × [0, height) × [0, depth)` in
  the planet's "logical sub-cell" coords, where sub-cells are at
  depth `planet_depth + 1` and span `3.0/3 = 1.0` planet-frame
  units each. So `width=20, height=10, depth=2` means the active
  region is *not* axis-aligned with the planet's 3³ child grid — it's
  expressed in deeper-cell units.

This is the same trick the cubed-sphere body uses: a node defines a
local frame; the active geometry inside that frame is finer than 3³.

For Phase 1.2 we keep things tractable by sizing the active region
at the immediate child-cell level: **active region = `width × height ×
depth` where each is in units of 1/3-sized child cells of the
WrappedPlanet node**. So `width=20, height=10, depth=2` requires
`20×10×2 / 27 ≈ 7.4` planet-frame units, exceeding `[0, 3)`.

**Conclusion: the active region can extend into deeper sub-cells.**
The cleanest formulation: `width` etc. are in units of child cells at
some "active depth" `A`, such that the active region fits inside a
single WrappedPlanet node's `[0, 3)³`. For `20 × 10 × 2`:

- One planet node spans `[0, 3)`.
- Subdivide each axis A levels deep so the active region fits.
  `3 · 3^A ≥ 20` → `A ≥ 2` (since `3 · 9 = 27 ≥ 20`). Use **A=2**.
- Active region in planet-local f32: `[0, 20·1/9) × [0, 10·1/9) ×
  [0, 2·1/9)` = `[0, 2.22) × [0, 1.11) × [0, 0.22)`.

So a single WrappedPlanet node at depth 22, width=20, height=10,
depth=2 (each in units of 1/9 of the planet frame, i.e. depth-24
cells) fits cleanly.

Let me revise the type:

```rust
WrappedPlanet {
    /// Active-region cell count along X. Cells are at depth
    /// `planet_depth + active_subdepth` in the tree (default
    /// `active_subdepth = 2`, so cells are 1/9 of the planet frame).
    width: u16,
    height: u16,
    depth: u16,
    /// Number of subdivisions below this node where one "active
    /// cell" is one tree cell. With `active_subdepth = 2`, each
    /// active cell spans 1/9 of the planet's [0, 3)³ frame.
    active_subdepth: u8,
},
```

For Phase 1.2 the preset hardcodes `active_subdepth = 2` and active
dims `20 × 10 × 2`. The exposed knob lets future content tune the
ratio without editing the type.

#### Files to touch

- `src/world/bootstrap.rs` — add `WorldPreset::WrappedPlanet` variant
  next to `WorldPreset::DemoSphere` at `bootstrap.rs:21`. Add
  dispatch arm in `bootstrap_world` (`bootstrap.rs:134-184`):

  ```rust
  WorldPreset::WrappedPlanet => bootstrap_wrapped_planet_world(),
  ```

  Add `surface_y_for_preset` arm at `bootstrap.rs:93-114` returning
  `Some(0.5)` (any flat-Y surface inside the active region works).

- `src/world/wrapped_planet.rs` — new module. Module declaration in
  `src/world/mod.rs` next to `cubesphere`-line declarations.

  Key function:

  ```rust
  /// Insert a WrappedPlanet node + its children. Returns the
  /// NodeId of the planet root and the path slot it should occupy
  /// in its parent.
  pub fn insert_wrapped_planet(
      lib: &mut NodeLibrary,
      width: u16,
      height: u16,
      depth: u16,
      active_subdepth: u8,
  ) -> (NodeId, u8 /* parent_slot */) {
      // 1. Build a flat slab inside [0, width) × [0, height) ×
      //    [0, depth) at active_subdepth resolution.
      // 2. Wrap the slab in a 3³ Cartesian "filler" node so the
      //    WrappedPlanet itself is the canonical root with children
      //    at active_subdepth=2.
      // 3. Insert with NodeKind::WrappedPlanet { width, height,
      //    depth, active_subdepth }.
  }
  ```

  Exact construction: walk x ∈ [0, width), y ∈ [0, height),
  z ∈ [0, depth). For each cell, decide block type by Y:
    - y == 0 → `block::DIRT` (bedrock band)
    - y == height-1 → `block::GRASS` (top band)
    - else → `block::STONE`
  Pack into a recursive 3³ tree using `slot_index` (`tree.rs:161`).

- `src/world/bootstrap.rs` — `bootstrap_wrapped_planet_world` function:

  ```rust
  fn bootstrap_wrapped_planet_world() -> WorldBootstrap {
      let mut lib = NodeLibrary::default();
      let (planet_id, _) = wrapped_planet::insert_wrapped_planet(
          &mut lib, 20, 10, 2, 2,
      );

      // Wrap planet in 22 layers of air-with-center-child so the
      // planet sits at depth 22 of the world root.
      let mut current = planet_id;
      let air = lib.insert(empty_children());
      for _ in 0..22 {
          let mut children = empty_children();
          children[CENTER_SLOT] = Child::Node(current);
          current = lib.insert(children);
      }
      lib.ref_inc(current);

      let world = WorldState { root: current, library: lib };
      // Spawn at planet's surface centre, looking +Z.
      let spawn_pos = WorldPos::from_frame_local(
          &Path::root(),
          [1.5, 1.5, 1.5],   // centre of root cell
          2,
      ).deepened_to(22);
      WorldBootstrap {
          world,
          planet_path: None,  // planet_path was sphere-only metadata
          default_spawn_pos: spawn_pos,
          default_spawn_yaw: 0.0,
          default_spawn_pitch: -0.2,
          plain_layers: 22,
          color_registry: ColorRegistry::new(),
      }
  }
  ```

- `src/app/test_runner/config.rs:187` — add `--wrapped-planet-world`
  flag next to `--sphere-world`:

  ```rust
  "--wrapped-planet-world" => {
      cfg.world_preset = WorldPreset::WrappedPlanet;
  }
  ```

#### Test

Unit test in `src/world/wrapped_planet.rs`:

```rust
#[test]
fn wrapped_planet_inserts_active_region_only() {
    let mut lib = NodeLibrary::default();
    let (id, _) = insert_wrapped_planet(&mut lib, 20, 10, 2, 2);
    let node = lib.get(id).unwrap();
    assert!(matches!(
        node.kind,
        NodeKind::WrappedPlanet { width: 20, height: 10, depth: 2, active_subdepth: 2 },
    ));
    // Sanity check: representative_block is non-empty (the planet
    // contains stone/grass/dirt).
    assert_ne!(node.representative_block, REPRESENTATIVE_EMPTY);
}
```

Visual harness test (Phase 1.4/1.5 acceptance):

```bash
cargo run -- --render-harness --wrapped-planet-world \
    --spawn-xyz 1.5 1.6 1.5 --spawn-depth 22 \
    --screenshot tmp/phase-1-inside.png \
    --exit-after-frames 60 --timeout-secs 6
```

Acceptance: top half is sky, bottom half is grass colour (block
GRASS = green). No banding.

Outside-the-planet harness (Phase 1.5):

```bash
cargo run -- --render-harness --wrapped-planet-world \
    --spawn-xyz 1.5 0.1 1.5 --spawn-depth 4 \
    --screenshot tmp/phase-1-outside.png \
    --exit-after-frames 60 --timeout-secs 6
```

Acceptance: a finite rectangular slab silhouette visible against sky.
Diff against pixel column histogram: the slab is bounded.

#### Risks

- `compute_render_frame` (`src/app/frame.rs:87-170`) currently knows
  about `Cartesian`, `CubedSphereBody`, `CubedSphereFace`. After
  Phase 0.2 deletes the sphere variants, only `Cartesian` remains.
  Phase 1.1 adds `WrappedPlanet`; the dispatch arm in
  `compute_render_frame` must accept it as "descend through, treat
  like Cartesian for path-walking purposes" (the wrap is only at
  ray-march time, not at compute_render_frame time). Add an arm
  identical to the current `NodeKind::Cartesian` arm at
  `frame.rs:107-117`.
- The CPU mirror (`cartesian.rs:127-150`) currently dispatches into
  `sphere::cs_raycast_in_body` on `NodeKind::CubedSphereBody`. After
  Phase 0.2, that arm vanishes. Phase 1.1 adds nothing to the CPU
  side here — `WrappedPlanet` *acts* like Cartesian for editing
  raycasts (no wrap needed for click-rays, see Phase 4.3). A future
  edit-raycast may want wrap-awareness, but Phase 1's preset doesn't.

---

### 1.3 Cartesian DDA respects banned cells (no-hit, no descend)

The active region `[0, width) × [0, height) × [0, depth)` lives inside
the WrappedPlanet's `[0, 3)³` frame. Cells outside this region — the
banned cells — must render as no-hit, the way the sphere body's
"outside the spherical shell" cells do today.

**Decision: reuse the parent-frame bbox check via the existing AABB
storage buffer (`assets/shaders/bindings.wgsl:169`).** Do NOT add a
per-cell "banned" tag.

#### Why parent-frame bbox

- Per-cell tags would require widening `GpuChild` (currently 8 B,
  `gpu/types.rs:32-40`) or adding a parallel buffer indexed by BFS
  position. New buffer = new bind-group entry, new pack code, new
  shader plumbing — three weeks of scope for a 12-bit AABB that
  already exists.
- The banned-region geometry is a single axis-aligned bbox per
  WrappedPlanet node. The shader's existing AABB cull at
  `march.wgsl:669-699` already reads `aabbs[child_idx]` and skips
  the descent if the ray misses. We hijack this: the planet's
  AABB stored there will be the active region, in planet-local
  units. The march already culls if the ray misses, which gives us
  banned cells for free.
- For *interior* banned cells the AABB cull doesn't apply (the cell
  is inside the planet root's own cull bbox). But there are no
  interior banned cells in the Phase 1 design — banned = "polar
  bands at top/bottom Y". The `height` row is bounded; everything
  outside is air at the planet root's children. The shader's
  existing empty-cell DDA advance handles that.

#### Files to touch

- `src/world/gpu/pack.rs` — find where `aabbs[]` is computed (see
  `src/world/aabb.rs` and the packer's content-AABB pass). For
  WrappedPlanet nodes, override the computed AABB with the active
  region's bbox in planet-local 12-bit units.

  The 12-bit format (`march.wgsl:670-688`) packs `(min_x, min_y,
  min_z, max_x, max_y, max_z)` with each value 2 bits in `[0, 3)`.
  Insufficient resolution — width=20 in planet-relative units is
  20·(1/9) ≈ 2.22 of the planet's `[0, 3)`, which lands at bit 1
  (truncated to "2"). `min` and `max` of `[0, 2)` → 0 and 2 in 2-bit.
  Good enough for Phase 1 (the cull is conservative — banned cells
  inside the active region are handled by the empty children
  themselves).

- `assets/shaders/march.wgsl` — at the dispatch site (`march.wgsl:558-580`
  block), the new arm:

  ```wgsl
  if kind == 3u {
      // WrappedPlanet root. Treat as Cartesian for now (Phase 1.3);
      // wrap dispatch lands in Phase 2. Banned cells outside the
      // active region are caught by the existing aabbs[] cull at
      // line 669-699 — no new code here.
      // Fall through to the existing "Cartesian Node" path.
  }
  ```

  No actual change yet at this site — the existing Cartesian node
  descent path (`march.wgsl:597-790`) handles WrappedPlanet
  identically until Phase 2. The AABB cull does the banning work.

- CPU mirror: same. `cartesian.rs:127-150` currently dispatches on
  `CubedSphereBody`. After Phase 0.2 cleanup that arm is gone. Add
  no new arm for WrappedPlanet — it descends through normally.

#### Test

```rust
#[test]
fn ray_through_banned_cell_misses() {
    use crate::world::wrapped_planet::insert_wrapped_planet;
    let mut lib = NodeLibrary::default();
    let (planet, _) = insert_wrapped_planet(&mut lib, 20, 10, 2, 2);
    let mut root_children = empty_children();
    root_children[13] = Child::Node(planet);
    let root = lib.insert(root_children);
    lib.ref_inc(root);

    // Planet sits in slot 13 (cell 1,1,1) of root, spanning [1,2)
    // in each root axis. Active region inside is [0, 20·1/9), …
    // Cast a ray through the banned region (high Y, above height=10
    // ⇒ y > 10·1/9 = 1.111 in planet-local).
    let ray_origin = [1.5, 2.5, 1.5];
    let ray_dir    = [0.0,  0.001, 1.0];
    let hit = cpu_raycast(&lib, root, ray_origin, ray_dir, 4);
    assert!(hit.is_none(), "ray through banned cell should miss");
}
```

#### Risks

- The 2-bit AABB packing is coarse. `width=20` packs to 2/3 of the
  planet frame, but `height=10` packs to 1/3 of the frame and
  `depth=2` packs to 1/27 — both round up to bit 1. The result is
  conservative (overestimates the active region), so banned-cell
  rays sometimes traverse the planet before missing. Acceptable
  for Phase 1.3 (one extra DDA step is in noise); revisit when
  measuring perf.
- The `aabbs[]` write happens at pack time. If `pack.rs` is
  unaware of WrappedPlanet, the planet's AABB falls back to the
  default `[0, 3)³` and the banning fails. Plumb `NodeKind` into
  the AABB computation explicitly.

---

### 1.4 / 1.5 Visual acceptance

Already specified above (Phase 1.2 test plan). Both screenshots are
artifacts of the harness; visual regression is the acceptance gate.

---

## Phase 2 — X-Wrap

End state: a ray exiting the planet's east face re-enters from the
west face. Walking east continuously sees the same terrain repeat.

### 2.1 Modular X-wrap in shader DDA

#### Where the wrap goes

The OOB (out-of-bounds) pop branch in `march_cartesian` lives at
`assets/shaders/march.wgsl:386-419`. Quoting verbatim
(lines 384-401, the entry of the OOB block):

```wgsl
let cell = unpack_cell(s_cell[depth]);

if cell.x < 0 || cell.x > 2 || cell.y < 0 || cell.y > 2 || cell.z < 0 || cell.z > 2 {
    if depth == 0u { break; }
    depth -= 1u;
    cur_cell_size = cur_cell_size * 3.0;
    let parent_cell = unpack_cell(s_cell[depth]);
    cur_node_origin = cur_node_origin - vec3<f32>(parent_cell) * cur_cell_size;
    let lc_pop = vec3<f32>(parent_cell);
    cur_side_dist = vec3<f32>(
        select((cur_node_origin.x + lc_pop.x * cur_cell_size - entry_pos.x) * inv_dir.x,
               (cur_node_origin.x + (lc_pop.x + 1.0) * cur_cell_size - entry_pos.x) * inv_dir.x, ray_dir.x >= 0.0),
        ...
    );
```

The condition `cell.x < 0 || cell.x > 2` triggers a pop. **For the X
axis inside a WrappedPlanet node, we want to wrap, not pop.** The
wrap is at the planet-frame level (`depth == 0` of the inner
`march_cartesian` call), NOT at sub-cell levels. So the modular
X-wrap is gated on:

- `depth == 0u` — we're at the planet's own root frame, not a
  descendant.
- `current_kind == ROOT_KIND_WRAPPED_PLANET` — added in Phase 1.1's
  GPU dispatch.

#### Proposed code

Replace the OOB X-axis check with a wrap-vs-pop decision. The wrap
happens before the pop logic. Insert at `march.wgsl:386` (replacing
the single `if cell.x < 0 || ...` predicate with two clauses):

```wgsl
// Phase 2: X-wrap inside WrappedPlanet's planet-root frame.
//
// The active region of the planet is [0, planet_w_cells) where
// planet_w_cells = width * 3^(active_subdepth-1). Inside that
// region, when a ray exits ±X at planet-root depth, it re-enters
// the opposite face instead of popping out of the planet node.
//
// Wrap is gated on:
//   - depth == 0u: only at the planet-root frame, not sub-cells
//     (the active region's shape is in planet-local coords; sub-
//     cell descents have their own [0, 3)³ frames).
//   - uniforms.root_kind == ROOT_KIND_WRAPPED_PLANET (or, when
//     dispatched as a child, kind == 3u at this descent level).
//
// The wrap uses cell coords in planet-frame [0, 3) units, but the
// "active width" in those units is planet_w_units = width / 3^a
// (a = active_subdepth - 1 for cell at depth 0). Walking east from
// cell.x = ceil(planet_w_units) lands at cell.x = 0.
let in_planet_frame = (depth == 0u) && (uniforms.root_kind == ROOT_KIND_WRAPPED_PLANET);
let planet_w_cells_at_d0 = uniforms.planet_dims.x;  // see uniform additions below
let oob_x = cell.x < 0 || cell.x >= i32(planet_w_cells_at_d0);
let oob_y = cell.y < 0 || cell.y > 2;
let oob_z = cell.z < 0 || cell.z > 2;

if in_planet_frame && oob_x && !oob_y && !oob_z {
    // Modular X-wrap at the planet-root level.
    var new_x = cell.x;
    if new_x < 0 { new_x = i32(planet_w_cells_at_d0) - 1; }
    if new_x >= i32(planet_w_cells_at_d0) { new_x = 0; }
    s_cell[depth] = pack_cell(vec3<i32>(new_x, cell.y, cell.z));
    // Adjust ray_origin so the wrap is geometrically consistent:
    // a ray exiting at x=W exits at world position
    // (planet_w_cells * cur_cell_size + node_origin.x, ...). After
    // wrap it should appear to be entering at x=0 instead. Update
    // cur_side_dist to reflect new x's side distances:
    let lc_wrap = vec3<f32>(f32(new_x), f32(cell.y), f32(cell.z));
    cur_side_dist.x = select(
        (cur_node_origin.x + lc_wrap.x * cur_cell_size - entry_pos.x) * inv_dir.x,
        (cur_node_origin.x + (lc_wrap.x + 1.0) * cur_cell_size - entry_pos.x) * inv_dir.x,
        ray_dir.x >= 0.0,
    );
    // Tile the ray's effective entry_pos by the planet width so future
    // side_dist updates land on the correct cell boundaries. Compute
    // x_phase = (entry_pos.x - origin) mod (planet_w_cells * cur_cell_size).
    // The simplest implementation reflects the entry into the wrapped
    // domain:
    entry_pos.x = entry_pos.x - f32(sign(cell.x - new_x)) * f32(planet_w_cells_at_d0) * cur_cell_size;
    continue;
}
// Fall through to the existing pop logic for non-wrapping OOB.
if oob_x || oob_y || oob_z {
    // ...existing pop code (unchanged)
}
```

Note: rewriting `entry_pos` is the load-bearing edit. The DDA's
`cur_side_dist` reads `entry_pos` as the reference point. After the
wrap we virtually shift the ray's history so that future cell
boundary computations stay correct in the wrapped domain. The
alternative is to convert side_dist into per-cell deltas — invasive.

#### Uniform additions

Add to `Uniforms` (`assets/shaders/bindings.wgsl:29-68`):

```wgsl
/// Active region of the WrappedPlanet root, in planet-local
/// integer cell coords at depth 0 (planet-root frame). x = width,
/// y = height, z = depth, w = active_subdepth (so renderer can
/// decode finer-cell widths during sub-cell descent if needed).
planet_dims: vec4<u32>,
```

Mirror in `src/renderer/init.rs` (`GpuUniforms` struct), and write
from `src/app/edit_actions/upload.rs` when uploading the per-frame
uniforms — read from the active frame's `WrappedPlanet` kind.

#### Edge case: don't double-wrap on sub-cell descent

When the DDA descends into a child of a wrap-adjacent cell (e.g.
the eastmost active X-cell), the child frame is its own `[0, 3)³`.
At that descent level `depth > 0u`, so the `in_planet_frame` gate
above is false — the child uses ordinary pop semantics, and pops
back up to the parent. The parent (which is `depth == 0u` again) then
applies the wrap. This is correct: the wrap is a planet-frame
property, not a child-frame property.

#### Risks

- `entry_pos` rewrite is f32-fragile if the ray traverses many wraps
  in a single march. Cap with the existing `max_iterations = 2048u`
  (`march.wgsl:377`) — at 2048 iterations and a 20-cell-wide planet,
  100 wraps max, which is well within f32 mantissa precision for
  `cur_cell_size ≥ 0.01`.
- The wrap must NOT apply when the ray was "popping out of the
  planet through a polar face (Y/Z)" — those still pop. The
  `!oob_y && !oob_z` gate enforces this.
- The wrap must not apply at sub-cell descent — gated by `depth == 0u`.

#### Test

Image-analysis test, headless harness:

```bash
cargo run -- --render-harness --wrapped-planet-world \
    --spawn-xyz 1.5 1.5 1.5 --spawn-depth 22 \
    --script "wait:30" --screenshot tmp/wrap-east-0.png \
    --exit-after-frames 60 --timeout-secs 6

# Walk east 5 cells (using a script command — needs new harness verb).
# Verify tmp/wrap-east-5.png has the same pixel histogram as east-0.
```

Phase 2.3 acceptance is a screenshot regression: tmp/wrap-east-0.png
and tmp/wrap-east-after-N-cells.png have the same dominant column
pattern (within ε noise from anti-aliasing).

---

### 2.2 Modular X-wrap in CPU `Path::step_neighbor_cartesian`

The CPU mirror at `src/world/anchor.rs:104-128` (quoted earlier) walks
parent slots on overflow. For the camera anchor inside a WrappedPlanet
node, walking east past the east edge must wrap, not bubble up.

The complication: `step_neighbor_cartesian` doesn't know what NodeKind
its enclosing path slot is. The path is just slot indices — semantics
live in the `NodeLibrary`.

#### Solution

Add a wrap-aware variant. Quote the current code at `anchor.rs:104-128`:

```rust
pub fn step_neighbor_cartesian(&mut self, axis: usize, direction: i32) {
    debug_assert!(axis < 3);
    debug_assert!(direction == 1 || direction == -1);
    if self.depth == 0 {
        return;
    }
    let d = self.depth as usize - 1;
    let slot = self.slots[d] as usize;
    let (x, y, z) = slot_coords(slot);
    let mut coords = [x, y, z];
    let v = coords[axis] as i32 + direction;
    if (0..3).contains(&v) {
        coords[axis] = v as usize;
        self.slots[d] = slot_index(coords[0], coords[1], coords[2]) as u8;
    } else {
        // Bubble up: pop, step parent, push the wrapped slot.
        self.depth -= 1;
        self.step_neighbor_cartesian(axis, direction);
        let wrapped = if direction < 0 { 2 } else { 0 };
        coords[axis] = wrapped;
        let new_slot = slot_index(coords[0], coords[1], coords[2]) as u8;
        self.slots[self.depth as usize] = new_slot;
        self.depth += 1;
    }
}
```

Replacement (extending the existing function with wrap-aware logic):

```rust
pub fn step_neighbor_cartesian(&mut self, axis: usize, direction: i32) {
    self.step_neighbor_with_lib(axis, direction, None);
}

pub fn step_neighbor_with_lib(
    &mut self,
    axis: usize,
    direction: i32,
    lib: Option<&NodeLibrary>,
) {
    // ... walk parents, building a stack of (node_id, slot) so we
    // know each level's NodeKind.
    // When bubble-up reaches a NodeKind::WrappedPlanet ancestor and
    // axis == 0, apply modular wrap at THAT level's active region
    // bounds, not at slot bounds.
    // Delegate to a recursive helper that carries the library.
}
```

The implementation must walk the path while resolving NodeIds via the
library (stepping in lockstep with what `compute_render_frame` does at
`src/app/frame.rs:99-133`). Walking the library is O(depth); doing it
on every player step is fine.

For Phase 2.2 the simpler shape: keep the original
`step_neighbor_cartesian` semantics, and add a sibling
`step_neighbor_in_planet(axis, direction, planet_path, planet_dims)`
that the camera-update code calls when it knows the camera is inside
a planet. The existing `add_local` (`anchor.rs:268-274`) calls
`renormalize_cartesian` which calls `step_neighbor_cartesian`. We
add a parallel path for in-planet motion.

The cleanest plumbing: `WorldPos::add_local` already takes a
`_lib: &NodeLibrary` (`anchor.rs:268`) but ignores it. Use it now —
walk the path, find the deepest `NodeKind::WrappedPlanet` ancestor on
self.anchor, and apply wrap at that depth's edge.

#### Test

Unit test in `src/world/anchor.rs::tests`:

```rust
#[test]
fn step_east_wraps_inside_wrapped_planet() {
    let mut lib = NodeLibrary::default();
    let (planet, _) = wrapped_planet::insert_wrapped_planet(
        &mut lib, 20, 10, 2, 2,
    );
    let mut root_children = empty_children();
    root_children[13] = Child::Node(planet);
    let root_id = lib.insert(root_children);

    // Build a path: root → planet (slot 13) → planet child (slot 0).
    let mut anchor = Path::root();
    anchor.push(13);
    anchor.push(0);  // far west, depth 1 inside planet (planet-frame x=0)

    // Step west 1: should wrap to far-east cell at the same depth.
    anchor.step_neighbor_with_lib(0, -1, Some(&lib));
    assert_eq!(anchor.depth(), 2);
    assert_eq!(anchor.slot(0), 13);
    // The wrapped slot at depth-1 inside the planet should be the
    // east-edge slot; for a 20-wide planet at active_subdepth=2,
    // east edge is the slot whose planet-frame x=ceil(20/9)=3.
    // ... (exact slot depends on canonicalization)
}
```

Plus a "walk N steps east, end at same physical position" test:

```rust
#[test]
fn walking_east_planet_width_returns_to_origin() {
    let mut lib = ...;
    let mut anchor = ... ;  // some position inside the planet
    let start = anchor;
    for _ in 0..(20 /* width in active cells */) {
        anchor.step_neighbor_with_lib(0, 1, Some(&lib));
    }
    // After 20 east steps in a width-20 planet, anchor at the same
    // active cell (mod 20) and Y/Z unchanged.
    assert_eq!(anchor.slot(0), start.slot(0));
    // ... (and identical sub-cell slots)
}
```

#### Risks

- Path arithmetic across NodeKinds is the load-bearing piece. The
  existing `step_neighbor_cartesian` is total: bubble-up reaches
  the root and clamps. Adding `WrappedPlanet`-aware bubble-up means
  the recursion can stop one level deep and apply wrap. The depth
  bookkeeping must not double-pop.
- The library lookup needs a path-of-NodeIds to know each level's
  `NodeKind`. The `Path` struct (`anchor.rs:30-33`) only stores
  slot indices, not NodeIds. Reconstruct NodeIds by walking from
  root each call. O(depth) is acceptable for a single step (called
  once per axis per frame).

---

## Phase 3 — Polar Treatment

End state: looking down at the planet from "above the pole" (looking
along -Y from outside the active region) doesn't reveal a hole. The
banned region renders something cosmetic, not sky.

### Decision: Option (a) — extended ground colour as a flat impostor disk

Three options:

| Option | What it does | Pros | Cons |
|---|---|---|---|
| (a) Extended ground colour as flat impostor disk | The shader returns a constant colour for any ray hitting the polar bbox above/below the active region | Cheapest; one extra branch in the shader; matches "ice cap" reading from orbit | Up close it's flat; player won't reach poles anyway |
| (b) Solid polar cap shader effect | Procedurally shade the cap with bevels / fake atmospheric haze | Looks better up close | More shader code; tunable; risk of drifting toward the "no atmosphere" non-goal in HIGH_LEVEL_PLAN.md |
| (c) Fixed colour fill of banned region | Same as (a) but applied to the full polar volume, not just the projected disk | Even cheaper than (a) | From oblique angles, looks like a brick of ice rather than a smooth cap |

**Recommend (a).** The high-level plan says explicitly "no atmosphere
or anything fancy" and "from above-pole, no hole visible". A flat
impostor disk meets both constraints with one shader branch.

#### Implementation

In `assets/shaders/march.wgsl`, after the dispatch arm for
`WrappedPlanet` lands its existing Cartesian descent, add a polar-cap
branch when the ray exits the active region's Y bounds at the
planet-root frame:

At the OOB block (`march.wgsl:386-419`), inside the new wrap branch
from Phase 2.1, add a Y-bounds check:

```wgsl
let oob_y_above = (cell.y >= i32(uniforms.planet_dims.y)) && (depth == 0u)
                  && in_planet_frame;
if oob_y_above {
    // Polar cap impostor: solid grey/white. Find the t-value of
    // the ray crossing the y = height plane.
    let plane_y = f32(uniforms.planet_dims.y) * cur_cell_size + cur_node_origin.y;
    let t_plane = (plane_y - ray_origin.y) * inv_dir.y;
    if t_plane > 0.0 {
        result.hit = true;
        result.t = t_plane;
        result.color = vec3<f32>(0.9, 0.92, 0.95);  // ice
        result.normal = vec3<f32>(0.0, 1.0, 0.0);
        result.cell_min = vec3<f32>(0.0, plane_y, 0.0);
        result.cell_size = cur_cell_size;
        return result;
    }
}
// Mirror for oob_y_below at y < 0.
```

#### Test

Visual harness, top-down view:

```bash
cargo run -- --render-harness --wrapped-planet-world \
    --spawn-xyz 1.5 2.9 1.5 --spawn-depth 22 \
    --script "wait:30" --screenshot tmp/phase-3-pole-top.png \
    --exit-after-frames 60 --timeout-secs 6
```

Acceptance: the screenshot shows ice-grey covering the planet
silhouette where active terrain ends, no sky-blue holes.

#### Risks

- The "active region's Y plane" must be in the SAME frame coords as
  `ray_origin` / `cur_node_origin`. After the wrap-and-walk in
  Phase 2, we're still at depth 0 of the planet root, so this works.
  Check that `cur_node_origin.y` is the planet's bottom-Y in the
  current frame.
- Edge case: a ray that misses the active region but enters the
  polar volume from outside the planet's `[0, 3)³` bbox should still
  render the polar cap. The AABB cull at `march.wgsl:669-699` can
  cull the planet entirely if the ray misses both the active region
  and the polar bbox. Widen the AABB to include the polar caps.

---

## Phase 4 — Curvature

End state: the same flat voxel data renders as a sphere from orbit.
Transition is smooth. Editing rays bypass the curvature.

### 4.1 `k(altitude)` curve in CPU, uploaded to shader

#### The curve

The user wants a smooth transition. Define `k(altitude) ∈ [0, 1]`
where altitude is the camera's distance above the active region's
top-Y (i.e. the highest player-reachable cell).

ASCII plot of recommended sigmoid:

```
k
1.0 |                            ___________________
    |                          /
    |                        /
0.5 |                      / <- inflection at altitude = 4·planet_height
    |                    /
    |                  /
0.0 |________________/
    |
    +----------------------------------------
       0  1  2  3  4  5  6  7  8  9  10
                         altitude / planet_height
```

Formula:

```rust
pub fn curvature_k(altitude: f32, planet_height: f32) -> f32 {
    let h_norm = altitude / planet_height.max(0.001);
    // Sigmoid centred at 4·planet_height, slope picked so k(0) ≈ 0
    // and k(8·planet_height) ≈ 1.
    let x = (h_norm - 4.0) * 0.8;
    let s = 1.0 / (1.0 + (-x).exp());
    s.clamp(0.0, 1.0)
}
```

Alternative: smoothstep with explicit endpoints — easier to tune.

```rust
pub fn curvature_k(altitude: f32, planet_height: f32) -> f32 {
    let lo = planet_height * 1.0;
    let hi = planet_height * 8.0;
    let t = ((altitude - lo) / (hi - lo)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)  // smoothstep
}
```

Pick smoothstep for tunability. Tune `lo`/`hi` via the harness in
Phase 4.4.

#### Files to touch

- `src/world/wrapped_planet.rs` — add `pub fn curvature_k(...)`.
- `src/app/edit_actions/upload.rs` — read camera altitude relative to
  the active frame's `WrappedPlanet` ancestor, compute `k`, write to
  uniforms.
- `assets/shaders/bindings.wgsl:29` — add `curvature_k: f32` (and a
  pad) to `Uniforms`.
- `src/world/gpu/types.rs` — mirror in `GpuUniforms`.

#### Test

Unit test the curve shape:

```rust
#[test]
fn curvature_k_is_zero_at_surface_and_one_high() {
    let h = 10.0;
    assert_eq!(curvature_k(0.0, h), 0.0);
    assert!(curvature_k(80.0, h) > 0.99);
    assert!((curvature_k(40.0, h) - 0.5).abs() < 0.1);
}
```

---

### 4.2 Parabolic ray-bend at sample point

#### Insertion site

Quote `march.wgsl:725-727`:

```wgsl
let ct_start = max(node_hit.t_enter, 0.0) + 0.0001 * child_cell_size;
let child_entry = ray_origin + ray_dir * ct_start;
let local_entry = (child_entry - child_origin) / child_cell_size;
```

This is per-descent (when the ray steps into a deeper child). Curvature
is conceptually per-step, but the ribbon-pop architecture means
`local_entry` is recomputed per descent — close enough for visual
purposes (the cell granularity at descent is the natural sample rate).

But the cleaner site is at the entry to the planet root frame —
i.e. inside `march()` (`march.wgsl:808-998`) at the dispatch site
right before calling `march_cartesian`. The bend is applied to the
entire ray relative to the planet's frame, then DDA proceeds straight
in the bent coords.

Quote `march.wgsl:840-851`:

```wgsl
} else {
    // Cartesian frame: no depth cap beyond the hardware
    // stack ceiling. ...
    r = march_cartesian(current_idx, ray_origin, ray_dir, MAX_STACK_DEPTH, skip_slot);
}
```

Replace with a wrap-aware dispatch that bends `ray_origin` / `ray_dir`
before the call when `current_kind == ROOT_KIND_WRAPPED_PLANET`:

```wgsl
} else if current_kind == ROOT_KIND_WRAPPED_PLANET {
    // Per-step parabolic bend approximation, applied via a one-shot
    // pre-warp of ray_origin's Y based on accumulated camera-frame
    // distance. The bend factor is k * dist² / (2R), where R is
    // the implied planet radius (set so the silhouette is round
    // from k=1 altitude). dist is approximated as |ray_origin -
    // camera_pos| in planet-frame units.
    let R = uniforms.planet_radius;  // see uniform additions
    let k = uniforms.curvature_k;
    // The march walks the data straight; the bend is the difference
    // between the straight ray's sampled Y and what the curved ray
    // sees. Implement by adjusting ray_origin.y on a per-step basis,
    // which inside march_cartesian we approximate by adjusting it
    // ONCE per descent at line 727 below.
    r = march_cartesian(current_idx, ray_origin, ray_dir,
                        MAX_STACK_DEPTH, skip_slot);
}
```

Then inside `march_cartesian`, at `march.wgsl:727`, replace:

```wgsl
let local_entry = (child_entry - child_origin) / child_cell_size;
```

with:

```wgsl
var local_entry = (child_entry - child_origin) / child_cell_size;
if uniforms.root_kind == ROOT_KIND_WRAPPED_PLANET && uniforms.curvature_k > 0.0 {
    // Parabolic bend: a ray at distance t from the camera has its
    // sampled Y reduced by t²·k / (2·R). t is approximated by
    // ct_start (the camera-frame ray distance to this descent).
    let bend = uniforms.curvature_k * ct_start * ct_start
               / (2.0 * uniforms.planet_radius);
    local_entry.y -= bend / child_cell_size;
}
```

Important: `ct_start` is in camera-frame units (after ribbon pops,
the dir has been preserved per `march.wgsl:938-947`), so the bend is
in camera-frame units too. Dividing by `child_cell_size` converts to
the child's local frame.

#### Composition across ribbon pops

The bend is a function of `t² · k / (2R)`. After a ribbon pop,
`ray_dir` is preserved (`march.wgsl:947` keeps `ray_dir` at camera-
frame magnitude), so subsequent `t` values in the popped frame
continue accumulating from the same camera origin. The bend value
computed inside `march_cartesian` uses `ct_start` from the inner
frame's dispatch — that's the camera-frame `t` for that descent
boundary, NOT a frame-local `t`. Therefore the bend composes
correctly.

Confirm by walking through: a ray that traverses 1 cell at depth=0
in frame F0, pops to ancestor F1, traverses 5 cells, hits.
- t at hit = (sum of segment lengths in camera-frame units).
- bend at hit = k · t² / (2R).
- The DDA inside F1 uses `ct_start_f1` to compute the bend at descent.
That's identical to the camera-frame distance to F1's entry.
Good.

The only risk: the bend depends on `ct_start` being camera-frame
distance, not a local-frame distance. Verify in code review that the
DDA's `t_start` (`march.wgsl:357`) is camera-frame after pops.
Reading `march.wgsl:946` ("ray_dir preserved at camera-frame
magnitude") confirms this.

#### Test

Image-analysis: silhouette curvature.

```rust
// From orbit (altitude = 8·planet_height, k → 1), the planet's
// silhouette pixels should fit a circle within ε.
fn silhouette_is_circular(image: &Image) -> bool {
    let edge = detect_silhouette_edge(image);
    let (cx, cy) = centroid(&edge);
    let radii: Vec<f32> = edge.iter().map(|p| dist(p, (cx, cy))).collect();
    let mean_r = radii.iter().sum::<f32>() / radii.len() as f32;
    let max_dev = radii.iter()
        .map(|r| (r - mean_r).abs() / mean_r)
        .fold(0.0, f32::max);
    max_dev < 0.05  // 5% tolerance for f32 + raster + curvature
}
```

The visual regression test (Phase 4.5) runs the harness from orbit:

```bash
cargo run -- --render-harness --wrapped-planet-world \
    --spawn-xyz 1.5 2.95 1.5 --spawn-depth 4 \
    --screenshot tmp/phase-4-orbit.png \
    --exit-after-frames 60 --timeout-secs 6
```

Acceptance: `silhouette_is_circular(tmp/phase-4-orbit.png) == true`.

#### Risks

- The bend uses `ct_start` only at descent boundaries. Inside a single
  descent, the bend is constant — visually this is fine because cells
  inside a descent are tiny in screen-space at orbit altitude. But
  at low altitudes, individual cells might span many bend deltas;
  cells appear "squashed" non-uniformly. Mitigation: tune the
  smoothstep so `k > 0` only at altitudes where individual cells
  are sub-pixel. Phase 4.4 verifies this.
- `planet_radius` is a free parameter. Set it so the planet is
  spherical when `k=1`: `R = planet_height + air_buffer`. Tune in
  Phase 4.4.

---

### 4.3 Click-ray for editing remains straight

The architecture proposal (line 59) says: cast two rays — one straight
for gameplay, one curved for visuals.

#### Files to touch

The editing code path enters via `cpu_raycast_in_frame` at
`src/world/raycast/mod.rs:58-123`. This is the CPU mirror; it doesn't
read the curvature uniform, so it's straight by default. **No change
needed for editing rays — they already bypass curvature.**

The shader-side click-ray (the highlight raycast for showing where
the cursor points) ALSO must bypass curvature. The highlight is a
GPU raycast originating from the same camera but for visualization
of the editing target, not pixel rendering. The current architecture
uses `cpu_raycast_in_frame` for highlight (`src/app/cursor.rs`,
`src/app/edit_actions/`) — already CPU, already straight.

**Confirm via code search**: the shader's `march()` (which applies
curvature in Phase 4.2) is called by `fs_main` for pixel rendering.
There is no GPU click-ray. The highlight bbox is computed CPU-side
and uploaded as a uniform (`uniforms.highlight_min/max` at
`bindings.wgsl:56-57`), then drawn as a wireframe in a separate pass.

So Phase 4.3 is a no-op acceptance check, not a new code path.

#### Test

```rust
#[test]
fn cpu_raycast_in_frame_unaffected_by_curvature() {
    // Same scene, two raycasts — verify hit cell is identical
    // regardless of camera altitude (which controls curvature_k).
    // Since cpu_raycast_in_frame doesn't read curvature_k, this is
    // a structural assertion.
    let world = bootstrap_wrapped_planet_world().world;
    let frame_path = [13u8];  // planet's slot in root
    let hit_low = cpu_raycast_in_frame(...);
    let hit_high = cpu_raycast_in_frame(...);
    assert_eq!(hit_low.unwrap().path, hit_high.unwrap().path);
}
```

---

### 4.4 Tune transition altitude band

Manual / visual. Run the harness at a sequence of altitudes:

```bash
for y in 1.5 1.8 2.1 2.4 2.7 2.95; do
    cargo run -- --render-harness --wrapped-planet-world \
        --spawn-xyz 1.5 $y 1.5 --spawn-depth 4 \
        --screenshot tmp/phase-4-y$y.png \
        --exit-after-frames 60 --timeout-secs 6
done
```

Visual check: smooth transition, no horizon-pop.

If a visible "the horizon just bent" moment appears, widen the
smoothstep's `(lo, hi)` band. Each lap of tuning is 6 screenshots.

---

### 4.5 Already covered above (silhouette_is_circular).

---

## Phase 5 — Integration

End state: lighting, entities, and other gameplay systems handle the
wrap correctly. The full descent (orbit → surface → orbit) renders
cleanly and the test harness passes.

### 5.1 Wrap-aware coord ops

Systems that need wrap awareness:

| System | File | Change |
|---|---|---|
| Lighting propagation | `src/world/lighting.rs` (if it exists; otherwise the renderer's per-pixel hit shading is the only lighting and it's wrap-implicit because rays already wrap) | Sun direction + per-cell light values must wrap on X. Current renderer has no per-cell lighting buffer; sky shading at pixel-level inherits wrap automatically. **Likely no-op in current architecture.** |
| Entity placement | `src/world/scene.rs`, `src/world/entity.rs` | When an entity wanders past x=width-1, snap to x=0. One call site. |
| Collision | `src/world/raycast/mod.rs:cpu_raycast_in_frame` | Currently unaware. Add wrap dispatch when the planet path is involved. **Defer to Phase 5 polish; not on the critical path.** |
| AI / pathfinding | (none in current codebase) | n/a |

For Phase 5.1 the **critical** change is entity placement — entities
that drift across the wrap must snap. All other systems are visual
or simulation-only and inherit wrap correctness from `step_neighbor_with_lib`.

### 5.2 End-to-end harness

Add to `tests/`:

```rust
// tests/wrapped_planet_descent.rs
#[test]
fn full_descent_orbit_to_surface() {
    // Orbit screenshot.
    let orbit = run_harness("--spawn-xyz 1.5 2.95 1.5 --spawn-depth 4");
    assert!(silhouette_is_circular(&orbit));

    // Mid-altitude.
    let mid = run_harness("--spawn-xyz 1.5 2.0 1.5 --spawn-depth 12");
    // (silhouette should be partial circle)

    // Surface.
    let surf = run_harness("--spawn-xyz 1.5 1.5 1.5 --spawn-depth 22");
    // (top half sky, bottom half flat ground)

    // No image is more than 50% sky-coloured (pole hole regression).
    for img in [&orbit, &mid, &surf] {
        assert!(sky_fraction(img) < 0.7);
    }
}
```

### 5.3 Final polish

- Delete dev-only `--wrapped-planet-world` debug prints.
- Confirm `surface_y_for_preset` in `bootstrap.rs:93` returns sensible
  Y for entity rest height.
- Pack `aabbs[]` correctly for WrappedPlanet — verify the AABB cull
  doesn't false-positive cull the polar caps (Phase 3 added them
  outside the active region's bbox).

---

## Cross-cutting risks

1. **Two precision pitfalls to watch**:
   - The wrap's `entry_pos.x -= W·cur_cell_size` rewrite introduces
     accumulated drift if the ray wraps many times. Bound max
     iterations (already 2048) and verify mantissa stays clean.
   - The bend's `ct_start²` is at camera-frame magnitude. At 8·planet_height
     altitude, `ct_start` could be ≈ 80 cells = 80 in camera-frame
     units. Squared: 6400. Divided by `R ≈ 10` and multiplied by
     `k ≤ 1`: 640. That's the bend in Y, in camera-frame units —
     huge. The bend must be applied in PLANET-FRAME units (where
     R is normalized to unity), or the formula must scale by
     `cur_cell_size`. Re-verify in Phase 4.2 implementation.

2. **Don't backslide into per-cell tags.** The temptation to add a
   per-cell "banned" or "polar" flag will recur. Resist it. The
   AABB cull + the polar-impostor branch + the wrap is enough.

3. **`compute_render_frame` must accept `WrappedPlanet`**. Currently
   it has explicit `NodeKind` arms (`frame.rs:107-129`). Phase 1.1
   adds an arm; without it, the camera path stalls at the planet
   root and the renderer goes black.

4. **Ribbon entries.** When the camera anchor is *inside* the
   planet, the ribbon at `assets/shaders/march.wgsl:914-987` walks
   *outward* — into the planet's parent, then grandparent, etc. The
   pop pop-back math (`ray_origin = slot_off + ray_origin / 3.0`)
   assumes pure 3³ subdivision. WrappedPlanet IS a 3³ subdivision
   at its own layer (the wrap is at deeper sub-cells via
   `active_subdepth`). The pop should work without changes; verify
   in Phase 2 testing.

---

## Acceptance summary

| Phase | Acceptance |
|---|---|
| 1.1 | `cargo test wrapped_planet_kind_distinct_from_cartesian` passes |
| 1.2 | `cargo run -- --wrapped-planet-world` boots; preset visible |
| 1.3 | `ray_through_banned_cell_misses` test passes |
| 1.4 | `tmp/phase-1-inside.png` shows top half sky, bottom half grass |
| 1.5 | `tmp/phase-1-outside.png` shows finite slab silhouette |
| 2.1 | wrap-east screenshot regression matches |
| 2.2 | `walking_east_planet_width_returns_to_origin` passes |
| 3 | `tmp/phase-3-pole-top.png` shows ice-grey, no sky leak |
| 4.2 | `silhouette_is_circular(tmp/phase-4-orbit.png)` is true |
| 4.3 | `cpu_raycast_in_frame_unaffected_by_curvature` passes |
| 5.2 | `tests/wrapped_planet_descent.rs::full_descent_orbit_to_surface` passes |

Total LoC estimate (post Phase 0):
- Phase 1: ~+250 LoC (tree.rs +30, gpu/types.rs +20, wrapped_planet.rs +150, bootstrap.rs +50)
- Phase 2: ~+300 LoC (march.wgsl +100, anchor.rs +150, bindings.wgsl +20, types/upload +30)
- Phase 3: ~+50 LoC (march.wgsl +30, tests +20)
- Phase 4: ~+150 LoC (curvature_k +20, march.wgsl +40, types/upload +20, tests +70)
- Phase 5: ~+100 LoC (entity wrap +20, e2e harness +80)

Net: ~+850 LoC in new code, mostly Rust + WGSL. The existing renderer
hot path adds 4-5 lines (the wrap branch in OOB, the polar branch, the
bend in `local_entry`). All other changes are at dispatch boundaries
(`compute_render_frame`, `march()` dispatch, NodeKind packing).
