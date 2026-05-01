# Coordinator Refinements to IMPLEMENTATION_PLAN.md

This doc supersedes specific sections of `IMPLEMENTATION_PLAN.md` where my critical review found errors or ambiguity. **Implementing agents should follow this doc where it overrides the original plan.**

The original plan is otherwise solid — strong architectural anchors, correct dispatch decisions, AABB cull reuse for banning, correct claim that curvature composes across ribbon pops because `ray_dir` is preserved at camera-frame magnitude.

## Issue 1 — Phase 1 dimensions: 18×9×3 (2:1 Mercator-correct, 3^N-aligned)

**Override Phase 1.2's dimension choice.**

The plan settles on `width=20, height=10, depth=2` with `active_subdepth=2`, which gives an active region of `[0, 2.22) × [0, 1.11) × [0, 0.22)` in planet-frame units. The active edge cuts through the middle of cells at every march-DDA depth — the hardest case to validate.

For Phase 1 we MUST validate the architecture against the simplest aligned case AND keep the Mercator-correct 2:1 aspect ratio (longitude spans 360°, latitude spans 180°; 2:1 grid → roughly square cells at the equator when wrapped onto the sphere).

**Use `width=18, height=9, depth=3` with `active_subdepth=2`.**

- **X axis (longitude):** 18 active cells = `[0, 18/9) = [0, 2)` planet-frame extent. At march-depth 0, cells `x=0, 1` are active; cell `x=2` is entirely banned. The wrap fires at the active edge (`cell.x >= 2 || cell.x < 0`), NOT at the frame edge.
- **Y axis (latitude):** 9 active cells = `[0, 1)` planet-frame extent. At march-depth 0, cell `y=0` is fully active; cells `y=1, 2` are entirely polar (banned, no children, no descent).
- **Z axis (depth):** 3 active cells = `[0, 1/3)` planet-frame extent. At march-depth 0, cell `z=0` is partially active (the first 1/3); rest banned. At march-depth 1 inside cell `z=0`, cells `z=0` is active, cells `z=1, 2` banned. Clean alignment at march-depth 1.

This is the **maximum-X 2:1 layout that fits in one planet-root frame** at active_subdepth=2 (caps at 27 active cells per axis). Larger 2:1 worlds require active_subdepth=3 (max 81 X cells but 9× finer per cell), or a follow-up architecture step for cross-sibling wrap.

**Why this is preferred over 27×9 (3:1):** 27×9 is 3:1, so cells render 1.5× wider than tall at the equator on a sphere — visibly oblong from orbit. 18×9 is 2:1, square cells at the equator.

**Why this is preferred over 20×10×2:** 20 doesn't divide cleanly by 9, so the active edge cuts through the middle of cells. The wrap dispatch and the AABB cull both need partial-cell handling — invasive for Phase 1.

**Z is tunable:** Z=3, 6, or 9 are all aligned. Start at Z=3 for Phase 1 — each Z cell recurses 27³ subcells so absolute dig depth is huge regardless. Bump if Phase 5 needs a thicker crust.

## Issue 2 — Phase 2.1 wrap dispatch: gate on march-depth 0 + active_width_d0 boundary

**Override Phase 2.1's wrap-at-frame-edge formulation.**

With 18×9×3 + active_subdepth=2 (Issue 1), the active X-region spans 2 of the 3 cells at march-depth 0. The wrap fires when the DDA tries to advance from `cell.x = 1 → 2` (or `0 → -1`) — at the active edge, NOT the frame edge.

Required uniform addition (`assets/shaders/bindings.wgsl` Uniforms struct):

```wgsl
/// At march-depth 0 inside a WrappedPlanet frame, X cells with
/// index >= active_width_d0 (or < 0) wrap modularly. For 18×9×3
/// with active_subdepth=2, active_width_d0 = 2.
active_width_d0: u32,
```

Wrap dispatch logic at `assets/shaders/march.wgsl:386` (replacing the OOB X-axis check inside WrappedPlanet):

```wgsl
let in_planet_frame = (depth == 0u) && (uniforms.root_kind == ROOT_KIND_WRAPPED_PLANET);
let active_w = i32(uniforms.active_width_d0);  // 2 for 18-cell planet

if in_planet_frame && (cell.x < 0 || cell.x >= active_w) {
    let new_x = ((cell.x % active_w) + active_w) % active_w;
    // ... rewrite s_cell, cur_side_dist, entry_pos exactly as plan describes
    s_cell[depth] = pack_cell(vec3<i32>(new_x, cell.y, cell.z));
    let dx = new_x - cell.x;  // signed wrap delta in cells
    entry_pos.x = entry_pos.x - f32(-dx) * cur_cell_size;
    // recompute cur_side_dist.x for the wrapped cell
    let lc_wrap = vec3<f32>(f32(new_x), f32(cell.y), f32(cell.z));
    cur_side_dist.x = select(
        (cur_node_origin.x + lc_wrap.x * cur_cell_size - entry_pos.x) * inv_dir.x,
        (cur_node_origin.x + (lc_wrap.x + 1.0) * cur_cell_size - entry_pos.x) * inv_dir.x,
        ray_dir.x >= 0.0,
    );
    continue;
}
```

The wrap depth IS `depth == 0u` for our aligned dimensions, but the boundary is `active_w` (uniform-supplied), not `3`. This is a small generalization of the original plan: the gate stays simple, the boundary is parameterized.

For Y/Z OOB at march-depth 0: keep the standard pop logic. The polar treatment in Phase 3 catches Y-banned rays before they pop; Z-banned rays are handled by empty-cell DDA advance through the banned region.

## Issue 3 — Phase 4.2 bend units are wrong

**Override Phase 4.2's bend formula.**

The plan flags but does not fix: `ct_start` is in camera-frame units. With `ct_start ≈ 80` and `R ≈ 10`, `bend = k · 6400 / 20 = 320 · k` in camera-frame units. That is huge — at `k=1` it bends the ray 320× the planet height downward.

The bend must be applied in **planet-frame units**, with `R` defined in those same units. Concretely:

- The shader's `march()` dispatcher tracks `cur_scale` (the cumulative `1/3^N` ribbon-pop scale; see `march.wgsl:914-987`). Inside the planet's frame, `cur_scale = 1.0` (ribbon pops happen below the planet, not at it). So the planet-frame conversion is `t_planet = ct_start * cur_scale` — and `cur_scale` is constant inside the planet root.
- Define `R_planet` in planet-frame units. With width=27 spanning the full planet frame `[0, 3)`, set `R_planet ≈ 1.5` (half the frame). Then `bend = k · t_planet² / (2 · R_planet)` is bounded for any `t_planet ∈ [0, ~3)`.
- The bend is added to `local_entry.y` in **child-frame** units, so divide by `child_cell_size`.

Final corrected formula at `march.wgsl:727`:

```wgsl
var local_entry = (child_entry - child_origin) / child_cell_size;
if uniforms.root_kind == ROOT_KIND_WRAPPED_PLANET && uniforms.curvature_k > 0.0 {
    let t_planet = ct_start * uniforms.planet_frame_scale;
    let bend_planet = uniforms.curvature_k
                    * t_planet * t_planet
                    / (2.0 * uniforms.planet_radius);
    local_entry.y -= bend_planet / child_cell_size;
}
```

`uniforms.planet_frame_scale = 1.0 / camera_frame_to_planet_frame_ratio` is uploaded from CPU per frame. When the camera is inside the planet's anchor frame, `planet_frame_scale = 1.0`. When the camera is outside (orbit), `planet_frame_scale = 3^N` where N is how many ribbons up from the camera the planet sits — uploaded by `compute_render_frame`.

## Issue 4 — Rename GpuNodeKind fields

**Override Phase 1.1's reuse of `face`/`inner_r`/`outer_r`.**

The plan packs `width/height/depth` into the existing `face`/`inner_r`/`outer_r` u32/f32 slots via bitcast. The bitcast trick works but is fragile and the slot names are wrong now (cubed-sphere is gone).

Rename the slots to neutral names **as a one-shot sweep before Phase 1 starts**:

```wgsl
struct NodeKindGpu {
    kind: u32,    // 0=Cartesian, 3=WrappedPlanet
    geom_a: u32,
    geom_b: u32,
    geom_c: u32,
}
```

Mirror in `src/world/gpu/types.rs`. This is ~20 LoC of mechanical rename, no behavior change. WrappedPlanet writes `geom_a=width, geom_b=height, geom_c=depth`. Cartesian leaves them zero. The bitcast trick disappears.

Schedule: do this rename as a "Phase 0.5" task before Phase 1.1.

## Issue 5 — Phase 5.1 lighting / shadows forward-compat note

The plan says lighting is "wrap-implicit because rays already wrap" — correct for the CURRENT renderer (per-pixel sun shading, no light buffer). But add a comment in `assets/shaders/march.wgsl` near the sun shading that **anyone adding a light buffer or shadow map MUST make it wrap-aware on X**. Single-line comment, easy to add when Phase 4 lands. Catches future regressions without blocking Phase 5.

## Issue 6 — Phase 1.3 acceptance test gap

The plan's `ray_through_banned_cell_misses` test (line 521) tests Y-banned (above-planet) rays. Add sibling tests for X-banned (cell.x=2 at march-depth 0) and Z-banned (z>=1/3 of planet frame) rays:

```rust
#[test]
fn ray_through_x_banned_cell_misses() {
    // 18-cell-wide planet: at march-depth 0 inside the planet,
    // cell.x = 0, 1 are active; cell.x = 2 is banned. Cast a ray
    // that lands entirely in cell.x = 2 of the planet root.
    let mut lib = NodeLibrary::default();
    let (planet, _) = insert_wrapped_planet(&mut lib, 18, 9, 3, 2);
    // ... wrap planet in 22 layers, anchor camera in cell.x=2
    let ray_origin = [2.5, 0.5, 0.1];  // planet-frame x=2.5 is in cell.x=2
    let ray_dir    = [0.0, 0.0, 1.0];
    let hit = cpu_raycast(&lib, root, ray_origin, ray_dir, 4);
    assert!(hit.is_none(), "rays inside the banned X column should miss");
}

#[test]
fn ray_through_z_banned_cell_misses() {
    let mut lib = NodeLibrary::default();
    let (planet, _) = insert_wrapped_planet(&mut lib, 18, 9, 3, 2);
    // active Z range is [0, 1/3) of planet frame; z=0.5 is banned.
    let ray_origin = [1.0, 0.5, 0.5];
    let ray_dir    = [0.001, 0.0, 1.0];
    let hit = cpu_raycast(&lib, root, ray_origin, ray_dir, 4);
    assert!(hit.is_none());
}
```

## Issue 7 — Phase 0.5 (new)

Insert a new "Phase 0.5" step before Phase 1.1:
- Issue 4: rename `GpuNodeKind` fields to `geom_a/b/c` (one PR).
- Audit and delete any remaining `world::sdf::Planet` and related dead code that Agent B flagged but didn't remove (1380-line plan never mentions; flagged in Phase 0.2 deletion report).

This is a 100-line cleanup PR before architectural work starts.

---

## Summary of overrides

| Original plan section | Issue | Override |
|---|---|---|
| Phase 1.2 dims `20×10×2` | Active edge mid-cell + 3:1 fallback breaks Mercator | Use **`18×9×3`** for Phase 1 — 2:1 + 3^N-aligned |
| Phase 2.1 wrap gate | Active edge ≠ frame edge under 18-cell X | Gate on `depth==0u` + parameterize bound by `active_width_d0=2` uniform |
| Phase 4.2 bend units `ct_start` | Camera-frame, breaks at orbit | Multiply by `planet_frame_scale` |
| Phase 1.1 GpuNodeKind reuse | Fragile bitcast, wrong field names | Rename to `geom_a/b/c` first |
| Phase 5.1 lighting | Implicit but undocumented | Add forward-compat shader comment |
| Phase 1.3 acceptance | Only tests Y-banned | Add X-banned and Z-banned tests |
| Sequencing | Plan jumps straight to 1.1 | Insert Phase 0.5 cleanup |

All other sections of the plan stand as written.
