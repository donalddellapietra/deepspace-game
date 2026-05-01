# Coordinator Refinements to IMPLEMENTATION_PLAN.md

This doc supersedes specific sections of `IMPLEMENTATION_PLAN.md` where my critical review found errors or ambiguity. **Implementing agents should follow this doc where it overrides the original plan.**

The original plan is otherwise solid — strong architectural anchors, correct dispatch decisions, AABB cull reuse for banning, correct claim that curvature composes across ribbon pops because `ray_dir` is preserved at camera-frame magnitude.

## Issue 1 — Phase 1 dimensions: use 3^N-aligned numbers, not 20×10×2

**Override Phase 1.2's dimension choice.**

The plan settles on `width=20, height=10, depth=2` with `active_subdepth=2`, which gives an active region of `[0, 2.22) × [0, 1.11) × [0, 0.22)` in planet-frame units. This means the active edge cuts through the middle of cells at every march-DDA depth. That is the single hardest case to validate.

For Phase 1 we MUST validate the architecture against the simplest case first. **Use dimensions that are powers of 3**:

- **Phase 1 (slab):** `width=27, height=9, depth=3` with `active_subdepth=2`.
  - X axis: 27 active cells = full `[0, 3)` planet frame at march-depth 0. The wrap fires at the planet-frame OOB boundary, exactly where `march.wgsl:386` already detects.
  - Y axis: 9 active cells = `[0, 1)` of the planet frame = exactly cell `y=0` at march-depth 0. Cells `y=1, 2` at march-depth 0 are entirely polar (banned).
  - Z axis: 3 active cells = `[0, 1/3)` of the planet frame = exactly cell `z=0` at march-depth 1, sub-cell `z=0` at march-depth 0. Aligned at depth 1.

- **Phase 4 (curvature):** keep 27×9×3 — the curvature math doesn't care about dimensions.

- **Later, when 20×10×2 is needed for art:** add a follow-up phase to handle partial-cell active regions. Out of scope for Phase 1–5.

**Why the user said "20×10×2":** that was illustrative ("something like 20×10×2"). The 2:1 aspect ratio matters; the exact integer doesn't. 27:9 is 3:1 — close enough for orbit visuals, since the user explicitly said they don't care about latitude travel distance. We can relax to non-3^N dims after the architecture is validated.

## Issue 2 — Phase 2.1 wrap dispatch depth is wrong

**Override Phase 2.1's `depth == 0u` gate.**

The plan gates the wrap on `depth == 0u` of the inner `march_cartesian`. With Issue-1 dimensions (27×9×3, active_subdepth=2), the active region is exactly `[0, 3)` along X — i.e., the planet-frame boundary IS the active boundary. So the wrap correctly fires at march-depth 0, when `cell.x ∈ {-1, 3}`.

But: the wrap depth is **dimension-dependent**, not a fixed `depth == 0u`. In general, the wrap fires at `march_depth == active_subdepth - log_3(width / 3)`, i.e., the depth at which `cell.x ∈ [0, 3)` covers exactly the active width.

For Phase 2 with 27×9×3 + active_subdepth=2: active width = 27 cells at depth 2, which is `27/9 = 3` cells at depth 0. So march-depth 0 it is. The plan's `depth == 0u` is correct **only because we picked aligned dimensions in Issue 1**. Document it but don't generalize until later phases need it.

The simpler restatement: **the wrap fires when march-DDA cell-coords exit the planet-frame `[0, 3)` boundary, AND we are inside a `WrappedPlanet` root frame.** No new uniform for `planet_w_cells_at_d0` is needed in Phase 2 — the X-OOB check at `march.wgsl:386` already catches `cell.x ∈ {-1, 3}`. Replace it with a wrap when in WrappedPlanet, pop otherwise.

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

The plan's `ray_through_banned_cell_misses` test (line 521) tests Y-banned (above-planet) rays. Add a sibling test for Z-banned rays (with 27×9×3 dims, the active region is only the first 1/3 of cell `z=0` at march-depth 0):

```rust
#[test]
fn ray_through_z_banned_cell_misses() {
    let mut lib = NodeLibrary::default();
    let (planet, _) = insert_wrapped_planet(&mut lib, 27, 9, 3, 2);
    // ...
    // cell z>=3 in active-cell coords = z>=1/3 in planet frame
    let ray_origin = [1.5, 0.5, 2.5];  // planet z=2.5, well outside [0, 1/3)
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
| Phase 1.2 dims `20×10×2` | Active edge mid-cell — hardest case | Use `27×9×3` for Phase 1; defer 20×10×2 |
| Phase 2.1 wrap gate `depth==0u` | Coincidentally correct under Issue 1 | Document, don't generalize |
| Phase 4.2 bend units `ct_start` | Camera-frame, breaks at orbit | Multiply by `planet_frame_scale` |
| Phase 1.1 GpuNodeKind reuse | Fragile bitcast, wrong field names | Rename to `geom_a/b/c` first |
| Phase 5.1 lighting | Implicit but undocumented | Add forward-compat shader comment |
| Phase 1.3 acceptance | Only tests Y-banned | Add Z-banned test |
| Sequencing | Plan jumps straight to 1.1 | Insert Phase 0.5 cleanup |

All other sections of the plan stand as written.
