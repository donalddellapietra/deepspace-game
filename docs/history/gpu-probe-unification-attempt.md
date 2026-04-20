# GPU cursor-probe unification — attempted, reverted

**Branch:** `occupancy-stack-slim-perf-sphere-debug`
**Session end state:** reset to `9f7ef04` (pre-attempt). Everything below
is learning captured before the revert so the next attempt doesn't
repeat the same dead ends.

## Starting state

- Pre-existing GPU cursor-probe infra (`cursor_probe.wgsl`,
  `CursorProbe_Gpu`) dispatched a single ray per frame through the same
  `march()` the fragment shader runs and copied the hit to a MAP_READ
  staging buffer.
- Highlight + break/place were still driven by the CPU side:
  `frame_aware_raycast()` in `src/app/edit_actions/mod.rs` called
  `cpu_raycast_in_frame` / `cpu_raycast_in_sphere_frame` from
  `src/world/raycast/`, truncated the hit to `edit_depth`, and gated
  it through `interaction_range_in_frame`.
- Observable bug that motivated the session:
  `sphere_highlight_glow_is_anchor_sized_not_collapsed_block`
  returning 0 yellow-ish pixels. Highlight glow was missing entirely.
- User framing (load-bearing): "the highlight is WRONG — it is at
  various random-sized layer blocks instead of the layer you are at"
  → later clarified: "it highlights a COLLAPSED cell in its
  entirety. some cells if they are full collapse to a single cell."

## Diagnosis of the 0-glow failure

Dumped subprocess stderr to `tmp/<dir>/stderr.log` from the test so
diagnostic `eprintln!` lines could be read:

```
frame_raycast_hit path_len=6 face=2 t=0.50752956 source=gpu
  slots=[13, 16, 13, 22, 7, 10]
```

GPU probe was hitting correctly. But the CPU path feeding
`set_highlight_path` was logging:

```
interaction_radius_reject t=0.5160 max_t=0.4400 cells=12
  anchor_depth=8 frame_depth=1
frame_raycast frame=0 ... hit=false
```

i.e. CPU raycast *did* hit the sphere at the same point as the GPU
probe, but the `interaction_radius` gate (`interaction_radius_cells
× anchor_cell_size_in_frame`) rejected it because the active render
frame was the shallow body frame (`[13, 16]`, `face_depth=0`) where
`max_t` collapsed to 0.44 < 0.508. So:

- Highlight ran in the body frame, CPU raycast got clamped → no hit →
  empty slots → shader had no anchor path to match.
- The GPU probe was seeing the sphere fine.

**Mixed-system conclusion.** Two sources of truth (CPU raycast, GPU
walker) were drifting on the exact scenario the test was probing.
User said: "shift it all to the gpu if so, or decide on CPU for
all" → "yes" (to GPU).

## The migration attempt

Committed as `BROKEN STATE` (`e11fed3`) → later
`probe: library-pad past walker cap + sync dispatch for direction
change` (`758909b`). Both were `git reset --hard`ed back to
`9f7ef04` at the end of the session.

### What got built

1. **`src/app/edit_actions/probe_hit.rs`** (new). `App::probe_hit()`
   dispatched a fresh sync compute (see #3) and converted the probe
   result into a `HitInfo`:
   - `walk_library_by_slots_padded(slots, target_depth, pad_slot)`
     walked the library by slot, then, if the walker terminated
     shallower than `edit_depth`, padded by descending a ray-aligned
     slot.
   - `pad_slot_for_face(face)` mapped the probe's encoded hit-normal
     face to the child slot the ray would enter next (`face == 3` →
     `slot_index(1, 0, 1)` for a -Y-travelling ray, etc.).
   - `derive_place_path()` for hits whose immediate parent was a
     `CubedSphereFace` — stepped `-r` (axis 2) via
     `step_neighbor_cartesian` to synthesise a placement target.

2. **Renderer** (`src/renderer/mod.rs`)
   - Added `dispatch_and_read_cursor_probe_sync()` that encodes a
     standalone compute pass + copy, submits, and reads the staging
     buffer synchronously. This was needed because `harness_probe_down`
     rotates the camera pitch *this frame* and the per-frame probe
     reads the *previous* frame's result.
   - `src/renderer/cursor_probe.rs` pipeline overrides
     `LOD_PIXEL_THRESHOLD = 0.0` and `ENABLE_STATS = 0.0` so the
     probe's walker deep-walks to the library leaf while the
     fragment shader keeps Nyquist LOD for perf.

3. **Edit paths rewritten**
   - `highlight.rs`: reads `read_cursor_probe()` directly, calls
     `set_highlight_path` with probe slots. No more
     `frame_aware_raycast`.
   - `break_place.rs`: `do_break` / `do_place` now call
     `self.probe_hit()`. Same downstream
     `edit::break_block` / `edit::place_block`.
   - `harness_emit.rs`: `harness_probe_down` / `_cursor` route through
     `probe_hit` too. `harness_probe_down` saves pitch, sets pitch to
     `-π/2`, probes, restores pitch. `fly_to_surface` likewise.

4. **Deletions** (all restored by revert):
   - `src/world/raycast/{mod, cartesian, sphere}.rs` (≈ 1089 LOC)
   - `src/world/aabb.rs` (≈ 241 LOC)
   - `frame_aware_raycast`, `interaction_range_in_frame`,
     `truncate_hit_to_edit_depth`, `ray_dir_in_frame` in
     `edit_actions/mod.rs`
   - `--interaction-radius` CLI flag, test_runner config field
   - Callers in `harness_emit.rs`, `worldgen.rs`, `pack.rs`, `edit.rs`
     tests — all rewritten to synthesise `HitInfo` directly.

5. **`HitInfo`** moved out of `src/world/raycast::` into
   `src/world/edit.rs` (its only remaining consumer).

### What the first attempt got wrong

Walker / CPU divergence surfaced as cascading failures:

1. **Edit depth mismatch.** `descent_sees_sky_and_breaks_at_every_layer`
   asserted `edit.anchor.len() == anchor_depth` at every layer. First
   failure was at d6: GPU walker returned 8 slots, `edit_depth` was 6,
   naive trim dropped to 6. Easy fix — trim `probe.slots` to
   `edit_depth` before walking the library.

2. **Walker terminates shallower than edit_depth.** Even after
   `LOD_PIXEL_THRESHOLD=0`, the walker caps at `MAX_STACK_DEPTH=20`
   and at packed-tag=1 uniform-collapse terminals. At d7+ the probe
   returned 6 slots for `edit_depth=7`. Fix: library-pad past the
   walker's terminal using `CENTER_SLOT`.

3. **CENTER_SLOT can land in a pre-carved air pocket.** At deep
   layers, `carve_air_pocket` leaves the centre child of many
   ancestors `Empty`. Padding with slot 13 ran into that Empty and
   stopped. Fix: choose `pad_slot` by probe face (ray direction), fall
   back to the first Node sibling if that slot is also Empty/Block.

4. **The divergence that killed the descent test.** Even with all
   three fixes, the full descent test (37 break/zoom/teleport cycles)
   diverged at d7→d8 because the GPU probe's DDA traversal lands on a
   *different cell* than the old CPU DDA for the same physical ray
   once the render-frame transitions from face_depth=0 (spherical
   walker) to face_depth≥1 (Cartesian walker).

   The sky-screenshot check (`look_up`, `screenshot`, assert
   top-half sky-dominance ≥ 0.05) fails at **d8** on the migration,
   fails at **d25** on pre-migration `9f7ef04`. So:

   - Migration regressed the threshold from d25 → d8.
   - The user's empirical "doesn't zoom below layer 23" (layer 23 ≈
     `anchor_depth 18`, with failure around d25 = layer 16) was a
     **pre-existing bug** that the migration made strictly worse.

## Lib test state during the attempt

Kept green throughout (97/97):

- Tests in `src/world/raycast/` deleted.
- Tests in `src/world/edit.rs`, `src/world/gpu/pack.rs` that used
  `cpu_raycast` rewritten to synthesise `HitInfo` by walking
  `pos.anchor` from root (with a `step_neighbor_cartesian(1, -1)` to
  target the ground cell below the carved air pocket, not the
  pocket itself).
- `src/world/worldgen.rs` test rewritten to walk library slot-wise
  instead of `is_solid_at`.

## The architectural realisation

Mid-session it became clear the real problem isn't "CPU vs GPU" but
the **face_depth=0 ↔ face_depth≥1 dispatch**:

- `march_face_root` (curved sphere walker) vs `march_cartesian` have
  different cell_min/cell_size conventions, different slot-semantics
  (u,v,r vs x,y,z for the **geometric** interpretation), different
  walker termination.
- Highlight sub-cell extension in `main.wgsl` assumes Cartesian slot
  arithmetic, which is wrong when the walker's frame is a face
  subtree rendered via `march_face_root` (which never populates
  `cell_min`/`cell_size`).
- Editing has matching duality: `place_child`'s face → xyz-delta
  fallback gives garbage in face subtrees.

The user's subsequent question: **"why do we even have cartesian? why
can't the sphere be fully u v r?"**

Key insight: slot arithmetic inside a `CubedSphereFace` subtree is
*already identical* to Cartesian — (u, v, r) maps to (x, y, z) of a
local `[0, 3]³` box. The duality is purely a **rendering** choice.
`march_face_root` exists only to draw a smooth curved silhouette at
zoomed-out views; everything else treats slots as cube cells with
slot_index = z*9 + y*3 + x.

And: **why u/v/r at all?** It's a 6-chart atlas giving every face
its own locally-Cartesian frame so surface content is locally
cube-aligned everywhere on the planet. Without it, a city stored in
pure world-Cartesian is only axis-aligned at the poles; at the
equator it's rotated 90° relative to gravity. A rotated camera fixes
the view but not the voxel grid — the grid still needs a per-face
frame to put an integer-cell building with "up = radial" anywhere on
the sphere. So the atlas has to stay; only the smooth-curved
*rendering* at `face_depth == 0` is what could be collapsed.

## Why we reverted

The migration worked for shallow scenarios
(`layer_37_break_below_is_registered_three_ways`,
`layers_37_to_36_descend_and_break` both green) but regressed the
deep-descent sky visibility from d25 → d8 because the GPU walker's
DDA drifts from CPU DDA across the `face_depth` transition. Rather
than keep patching the divergence with padding heuristics (pad by
face, fall back to first Node sibling, step back along `-r` for
place paths…), user called it: revert, address the underlying
dispatch split first, then reconsider CPU/GPU unification on top of a
single walker.

## Things worth keeping when we try again

1. **GPU probe has a one-shot sync dispatch API** that re-writes the
   camera uniform before the compute pass. Without that,
   `harness_probe_down` would read stale data. Design preserved.

2. **Probe should override `LOD_PIXEL_THRESHOLD` to 0** — the
   fragment-shader pixel-cutoff LOD is a *rendering* optimisation,
   not a *hit-resolution* one. Edit targeting needs the walker to
   descend to the library leaf.

3. **Walker termination ≠ edit depth.** Even with LOD off, the
   walker caps at `MAX_STACK_DEPTH` and at packed-tag=1 uniform
   collapse. CPU-side library padding is unavoidable if we want
   edits at arbitrary user-layer depth.

4. **`CENTER_SLOT` padding is wrong** at scenes with carved air
   pockets. Pad by face-direction, fall back to first Node sibling.
   Better still: emit `place_path` from the GPU walker itself (track
   last-empty cell during DDA) so the CPU never has to heuristic-pad.

5. **The sky-dominance test has been broken since at least
   `9f7ef04`** (pre-existing bug, fails at d25). It's the right test
   to drive the unification fix, but its current failure isn't a
   migration regression to worry about independently.

## Things to address before attempting again

1. **Unify `march_face_root` into the generic walker** — or keep it
   strictly as a cosmetic SDF pass at `face_depth == 0` that doesn't
   participate in hit resolution at all. One walker, one slot
   convention, one cell_min/cell_size contract.

2. **Shader-side `place_path` emission** — track the last-empty cell
   along the DDA in `HitResult`, pack it like `hit_path`. Removes
   all the CPU step-back heuristics for face-subtree placement.

3. **Fix the descent test's "below layer 23" bug** at the root (likely
   in the walker's frame transition, not in the CPU raycast) so we
   have a single clean reference point to migrate against.
