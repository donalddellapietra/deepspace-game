# Testing-infra audit: `sphere-attempt-2-2-3-2` -> `sphere-mercator-3`

Snapshot of the gap between the reference worktree (where the cubed-sphere
testing infra was originally built) and the current worktree (cubed-sphere
deleted, redesigning around a wrapped-Cartesian planet).

The current worktree is far closer to feature-parity than expected. Most
sphere-related testing additions in the reference were guarded by sphere-
specific code paths (`CubedSphereBody`, `SphereState`, `pick_face`,
`sphere_in_cell`) that no longer exist here, so they cannot be ported
without re-introducing sphere infrastructure.

Sections below: what's in BOTH (already-ported), ONLY in REF (porting
candidates), ONLY in CURRENT (already-ahead, leave alone).

## 1. `tests/` directory

| File                              | Both | Ref-only | Cur-only |
| --------------------------------- | :--: | :------: | :------: |
| `CONTEXT.md`                      |  Y   |          |          |
| `e2e_layer_descent/harness.rs`    |  Y   |          |          |
| `e2e_layer_descent.rs`            |  Y   |          |          |
| `heightmap.rs`                    |  Y   |          |          |
| `render_entities.rs`              |  Y   |          |          |
| `render_perf.rs`                  |  Y   |          |          |
| `render_visibility.rs`            |  Y   |          |          |
| `e2e_sphere_descent.rs`           |      |    Y     |          |
| `sphere_ribbon_pop_precision.rs`  |      |    Y     |          |
| `sphere_zoom_seamless.rs`         |      |    Y     |          |

All three ref-only files are sphere-specific:

- `e2e_sphere_descent.rs` — exercises `--sphere-world` descent
- `sphere_ribbon_pop_precision.rs` — face-EA-coord plane-normal precision
- `sphere_zoom_seamless.rs` — sub-frame transition across `MIN_SPHERE_SUB_DEPTH`

Files in BOTH are byte-identical (verified via `diff`).

`sphere_zoom_seamless.rs` does contain ONE generic image-analysis primitive
worth lifting: a `planet_fraction()` function (sky/non-sky pixel ratio)
that complements the existing `sky_dominance_top_half()` helper in
`tests/e2e_layer_descent/harness.rs`.

## 2. Render harness (`src/app/test_runner/`, `src/app/harness_emit.rs`)

Files exist in both. CLI flags differ:

| Flag                          | Both | Ref-only | Cur-only | Notes                          |
| ----------------------------- | :--: | :------: | :------: | ------------------------------ |
| `--render-harness`            |  Y   |          |          |                                |
| `--show-window`               |  Y   |          |          |                                |
| `--disable-overlay`           |  Y   |          |          |                                |
| `--disable-highlight`         |  Y   |          |          |                                |
| `--suppress-startup-logs`     |  Y   |          |          |                                |
| `--harness-width/-height`     |  Y   |          |          |                                |
| `--screenshot PATH`           |  Y   |          |          |                                |
| `--exit-after-frames N`       |  Y   |          |          |                                |
| `--timeout-secs N` (def 5.0)  |  Y   |          |          | already at desired default     |
| `--script CMDS`               |  Y   |          |          |                                |
| `--spawn-xyz X Y Z`           |  Y   |          |          |                                |
| `--spawn-depth N`             |  Y   |          |          |                                |
| `--spawn-yaw RAD`             |  Y   |          |          |                                |
| `--spawn-pitch RAD`           |  Y   |          |          |                                |
| `--plain-world`               |  Y   |          |          | (default; flag explicit-only)  |
| fractal-preset flags          |  Y   |          |          |                                |
| `--vox-model`/`--scene` etc.  |  Y   |          |          |                                |
| `--shader-stats`              |  Y   |          |          |                                |
| `--lod-pixels`                |  Y   |          |          |                                |
| `--interaction-radius`        |  Y   |          |          |                                |
| `--perf-trace`                |  Y   |          |          |                                |
| `--min-fps`/cadence/etc       |  Y   |          |          |                                |
| `--spawn-elevation-cells N`   |      |    Y     |          | sphere-coupled impl, see below |
| `--sphere-debug-mode N`       |      |    Y     |          | sphere-only                    |
| `--sphere-world`              |      |    Y     |          | sphere-only                    |

ScriptCmd grammar differs:

| Command                          | Both | Ref-only | Notes                                     |
| -------------------------------- | :--: | :------: | ----------------------------------------- |
| `wait:N`, `break`, `place`       |  Y   |          |                                           |
| `screenshot:PATH`                |  Y   |          |                                           |
| `pitch:R`, `yaw:R`               |  Y   |          |                                           |
| `zoom_in:N`, `zoom_out:N`        |  Y   |          |                                           |
| `probe_down`                     |  Y   |          |                                           |
| `emit:LABEL`                     |  Y   |          |                                           |
| `fly_to_surface`                 |  Y   |          |                                           |
| `teleport_above_last_edit`       |  Y   |          |                                           |
| `step:axis:delta`                |  Y   |          |                                           |
| `probe_at:pitch:yaw`             |      |    Y     | references deleted Sphere face/cell types |
| `probe_gpu:x:y`                  |      |    Y     | sphere_in_cell GPU debug                  |
| `fly_to_surface_elevation:N`     |      |    Y     | sphere-coupled impl, see below            |
| `teleport_into_last_edit`        |      |    Y     | uses `path_crosses_sphere_body`           |
| `dump_position`                  |      |    Y     | prints sphere-shell distances             |

Generic harness emit (`HARNESS_MARK`, `HARNESS_EDIT`, `HARNESS_PROBE`)
machinery is byte-identical between both worktrees.

## 3. `scripts/` repro patterns

| Script                              | Both | Ref-only | Cur-only |
| ----------------------------------- | :--: | :------: | :------: |
| `repro-jerusalem-lag.sh`            |  Y   |          |          |
| `repro-sphere-d10-bug.sh`           |      |    Y     |          |
| `repro-sphere-d10-elevation.sh`     |      |    Y     |          |
| `replicate_*` perf scripts          |  Y   |          |          |
| `compare_stack_depth.sh`            |  Y   |          |          |
| `perf-breakdown.sh`                 |  Y   |          |          |
| `perf-entity-raster.sh`             |  Y   |          |          |
| `replicate_entity_cutoff.sh`        |      |          |    Y     |
| `regen-vox-entities.sh`             |      |          |    Y     |

The two ref-only repro scripts are sphere-specific. The PATTERN they use
(spawn camera at fixed coords + depth, run harness, save screenshot to
`tmp/`, optionally diff against reference) is already exemplified in the
existing `repro-jerusalem-lag.sh`.

`compare-place-induced.sh` is in ref-only; it scripts back-to-back
break/place comparisons against the sphere-d10 bug — sphere-specific.

## 4. `src/bin/` test binaries

| File                  | Both | Ref-only | Cur-only |
| --------------------- | :--: | :------: | :------: |
| `compare_pngs.rs`     |  Y   |          |          |
| `df_analysis.rs`      |  Y*  |          |          |
| `perf_opt_analysis.rs`|  Y   |          |          |
| `winit_probe.rs`      |  Y   |          |          |

`*` `df_analysis.rs` differs by one comment line that mentions sphere
body/face tags — current's wording correctly drops the stale references.
No functional drift.

`compare_pngs` is the byte-equal pixel-diff regression tool; usable
unchanged for wrapped-planet visual regression.

## 5. Image-analysis primitives

In CURRENT (`tests/e2e_layer_descent/harness.rs`):

- `tmp_dir(name) -> PathBuf` — gitignored per-scenario artifact dir
- `sky_dominance_top_half(path) -> f32` — fraction of top-half pixels
  with `b > r && b > g`. Generic sky/planet-silhouette discriminator.

In REF only (`tests/sphere_zoom_seamless.rs`):

- `planet_fraction(path) -> f32` — over the entire image, fraction of
  pixels that are NOT sky-blue. The mirror of `sky_dominance_top_half`,
  except (a) computed over the whole frame, not just top half, and (b)
  returns "ground fraction" rather than "sky fraction". Useful for
  silhouette tests where the camera looks down at the planet.

NO file in either worktree contains:

- silhouette curvature analysis
- pixel-row-count (e.g. "row N has K solid pixels")
- generic color histograms
- region masks beyond sky/non-sky

So the agent-facing memory note about silhouette curvature analysis was
aspirational; that primitive was never written for the cubed-sphere work.
Per scope, do not invent new image-analysis primitives — only port what
exists.

## 6. `harness_emit.rs` / structured test reporting

Both worktrees emit the same machine-parseable lines:

```
HARNESS_MARK  label=<str> ui_layer=<u32> anchor_depth=<u32> frame=<u64>
HARNESS_EDIT  action=<broke|placed> anchor=[..] changed=<bool> ui_layer anchor_depth
HARNESS_PROBE direction=<str> hit=<bool> anchor=[..] ui_layer anchor_depth
```

The reference also emits sphere-specific debug eprintlns from inside the
deleted ScriptCmd handlers (`probe_at`, `probe_gpu`, `dump_position`,
`teleport_into_last_edit`'s sphere branch). These reference deleted
Sphere/Face/SphereState types and cannot be ported without the sphere
modules.

## What to port

After audit, the only generic gap is:

1. **`planet_fraction()` image-analysis helper** — port into existing
   `tests/e2e_layer_descent/harness.rs` next to `sky_dominance_top_half`.

Everything else either exists already or is sphere-coupled.

## What NOT to port (and why)

| Item                              | Reason                                               |
| --------------------------------- | ---------------------------------------------------- |
| `--spawn-elevation-cells` flag    | impl uses `WorldPos::new_with_sphere_resolved`,      |
|                                   | `path_crosses_sphere_body`; would need de-sphering   |
|                                   | rewrite, and `--spawn-xyz`+`--spawn-depth` already   |
|                                   | covers the common case                               |
| `--sphere-debug-mode` flag        | sphere-only                                          |
| `--sphere-world` flag             | sphere preset deleted                                |
| `probe_at:pitch:yaw` script cmd   | references deleted Sphere face/cell on hit struct    |
| `probe_gpu:x:y` script cmd        | sphere `sphere_in_cell` debug paint                  |
| `fly_to_surface_elevation:N`      | impl is sphere-aware                                 |
| `teleport_into_last_edit`         | impl uses `path_crosses_sphere_body`                 |
| `dump_position`                   | prints sphere-shell distances                        |
| `e2e_sphere_descent.rs`           | sphere preset deleted                                |
| `sphere_zoom_seamless.rs`         | sphere preset deleted                                |
| `sphere_ribbon_pop_precision.rs`  | face-EA-plane precision; not relevant to wrapped     |
| `repro-sphere-d10-bug.sh`         | sphere preset deleted                                |
| `repro-sphere-d10-elevation.sh`   | sphere preset deleted                                |
| `compare-place-induced.sh`        | wraps `repro-sphere-d10-bug.sh`                      |

## Blocked by API drift (none)

No items in scope are blocked by API drift; everything is either already
ported or sphere-coupled (out of scope).
