# Sphere shader — visible rendering failure after two-step ribbon-pop

Commit `0c1fae9` (feat(sphere): two-step ribbon-pop, CPU complete, shader WIP) landed a CPU-correct deep-UVR sphere DDA plus uniform plumbing + WGSL scaffolding for the GPU mirror. The CPU passes all sphere tests. The shader does NOT render visible geometry during the dig-down descent — every pixel falls through to the background gradient.

This doc is a runbook for the next session to track down the GPU-side bug.

## TL;DR

- Branch: `sphere-attempt-2-2`, worktree `.claude/worktrees/sphere-attempt-2-2`
- All commands run from that worktree directory (verify with `pwd`).
- `cargo test --test e2e_sphere_descent` passes (CPU correct).
- `tmp/sphere_descent/d{6..25}.png` show a smeared grey diamond instead of the dug-pit walls that the CPU raycast can see.
- `render_harness_shader … hit_fraction=0.0000` — the shader's `sphere_in_sub_frame` isn't producing hits.

## Files of interest

- `assets/shaders/sphere.wgsl` — the shader mirror of CPU `cs_raycast_local`. `sphere_in_sub_frame()` is the entry point (line ~850). New helpers: `face_frame_jacobian_shader`, `mat3_inv_shader`, `mat3_mul_vec_shader`, `slot_to_coords`, `coords_to_slot`, `walk_from_deep_sub_frame_dyn`.
- `assets/shaders/bindings.wgsl` — `Uniforms` struct. `sub_uvr_slots: array<vec4<u32>, 16>`, `sub_meta.y` = prefix_len, `sub_meta.z` = face_root_depth.
- `assets/shaders/march.wgsl` — dispatches `sphere_in_sub_frame` at line ~843 when `ribbon_level == 0u && uniforms.root_kind == ROOT_KIND_SPHERE_SUB`.
- `src/app/edit_actions/upload.rs` — populates sphere uniforms from `active_frame.kind == SphereSub(sub)` at line ~320. Calls `renderer.set_root_kind_sphere_sub(...)` with the sub-frame's data.
- `src/renderer/mod.rs` — `set_root_kind_sphere_sub` sets `self.root_kind = ROOT_KIND_SPHERE_SUB` and writes the sphere-side uniform fields.
- `src/renderer/buffers.rs` + `src/renderer/init.rs` — GPU uniform upload path.
- `src/world/raycast/sphere_sub.rs` — the CPU reference (`cs_raycast_local`). Whatever WGSL is missing, this file shows what the correct algorithm is.
- `src/app/frame.rs` — `SphereSubFrame` struct + `with_neighbor_stepped`.

## 1. Build

```sh
cd /Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/sphere-attempt-2-2
cargo build --bin deepspace-game
```

Expected: `Finished dev profile … generated 7 warnings`. No errors.

## 2. Confirm CPU is correct

```sh
cargo test --lib -p deepspace-game sphere_sub 2>&1 | tail -10
cargo test --test e2e_sphere_descent 2>&1 | tail -10
```

Expected: all green, including `sphere_dig_down_descent`, `sphere_probe_anchor_equals_break_anchor`, `sphere_break_path_length_equals_anchor_depth`, `cs_raycast_local_neighbor_transition`.

## 3. Reproduce the visible failure

```sh
rm -f tmp/sphere_descent/*.png
cargo test --test e2e_sphere_descent sphere_dig_down_descent
```

Screenshots land in `tmp/sphere_descent/d{5..25}.png`. Expected:
- `d5.png` — crisp sphere surface with face grid + yellow highlight box (rendered via body march before sphere state is set).
- `d6.png … d25.png` — smeared grey gradient. ALL of them look identical. No visible pit walls, no cells, nothing.

The CPU probe hits correctly at every layer (the test passes), so it's proof of a GPU-only bug: raycasts from any pixel through the same camera+frame return `None` in WGSL.

## 4. Instrumented trace

This spits out the in-pipeline state for the first few descent steps:

```sh
timeout 30 ./target/debug/deepspace-game \
  --render-harness --sphere-world \
  --spawn-depth 5 --spawn-xyz 1.5 1.804938272 1.5 \
  --spawn-pitch -1.5707 --spawn-yaw 0 \
  --interaction-radius 12 \
  --harness-width 480 --harness-height 320 \
  --exit-after-frames 400 --timeout-secs 25 \
  --suppress-startup-logs \
  --script "emit:d5,probe_down,break,wait:10,zoom_in:1,teleport_above_last_edit,wait:10,emit:d6,probe_down,break,wait:10,zoom_in:1,teleport_above_last_edit,wait:10,emit:d7,probe_down,screenshot:tmp/sphere_descent/debug_d7.png,wait:10,emit:end" 2>&1 \
  | grep -E "HARNESS_|CRF sphere|TRF_FINAL|UPLOAD_ENTRY|SUB_RAYCAST|NEIGHBOR_STEP|render_harness_shader" \
  | head -40
```

Look for:
- `CRF sphere …` — confirms `compute_render_frame` built a SphereSub at the deep m_truncated.
- `TRF_FINAL kind=SphereSub(…)` — confirms `target_render_frame` returns SphereSub, not Body.
- `UPLOAD_ENTRY intended_kind=SphereSub(…) intended_render_path=[13, 16, 13, 13, …]` — confirms upload receives the deep sub-frame.
- `SUB_RAYCAST hit_some=true` — confirms CPU raycast finds hits.
- `NEIGHBOR_STEP axis=… sign=…` — confirms CPU neighbor transitions fire.
- `render_harness_shader … hit_fraction=0.0000 avg_steps=0.00` — confirms the SHADER produces no hits and takes no DDA steps.

## 5. Isolate the shader dispatch

Single-frame instrumented run — don't do the full descent, just spawn already inside the sphere:

```sh
timeout 15 ./target/debug/deepspace-game \
  --render-harness --sphere-world \
  --spawn-depth 5 --spawn-xyz 1.5 1.804938272 1.5 \
  --spawn-pitch -1.5707 --spawn-yaw 0 \
  --interaction-radius 12 \
  --harness-width 480 --harness-height 320 \
  --exit-after-frames 100 --timeout-secs 12 \
  --suppress-startup-logs \
  --script "probe_down,break,wait:10,zoom_in:1,teleport_above_last_edit,wait:10,screenshot:tmp/sphere_descent/post_teleport.png,wait:30" 2>&1 \
  | grep -E "root_kind|set_root_kind_sphere_sub|sub_meta|render_harness_shader"
```

Verify `set_root_kind_sphere_sub` is called, `root_kind = 3` is sent to the GPU, and check the shader's per-frame stats.

## 6. What's known to work / break

### Works
- Uniform layout is accepted by naga (the test binary builds; the shader would reject if it were malformed).
- `set_root_kind_sphere_sub` is called on each frame where SphereSub is active (trace via the `upload.rs` path).
- CPU `cs_raycast_local` has the correct algorithm (tests).

### Suspected bug sites (descending probability)

**1. Uniform write path for `sub_uvr_slots`** — most likely. The agent added the array as `array<vec4<u32>, 16>` (one slot per `vec4.x`, 16-byte stride). Check `src/renderer/buffers.rs::write_uniforms` and the corresponding `assets/shaders/bindings.wgsl` layout. A std140/uniform alignment mismatch would manifest as the shader reading zeros.

```sh
grep -n "sub_uvr_slots\|sub_meta" src/renderer/buffers.rs src/renderer/mod.rs assets/shaders/bindings.wgsl
```

**2. `sphere_in_sub_frame`'s early-exit logic** — the new loop in `sphere.wgsl` (line ~915) does `if t_exit <= 0.0 || t_enter >= t_exit { return result; }`. If `rd_local` is wrong (zero, NaN, or wrong sign), the interval check fails and the function returns `hit=false` immediately. That gives `hit_fraction=0` and `avg_steps=0`.

Possible cause: `ray_dir_local` passed into the function is not what the shader expects. The caller (in `march.wgsl` around line 843) passes `ray_dir_local`, which should be `J_inv · rd_body` — if `J_inv` on the GPU differs from the CPU's, the ray direction is wrong.

**3. `face_frame_jacobian_shader`** — WGSL port of the Rust `face_frame_jacobian`. Ported by hand, easy to swap tangent axes or sign conventions. Check the output against the CPU's `sub.j` printed in `TRF_FINAL` logs for a known (un, vn, rn, frame_size).

```sh
grep -n "face_frame_jacobian_shader\|ea_to_cube\|face_tangents" assets/shaders/sphere.wgsl | head -20
```

Print the CPU J for comparison via the TRF_FINAL eprintln (already emitted). Plug the same inputs into the shader-side `face_frame_jacobian_shader` in a disposable compute test.

**4. `walk_from_deep_sub_frame_dyn`** — the WGSL walker that pre-descends `uvr_slots[0..uvr_prefix_len]` from `face_root_idx`. If `face_root_idx` isn't a valid BFS index in the GPU-packed tree, every walk returns empty. Check:

```sh
grep -n "face_root_idx\|root_index\|set_frame_root\|scene_frame_bfs" src/app/edit_actions/upload.rs | head -20
```

Specifically: `upload.rs` sets `scene_frame_bfs = cache.ensure_root(scene_frame_id)` (line ~209) where `scene_frame_id` walked through the clipped `ribbon_intended_path` (body + face_slot). Then `renderer.set_frame_root(scene_frame_bfs)`. This should point the shader at the face root BFS node. If it points somewhere else, the walker starts from the wrong place.

**5. `mat3_inv_shader`** — WGSL `mat3_inv` using f32. The CPU uses f64 internally. At deep m, J's determinant is `O((1/3^m)^3)` = tiny in f32. If the shader's f32 mat3_inv underflows, J_inv is garbage → ray_dir_local is garbage → ray interval fails → hit=false.

## 7. Minimal instrumentation to add

A single-line shader debug — write a known value to `result.color` when a dispatch path is taken. E.g., inside `sphere_in_sub_frame` after computing `(t_enter, t_exit)`:

```wgsl
// DEBUG: visualize which exit path fires
result.hit = true;
result.t = 0.01;
if t_exit <= 0.0 {
    result.color = vec3<f32>(1.0, 0.0, 0.0);  // red: negative t_exit
    return result;
} else if t_enter >= t_exit {
    result.color = vec3<f32>(1.0, 1.0, 0.0);  // yellow: entered past exit
    return result;
}
// ... continue
```

After the descent, `d18.png` should be red or yellow instead of gradient — that tells you which early-exit path fires. Iterate.

Similarly, a dispatch probe in `march.wgsl` at the SphereSub branch:

```wgsl
if ribbon_level == 0u && uniforms.root_kind == ROOT_KIND_SPHERE_SUB {
    r.hit = true;
    r.color = vec3<f32>(0.0, 1.0, 1.0);  // cyan: SphereSub dispatched
    r.t = 0.01;
    return r;
    // … actual dispatch below
}
```

If `d18.png` stays smeared grey (not cyan), the shader dispatch itself isn't firing → `uniforms.root_kind` isn't SPHERE_SUB at GPU time. Check the upload → write path again.

If `d18.png` becomes cyan, dispatch IS firing → bug is inside `sphere_in_sub_frame` itself. Proceed with the t_enter/t_exit colors above.

## 8. Comparing CPU vs GPU for a single ray

`src/world/raycast/sphere_sub.rs::tests::cs_raycast_local_neighbor_transition` is a hand-built synthetic world with a known-good ray. To cross-check the shader, add a shader-side compute pass (or reuse the ray-march with a single pixel) against the same uniform state and compare:

- `ro_local` at function entry (CPU prints via `SUB_RAYCAST cam_local=[…]`).
- `rd_local` after `J_inv · rd_body` (add a CPU eprintln in `cs_raycast_local` before the box interval).
- `(t_enter, t_exit)` (add CPU eprintln after `ray_local_box_interval`).
- Walker result after `walk_from_deep_sub_frame` (the block + local bounds).

Any of these diverging between CPU and GPU narrows the bug to that step.

## 9. Reverting debug logging

Once the shader bug is found + fixed, strip the `eprintln!` traces. Grep for:

```sh
grep -rn "CRF sphere\|TRF_FINAL\|UPLOAD_ENTRY\|SUB_RAYCAST\|NEIGHBOR_STEP\|CAMERA_FITS" src/ | wc -l
```

Should return a high count now. Remove each one or gate behind a `debug_sphere_frames: bool` flag on `App` if you want to keep the tool around.

## 10. Reference invariants

Anything the shader produces should match these CPU-side checks (all passing today):

- `active_frame.kind == SphereSub(_)` whenever `camera.sphere.is_some() && m_truncated >= 1`.
- `sub.face_root_id` = BFS index pointing at the face subtree root (`body + face_slot` in tree coords).
- `sub.render_path.depth() == body_path.depth() + 1 + m_truncated`.
- Sum of `sub.un_corner + (uvr-offset sub * frame_size)` over the camera's uvr_path past m_truncated = the absolute face-normalized un at the camera's deepest cell.
- `J · [3, 0, 0]` (and similar on the other axes) = the body-coord displacement from the current cell's corner to its `+u` neighbor's corner.

If a shader-side probe shows one of these broken, that's the culprit.

## Contact

The commit message of `0c1fae9` has the two-step ribbon-pop overview. `docs/design/sphere-ribbon-pop-two-step.md` has the full precision analysis + algorithmic spec. Both are kept in-tree.
