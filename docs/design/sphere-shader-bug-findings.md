# Sphere shader — investigation findings for sphere-attempt-2-2-2

Follow-up to `docs/design/sphere-shader-bug-repro.md`. This session ran
the runbook and narrowed the bug substantially. The original claim "the
shader produces no hits (`hit_fraction=0.0000`) — every pixel falls
through to sky" turns out to be **wrong at the current branch state**.
The shader IS dispatching and IS finding hits. What's broken is which
cell it's hitting.

## What we confirmed works

A sequence of distinct-color debug probes, injected into
`sphere_in_sub_frame` and `march_cartesian`'s SphereSub branch, proved:

1. `uniforms.root_kind == ROOT_KIND_SPHERE_SUB` at d6, d10, d18 (all
   depths where the CPU produces `SphereSub`). An unconditional
   magenta return inside the dispatch branch paints the whole frame
   magenta. **Dispatch is correct.**
2. Uniform layout is correct. A probe that reads
   `sub_uvr_slot_at(4u)` (first slot in row 1 of the 16-vec4 array)
   at `sub_meta.y ≥ 5` returned `13u` as expected at both d10 and
   d18 — no std140/f32-alignment mismatch.
3. An unconditional-red return at the top of `sphere_in_sub_frame`
   produces solid red at d6/d10/d18. The function is reached for
   every pixel and its return value propagates correctly to the
   framebuffer.
4. Walker-limit-encoded colors (orange/yellow/cyan/purple) identified
   the GPU's walker budget per depth: walker_limit = 2–3 at d6/d10,
   7+ at d18 — matching `visual_depth = edit_depth - render_path.depth()`.

## What's actually broken

With debug colors injected into each exit path of `sphere_in_sub_frame`:

- At **d6**, the in-loop hit branch fires for most pixels and the
  loop fall-through fires on narrow grid lines. Colouring by `w.size`
  shows the hits are **full-sub-frame uniform hits** (`w.size >= 2.99`,
  mid-prefix `Child::Block` branch in `walk_from_deep_sub_frame_dyn`).
- At **d10**, **d18**, every pixel hits the walker's deep-cell path
  (`w.size < 0.5`), and the returned block is the same across pixels —
  hence the uniform gray appearance.

The uniformity is the tell. Every ray in the fragment pipeline, on its
first DDA iteration, lands at almost the same sub-frame-local point
(roughly `camera.pos` plus a tiny `t_nudge` step). The walker resolves
that position to one cell, and that cell is returned for every pixel.
Per-pixel variation is entirely in the shading math downstream — which
is why the result *looks* like a smooth grey gradient rather than a
solid color.

**The walker is finding solid content at `camera.pos`** even though the
test's teleport has placed the camera inside the dug-out cavity.

## Top open hypothesis

`camera.pos` in sub-frame local coords (produced by
`WorldPos::in_sub_frame`) is evaluating to `(1.5, 1.5, 1.5)` — the
geometric centre of the local `[0, 3)³` box. For a sub-frame at world
depth 7 when the camera's physical location is at world depth ≥ 10,
the walker descends `walker_limit` extra UVR levels at `(1.5,1.5,1.5)`.
It always picks slot 13 (= centre), so it visits the `[1,1,1]` child
at each depth — the chain of central cells, all of which are still
solid at d10 screenshot time (only the *parent* chain through slots
≠ 13 has been broken by the descent).

If that's right, the render frame's *local origin* is not actually
where the camera is. The CPU `SphereSub` frame captures the Jacobian
at a face-normalised corner that matches the camera's deep UVR path,
but the camera's `in_sub_frame` output may not be placing the camera at
the corresponding `[0, 3)³` point — it's pinning it to the frame
centre, which is a different deep cell on the face.

Verification plan for next session:

1. Add CPU eprintlns printing `cam_local` from
   `WorldPos::in_sub_frame(&sub)` alongside the sub-frame's
   `(un_corner, vn_corner, rn_corner, frame_size)` and the camera's
   full `uvr_path`. Confirm whether the local-coord derivation
   re-projects the camera's deep UVR offset correctly.
2. Cross-check against the existing passing
   `local_march_hits_same_cell_as_body_march` test — that test
   constructs a `cam_local` explicitly and hits correctly, so it
   proves the in-loop maths work when the local-coord input is
   right. The dig-down flow is where the `in_sub_frame` projection
   is suspect.
3. If CPU `in_sub_frame` is off, the GPU sees the same (correct)
   wrong value; the fix lives in `WorldPos::in_sub_frame` /
   whoever feeds the camera's uvr offset into it, not in the WGSL.

## Secondary hypothesis (less likely)

First-iteration DDA nudge. With `rd_local` magnitude `~O(3^m)` at the
sub-frame depth (from `J_inv · rd_body`), `t_nudge = t_span * 1e-5`
might leave the first iteration's `pos` still exactly in the camera's
starting cell. A larger nudge, or a "skip walker on first iteration if
we're inside the camera's own cell" rule, would let the ray actually
move before the walker runs.

This is a less satisfying hypothesis because the CPU runs with the
same nudge and produces correct hits. But worth ruling out if
`in_sub_frame` turns out to be fine.

## Observed CPU state at each depth (from instrumented run)

```
d5  cam_sphere=None             kind=Body       render_path=[13]
d6  cam_sphere=Some((PosY, 4))  kind=SphereSub  render_path=[13,16,13,13,22,22]
d7  cam_sphere=Some((PosY, 5))  kind=SphereSub  render_path=[13,16,13,13,13,22,22]
d8  cam_sphere=Some((PosY, 6))  kind=SphereSub  render_path=[13,16,13,13,13,13,22]
d9  cam_sphere=Some((PosY, 7))  kind=SphereSub  render_path=[13,16,13,13,13,13,13]
d10 cam_sphere=Some((PosY, 8))  kind=SphereSub  render_path=[13,16,13,13,13,13,13]
d11 cam_sphere=Some((PosY, 9))  kind=SphereSub  render_path=[13,16,13,13,13,13,13]
d12 cam_sphere=Some((PosY, 10)) kind=SphereSub  render_path=[13,16,13,13,13,13,13]
```

Note the render_path stabilises at length 7 from d9 onward while the
camera's logical uvr_path keeps growing. `m_truncated` is capped by
`target_depth - body_depth - 1` = `visual_depth - 2`, and the test's
`visual_depth` caps at ~7 for the harness dimensions. That's the
expected LOD behaviour. The sub-frame's `(un,vn,rn,frame_size)` does
move slightly per depth (e.g. `rn_corner` 0.514 → 0.502 → 0.498) to
follow the camera's drift deeper, so the frame *is* repositioning; it
just doesn't descend further.

## Runbook correction

Step 6 of `sphere-shader-bug-repro.md` describes `hit_fraction=0.0000`.
After this session's probes the GPU is clearly producing hits — the
earlier report must have been from a stale build or measured before
some fix landed. Section 1 ("Uniform write path") is also disproved by
the slot-read probe.

**Next session should re-prioritise around the `in_sub_frame`
hypothesis** and skip the uniform-alignment / dispatch-fires checks
above — they've been verified clean.
