# Render Frame Architecture: Status & Open Issues

*Branch: `sphere-mercator-1-2-2-1`. As of 2026-05-03.*

## Architecture overview

The world is a sparse 27-ary octree (`NodeLibrary`). Tree depths reach
30+. The fragment shader does ray-marching via stack-based DDA in
`assets/shaders/march.wgsl`'s `march_cartesian` function.

Every frame the CPU picks a **render frame**: a specific tree node
that becomes the GPU shader's starting point for ray-marching. The
shader's per-ray DDA descends from this frame, with a hard
`MAX_STACK_DEPTH = 8` cap on per-invocation descent. When a ray exits
the frame's `[0, 3)³` local box, a **ribbon-pop** transitions it to
an ancestor frame (a separate march invocation, also bounded by 8
levels of descent).

## The constraint that drives every issue

`MAX_STACK_DEPTH = 8` exists because the shader's DDA accumulates
`cur_node_origin` (vec3) and `cur_side_dist` (vec3) in absolute
`[0, 3)³` frame coords, with `cur_cell_size` dividing by 3 each push.
At depth 8, `cur_cell_size ≈ 4.6e-4`; at depth 16 it hits f32 epsilon
(~1.2e-7). Adding tiny values to bounded ones loses precision —
visible jitter at deep depths.

**Consequence**: the shader can only see content within
`MAX_STACK_DEPTH = 8` tree levels of the chosen render frame. Content
deeper than `frame_depth + 8` is invisible. After a ribbon-pop to a
parent, the new invocation has its own 8-level budget — but only
into ITS subtree, not back into deeper sibling branches.

## Current architecture (commit `c5bd692`)

`compute_render_frame` walks the camera's anchor path (slot indices
derived from `WorldPos.offset` arithmetic) up to depth
`anchor_depth - RENDER_FRAME_K` (K = 3). It stops at the first
non-Node child along that path.

`visual_depth` (passed to shader as `depth_limit`) =
`edit_depth - render_path.depth()`, capped by various pixel-size
heuristics.

## The bugs this architecture causes

### Bug 1: Render path stops shallow when anchor enters Empty

When the camera is in an empty area (e.g. just dug a tunnel), the
anchor's slot path goes through Empty cells in the tree.
`compute_render_frame` stops at the first non-Node it finds — could
be many levels above the desired depth. With shallow render path,
the shader can't reach the carved structure.

### Bug 2: Cell-boundary instability

When the camera is geometrically near a cell boundary, f32 offset
arithmetic decides which slot the anchor "rounds to." With small
camera movement, the anchor flips between sibling slots. If one
slot has carved Node content and the other is uniform-stone or
Empty, the rendering changes drastically for what is essentially
the same world position.

**Reproduction**: stand right next to a wall containing carved
detail. Move 0.01 units; render shifts from "uniform stone wall" to
"detailed carving" because the anchor's slot at some depth flipped.

### Bug 3: Block break invisible until you cross a boundary

You break a block in a wall. The CPU raycast finds it, the tree is
mutated, the GPU pack is updated. But your camera anchor still goes
through the empty room cells (you're standing in front of the wall,
not inside it), so `compute_render_frame` stops at the deepest Node
on the room's branch — which doesn't include the new structure (in
a sibling branch).

When you move slightly and your anchor's slot at the relevant depth
flips to one in the wall's branch, the render path suddenly goes
deep into the wall and the break is visible.

### Bug 4: Sibling-branch deep content invisible

Camera in slot A at some depth N (empty room). Carved tunnel in
sibling slot B at depth N (depths N+1..N+M of carving). Looking at
the tunnel from outside requires the shader to render content in B's
subtree from a frame in A's subtree. Best case path: ribbon-pop up to
the common ancestor (depth N-1), then descend into slot B. But each
march invocation can only descend 8 levels. If carving is at depth
N+M with M > 8, the deep parts are unreachable from the popped
frame regardless of CPU frame choice.

This is the **fundamental architectural limit** — irrespective of
where the CPU places the frame, MAX_STACK_DEPTH=8 caps total
visible depth in any sibling branch.

## What we tried, and why each failed

### Attempt 1: Position-aware compute_render_frame

When anchor's slot leads to non-Node, scan boundary-neighbor slots
for a Node and follow that. *Failed*: introduced "floating block"
artifacts because camera coords were computed for the anchor frame
but rendered against the chosen-neighbor frame; the camera was
geometrically OUTSIDE the rendered subtree.

### Attempt 2: snap-on-edit

After each break/place, set camera anchor = dig path. *Failed*:
teleported the camera to the broken block's cell — significant
visual jump at every edit.

### Attempt 3: tree-aware zoom-in

When `zoom_in` is on a cell boundary and one neighbor has Node
content while the other is Empty/Block, prefer the neighbor.
*Partial success*: helps at zoom-in time but doesn't address
boundary issues from movement (`add_local`) or steady-state drift.

### Attempt 4: uniform-empty Nodes as tag=2

Make uniform-empty subtrees traversible by the GPU ribbon walker
(previously they were collapsed to absent slab entries). *Worked*
for ribbon traversal but caused massive perf regression because
the shader's DDA descended into every empty subtree before realizing
it had no content. Mitigated with an `aabb_bits == 0` skip in the
shader, but still added baseline cost.

### Attempt 5: Force-path-traversible / pack repair

When ribbon can't follow the intended path, evict path NodeIds from
the dedup cache and re-emit. *Worked* mechanically but content-
addressed dedup means re-emit re-applies uniform-flatten — same
problem on the next frame.

### Attempt 6: MAX_STACK_DEPTH bump (8 → 16)

Just raise the cap. *Failed*: hit the f32 precision wall at depth
12+; visible jitter near walls.

### Attempt 7: Per-push local-frame DDA rewrite (Opus subagent)

Rewrite march_cartesian so each push transforms the ray to the
child's local `[0,3)³` frame, keeping `cur_cell_size = 1.0` always.
This avoids the precision wall and allows MAX_STACK_DEPTH > 8.

*Failed*:
- Per-thread state grew significantly (`s_origin`, `s_dir`,
  `s_side_dist` arrays at MAX_STACK_DEPTH=16 = ~600 bytes new).
  Likely register spilling on Apple Silicon → significant perf
  regression.
- Block-break artifacts at deep depths (block geometry rendering
  incorrectly).
- Even after replacing absolute-coord `cell_min`/`cell_size` outputs
  with local-frame `local_in_cell`, the perf hit and remaining
  artifacts made it unviable.

### Attempt 8: Multi-ray content sampling for frame selection

CPU casts a 16×16 grid of rays through the FOV per frame. Compute
the deepest common-ancestor tree path of all hits — that becomes
the render frame.

*Failed*: caused new artifacts (solid objects rendering as
fragmented or hollow when approaching). Likely cause: the deep
common-ancestor frame has `visual_depth = edit_depth - frame_depth`
becoming very small (1-3), so ribbon-popped sibling content
LOD-terminates at almost no depth → fragmented appearance.

A follow-up patch widened `visual_depth` to use the full shader
budget, which didn't fully fix it and introduced new LOD issues.
Both reverted.

## Where we are now

Back at commit `c5bd692` — the camera-anchor-based render frame,
with all the bugs listed above, but at least it's stable and
performant.

## What we know works (from history)

- The shader's per-ray `LOD_PIXEL_THRESHOLD` correctly handles
  per-cell nyquist termination (descent stops when cells are
  sub-pixel).
- The shader's ribbon-pop correctly transitions ray frames.
- The `march_in_tangent_cube` function has its own
  `TANGENT_STACK_DEPTH = 24` and works at deeper local depths
  because it operates entirely in a `[0, 3)³` local frame from the
  TB node.

## Possible future paths

These are not endorsed; just documented as possible directions.

1. **Lean local-frame DDA**: restructure march_cartesian like
   attempt 7, but recompute `ray_dir` / `side_dist` on pop instead
   of saving on stack. Minimal additional state. Requires careful
   numeric analysis to avoid the perf hit.

2. **Variable per-block depth budget**: on the GPU side, when the
   DDA encounters a child Node, dynamically set a depth budget for
   that sub-march based on the cell's projected pixel size. Already
   partly done by `LOD_PIXEL_THRESHOLD`; could extend to give each
   sibling its own budget.

3. **Multi-pass rendering**: render multiple frames per output
   pixel and composite. E.g., render the anchor frame and one
   "deep content" frame separately, depth-test composite. Major
   shader / pipeline changes.

4. **Multi-ray frame selection (better tuning)**: re-attempt
   attempt 8 with a different aggregation than common-ancestor
   (maybe weighted-by-distance, or pick frame depth such that the
   N closest hits are within MAX_STACK).

5. **CPU pre-pass at coarse resolution**: like attempt 8 but
   instead of fixing one frame, output a per-tile frame depth
   (8×8 tiles = 256 frames), let each tile use its own. Requires
   shader changes to accept per-tile frame uniforms.

6. **Accept the architectural limit**: leave bugs 1-3 unfixed
   except via UX workarounds (visual indicators, snap-to-content
   buttons, etc.). Bug 4 (sibling-branch deep content) is
   fundamentally unfixable without one of the above.

## Files of interest

- `src/app/frame.rs` — `compute_render_frame`, `with_render_margin`
- `src/app/mod.rs` — `App::render_frame`
- `src/app/edit_actions/zoom.rs` — `target_render_frame`,
  `visual_depth`, `frame_projected_pixels`, `camera_fits_frame`
- `src/app/edit_actions/upload.rs` — `upload_tree_lod` (uses the
  computed frame)
- `assets/shaders/march.wgsl` — `march_cartesian`, the outer
  `march()` ribbon-pop loop
- `assets/shaders/bindings.wgsl` — `MAX_STACK_DEPTH = 8` cap
- `src/world/raycast/cartesian.rs` — CPU mirror of the DDA, used
  for click/break/place
