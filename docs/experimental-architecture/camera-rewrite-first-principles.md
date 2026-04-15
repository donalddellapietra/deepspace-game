# Camera rewrite: first-principles analysis

Prerequisites: `anchor-refactor-decisions.md` in the same directory.
This doc starts from that model and works out what a correct
implementation on top of the current code **actually requires** —
with special attention to the planet/face geometry that my last
attempt ignored.

## 1. What "no absolute coordinates" actually means

A coordinate system is an **anchor** + an **offset within that
anchor**. The anchor has a frame; the offset lives inside `[0, 1)³`
of that frame. Nothing in the game holds a coordinate whose frame is
"the tree root" unless the camera happens to be right under the root.

The camera already stores its location as `Position { path, depth,
offset }`. That part is fine.

The "absolute coordinates" we need to eliminate are the XYZ numbers
that get **passed around** as `[f32; 3]`:

1. The camera XYZ pushed to the GPU each frame.
2. The camera XYZ fed to LOD culling in `pack_tree_lod`.
3. The camera XYZ fed to `cpu_raycast` for cursor targeting.
4. The camera XYZ fed to gravity math (`camera - planet.center`).
5. The AABB XYZ fed to the highlight uniform.

Each of these numbers lives **in some frame**. Today every single one
is in the tree-root frame `[0, 3)³`. The refactor must change them to
be in a frame **close to the camera's anchor** — only then do the
numbers stay small enough for f32.

## 2. The precision budget

f32 has ~23 bits of mantissa. Inside a cell of extent `E`, the
smallest distinguishable sub-voxel has size `E · 2⁻²³ ≈ 1.2e-7 · E`.

If the render frame is N levels above the camera, the camera's cell
has extent `3⁻ᴺ · E` in the render frame, so one camera-cell voxel is
`3⁻ᴺ · E / 2²³` in f32 at that scale. We need the per-frame rendering
to resolve sub-voxel positions cleanly, which means roughly:

    camera_voxel_size > 4 × f32_ulp_at_render_frame_magnitude

Working that through, **the render-frame-to-camera depth delta must
be ≤ ~14** for f32 to cleanly resolve the camera's voxel. Any deeper
and the shader's floor/ceil operations on the camera position start
snapping.

The current spawn is at `camera.depth = 11`. With `render_root_depth
= 0` (tree root), delta = 11. That's inside the precision budget but
close to the edge, and motion that accumulates per-frame rounding
shows up as jitter — the symptom we're chasing.

## 3. What can actually be a render root

This is the part I got wrong last attempt. The shader doesn't
render arbitrary tree subtrees. It supports:

| Root NodeKind        | Shader handles? | Why |
|----------------------|-----------------|-----|
| `Cartesian`          | Yes             | DDA walks children as 3×3×3 Cartesian grid |
| `CubedSphereBody`    | Yes             | Dispatches into cubed-sphere DDA over the 6 face children |
| `CubedSphereFace`    | **No**          | Needs the containing body's context (center, radii, face orientation); can't stand alone |
| `Empty` / `Block`    | Nothing to walk | — |

So the render root must be a `Cartesian` or `CubedSphereBody` node.
For a camera inside a face subtree (the normal case on a planet), the
deepest usable render root is the **body node itself** — you can't
descend further without losing the body context.

That means: for a player standing on a planet, the render frame is
fixed at the body's depth (depth 1 in the demo scene). As the player
zooms in deeper, the render frame stays at depth 1; the delta grows.
Once delta > ~14, f32 stops being enough.

**This is a real architectural constraint, not just an implementation
bug.** There are two ways to break through it:

1. **Make the shader able to root at a face subtree** — shader learns
   the body context from a small side-channel (face index, body
   radii, body center-in-render-frame). The render root can then
   descend into a face and the delta stays small.

2. **Render in two passes or a dynamic inner frame** — near the
   camera, render a shallow subtree in its own f32-safe frame; past
   that, render the body/ancestor tree at lower precision for the
   far field. Composite by depth.

Option 1 is closer to the anchor-refactor-decisions.md language
("shader walks that subtree in its local frame") and is the
right first pass. Option 2 is more work and a future concern.

## 4. The gravity problem

Gravity reads `planet.center` (root-frame XYZ) and computes
`camera_pos - planet.center`. At deep camera depth the camera XYZ in
root frame also loses precision.

The correct frame for gravity math is the **common ancestor** of the
camera anchor and the planet-body anchor — which, for the demo scene,
is the tree root for ancestors ≤ depth 1 and the body node for
ancestors ≥ depth 1. In practice:

- Store `planet.center` as a `Position` (not `[f32; 3]`).
- Each tick, compute camera→planet vector in the body's frame (or the
  render frame, which coincides with the body for typical cases).
- That vector is `[f32; 3]` in a bounded-magnitude frame, so f32 is
  fine.

We can defer this until the rendering fix lands; gravity at depth 11
is not the visible jitter the user reports. But it's on the same
critical path.

## 5. Correct restructured plan

Drop the "simple dynamic render frame" idea. Do this instead:

### 5a. Phase 1 — render root at body (no shader change)

1. `render_root_depth` returns the depth of the **deepest Cartesian-
   or-body ancestor** of the camera, via a walk from the tree root
   that stops the moment it would descend into a `CubedSphereFace`.
   For the demo scene the result is `1` while the camera is on the
   planet, `0` elsewhere.

2. `render_root_id` walks the same prefix and returns the node at
   that depth.

3. `camera_pos_in_render_frame()` calls
   `Position::pos_in_ancestor_frame(render_root_depth)`. This uses
   the existing f64 reverse-Horner — precision-safe.

4. `pack_tree_lod(library, render_root_id, camera_pos_in_render_frame,
   …)` — rooted at the body (or root), which the shader handles.

5. `cpu_raycast(library, render_root_id, camera_pos_in_render_frame,
   …)` — same root, same frame. Hit paths are relative to render
   root; break/place operate on the returned `(NodeId, slot)` pairs
   which are library-absolute, so this Just Works.

6. Highlight AABB from `hit_aabb` is already in the cpu_raycast's
   `[0, 3)³` frame, which is the render frame. Already correct once
   the shader is also in that frame.

7. Camera uniform: `camera.gpu_camera(fov, render_root_depth)`.
   Shader sees camera inside `[0, 3)³`.

This shrinks the delta from "camera depth" to "camera depth minus
render-root depth". For a camera at depth 11 standing on the demo
planet, delta = 10 — better than 11, not much. **Does not fix jitter
alone**, but sets up the plumbing correctly and is verifiable (game
still renders, edit/place still work).

### 5b. Phase 2 — shader-side face context (the real jitter fix)

1. Extend the shader's tree-walk dispatch so a `CubedSphereFace` root
   is rendered correctly given a small "face context" side-channel:
   face index, body center/radii in render frame, up direction.
2. Extend `render_root_depth` to descend further, into the face
   subtree, by one to two levels. That makes the camera-to-render-
   root delta ≤ `K` (3 or 4), which puts us firmly in the f32-safe
   zone.
3. Verify by zooming into a face subtree and confirming no jitter.

### 5c. Phase 3 — gravity in body frame

1. Replace `SphericalPlanet.center: [f32; 3]` with
   `SphericalPlanet.center_anchor: Position` (or equivalent).
2. In `player::update`, compute
   `planet_center_in_render_frame = planet.center_anchor
       .pos_in_ancestor_frame(render_root_depth)`
   and `camera_pos_in_render_frame = camera.position
       .pos_in_ancestor_frame(render_root_depth)`.
3. Subtract in render-frame f32 — bounded magnitude, clean result.
4. Delete the `[f32; 3]` center field.

### 5d. Phase 4 — purge the remaining absolute-XYZ leaks

1. Delete `Position::world_pos()` and the `(0)` call sites.
2. Debug-overlay `camera_pos` becomes `(depth, offset)` or a pretty
   path string.
3. Spawn: add `Camera::at_position(Position, …)` and return the
   spawn position from worldgen as a `Position` instead of XYZ.
4. Grep for `pos_in_ancestor_frame(0` and `world_pos(` — both should
   be zero in runtime paths when done.

## 6. Why this order

Phase 1 is a pure plumbing change. It doesn't fix the jitter but it
doesn't break anything either (the shader still sees a Cartesian or
body root). That's the "green-at-every-step" discipline the refactor
doc talks about.

Phase 2 is where the jitter actually dies. It requires shader work,
which is riskier, so it has the stable phase-1 plumbing underneath.

Phases 3 and 4 are cleanup — not behavior changes, but the reason the
refactor exists.

## 7. Testable outcomes per phase

- **Phase 1**: game still runs; `render_root_depth()` returns 1
  while on the demo planet, 0 otherwise; no visual change; break/
  place/highlight still work at every tested depth.
- **Phase 2**: zoom-in shows no per-frame pixel jitter at camera
  depths up to 14 or so; shader correctly renders when rooted in a
  face subtree.
- **Phase 3**: gravity is numerically stable at the deepest zoom the
  player can reach; falling toward the surface produces a smooth path.
- **Phase 4**: grep shows the intended zero hits; spawn reads a
  `Position` from worldgen rather than constructing from XYZ.

## 8. What my last attempt got wrong

- Treated "deepest Node along the path" as the render root, which
  dropped into a `CubedSphereFace` subtree that the shader can't
  render. Result: the shader was handed a subtree whose semantics
  it doesn't understand, and the whole render broke.
- Didn't consult the actual shader dispatch rules before deciding
  what "render root" meant.
- Ignored that the body's cubed-sphere handling is a real constraint
  on which nodes can be a render root.

The lesson: a "render root" is not just a Node — it's a Node whose
**NodeKind the shader can start a walk from**. Cartesian or Body.
Not Face.
