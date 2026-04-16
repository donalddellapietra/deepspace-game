# Zoom

Zoom is tree navigation, not camera manipulation. Scrolling the mouse
wheel moves the camera's anchor up or down the tree. The renderer
always renders at full per-pixel LOD — zoom doesn't change the FOV or
the ray-march budget.

Source of truth:
- `src/world/anchor.rs` — `WorldPos::zoom_in` / `zoom_out` primitives.
- `src/app/event_loop.rs` — scroll handler + `zoom_anchor(step)`.
- `src/app/harness_emit.rs` — script `zoom_in:N` / `zoom_out:N`.

## What the code does

```rust
pub(super) fn zoom_anchor(&mut self, step: i32) {
    let cur = self.anchor_depth() as i32;
    let new_depth = (cur + step).clamp(1, MAX_DEPTH as i32);
    if new_depth == cur { return; }
    if step > 0 { self.camera.position.zoom_in(); }
    else        { self.camera.position.zoom_out(); }
    self.apply_zoom();
}
```

- **Zoom in (step > 0)**: `WorldPos::zoom_in` pushes a slot equal to
  `floor(offset * 3)` onto the anchor path and rescales offset into
  the child cell. Anchor depth increases.
- **Zoom out (step < 0)**: pops the last slot and rescales offset
  back out. Anchor depth decreases. Clamps at depth 1.

Both primitives are O(1) with no tree reads. They do not *move* the
player — they re-express the same position at a different granularity.

`apply_zoom` follows by recomputing the active frame, resetting the
LOD cache key, and repacking the GPU tree.

## What changes

| What | How |
|---|---|
| Anchor depth | ±1 per scroll step. |
| UI `layer` | `tree_depth − anchor_depth + 1`. Shown on the hotbar overlay. |
| Edit granularity | The CPU raycast descends up to `edit_depth()` = current `anchor_depth` (overridable via `--force-edit-depth`). Shallower anchor ⇒ coarser edits. |
| Render frame | `compute_render_frame` picks a new frame root when the camera's ancestor chain changes. |

## What doesn't change

- **Visual detail.** The renderer runs the same DDA per pixel; LOD
  cut-off is a function of cell-on-screen size, not zoom. Zooming out
  just means the same pixel covers more world.
- **FOV or camera orientation.**
- **The tree.** No nodes are allocated or freed on zoom.

## What's explicitly *not* implemented

The code has zoom primitives and the anchor-depth change — that's it.
Aspirational side effects described in older docs (walk speed ∝
cell size, gravity ∝ cell size, AABB scaling, collision layer at
`anchor_depth − 1`) are **not in the current code**. Player motion
today is debug-only: WASD teleports one child cell at the current
anchor depth (`App::step_chunk`); `src/player.rs::update` is a
no-op. See [../design/collision.md](../design/collision.md) for the
intended eventual behavior.

## Controls

| Input | Effect |
|---|---|
| Mouse wheel up | `zoom_anchor(+1)` — anchor deeper. |
| Mouse wheel down | `zoom_anchor(-1)` — anchor shallower. |
| Script `zoom_in:N` / `zoom_out:N` | Repeated `zoom_anchor` from the test harness. |

`E` / `Q` keys are *not* bound to zoom today (older docs claimed they
were).
