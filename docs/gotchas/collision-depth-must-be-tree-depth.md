# Collision probes must walk the full tree depth, not `edit_depth`

## Symptom

On-planet movement is completely frozen. WASD do nothing; gravity
still pulls the player down, but horizontal motion is blocked in
every direction even though the player is clearly standing in open
air.

## Root cause

`world::edit::is_solid_at(library, root, pos, max_depth)` returns
**true** when it runs out of `max_depth` before reaching a terminal
(`Block` or `Empty`). That's fine for a shallow probe against a
tree whose subtrees are uniformly content-addressed — you *know* a
big filler subtree is "solid enough" and can stop early.

It is **not** fine for SDF-sampled planet content. Near-surface
cells contain mixed subtrees — one child is `Empty`, another is a
`Node` leading to more mixed content, etc. If collision probes at
a shallow `max_depth` (e.g. `edit_depth` = 6 at the default zoom
level), every probe near the planet's surface bottoms out at a
mixed `Node` child and returns `true`. Every X/Z sweep reads the
terrain as solid. Nothing moves.

## Fix

Collision **always** passes `world.tree_depth()` (and the matching
`cell_size = 1 / 3^tree_depth`) to `move_and_collide`, independent
of the renderer's zoom / edit depth. Rendering zoom controls how
far the raymarcher descends for display; it has no business
affecting physics resolution.

```rust
let coll_depth = self.tree_depth;
let coll_cs = 1.0 / 3.0f32.powi(coll_depth as i32);
collision::move_and_collide(..., coll_cs, coll_depth);
```

## Why the unit tests passed

The collision tests directly passed `world.tree_depth()` when
calling `move_and_collide`, so they never hit the bug. The game
code, by contrast, was passing `self.edit_depth()` on the theory
that collision should match the interaction grain. The tests
agreed with themselves but not with the game — classic "greenlit
by tests that don't mirror the call site."

**Lesson:** when wiring a new piece into `main.rs` (or any
integration point), add at least one test that constructs state
the same way the integration does, and exercises the same
parameters. Matching unit inputs to production inputs is a free
check against category-of-arg mistakes like this one.

## Related

- `src/world/edit.rs::is_solid_at` — the line `return true;` at
  max_depth is the hazardous fallback. If the probe ever hits a
  Node at max_depth, collision thinks there's stuff there.
- `src/world/collision.rs::move_and_collide` — takes `max_depth`
  as an explicit parameter; callers must pass the *tree*'s depth,
  not the interaction depth.
