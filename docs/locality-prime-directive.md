# Locality Is The Prime Directive

This branch is only correct if **all precision-critical runtime decisions are
made in the active frame's local coordinates**.

This is not just a math preference. It is the governing rewrite constraint for
the entire branch.

That applies to:

- render camera position
- render ray direction and basis
- CPU cursor raycast
- highlight bounds
- block break/place targeting
- GPU LOD decisions

It explicitly does **not** allow a fallback to "root-style" or "absolute"
reasoning just because a path is deep.

## Rewrite Directives

These are the branch-level directives for all further work:

1. Rewrite the broken structure end to end instead of layering patches onto it.
2. Do not stop at intermediate half-ported states and call them fixes.
3. If a file needs structural correction, rewrite it cleanly from scratch.
4. Before rewriting a file:
   - read the file
   - if it is too long, refactor into smaller files first
   - then write the resulting file(s) from scratch in the correct structure
5. Do not keep root-special or absolute-coordinate shortcuts "for now" if they
   violate locality.
6. CPU and GPU must follow the same frame contract. A local CPU path plus a
   root-metric GPU path is still broken.
7. Cartesian and sphere paths must be symmetric. No separate "temporary"
   Cartesian root behavior and no separate sphere-depth hacks.

## What Counts As Broken

The branch is still broken if any of the following are true:

- deep edits mutate the world but do not render
- deep rendering collapses into representative blocks or a uniform field
- raycast/highlight/edit and rendering disagree about which layer is active
- layer depth changes performance for reasons other than local visible work
- any runtime path still relies on world-root scale or absolute XYZ thinking
  for a local-frame decision

## Rule

If a runtime path uses:

- a local/frame-relative origin
- but a direction, distance, depth budget, or visibility heuristic that still
  assumes root scale

then the system is still broken.

That kind of partial port creates the exact class of bugs seen in this branch:

- edits work but do not render
- deep layers flatten into representative blocks
- interaction and rendering disagree below some threshold layer
- performance changes with absolute layer depth instead of only local visual
  depth

## Correct Model

For both Cartesian and sphere content:

1. The active render frame is local to the camera.
2. The renderer works in that frame's metric.
3. CPU interaction works in that same frame's metric.
4. Rays can pop outward through ancestors for distance visibility.
5. Visible detail is controlled by **local visual depth**, not absolute tree
   depth.

## Anti-Patterns

These are not acceptable final solutions:

- forcing Cartesian rendering to stay rooted at world root
- raising global depth caps from the root to "make edits visible"
- keeping edit/raycast local while leaving GPU ray/basis at root scale
- preserving edited leaves as a special-case hack instead of fixing the frame
  budget
- making one subsystem local while leaving another one half-ported
- "one more level" tweaks used as a substitute for a structural local rewrite
- performance tuning on top of a non-local render path

## File Rewrite Procedure

When a file is structurally wrong, the procedure is:

1. Read the file fully enough to understand the actual responsibilities in it.
2. If the file is carrying too many responsibilities, split it into smaller
   files with clear ownership.
3. Rewrite the resulting file(s) from scratch around the locality model.
4. Only after the full local path exists should validation be treated as
   meaningful.

The purpose of this procedure is to avoid accumulating patch stacks on top of a
broken architecture.

## Practical Consequence

For a deep camera anchor, the renderer must satisfy:

`render_path.depth + local_visual_depth >= nearest_visible_edit_depth`

If that is false, the world can mutate correctly while the renderer still
shows only a coarser representative block. That is a correctness failure, not
just a quality tradeoff.
