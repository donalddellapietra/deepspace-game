# Critical evaluation of the gnomonic rewrite proposal

Self-review after the opus agent's round. Concluding the proposal
is NOT the right fix for the user's reported d=10 bug, and should
be shelved here as a future-architecture note rather than pursued
as the immediate response.

## Why the proposal doesn't fix the reported bug

**The d=10 bug is tree-state-dependent, not a sphere-DDA precision issue.**

Hard evidence from `sphere-attempt-2-2-3-2`:
- Before placing a block: uniform r_lo winning on the ground (mode 4
  clean light-blue).
- After placing a block: striped r_lo / v_lo / u_lo winning across
  adjacent pixel rows.
- Mode 6 (walker ratio-u/ratio-v/ratio-depth as RGB) differs between
  pre- and post-place states for **rays that don't pass through
  the placed cell**.
- Reverting the place (break the same cell) restores the pre-place
  render.

This is a walker/pack interaction where placing a block alters the
walker's output for UNRELATED rays. The sphere DDA's trig math is
numerically stable at d=10 — both equal-angle and gnomonic would
give plane normals with ~200–500 ULPs of f32 headroom. Dropping the
`tan` wrap doesn't address why the walker's slot-pick output differs
for identical rays between two tree states.

## Why the proposal's claims don't hold

1. **"Precision fix"**: at d=10, equal-angle and gnomonic have
   comparable numeric stability. The trig isn't the bottleneck here.
2. **"d=30+ support"**: the proposal handwaves frame-local rescale.
   `march_cartesian` achieves this by having each descent step's
   current frame be `[0, 3)³` relative to its parent. The sphere
   body's fixed geometry (cs_center, inner_r, outer_r) doesn't
   rescale on descent — you'd need to transition face-subtree cells
   into flat Cartesian cubes at some depth. The proposal doesn't
   describe this.
3. **"Cosmetic distortion"**: 5.2× area at cube corners is visible
   at wide framings. The dismissal is too quick.
4. **Conflated independent fixes**: mapping change (ea vs gnomonic)
   and depth-extension (frame-local rescale) are orthogonal. Neither
   addresses the confirmed bug.

## What the proposal IS good for

It's a valid long-term architecture note — a future "sphere v2"
effort could pursue gnomonic + frame-local-descent as a unified
rewrite. But the d=10 bug has a more specific root cause that
needs pinpointing first.

## Actual next step

Return to `sphere-attempt-2-2-3-2` and investigate the pre-vs-post
place walker-output divergence. The tree only changes on the edit
path; why does the walker's descent produce different cells for
rays off that path? That's the concrete bug. Once pinpointed, the
fix is likely targeted (one or two edit-path-aware GPU upload
changes) and far smaller than the gnomonic rewrite.

This worktree (`sphere-attempt-2-2-3-4`) is parked. Resume here
only when a sphere-architecture modernization is the actual target.
