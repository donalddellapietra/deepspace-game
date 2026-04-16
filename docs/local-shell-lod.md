# Local Shell LOD Design

This document defines the intended render architecture for deep layers.

It exists to replace the current half-local behavior where:

- the camera/edit path is local
- but rendering still bottoms out at a coarse parent and shows a representative
  block instead of the real deep cells

The design here is the standard nested-shell LOD model, but fully local and
path-based.

It must obey [docs/locality-prime-directive.md](/Users/donalddellapietra/GitHub/deepspace-game/.claude/worktrees/deep-layers-asymmetry-fix/docs/locality-prime-directive.md).

## Problem In One Sentence

The current renderer starts too coarse and is only allowed to descend a bounded
number of levels, so once the real visible cells are deeper than that budget,
the ray never reaches them and shades a coarse representative block instead.

## Design Goals

1. All precision-critical decisions are local to a frame path, never absolute
   world coordinates.
2. Cartesian and sphere content use the same shell model.
3. Nearby visible/editable cells render as real 3D cells, not representative
   parents.
4. Distant visibility remains good.
5. Per-pixel cost is bounded by shell count and local shell work, not by total
   absolute tree depth.

## Core Idea

The renderer uses a stack of nested shells around the camera.

Each shell is:

- a frame path
- a bounded local descent budget inside that frame
- a local subtree upload for that frame

A ray does not start from world root. It starts in the innermost shell. If it
leaves that shell without hitting anything, it transforms to the next outer
shell and continues.

Within a shell, the normal pixel-size stopping rule still applies:

- descend while the current cell is still visibly larger than a pixel
- stop descending once the cell is subpixel or terminal

So:

- pixel threshold answers "is this cell detailed enough?"
- shell selection answers "what scale of world is this ray traversing right
  now?"

Both are required.

## What A Shell Is

For a shell `i`, define:

- `shell_path_i`: the ancestor prefix that owns the shell
- `shell_span_i`: how many tree levels this shell covers
- `shell_budget_i`: how many levels the marcher may descend within this shell

The simplest general rule is fixed-width shells:

- shell 0 is the deepest shell near the camera
- shell 1 is its parent shell
- shell 2 is the parent of shell 1
- each shell covers the same number of levels

Example at anchor depth 33, with 4-level shells:

- shell 0: depths 29..33
- shell 1: depths 25..29
- shell 2: depths 21..25
- shell 3: depths 17..21

A ray first marches shell 0. If it exits shell 0, it is transformed into shell
1 coordinates. Then shell 1 continues the ray, and so on.

## Required Invariant

The innermost shell must always be able to reach the nearest visible/editable
cells.

In precise terms:

`shell_0_render_depth + shell_0_budget >= nearest_visible_edit_depth`

If that is false, the world can mutate correctly while the renderer still shows
only a coarser representative block. That is a correctness failure.

This is the exact failure mode of the current branch.

## How Rays Move Between Shells

Shell transitions are path-based, not world-coordinate based.

When a ray exits a shell, transform it from child frame coordinates to parent
frame coordinates using slot ancestry:

`parent_pos = slot_offset + child_pos / 3`

`parent_dir = child_dir / 3`

Applying this repeatedly moves the ray outward through ancestor shells.

This is local because:

- the transform is derived only from path prefixes and slot ancestry
- no `WorldPos -> absolute XYZ -> shell selection` step is allowed

## What Stops Descent

Inside a shell, a ray stops descending when any of the following is true:

1. the current node is terminal
2. the current cell is subpixel
3. the shell-local descent budget is exhausted

The shell budget is not a replacement for the pixel rule. It exists so that one
shell only covers its intended local band of tree depth.

## What The Shell Architecture Prevents

This design prevents both bad extremes:

### Bad extreme 1: one coarse root with a bounded descent budget

Failure mode:

- the ray never reaches the real deep cells
- the renderer shades the representative block instead
- the scene turns into a flat field at some threshold layer

### Bad extreme 2: one very deep root for the whole screen

Failure mode:

- all pixels start in an overly tiny frame
- distant rays still pay deep local traversal/pop costs
- frame time explodes

Nested shells solve this by giving:

- deep detail near the camera
- coarse traversal farther away

## Distant Features

A correct shell system does not make distant objects disappear just because
they are far away.

If a distant tree is still larger than a pixel, the outer shell that covers it
must still descend enough to resolve it.

So the rule is:

- inner shells provide nearby deep detail
- outer shells provide distant context
- every shell still obeys the normal pixel-size stopping rule

If a distant object larger than a pixel disappears, the shell schedule is wrong.

## CPU And GPU Contract

The CPU raycast/highlight/edit path and the GPU renderer must use the same shell
contract.

That means both sides must agree on:

- shell paths
- shell ordering
- shell-local coordinates
- shell transition transforms
- what counts as the nearest visible/editable depth

It is not acceptable for CPU interaction to be shell-local while GPU rendering
still uses a different coarse-root heuristic.

## Packing Model

Each shell should upload only the subtree needed for that shell.

That means:

- shell-local packing, not one giant root-oriented pack
- shell-local representative blocks only for cells that are truly outside the
  shell budget or already subpixel

Representative blocks are not allowed to mask a shell that should have reached
the real cells.

## Recommended First Implementation

Start with a fixed-width shell schedule.

For example:

- constant shell span `S`
- shell 0 is the deepest ancestor prefix that still reaches the nearest
  visible/editable cells
- shell `i + 1` is the parent prefix `S` levels above shell `i`

This is general, symmetric, and easier to reason about than adaptive
per-frame heuristics.

Once that works correctly, the shell schedule can be tuned. But correctness
comes first.

## Non-Goals

These are not valid substitutes for the shell design:

- increasing one global depth cap
- decreasing render margin until the symptom moves
- preserving more edited leaves in the packer
- using representative blocks to hide missing deep geometry
- using absolute world coordinates to choose shells

Those only move the breakpoint. They do not fix the architecture.

## Rewrite Consequence

The renderer should no longer be modeled as:

- one active frame
- one render root
- one global visual depth

Instead it should be modeled as:

- an ordered shell stack
- shell-local traversal in each shell
- outward transitions between shells

That is the correct local LOD architecture for deep layers.
