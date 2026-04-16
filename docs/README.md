# Deepspace game — docs

## Product

- [vision.md](vision.md) — what the game is and why it exists.

## Principles

Load-bearing invariants. Every subsystem must honor these.

- [principles/no-absolute-coordinates.md](principles/no-absolute-coordinates.md)
- [principles/locality-prime-directive.md](principles/locality-prime-directive.md)
- [principles/scaling-deep-trees.md](principles/scaling-deep-trees.md)

## Architecture

How the code works today. Normative — if code and doc disagree, the
doc is wrong.

- [architecture/README.md](architecture/README.md) — five-minute tour.
- [architecture/tree.md](architecture/tree.md) — content-addressed 27-child node tree.
- [architecture/coordinates.md](architecture/coordinates.md) — path-anchored `WorldPos`.
- [architecture/rendering.md](architecture/rendering.md) — GPU ray march, render frame, ribbon pop.
- [architecture/editing.md](architecture/editing.md) — CPU raycast + `propagate_edit`.
- [architecture/collision.md](architecture/collision.md) — swept-AABB physics.
- [architecture/zoom.md](architecture/zoom.md) — zoom = anchor depth change.
- [architecture/cubed-sphere.md](architecture/cubed-sphere.md) — planetary bodies.
- [architecture/content-pipeline.md](architecture/content-pipeline.md) — voxelizing `.vox` into subtrees.
- [architecture/streaming.md](architecture/streaming.md) — content-addressed streaming model.
- [architecture/scale.md](architecture/scale.md) — per-layer real-world scale reference.

## Testing

- [testing/README.md](testing/README.md) — tier overview + quick commands.
- [testing/harness.md](testing/harness.md) — render-harness CLI + `HARNESS_*` protocol.
- [testing/e2e-layer-descent.md](testing/e2e-layer-descent.md) — flagship self-similarity test.
- [testing/playwright.md](testing/playwright.md) — browser UI specs.
- [testing/perf-isolation.md](testing/perf-isolation.md) — perf regression playbook.
- [testing/screenshot.md](testing/screenshot.md) — macOS headless window capture.

## Workflow

- [workflow/worktree-dev.md](workflow/worktree-dev.md) — worktree + dev-loop setup.
- [workflow/gotchas/](workflow/gotchas/) — sharp edges you'll run into.

## History

Landed refactors and superseded plans. For context only — use the
architecture docs as source of truth.

- [history/README.md](history/README.md)
