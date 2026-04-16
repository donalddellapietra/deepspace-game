# History

Landed refactors, superseded plans, and cautionary tales.

These docs are kept for context — they describe *how* we got to the
current architecture, not *what* the architecture is now. For the
live architecture, see [../architecture/](../architecture/). If a
history doc contradicts code today, trust the code.

## Contents

- [anchor-refactor-decisions.md](anchor-refactor-decisions.md) —
  the 13-part decision log for the path-anchored coordinate system
  (landed). Canonical reference: [../architecture/coordinates.md](../architecture/coordinates.md).

- [camera-rewrite-first-principles.md](camera-rewrite-first-principles.md) —
  the precision-budget analysis that led to the render-frame
  rewrite.

- [camera-rewrite-plan-v2.md](camera-rewrite-plan-v2.md) —
  the concrete plan for making the render root dynamic (CubedSphereBody
  + Cartesian cap). Canonical reference: [../architecture/rendering.md](../architecture/rendering.md).

- [local-shell-lod.md](local-shell-lod.md) — the nested-shell LOD
  design that informed today's ribbon-pop approach.
