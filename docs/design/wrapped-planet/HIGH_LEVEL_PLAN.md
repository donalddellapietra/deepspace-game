# Wrapped-Cartesian Planet — High-Level Plan

This is the agnostic-of-implementation plan. The detailed implementation plan with file:line citations lives next to it as `IMPLEMENTATION_PLAN.md` (produced by Agent A).

Reference: `/Users/donalddellapietra/Downloads/wrapped-cartesian-planet-architecture.md`.

## End State

Player starts in space, flies down to a planet, lands and walks on the surface. The planet **looks like a planet the entire flight** — no LOD pop, no atmosphere required, no fancy effects. Poles are non-buildable but visually continuous.

## Core Idea

Don't bend the simulation. Bend the rendering.

- **Storage and simulation** — flat Cartesian voxel grid (uses the existing perfect-precision recursive DDA).
- **Wrap** — east-west axis wraps modularly. The other axes are bounded with the polar bands non-buildable.
- **Curvature** — applied only at ray-sample time as a smooth function of camera altitude. From the surface, the world is flat. From orbit, the world is a sphere. The transition is continuous.

## Phases

### Phase 0 — Foundation (parallel + sequential)

| Step | Owner | Mode |
|---|---|---|
| 0.1 Codebase audit + write `IMPLEMENTATION_PLAN.md` | Opus agent A | parallel |
| 0.2 Remove legacy cubed-sphere code from this branch | Opus agent B | parallel |
| 0.3 Port testing infra from `sphere-attempt-2-2-3-2` | Opus agent C | sequential after B |
| 0.4 Coordinator reviews and refines plan | Coordinator | sequential after A/B/C |

### Phase 1 — Hardcoded Flat Slab (one-step-at-a-time)

| Step | Owner | Acceptance |
|---|---|---|
| 1.1 New `NodeKind` for the planet's anchor cell, with active-region bounds | Opus agent | builds; no behavior change yet |
| 1.2 Hardcoded preset that drops one planet (fixed dims, fixed layer) into the world | Opus agent | preset loads; planet exists in tree |
| 1.3 Cartesian DDA respects banned cells (return no-hit, do not descend) | **Coordinator** | rays through banned cells miss |
| 1.4 Visual: camera inside planet, walk on flat slab | Opus agent (test) | screenshot matches flat preset |
| 1.5 Visual: camera in space, planet renders as a brick | Opus agent (test) | screenshot shows finite slab |

### Phase 2 — X-Wrap (full implementation)

| Step | Owner | Acceptance |
|---|---|---|
| 2.1 Modular X-wrap in shader DDA inside `WrappedPlanet` frame | **Coordinator** | ray exits east, re-enters west |
| 2.2 Modular X-wrap in CPU `Path::step_neighbor_cartesian` | Opus agent | unit test: walking east N times returns to start |
| 2.3 Visual: walk east continuously, see the same terrain repeat | Opus agent (test) | screenshot regression |

### Phase 3 — Polar Treatment

| Step | Owner | Acceptance |
|---|---|---|
| 3.1 Decide cosmetic fill (extended terrain colour / solid cap / impostor) | Coordinator | design note in PLAN.md |
| 3.2 Implement chosen fill | Opus agent | from above-pole, no hole visible |
| 3.3 Visual regression | Opus agent (test) | screenshot at pole reads as planet surface |

### Phase 4 — Curvature (carefully sub-stepped)

| Step | Owner | Acceptance |
|---|---|---|
| 4.1 `k(altitude)` curve in CPU, uploaded to shader | Opus agent | uniform reaches GPU |
| 4.2 Parabolic ray-bend at sample point in `march_cartesian` | **Coordinator** | k=0 unchanged; k>0 bends correctly |
| 4.3 Click-ray for editing remains straight | Opus agent | place block where reticle points, not where ray bends |
| 4.4 Tune transition altitude band | Coordinator | smooth zoom, no horizon-pop |
| 4.5 Visual regression: silhouette circular at orbit | Opus agent (test) | image-analysis test |

### Phase 5 — Integration

| Step | Owner | Acceptance |
|---|---|---|
| 5.1 Lighting / entity systems wrap-aware | Opus agent | no boundary glitches |
| 5.2 End-to-end harness: orbit → surface → orbit | Opus agent | full descent screenshots OK |
| 5.3 Final polish + cleanup | Coordinator | ready to merge |

## Coordination Model

- **Coordinator (me)** — plan refinement, ray-marching shader edits, critical review of every agent diff.
- **Opus implementation agents** — file-level changes, deletions, additions, tests, repro scripts, WGSL outside the core DDA.
- **Self-contained briefings** — every agent prompt includes the goal, the why, the relevant file:line citations, and acceptance criteria. The agent does not see this conversation.
- **Critical review** — after every agent, the coordinator reads the diff before unblocking the next phase. Agents do not get to mark phases done.

## Non-Goals (Explicitly Out of Scope)

- Atmosphere or weather effects.
- Polar gameplay (poles are banned).
- Latitude-correct travel distances (we accept that walking around at lat 70 is the same number of steps as at the equator).
- Structure aesthetics from orbit (cosine compression accepted; structures are too small to see individually).
- Real spherical voxels at any layer.
