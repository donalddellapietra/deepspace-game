# Root Cause: AO Discontinuity at Child Boundaries

## Summary

The seams are caused by **ambient occlusion (AO) discontinuities at the 25-voxel child boundaries** within each 125×125×125 emit-level mesh. They are NOT caused by geometry gaps, float precision, transform misalignment, or shadow mapping.

## How we proved it

### Step 1: Identical render pipeline at all zoom levels

Diagnostic logging showed that zoom layers 10, 11, and 12 all use the same render parameters:

| Zoom layer | emit_layer | target_layer | scale | Visits |
|-----------|-----------|-------------|-------|--------|
| 12 | 11 | 12 | 1.0 | 2 |
| 11 | 11 | 12 | 1.0 | 35 |
| 10 | 11 | 12 | 1.0 | 1530 |

The only difference is the number of visible entities (driven by render radius). The seams exist at all layers but are only visible at layer 10+ because ~1530 entities tile the world, exposing many boundaries.

### Step 2: Perfect tile alignment

- All tile origins are exact integers (no f32 precision loss from the i64→f32 cast)
- Tile spacing is exactly 125.0 between adjacent tiles (0 gaps detected)
- Mesh vertex AABBs span exactly 0–125 on each axis

### Step 3: AO is the sole cause

- **Disabling AO entirely** (setting all vertex colors to `[1.0, 1.0, 1.0, 1.0]` in `merge_child_faces`) → **seams disappear completely**
- Re-enabling AO → seams return

### Step 4: Child boundaries, not tile boundaries

- Disabling AO only at **child boundaries** (every 25 voxels) → seams disappear
- The seams are inside each 125³ mesh, not between separate entities
- The grid pattern is at 25-voxel intervals (child edges), not 125-voxel intervals (tile edges)

### Step 5: AO values, not greedy merge structure

- Forcing `ao = [3,3,3,3]` during `compute_face_ao` for faces at child boundaries → seams disappear underground
- But this approach breaks incremental baking (changes which faces are uniform-AO vs per-vertex-AO, invalidating the per-child cache that enables instant block placement)

## Why the AO is wrong at child boundaries

The greedy mesher runs per-child (25³ regions via `bake_child_faces`). Each child produces its own set of quads. At the boundary between two children:

1. **AO is computed correctly** — the `get` closure sees the full 125³ flat grid, so neighbor lookups across child boundaries return correct voxel data.

2. **But the greedy merge can't cross children** — it operates within each 25³ region. Two adjacent children produce separate quads that meet edge-to-edge at the boundary.

3. **The quads can have different AO levels** — Child A's merged quad might have uniform AO=3 (brightness 1.0), while child B's boundary face has per-vertex AO=[3,2,3,2] (brightness gradient from 1.0 to 0.9). At the shared edge, the brightness jumps from 1.0 to 0.9, creating a visible dark line.

4. **The discontinuity is amplified at distance** — at zoom layer 10, the camera is far enough that sub-pixel AO gradients within a quad are invisible, but the sharp brightness jump at the child boundary becomes a visible line.

## What does NOT cause the seams

- **Geometry gaps** — vertices at child boundaries are at exact integer positions (25.0, 50.0, etc.), identical on both sides
- **Float precision** — all origins are exact integers, no precision loss
- **Entity transform mismatch** — irrelevant since the seams are within a single mesh/entity
- **Shadow mapping** — disabling shadows does not fix the seams (attempt #6)
- **Tile-level issues** — the seams are at 25-voxel child boundaries, not 125-voxel tile boundaries
