# TangentBlock unification — PROGRESS POINT 4

The story up to PROGRESS POINT 3 is in `tangent-block-rotation-fix.md`.
This doc covers what came after: unifying the rotation-handling so
the camera renders consistently across all zoom layers.

## The problem PP3 left

PP3 had three different code paths for TB rotation, and each fix
to one of them produced a new failure mode in another:

1. `march_cartesian` line ~778 — applied `R^T` at TB child entry from a Cartesian parent.
2. `march()` start — applied `R^T` at frame entry when frame root is TB.
3. `march_cartesian` running inside a TB descendant — no rotation applied.

When the user zoomed across layers, the render frame depth crossed
the boundaries between these paths. At each crossing, a different
combination of transforms was applied, and the camera "jumped" because
the render coordinates were inconsistent. Specifically:

- Layer 28 (frame=[]): line ~778 applies rotation. Cube renders correctly.
- Layer 27 (frame=[13]): frame-root case. Bevels broke because the inline cell_min/cell_size was in TB-storage but downstream consumed it with world hit_pos.
- Layer 26 (frame=[13, 15]): no rotation applied at frame root. Cube rendered axis-aligned. Camera "transported" elsewhere.

PP3 also had no inscribed-cube shrink. Without `tb_scale`, a 45°
rotated cube extended outside slot bounds at corners, but cube render
worked. Adding the shrink (PP2-era `c529493`) collided with these
inconsistencies and amplified the camera jumps.

## The unification

**One rule applied symmetrically at every TB boundary**:
- TB descent: `local' = R^T · (local − 1.5) / tb_scale + 1.5`, `dir' = R^T · dir / tb_scale`.
- TB pop:    inverse of the above.

Where `tb_scale = 1 / max_i(Σ_j |R[j][i]|)` — the largest uniform
shrink such that the rotated cube `R · ([0, 3)³ − 1.5) · tb_scale + 1.5`
fits inside `[0, 3)³`. For 45° about Y: `tb_scale = 1/√2 ≈ 0.707`.

This rule is applied at exactly one location per chain — every system
that touches the camera position applies the same transform at the
same edge:

| System | Where the rule fires |
|---|---|
| Anchor descent (`pop_one_level_rot_aware`, `descend_one_level_rot_aware`, `zoom_in_in_world`, `zoom_out_in_world`) | TB cell ↔ TB-storage transition during cell-local pop / descend |
| `in_frame_rot` (camera position projection) | Per-level slot offsets and final centred offset rotated/scaled by accumulated `cur_rot`/`cur_scale` over TBs in the path |
| WSAD step (`app/mod.rs::update`) | `R^T_chain · step_world / chain_scale` so step axes match the offset's chain-rotated frame |
| Shader `march_cartesian` line ~778 | TB child dispatch from Cartesian parent |
| Shader ribbon pop | TB exit during ribbon walk |
| CPU raycast `cpu_raycast_inner` | Same as shader — TB child dispatch |
| CPU raycast `cpu_raycast_in_frame` | Same as shader — TB ribbon pop |

`tb_scale` is stored in `rot_col0.w` of `GpuNodeKind` (was unused
padding) and computed by `inscribed_cube_scale(R)` in `world/gpu/types.rs`.

## Letting offset go out of `[0, 1)³`

A subtle consequence: when the camera is in the parent slot's corner
*outside* the inscribed diamond (the rotated cube extends from
`[0.5 − 0.5·tb_scale, 0.5 + 0.5·tb_scale]³`, leaving corner regions
that are physically inside the parent slot but outside the rotated
content), the post-`R^T·/tb_scale` offset lands outside `[0, 1)³`.

Earlier attempts handled this by clamping the offset back into
`[0, 1)³` (snapping the camera onto the inscribed boundary) or by
refusing the descent (dropping anchor depth). Both produced visible
artifacts:
- Clamp: camera trapped inside the diamond — moving toward the corner snapped back.
- Refuse: anchor depth dropped, the renderer briefly used a shallow render frame, the camera APPEARED to teleport to "layer 31 sky".

**The fix**: relax the `offset ∈ [0, 1)³` invariant. The offset is
allowed to go outside `[0, 1)³` for TB descendants. The TB cell's
"logical" extent is now the full parent slot (not just the inscribed
diamond). The mathematical formulas in `in_frame_rot` continue to
produce the correct world position for OOB offsets. The shader's
ribbon-pop mechanism handles cam.pos outside `[0, 3)³` by walking
rendering up to a containing ancestor frame.

`renormalize_world` is now two-phase to avoid infinite loops:
- **Phase 1**: pop while offset OOB until in range or root reached. WrappedPlane wraps fire here.
- **Phase 2**: descend back to `target_depth`. Each TB descent may produce OOB offset; that's accepted, no clamp, no re-pop.

Both phases are monotonic in `anchor.depth()`, so termination is
trivial.

## World position preservation across zoom

`zoom_in_in_world` had a bug where `coords` was clamped to `[0, 2]`
but `new_offset` was computed against the *unclamped* floor:

```rust
let s = (self.offset[i] * 3.0).floor();          // 6.0 for OOB offset 2.041
coords[i] = s.clamp(0.0, 2.0) as usize;          // CLAMPED to 2
new_offset[i] = (self.offset[i] * 3.0 - s)       // uses UNCLAMPED s = 6
                .clamp(0.0, 1.0 - f32::EPSILON); // re-clamps
```

For OOB input, this packed nothing into `coords` (just clamped)
while subtracting the unclamped value from `new_offset` and then
re-clamping `new_offset` to `[0, 1)`. The resulting `(slot, offset)`
pair represented a different world position from the input —
camera "jumped" by `(unclamped − clamped) / 3` per zoom level.

Fix: use the *clamped* slot for the subtraction and don't re-clamp
`new_offset`. The new pair preserves world position bit-exactly:
`storage_pos_in_parent = slot + new_offset = original_offset · 3` for
all offsets, OOB or not.

```rust
let s_unclamped = (self.offset[i] * 3.0).floor();
let s_clamped = s_unclamped.clamp(0.0, 2.0);
coords[i] = s_clamped as usize;
new_offset[i] = self.offset[i] * 3.0 - s_clamped;  // can be OOB
```

## Sequence of states

| State | Geometry | Shrink | Anchor | Camera-jump on zoom |
|---|---|---|---|---|
| `BROKEN STATE 1` (`3691e40`) | Direction-only `R^T` | none | Cartesian | Cube translates with camera |
| `PROGRESS POINT 2` (`872001b`) | Centred `R^T` (entry/pop/frame-root) | partial | Cartesian | Couldn't dig deep enough to expose |
| `BROKEN STATE 2` (`b2b574`) | Centred `R^T` everywhere | partial | World-XYZ snap (precision wall) | None — but movement frozen at depth 30 |
| `PROGRESS POINT 3` (`74079c3`) | Centred `R^T` entry+pop, no frame-root | none | Cell-local rotation-aware, clamp-on-OOB-descent | Layer 28↔27 stable; cube extended past slot bounds |
| Hybrid attempts (multiple) | Various | with shrink | Various clamp/refuse strategies | Different camera jumps at different layers |
| **PROGRESS POINT 4** (this) | **Single rule symmetric everywhere** | **with shrink, OOB allowed** | **Cell-local, no clamp, world-preserving zoom** | **None — camera stays put when zooming in place** |

## Files touched

- `src/world/gpu/types.rs` + `mod.rs`: `inscribed_cube_scale` helper, `tb_scale` in `rot_col0.w`.
- `src/world/anchor/world_pos.rs`:
  - `pop_one_level_rot_aware`, `descend_one_level_rot_aware`: rotation-aware with `tb_scale`, no clamp.
  - `zoom_in_in_world`, `zoom_out_in_world`: same rule, world-position-preserving.
  - `in_frame_rot`: walks tail with `cur_rot · cur_scale` per TB.
  - `renormalize_world`: two-phase (pop-then-descend), handles OOB offsets.
- `assets/shaders/march.wgsl`: line-778 dispatch + ribbon pop apply `R^T·/tb_scale` / inverse. No frame-root case.
- `src/world/raycast/cartesian.rs`, `mod.rs`: CPU mirrors of shader transforms.
- `src/app/mod.rs`: `frame_path_scale` helper. WSAD step gets `R^T_chain · step / chain_scale`.

## Properties

- **Generality**: the math depends on `R` only via standard rotation algebra and `tb_scale = f(R)`. Works for any rotation matrix, not just 45° about Y.
- **Precision**: cell-local arithmetic, magnitudes bounded by cell-fraction at any anchor depth.
- **Symmetry**: pop is the inverse of descent at every level. Round-trip pop+descend recovers the original `(anchor, offset)` bit-exactly modulo rotation roundoff.
- **World position invariance under zoom**: bit-exact for OOB and in-range offsets.
- **No depth drops**: anchor depth follows `target_depth` monotonically; OOB doesn't trigger pops.
