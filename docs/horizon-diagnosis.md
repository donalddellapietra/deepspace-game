# Horizon Problem Diagnosis

## What the user sees

"Gray block squares at the horizon" — visible when elevated/jumping,
not when flat on the ground.

## What was diagnosed

### The gray band (ground level)

At ground level, there's a visible gray horizontal band between the
terrain and the sky. Pixel analysis shows it is **the ClearColor**
showing through a gap between where the atmosphere sky rendering ends
and where terrain begins.

Evidence:
- Disabling atmosphere → band disappears (ClearColor matches terrain)
- Changing ClearColor → band color changes proportionally
- Band color is uniform horizontally (no block patterns)
- Band is at the same Y position regardless of clip_radius changes

The atmosphere renders a sky gradient down to a certain angle. Below
that, the ClearColor fills until terrain starts. The ClearColor after
tonemapping appeared as `#8E9D81` (gray), contrasting with the bright
atmosphere sky above (`#D4E0C9`).

### Approaches tried and failed

1. **BSL distance fog** — blending terrain toward sky color in the
   fragment shader, before atmosphere post-process. Failed because the
   atmosphere inscattering overpowers whatever the fragment shader
   outputs at that distance. Made terrain wash out to white at elevated
   angles.

2. **Dithered discard** — screen-door transparency near clip boundary.
   Didn't help because the gap is ClearColor, not terrain.

3. **Tighter clip_radius** — clipping terrain closer. Made the gray
   band WIDER because more ClearColor gap is exposed.

4. **Extended atmosphere LUT range** — `aerial_view_lut_max_distance *
   3.0`. No visible effect on the gray band (confirmed: it's ClearColor,
   not atmosphere inscattering).

5. **ClearColor tuning** — matching ClearColor to atmosphere horizon.
   Successfully reduces the band visibility but is not scalable (color
   depends on time of day, biome, etc.).

### The real problem (elevated view)

The gray band at ground level is a cosmetic issue. The user's actual
complaint is about **jagged block-shaped artifacts when elevated**.
This has NOT been properly diagnosed yet. Hypotheses:

- SSAO depth-discontinuity outlines on individual blocks near the clip
  boundary, visible when looking down from elevation
- Shadow cascade artifacts at the horizon distance
- Chunk mesh boundaries creating visible steps

The AO fade (both `diffuse_occlusion` and BSL AO) starting at 40%
radius was meant to address SSAO artifacts, but it may not be
aggressive enough or may not affect the right distance range when
viewed from above.

## Current state of the fix

Working:
- Shader clip at clip_radius (smooth circular terrain boundary)
- SSAO + diffuse_occlusion fade near clip boundary
- Shadow cascade extension to 2x radius

Not yet fixed:
- Jagged artifacts when elevated (needs elevated camera testing)
- ClearColor gap color (cosmetic, not the primary complaint)
