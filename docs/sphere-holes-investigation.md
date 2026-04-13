# Sphere Holes Investigation

## Symptom

Rectangular holes visible in the sphere terrain at zoom layers 10-11.
You can see through to the sky. The holes always appear at the same
positions on the sphere (deterministic). Breaking a block near a hole
causes the terrain to render correctly.

## What we ruled out

| Theory | Test | Result |
|--------|------|--------|
| Prebaked data corruption | `verify_meshes` binary compared all 10,905 nodes | All match (when sorted by voxel type) |
| Compact serialization (f32→u8 round trip) | Tested with raw f32 derives | Same holes |
| Streamer vs monolithic loader | Tested both | Same holes |
| Bake cache reuse across zoom | Tested with `clear_all` on zoom change | Same holes |
| AABB outside check (terrain margin) | Fixed to always use terrain margin | Same holes (but fix was reverted by git chaos — needs re-verification) |
| Water transparency | Made water opaque | Same holes |
| Cold bake budget exhaustion | Prebaked loader has no budget | Same holes |
| Missing nodes in prebaked index | All 10,905 nodes present | No missing |
| Backface culling | Set `double_sided: true, cull_mode: None` | Same holes |
| Shadow rendering | Disabled shadows entirely | Same holes |
| Walk radius culling | Holes are in the middle of view, not at edge | N/A |
| Missing palette materials | Logged `palette.material()` returns None | Zero misses |
| Nodes not in bake cache | Logged `get_merged()` returns None | Zero misses |
| Empty meshes at hole positions | Logged nodes with 0 submeshes but solid voxels | Zero found |

## Key finding

**Cold baking (no prebaked data) with unlimited budget eliminates
holes at layer 11.** Layer 10 couldn't be tested because unlimited
cold baking is too slow (hundreds of nodes × ~50ms each).

This means:
1. The tree data (world.bin) is correct — cold baking from the same
   tree produces correct meshes
2. The prebaked mesh data (meshes_mono.bin) is wrong for some nodes
3. `verify_meshes` showed the data matches — but it compared fresh
   bakes against prebaked data from the SAME gen_world run. If the
   prebaked data was generated from a DIFFERENT sphere (before AABB
   fix, different git state), NodeIds would mismatch

## Root cause: FOUND

**The compact mesh serialization (`CompactFaceData`) was corrupting
vertex positions.**

The compact format cast `f32` positions to `u8` for storage:
```rust
positions.push(p[0] as u8);  // f32 → u8 truncation
```

This works for integer positions (0, 1, 25, 125), but the greedy
mesher produces positions with fractional values in some cases —
specifically at merged quad boundaries where vertex positions can be
non-integer. The `as u8` cast truncates fractional parts, shifting
vertices by up to 1 voxel. This creates gaps between adjacent quads
within a single node's mesh.

### Why cold baking works

Cold baking at runtime (`BakedNode::new_cold`) calls
`merge_child_faces` → `meshes.add(data.build())` directly. The
`FaceData` is never serialized — the f32 positions go straight to
the GPU mesh with full precision. No truncation, no gaps.

### Why breaking a block fixes it

Breaking a block creates a NEW NodeId via `install_subtree`. The
new node isn't in the prebaked cache, so it falls through to cold
baking — which produces a correct mesh with full f32 precision.

### Why grassland doesn't have the issue

Grassland has only ~25 unique nodes via dedup. All surface nodes
are flat horizontal planes where greedy merging produces quads with
integer-only vertex positions. The u8 truncation has no effect.

The sphere has curved terrain with diagonal surfaces where greedy
merged quads have non-integer vertex positions at merge boundaries.

## Fix

Disable the `CompactFaceData` custom Serialize/Deserialize impls.
Use raw `#[derive(Serialize, Deserialize)]` on `FaceData` instead.
This increases meshes_mono.bin from 93MB to 139MB but eliminates
the vertex corruption.

A proper fix would use a lossless compact format (e.g., fixed-point
with enough precision, or only compact the components that are
guaranteed integer). The current compact format assumed ALL positions
are integers, which is wrong for greedy-merged quads.

## Version hash validation

Added `world_hash` field to `PrebakedFile`. gen_world embeds the
hash of the world tree in the mesh file. At load time, the game
verifies the mesh file's hash matches the loaded world.bin. Stale
mesh files are rejected with an error message.

## Related issues

- `docs/seam-problem/` — AO discontinuities at child boundaries
  (thin dark lines, separate from the holes issue)
- `docs/sphere-terrain-postmortem.md` — earlier terrain generation
  failures
