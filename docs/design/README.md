# Design (not yet implemented)

These docs describe subsystems that the architecture has *room for*
but no live code path actually implements. Keep them around because
the shape of the eventual solution is worth preserving; treat them as
**design sketches, not references for current behavior**.

If you're trying to understand how the code works today, go to
[`../architecture/`](../architecture/). If a design doc looks live,
it's not — verify against `src/` before trusting it.

## Contents

- [collision.md](collision.md) — swept-AABB physics at `anchor_depth −
  1`. Today `src/player.rs::update` is a no-op; the `sdf::Planet`
  gravity field is defined but never integrated. Motion is WASD-
  teleport one child cell per keypress.

- [content-pipeline.md](content-pipeline.md) — GLB voxelization into
  tree subtrees. Today `src/import/` parses MagicaVoxel `.vox` files
  and nothing else; no callers. The saved-mesh flow in
  `game_state::SavedMeshes` stashes `NodeId`s that were created via
  the normal edit path, not via import.

- [streaming.md](streaming.md) — CDN-native streaming for multiplayer
  and offline worlds. Today the `NodeLibrary` has content-addressed
  dedup (real, see [../architecture/tree.md](../architecture/tree.md)),
  but there is no network code, no server, no client cache.

## When to promote

Move a doc back into `architecture/` the moment the code actually
implements what it describes — and rewrite it then against the real
implementation. Don't port aspirational prose unchanged; the two
always diverge in detail.
