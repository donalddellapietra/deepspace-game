# Editing

Break, place, and install-subtree. All edits flow through a CPU
raycast that mirrors the shader's tree walk, a `HitInfo` describing
the cell that was hit, and `propagate_edit`, which rebuilds the
ancestor chain clone-on-write.

Source of truth:
- `src/world/raycast/` — CPU ray-march (mirrors the shader).
- `src/world/edit.rs` — `break_block`, `place_child`, `propagate_edit`.
- `src/app/edit_actions.rs` — input wiring and GPU re-upload.

## CPU raycast

The CPU ray-march runs in the **same frame** as the GPU render so that
the cell under the crosshair is the same cell the shader shaded. It
lives in two files:

- `raycast/cartesian.rs` — iterative stack-based Cartesian DDA over
  the unified tree. Dispatches to `sphere::cs_raycast_in_body` when it
  descends into a `CubedSphereBody` child.
- `raycast/sphere.rs` — cubed-sphere DDA. Step-based march; accuracy
  is tuned for cursor targeting, not rendering fidelity.

Entry points:

```rust
pub fn cpu_raycast(...) -> Option<HitInfo>;
pub fn cpu_raycast_in_frame(...) -> Option<HitInfo>;
pub fn cpu_raycast_in_frame_with_budget(...) -> Option<HitInfo>;
pub fn is_solid_at(...) -> bool;
```

`max_depth` caps how far the walker descends. This is how **zoom
controls edit granularity**: at shallow zoom the raycast bottoms out
on a large cell, and a single break removes a 3×3×3 (or 3⁶, …) chunk.
At deep zoom it reaches a single terminal block.

## `HitInfo`

```rust
pub struct HitInfo {
    pub path: Vec<(NodeId, usize)>,   // root → hit cell, each (node, slot)
    pub face: u32,                    // 0=+X … 5=-Z
    pub t: f32,                       // distance along ray
    pub place_path: Option<Vec<(NodeId, usize)>>,
}
```

- `path` is root-to-hit, with the last entry's `slot` being the cell
  the ray stopped at.
- `face` is which face of the hit cell was crossed; place resolves
  the adjacent cell by `slot + face-delta`.
- `place_path` is an **explicit placement path**. Sphere hits carry
  one — the last empty cell the ray traversed before hitting the
  block — because face-subtree `(u, v, r)` slots don't admit simple
  `face → xyz-delta` arithmetic. Cartesian hits leave `place_path =
  None` and let `place_child` derive the neighbor.

## `propagate_edit`

The one-and-only edit primitive (`src/world/edit.rs`):

```rust
fn propagate_edit(world: &mut WorldState, hit: &HitInfo, new_child: Child) -> bool;
```

Walks `hit.path` from the deepest parent back up to the root,
building new nodes with the edited child substituted. Key invariants:

- **`NodeKind` is preserved** on rebuild. Without that, the shader's
  kind dispatch stops firing past an edited sphere ancestor and the
  renderer walks the body Cartesian-style.
- Each new node goes through `NodeLibrary::insert_with_kind`, so
  dedup still applies — an edit that happens to collapse a uniform
  subtree may re-point at an existing node rather than allocate.
- The final `swap_root(new_root)` refcounts the old root down and
  the new root up.

## `break_block` and `place_child`

```rust
pub fn break_block(world: &mut WorldState, hit: &HitInfo) -> bool;
pub fn place_child(world: &mut WorldState, hit: &HitInfo, new_child: Child) -> bool;
```

`break` is sugar for `propagate_edit(hit, Child::Empty)`. `place`
derives the placement path (from `place_path` for sphere hits, from
`slot + face-delta` for Cartesian), then calls `propagate_edit`. Both
return `true` iff the tree actually changed.

## Input → GPU flow

1. `apply_mouse(button)` in `src/app/input_handlers.rs` fires on
   click.
2. `do_break` / `do_place` in `src/app/edit_actions.rs` runs
   `frame_aware_raycast` — a `cpu_raycast_in_frame` with the camera's
   render frame + current visual-depth budget.
3. On hit: `break_block` / `place_child` mutates `WorldState`,
   emits a `HARNESS_EDIT` line (see [../testing/harness.md](../testing/harness.md)),
   and updates `last_edit_slots` (used by `teleport_above_last_edit`).
4. The next render tick computes a new `LodUploadKey`; since the tree
   changed, it misses the cache and the packer re-uploads.

## No separate "edit depth"

The depth at which you edit is the depth the raycast resolved to.
There is no `cs_edit_depth`, no separate `edit_depth` slider. `zoom`
controls `max_depth` for the raycast (via `visual_depth` / camera
anchor depth) and that's the whole story. See
[coordinates.md](coordinates.md#what-this-replaces) for what went
away.
