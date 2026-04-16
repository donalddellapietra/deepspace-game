# Layer vs. depth (and `plain_layers`)

**Symptom**: You think "layer 37" and "anchor depth 37" mean the same
thing, so you reach for `--spawn-depth 37` to start the player at
layer 37. The camera lands fully zoomed in instead, every cell is
tiny, and nothing about the scene matches what you expected.

**Cause**: The codebase has three numerically distinct quantities that
are all informally called "layer" or "depth." They are not the same
number and often move in opposite directions.

| Name | Type | Meaning | Where |
|---|---|---|---|
| `tree_depth` (a.k.a. `plain_layers` at spawn) | `u8` | Total height of the world tree. `--plain-layers N` builds a tree `N` levels deep. | `world::bootstrap::plain_world`, `WorldState::tree_depth()` |
| `anchor_depth` | `u32` | Absolute tree depth of the camera's `Path` anchor. `0 = root`, grows downward. | `Path::depth()`, `App::anchor_depth()`, `--spawn-depth` |
| `layer` / `zoom_level` | `i32` | The UI-facing zoom indicator. Shown on the hotbar/mode overlay. What the player means when they say "I'm at layer 37." | `App::zoom_level()`, `GameUiState::zoom_level` |

The relationship is an **inverse**:

```text
zoom_level = tree_depth - anchor_depth + 1
anchor_depth = tree_depth - zoom_level + 1
```

(See `src/app/mod.rs::App::zoom_level` and the `apply_zoom` log line
in `src/app/edit_actions.rs`.)

## What this means in practice

With the default `--plain-layers 40`:

| UI layer | `anchor_depth` | Cell size (world units / `3^anchor_depth`) |
|---:|---:|---:|
| 40 | 1 | `WORLD_SIZE / 3` — huge, near root |
| 37 | 4 | `WORLD_SIZE / 81` |
| 1 | 40 | `WORLD_SIZE / 3^40` — tiny, fully zoomed in |

- **Higher layer number = shallower anchor = BIGGER cells = more zoomed out.**
- **Lower layer number = deeper anchor = SMALLER cells = more zoomed in.**
- **Zooming in** decreases `layer` and increases `anchor_depth`.
- **Zooming out** increases `layer` and decreases `anchor_depth`.

## Common traps

1. `--spawn-depth 37` in a 40-layer world puts the camera at
   **layer 4** (almost fully zoomed in), not layer 37. To start at
   **UI layer 37** in a 40-layer world, pass `--spawn-depth 4`.
2. "Digging all the way to depth 1" in conversational shorthand almost
   always means "UI layer 1" (fully zoomed in, `anchor_depth = tree_depth`),
   not `anchor_depth = 1` (root's direct child, gigantic cells).
3. `--plain-layers N` is a world-gen parameter — the tree's total
   height. It is not the starting zoom, the starting anchor depth, or
   the target UI layer. It caps what the other two can be.
4. Logs, code comments, and UI sometimes use the word "layer" for
   `zoom_level` and sometimes for `tree_depth` (e.g.,
   `"plain world must have at least one layer"` refers to
   `tree_depth`). Read the variable, not the word.

## Rule of thumb

If a CLI flag, log line, or function signature uses the word "depth,"
assume it's `anchor_depth` (or `path.depth()`) — absolute, grows with
zoom-in.

If the UI, a conversation, or a design doc uses the word "layer,"
assume it's the `zoom_level` — relative, grows with zoom-out, inverse
of `anchor_depth` against `tree_depth`.

Convert explicitly at the boundary:

```rust
let anchor_depth = tree_depth - ui_layer + 1;
let ui_layer    = tree_depth - anchor_depth + 1;
```
