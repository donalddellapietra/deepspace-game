# Color Picker Investigation — 2026-04-12

## Status

**Blocked.** All changes discarded. The codebase is back to the original broken state with the bevy_ui `Slider` widget that doesn't respond to drags.

## Original Problem

The color picker uses Bevy 0.18's built-in `Slider` widget (`bevy::ui_widgets`). Sliders render correctly but do not respond to mouse drags or clicks. The original code had 6 debug systems added trying to diagnose why (`debug_pointer_events`, `debug_global_press`, `debug_global_drag`, `debug_slider_values`, etc.) — none of which revealed the cause.

## Root Cause: CursorMoved events stop after cursor unlock

On macOS, after transitioning from `CursorGrabMode::Locked` → `CursorGrabMode::None` (which happens when any panel opens), **winit stops delivering `CursorMoved` events**. This was confirmed with diagnostic logging:

```
cursor_msgs=0 btn_msgs=0 pointer=Some([544.3 659.6])
```

The pointer position freezes at its last known value. No new `CursorMoved` or `MouseButtonInput` messages reach any UI system. This breaks ALL pointer-based UI in Bevy — both native `bevy_ui` interaction (`Interaction` component, `Slider` widget) and any third-party UI library that depends on these events.

Keyboard input (`KeyboardInput` messages, `ButtonInput<KeyCode>`) continues to work normally. Only mouse/pointer input is affected.

This is a winit/Bevy bug on macOS, not a UI library bug.

## bevy_egui Attempt

We replaced the bevy_ui color picker with `bevy_egui` 0.39. The egui sliders also didn't work initially — same root cause (no cursor events).

### Workaround 1: Track cursor via AccumulatedMouseMotion

`AccumulatedMouseMotion` (from `DeviceEvent::MouseMotion`) continues to flow even after cursor unlock — it's how camera rotation works while the cursor is locked. We tracked cursor position ourselves:

```rust
// In PreUpdate, after EguiPreUpdateSet::ProcessInput, before BeginPass:
tracked.pos += motion.delta / scale;
egui_input.0.events.push(egui::Event::PointerMoved(egui_pos));
```

This successfully delivered pointer position to egui.

### Problem 2: bevy_egui begin_pass/end_pass hit-testing bug

Even with correct pointer position and button state reaching egui, `Response::contains_pointer()` was always `false` for all widgets. Diagnostic output confirmed:

- Pointer at `(546, 389)` — geometrically inside slider rect `[[496, 380] - [596, 398]]`
- `pointer.latest_pos()`, `pointer.interact_pos()`, `pointer.hover_pos()` all correct
- `pointer.primary_pressed() = true`
- Layer IDs match
- But `contains_pointer = false`, `hovered = false`

The issue is in bevy_egui's split `begin_pass` (PreUpdate) / `end_pass` (PostUpdate) pipeline. Hit-testing in `begin_pass` uses `prev_pass.widgets` from the previous frame, but something in the split pipeline prevents the widget rects from being correctly matched to the pointer position.

### Workaround 2: ctx.run() with run_manually

Using `ctx.run()` (which processes input and renders UI in a single call — how egui is designed to work) instead of the split pipeline **fixed the slider interaction**. This was confirmed by an automated test:

```
TEST [f45]: press at x=546, r=0.500
TEST [f57]: release at x=711, r=1.000
=== TEST PASSED ===
```

### Problem 3: run_manually breaks bevy_egui rendering

Setting `run_manually = true` on the egui context causes secondary failures:

1. **When picker is closed:** bevy_egui's `process_output_system` expects `EguiFullOutput` every frame. If we skip `ctx.run()` when the picker is closed, it logs errors. If we call `ctx.run()` with empty UI, it works but has other side effects.

2. **Without `bevy_ui` feature:** egui renders directly over the 3D scene, causing a black screen.

3. **With `bevy_ui` feature:** The picker doesn't render at all (the egui-to-bevy_ui bridge doesn't work correctly with `run_manually`).

4. **Toggling `run_manually` dynamically:** Setting it to `true` only when the picker is open and `false` when closed causes the picker to not appear.

No combination of `run_manually`, `bevy_ui` feature, and render configuration produced a working result where:
- The game renders normally when the picker is closed
- The picker appears when opened
- Sliders respond to drag interaction

## Testing Infrastructure

An automated test pipeline was built during this investigation and remains in the codebase:

- **`tests/color_picker_input.rs`** — Standalone egui test (no Bevy). Proves egui's slider works correctly when driven with `ctx.run()` directly. Always passes.

- **`examples/test_color_picker.rs`** — Full game integration test. Opens the picker, drives `TrackedCursor` + `MouseButtonInput` messages to simulate a slider drag, asserts `state.r` changed. Requires `bevy_egui` dependency and `src/lib.rs` (both currently removed). The pattern: run with `timeout 30 cargo run --example test_color_picker --features dev` and check for `PASSED`/`FAILED`.

These files are currently non-functional since bevy_egui was removed. They can be restored as a starting point for future attempts.

## Possible Paths Forward

1. **File bugs upstream:**
   - **Bevy/winit:** `CursorMoved` events stop after `CursorGrabMode::Locked` → `None` on macOS. This is the root cause and blocks all pointer-based UI.
   - **bevy_egui:** `begin_pass`/`end_pass` split causes `contains_pointer` to always return false. `ctx.run()` works but `run_manually` mode has rendering issues.

2. **Avoid sliders entirely:** Use clickable color preset buttons or a grid-based color picker that doesn't need drag interaction. Bevy's `Interaction::Pressed` (click detection) may work if the initial cursor position is valid.

3. **Use a different cursor strategy:** Instead of `CursorGrabMode::Locked`, use `CursorGrabMode::Confined` which may not break `CursorMoved` events. Trade off: camera rotation would need a different approach.

4. **Wait for Bevy 0.19 / bevy_egui updates:** The winit and bevy_egui issues may be fixed in future versions.
