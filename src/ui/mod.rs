//! Game HUD: hotbar, layer indicator, and color picker.
//!
//! Palette-ready: when the Palette resource lands, replace
//! `BlockType::ALL` iteration with `palette.iter()` and
//! `HotbarItem::Block(BlockType)` with `HotbarItem::Block(u8)`.

pub mod color_picker;

use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions, PrimaryWindow};

use crate::block::{BlockType, Palette};
use crate::camera::CursorLocked;
use crate::editor::save_mode::{save_mode_eligible, SaveMode};
use crate::editor::{Hotbar, HotbarItem};
use crate::inventory::InventoryState;
use crate::world::CameraZoom;

// ── Design tokens ──────────────────────────────────────────────────

const HUD_BG: Color = Color::srgba(0.08, 0.08, 0.12, 0.88);
const HUD_BORDER: Color = Color::srgba(0.30, 0.55, 0.85, 0.25);
const ACCENT: Color = Color::srgba(0.50, 0.78, 1.0, 1.0);
const ACTIVE_BORDER: Color = Color::srgba(0.55, 0.82, 1.0, 0.95);
const INACTIVE_BORDER: Color = Color::srgba(1.0, 1.0, 1.0, 0.10);
const TEXT_PRIMARY: Color = Color::srgba(1.0, 1.0, 1.0, 1.0);
const TEXT_HINT: Color = Color::srgba(1.0, 1.0, 1.0, 0.35);
const SAVE_MODE_COLOR: Color = Color::srgba(0.3, 0.85, 1.0, 1.0);
const SAVE_WARNING_COLOR: Color = Color::srgba(1.0, 0.75, 0.3, 1.0);

// ── Plugin ─────────────────────────────────────────────────────────

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(color_picker::ColorPickerPlugin)
            .add_systems(Startup, (spawn_hotbar, spawn_mode_indicator))
            .add_systems(Update, (update_hotbar, update_mode_indicator, sync_cursor));
    }
}

// ── Components ─────────────────────────────────────────────────────

#[derive(Component)]
struct HotbarRoot;

#[derive(Component)]
struct HotbarSlotUi(u8);

#[derive(Component)]
struct HotbarSlotKey(u8);

#[derive(Component)]
struct HotbarLabel;

#[derive(Component)]
struct HotbarHint;

#[derive(Component)]
struct ModeIndicator;

#[derive(Component)]
struct ModeIndicatorSave;

// ── Hotbar spawn ───────────────────────────────────────────────────

fn spawn_hotbar(mut commands: Commands) {
    // Outer wrapper: pinned to bottom-center
    commands
        .spawn((
            HotbarRoot,
            Node {
                position_type: PositionType::Absolute,
                bottom: Val::Px(16.0),
                left: Val::Percent(50.0),
                // Shift left by half the expected width (~480px)
                margin: UiRect {
                    left: Val::Px(-244.0),
                    ..default()
                },
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                row_gap: Val::Px(6.0),
                ..default()
            },
        ))
        .with_children(|wrapper| {
            // ── Active item label ──
            wrapper.spawn((
                HotbarLabel,
                Text::new("Stone"),
                TextFont { font_size: 15.0, ..default() },
                TextColor(ACCENT),
            ));

            // ── Slot tray: dark background bar ──
            wrapper
                .spawn((
                    Node {
                        flex_direction: FlexDirection::Row,
                        column_gap: Val::Px(4.0),
                        padding: UiRect::axes(Val::Px(8.0), Val::Px(6.0)),
                        border: UiRect::all(Val::Px(1.0)),
                        border_radius: BorderRadius::all(Val::Px(10.0)),
                        ..default()
                    },
                    BackgroundColor(HUD_BG),
                    BorderColor::all(HUD_BORDER),
                ))
                .with_children(|tray| {
                    // Palette-ready: when Palette lands, replace BlockType::ALL
                    // iteration with palette.iter()
                    for i in 0..10u8 {
                        let bt = BlockType::ALL[i as usize];
                        let key_label = if i == 9 {
                            "0".to_string()
                        } else {
                            format!("{}", i + 1)
                        };

                        tray.spawn(Node {
                            flex_direction: FlexDirection::Column,
                            align_items: AlignItems::Center,
                            row_gap: Val::Px(2.0),
                            ..default()
                        })
                        .with_children(|col| {
                            // Key number hint
                            col.spawn((
                                HotbarSlotKey(i),
                                Text::new(key_label),
                                TextFont { font_size: 9.0, ..default() },
                                TextColor(TEXT_HINT),
                            ));

                            // Color swatch
                            col.spawn((
                                HotbarSlotUi(i),
                                Node {
                                    width: Val::Px(40.0),
                                    height: Val::Px(40.0),
                                    border: UiRect::all(Val::Px(2.0)),
                                    border_radius: BorderRadius::all(Val::Px(6.0)),
                                    ..default()
                                },
                                BackgroundColor(bt.color()),
                                BorderColor::all(INACTIVE_BORDER),
                            ));
                        });
                    }
                });

            // ── Hint text ──
            wrapper.spawn((
                HotbarHint,
                Text::new("1-0: select  |  E: inventory  |  C: color picker  |  Q/F: zoom  |  V: save"),
                TextFont { font_size: 10.0, ..default() },
                TextColor(TEXT_HINT),
            ));
        });
}

// ── Mode indicator spawn ───────────────────────────────────────────

fn spawn_mode_indicator(mut commands: Commands) {
    commands
        .spawn((
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(16.0),
                left: Val::Px(16.0),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                padding: UiRect::axes(Val::Px(14.0), Val::Px(10.0)),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(8.0)),
                ..default()
            },
            BackgroundColor(HUD_BG),
            BorderColor::all(HUD_BORDER),
        ))
        .with_children(|container| {
            container.spawn((
                ModeIndicator,
                Text::new("Layer 0"),
                TextFont { font_size: 18.0, ..default() },
                TextColor(TEXT_PRIMARY),
            ));
            container.spawn((
                ModeIndicatorSave,
                Text::new(""),
                TextFont { font_size: 12.0, ..default() },
                TextColor(SAVE_MODE_COLOR),
            ));
        });
}

// ── Hotbar update ──────────────────────────────────────────────────

fn update_hotbar(
    hotbar: Res<Hotbar>,
    zoom: Res<CameraZoom>,
    palette: Option<Res<Palette>>,
    mut slots: Query<(&HotbarSlotUi, &mut BorderColor, &mut BackgroundColor)>,
    mut label: Query<&mut Text, (With<HotbarLabel>, Without<HotbarSlotKey>)>,
    mut key_labels: Query<(&HotbarSlotKey, &mut TextColor)>,
) {
    let layer_slots = hotbar.slots(zoom.layer);

    for (slot, mut border, mut bg) in &mut slots {
        let i = slot.0 as usize;
        let is_active = i == hotbar.active;

        match &layer_slots[i] {
            HotbarItem::Block(voxel) => {
                bg.0 = palette
                    .as_ref()
                    .and_then(|p| p.get(*voxel))
                    .map(|e| e.color)
                    .unwrap_or(Color::srgba(0.3, 0.3, 0.3, 1.0));
            }
            HotbarItem::Model(_) => {
                bg.0 = Color::srgba(0.20, 0.85, 0.75, 0.7);
            }
        }

        *border = if is_active {
            BorderColor::all(ACTIVE_BORDER)
        } else {
            BorderColor::all(INACTIVE_BORDER)
        };
    }

    // Highlight the key label of the active slot
    for (key, mut tc) in &mut key_labels {
        tc.0 = if key.0 as usize == hotbar.active {
            ACCENT
        } else {
            TEXT_HINT
        };
    }

    // Update active item name
    if let Ok(mut t) = label.single_mut() {
        let name = match hotbar.active_item(zoom.layer) {
            HotbarItem::Block(voxel) => palette
                .as_ref()
                .and_then(|p| p.get(*voxel))
                .map(|e| e.name.clone())
                .unwrap_or_else(|| format!("Block {}", voxel)),
            HotbarItem::Model(idx) => format!("Mesh #{}", idx),
        };
        *t = Text::new(name);
    }
}

// ── Mode indicator update ──────────────────────────────────────────

fn update_mode_indicator(
    zoom: Res<CameraZoom>,
    save_mode: Res<SaveMode>,
    mut layer_q: Query<&mut Text, (With<ModeIndicator>, Without<ModeIndicatorSave>)>,
    mut save_q: Query<(&mut Text, &mut TextColor), (With<ModeIndicatorSave>, Without<ModeIndicator>)>,
) {
    let Ok(mut layer_text) = layer_q.single_mut() else {
        return;
    };
    *layer_text = Text::new(format!("Layer {}", zoom.layer));

    let Ok((mut save_text, mut save_color)) = save_q.single_mut() else {
        return;
    };
    if save_mode.active {
        if save_mode_eligible(zoom.layer) {
            *save_text = Text::new("SAVE MODE");
            save_color.0 = SAVE_MODE_COLOR;
        } else {
            *save_text = Text::new("SAVE — zoom out (Q)");
            save_color.0 = SAVE_WARNING_COLOR;
        }
    } else {
        *save_text = Text::new("");
    }
}

// ── Cursor sync ──────────────────────────────────────────────────

/// Single owner of cursor lock state. Panels set their `open` bool;
/// click-to-grab and Escape-to-release are handled here too.
pub fn sync_cursor(
    inv: Res<InventoryState>,
    picker: Res<color_picker::ColorPickerState>,
    mouse: Res<ButtonInput<MouseButton>>,
    key: Res<ButtonInput<KeyCode>>,
    mut cursor_options: Single<&mut CursorOptions>,
    mut cursor_locked: ResMut<CursorLocked>,
    mut window: Single<&mut Window, With<PrimaryWindow>>,
) {
    let any_panel_open = inv.open || picker.open;

    if any_panel_open {
        let was_locked = cursor_locked.0;
        cursor_options.visible = true;
        cursor_options.grab_mode = CursorGrabMode::None;
        cursor_locked.0 = false;

        // When transitioning from locked → unlocked, warp the cursor
        // to screen center so Bevy's picking and egui immediately
        // have a valid position (CursorMoved won't fire until the
        // user physically moves the mouse otherwise).
        if was_locked {
            let center = Vec2::new(window.width() / 2.0, window.height() / 2.0);
            window.set_cursor_position(Some(center));
        }
        return;
    }

    // Don't grab if egui is consuming the pointer (e.g. dragging a slider)
    if picker.egui_wants_pointer {
        return;
    }

    // Click to grab
    if mouse.just_pressed(MouseButton::Left) && !cursor_locked.0 {
        cursor_options.visible = false;
        cursor_options.grab_mode = CursorGrabMode::Locked;
        cursor_locked.0 = true;
        return;
    }

    // Escape to release
    if key.just_pressed(KeyCode::Escape) {
        cursor_options.visible = true;
        cursor_options.grab_mode = CursorGrabMode::None;
        cursor_locked.0 = false;
    }
}
