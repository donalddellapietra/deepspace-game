//! Minecraft-style inventory panel.
//!
//! Press E to open/close. Shows all block types. Click a block to
//! assign it to the active hotbar slot.

use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};

use crate::block::BlockType;
use crate::camera::CursorLocked;
use crate::editor::save_mode::SavedMeshes;
use crate::editor::{Hotbar, HotbarItem};
use crate::world::view::target_layer_for;
use crate::world::CameraZoom;

pub struct InventoryPlugin;

impl Plugin for InventoryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InventoryState>()
            .add_systems(
                Update,
                (toggle_inventory, handle_clicks, refresh_saved_meshes),
            );
    }
}

#[derive(Resource, Default)]
pub struct InventoryState {
    pub open: bool,
    /// Root entity for the whole inventory panel.
    root: Option<Entity>,
}

// --- Marker components ---
#[derive(Component)]
struct InventoryRoot;

/// A block type slot. Index 0-9 maps to BlockType::ALL.
#[derive(Component)]
struct BlockSlot(u8);

/// A saved-mesh slot. The `usize` indexes into `SavedMeshes.items`.
#[derive(Component)]
struct MeshSlot(usize);

/// Parent node of the saved-mesh row. Children are rebuilt whenever
/// `SavedMeshes` changes so new captures appear in the panel.
#[derive(Component)]
struct SavedMeshRow;

/// Text label above the saved-mesh row; updated to show the count.
#[derive(Component)]
struct SavedMeshLabel;

// ============================================================
// Toggle
// ============================================================

fn toggle_inventory(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<InventoryState>,
    mut cursor_options: Single<&mut CursorOptions>,
    mut cursor_locked: ResMut<CursorLocked>,
    mut commands: Commands,
    root_q: Query<Entity, With<InventoryRoot>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyE) { return }

    state.open = !state.open;

    if state.open {
        cursor_options.visible = true;
        cursor_options.grab_mode = CursorGrabMode::None;
        cursor_locked.0 = false;

        if state.root.is_none() {
            let root = spawn_panel(&mut commands);
            state.root = Some(root);
        } else if let Ok(entity) = root_q.single() {
            commands.entity(entity).insert(Visibility::Inherited);
        }
    } else {
        cursor_options.visible = false;
        cursor_options.grab_mode = CursorGrabMode::Locked;
        cursor_locked.0 = true;

        if let Ok(entity) = root_q.single() {
            commands.entity(entity).insert(Visibility::Hidden);
        }
    }
}

// ============================================================
// Click handling
// ============================================================

fn handle_clicks(
    state: Res<InventoryState>,
    zoom: Res<CameraZoom>,
    mut hotbar: ResMut<Hotbar>,
    mut block_slots: Query<
        (&Interaction, &BlockSlot, &mut BorderColor),
        (Changed<Interaction>, Without<MeshSlot>),
    >,
    mut mesh_slots: Query<
        (&Interaction, &MeshSlot, &mut BorderColor),
        (Changed<Interaction>, Without<BlockSlot>),
    >,
) {
    if !state.open { return }

    let active_slot = hotbar.active;
    let layer = zoom.layer;

    for (interaction, slot, mut border) in &mut block_slots {
        match interaction {
            Interaction::Pressed => {
                if let Some(bt) = BlockType::from_index(slot.0) {
                    hotbar.slots_mut(layer)[active_slot] = HotbarItem::Block(bt);
                }
            }
            Interaction::Hovered => *border = BorderColor::all(Color::WHITE),
            Interaction::None => *border = BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.2)),
        }
    }

    for (interaction, slot, mut border) in &mut mesh_slots {
        match interaction {
            Interaction::Pressed => {
                hotbar.slots_mut(layer)[active_slot] = HotbarItem::Model(slot.0);
            }
            Interaction::Hovered => {
                *border = BorderColor::all(Color::srgba(0.6, 1.0, 0.95, 1.0));
            }
            Interaction::None => {
                *border = BorderColor::all(Color::srgba(0.4, 1.0, 0.9, 0.9));
            }
        }
    }
}

// ============================================================
// Panel spawning
// ============================================================

fn spawn_panel(commands: &mut Commands) -> Entity {
    commands
        .spawn((
            InventoryRoot,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(50.0),
                top: Val::Percent(50.0),
                margin: UiRect {
                    left: Val::Px(-210.0),
                    top: Val::Px(-220.0),
                    ..default()
                },
                width: Val::Px(420.0),
                padding: UiRect::all(Val::Px(16.0)),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                row_gap: Val::Px(12.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.1, 0.1, 0.15, 0.92)),
            Visibility::Inherited,
        ))
        .with_children(|panel| {
            panel.spawn((
                Text::new("INVENTORY"),
                TextFont { font_size: 22.0, ..default() },
                TextColor(Color::WHITE),
            ));

            panel.spawn((
                Text::new("Click to select block type"),
                TextFont { font_size: 13.0, ..default() },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.5)),
            ));

            // Block types — 2 rows of 5
            for row_start in [0u8, 5] {
                panel.spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(8.0),
                    ..default()
                }).with_children(|row| {
                    for i in row_start..row_start + 5 {
                        let bt = BlockType::ALL[i as usize];
                        row.spawn((
                            BlockSlot(i),
                            Button,
                            Node {
                                width: Val::Px(60.0),
                                height: Val::Px(60.0),
                                border: UiRect::all(Val::Px(3.0)),
                                justify_content: JustifyContent::Center,
                                align_items: AlignItems::Center,
                                ..default()
                            },
                            BackgroundColor(bt.color().into()),
                            BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.2)),
                        )).with_children(|btn| {
                            btn.spawn((
                                Text::new(format!("{:?}", bt)),
                                TextFont { font_size: 10.0, ..default() },
                                TextColor(Color::WHITE),
                            ));
                        });
                    }
                });
            }

            // Saved meshes section (populated by `refresh_saved_meshes`).
            panel.spawn((
                SavedMeshLabel,
                Text::new("Saved meshes (0)"),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::srgba(0.6, 1.0, 0.9, 0.9)),
            ));
            panel.spawn((
                SavedMeshRow,
                Node {
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: Val::Px(6.0),
                    row_gap: Val::Px(6.0),
                    max_width: Val::Px(388.0),
                    ..default()
                },
            ));

            // Hints
            panel.spawn((
                Text::new("E: close | Q/F: zoom | R: reset | V: save mode | LClick in save mode: capture"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.4)),
            ));
        })
        .id()
}

// ============================================================
// Saved mesh row
// ============================================================

/// Rebuild the saved-mesh tiles whenever `SavedMeshes`, the current
/// zoom layer, or the row entity itself changes. We filter saved
/// meshes by the current layer's placement target — saved subtrees
/// from other zooms would fail to place anyway, so showing them
/// here would be misleading. Each tile is a neon swatch labelled
/// with the save's original index; the real 3D mesh isn't renderable
/// inside a UI node.
fn refresh_saved_meshes(
    saved: Res<SavedMeshes>,
    zoom: Res<CameraZoom>,
    mut commands: Commands,
    row_q: Query<Entity, With<SavedMeshRow>>,
    added_row: Query<(), Added<SavedMeshRow>>,
    mut label_q: Query<&mut Text, With<SavedMeshLabel>>,
) {
    // Rebuild when any input changes, OR when the row entity was
    // just spawned (the panel is lazily created on first E press, so
    // meshes saved before that need to be backfilled).
    if !saved.is_changed() && !zoom.is_changed() && added_row.is_empty() {
        return;
    }
    let Ok(row) = row_q.single() else {
        return;
    };

    let target = target_layer_for(zoom.layer);
    let matching: Vec<(usize, &crate::editor::save_mode::SavedMesh)> = saved
        .items
        .iter()
        .enumerate()
        .filter(|(_, m)| m.layer == target)
        .collect();

    if let Ok(mut label) = label_q.single_mut() {
        *label = Text::new(format!(
            "Saved meshes for Layer {} ({})",
            zoom.layer,
            matching.len()
        ));
    }
    commands.entity(row).despawn_related::<Children>();
    commands.entity(row).with_children(|parent| {
        for (i, mesh) in &matching {
            parent
                .spawn((
                    MeshSlot(*i),
                    Button,
                    Node {
                        width: Val::Px(48.0),
                        height: Val::Px(48.0),
                        border: UiRect::all(Val::Px(2.0)),
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        ..default()
                    },
                    BackgroundColor(Color::srgba(0.25, 1.0, 0.85, 0.35)),
                    BorderColor::all(Color::srgba(0.4, 1.0, 0.9, 0.9)),
                ))
                .with_children(|tile| {
                    tile.spawn((
                        Text::new(format!("#{}\nL{}", i, mesh.layer)),
                        TextFont { font_size: 10.0, ..default() },
                        TextColor(Color::WHITE),
                    ));
                });
        }
    });
}
