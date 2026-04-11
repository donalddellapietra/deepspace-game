//! Minecraft-style inventory panel.
//!
//! Press E to open/close. Shows all block types. Click a block to
//! assign it to the active hotbar slot.

use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};

use crate::block::BlockType;
use crate::camera::CursorLocked;
use crate::editor::{Hotbar, HotbarItem};

pub struct InventoryPlugin;

impl Plugin for InventoryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InventoryState>()
            .add_systems(Update, (toggle_inventory, handle_clicks));
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
    mut hotbar: ResMut<Hotbar>,
    mut block_slots: Query<(&Interaction, &BlockSlot, &mut BorderColor), Changed<Interaction>>,
) {
    if !state.open { return }

    let active_slot = hotbar.active;

    for (interaction, slot, mut border) in &mut block_slots {
        match interaction {
            Interaction::Pressed => {
                if let Some(bt) = BlockType::from_index(slot.0) {
                    hotbar.slots[active_slot] = HotbarItem::Block(bt);
                }
            }
            Interaction::Hovered => *border = BorderColor::all(Color::WHITE),
            Interaction::None => *border = BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.2)),
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

            // Hints
            panel.spawn((
                Text::new("E: close | F: reset player"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.4)),
            ));
        })
        .id()
}
