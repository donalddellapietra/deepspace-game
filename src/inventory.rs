//! Minecraft-style inventory panel.
//!
//! Press E to open/close. Shows all block types and saved models.
//! Click a block to select it. Click a saved model to set it for placement.

use bevy::prelude::*;
use bevy::window::{CursorGrabMode, CursorOptions};

use crate::block::BlockType;
use crate::camera::CursorLocked;
use crate::editor::{Hotbar, HotbarItem};
use crate::model::ModelRegistry;

pub struct InventoryPlugin;

impl Plugin for InventoryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InventoryState>()
            .add_systems(Update, (toggle_inventory, handle_clicks, refresh_saved_models));
    }
}

#[derive(Resource)]
pub struct InventoryState {
    pub open: bool,
    /// How many models were shown last refresh. Triggers rebuild when changed.
    last_model_count: usize,
    /// Root entity for the whole inventory panel.
    root: Option<Entity>,
}

impl Default for InventoryState {
    fn default() -> Self {
        Self { open: false, last_model_count: 0, root: None }
    }
}

// --- Marker components ---
#[derive(Component)]
struct InventoryRoot;

/// A block type slot. Index 0-9 maps to BlockType::ALL.
#[derive(Component)]
struct BlockSlot(u8);

/// A saved model slot. Index into ModelRegistry.models.
#[derive(Component)]
struct ModelSlot(usize);

/// Container for saved model buttons (rebuilt when model count changes).
#[derive(Component)]
struct SavedModelsContainer;

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

        // Spawn the inventory panel if it doesn't exist
        if state.root.is_none() {
            let root = spawn_panel(&mut commands);
            state.root = Some(root);
            state.last_model_count = 0; // force refresh of saved models
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
    mut block_slots: Query<(&Interaction, &BlockSlot, &mut BorderColor), (Changed<Interaction>, Without<ModelSlot>)>,
    mut model_slots: Query<(&Interaction, &ModelSlot, &mut BorderColor), (Changed<Interaction>, Without<BlockSlot>)>,
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

    for (interaction, slot, mut border) in &mut model_slots {
        match interaction {
            Interaction::Pressed => {
                hotbar.slots[active_slot] = HotbarItem::SavedModel(slot.0);
            }
            Interaction::Hovered => *border = BorderColor::all(Color::srgb(0.3, 1.0, 0.3)),
            Interaction::None => *border = BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.2)),
        }
    }
}

// ============================================================
// Dynamic refresh of saved models section
// ============================================================

fn refresh_saved_models(
    mut commands: Commands,
    mut state: ResMut<InventoryState>,
    registry: Res<ModelRegistry>,
    container_q: Query<Entity, With<SavedModelsContainer>>,
) {
    if !state.open { return }
    if registry.models.len() == state.last_model_count { return }
    state.last_model_count = registry.models.len();

    // Despawn old container
    for entity in &container_q {
        commands.entity(entity).despawn();
    }

    // Spawn new container with current models
    let Some(root) = state.root else { return };

    let container = commands.spawn((
        SavedModelsContainer,
        Node {
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            row_gap: Val::Px(8.0),
            width: Val::Percent(100.0),
            ..default()
        },
    )).with_children(|parent| {
        parent.spawn((
            Text::new(format!("SAVED MODELS ({})", registry.models.len())),
            TextFont { font_size: 16.0, ..default() },
            TextColor(Color::srgb(0.7, 1.0, 0.7)),
        ));

        if registry.models.is_empty() {
            parent.spawn((
                Text::new("Press P while editing to save a model"),
                TextFont { font_size: 11.0, ..default() },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.4)),
            ));
        } else {
            // Row of saved model buttons
            parent.spawn(Node {
                flex_direction: FlexDirection::Row,
                column_gap: Val::Px(8.0),
                flex_wrap: FlexWrap::Wrap,
                justify_content: JustifyContent::Center,
                ..default()
            }).with_children(|row| {
                for (i, model) in registry.models.iter().enumerate() {
                    // Use the model's dominant block color
                    let color = model.blocks.iter().flatten().flatten()
                        .find_map(|b| *b)
                        .map(|bt| bt.color())
                        .unwrap_or(Color::srgb(0.3, 0.3, 0.3));

                    row.spawn((
                        ModelSlot(i),
                        Button,
                        Node {
                            width: Val::Px(70.0),
                            height: Val::Px(50.0),
                            border: UiRect::all(Val::Px(2.0)),
                            justify_content: JustifyContent::Center,
                            align_items: AlignItems::Center,
                            ..default()
                        },
                        BackgroundColor(color.into()),
                        BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.2)),
                    )).with_children(|btn| {
                        btn.spawn((
                            Text::new(&model.name),
                            TextFont { font_size: 9.0, ..default() },
                            TextColor(Color::WHITE),
                        ));
                    });
                }
            });
        }
    }).id();

    commands.entity(root).add_child(container);
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

            // Saved models section gets added dynamically by refresh_saved_models

            // Hints
            panel.spawn((
                Text::new("E: close | F: drill in | Q: drill out | P: save model"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.4)),
            ));
        })
        .id()
}
