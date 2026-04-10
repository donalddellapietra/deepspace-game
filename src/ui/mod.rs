use bevy::prelude::*;

use crate::block::BlockType;
use crate::editor::{Hotbar, HotbarItem};
use crate::model::ModelRegistry;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (spawn_hotbar, spawn_mode_indicator))
            .add_systems(Update, (update_hotbar, update_mode_indicator));
    }
}

#[derive(Component)]
struct HotbarSlotUi(u8);
#[derive(Component)]
struct HotbarLabel;
#[derive(Component)]
struct ModeIndicator;

fn spawn_hotbar(mut commands: Commands) {
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(20.0), left: Val::Percent(50.0),
            margin: UiRect { left: Val::Px(-220.0), ..default() },
            flex_direction: FlexDirection::Column, align_items: AlignItems::Center,
            row_gap: Val::Px(6.0), ..default()
        })
        .with_children(|p| {
            p.spawn((HotbarLabel, Text::new("Stone"),
                TextFont { font_size: 16.0, ..default() }, TextColor(Color::WHITE)));
            p.spawn(Node { flex_direction: FlexDirection::Row, column_gap: Val::Px(4.0), ..default() })
                .with_children(|row| {
                    for i in 0..10u8 {
                        let bt = BlockType::ALL[i as usize];
                        row.spawn((
                            HotbarSlotUi(i),
                            Node { width: Val::Px(40.0), height: Val::Px(40.0),
                                border: UiRect::all(Val::Px(2.0)), ..default() },
                            BackgroundColor(bt.color().into()),
                            BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.2)),
                        ));
                    }
                });
            p.spawn((
                Text::new("E: inventory | F: drill in | Q: drill out | LClick: break | RClick: place | 1-0: hotbar"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.5)),
            ));
        });
}

fn spawn_mode_indicator(mut commands: Commands) {
    commands.spawn((
        ModeIndicator,
        Node { position_type: PositionType::Absolute, top: Val::Px(16.0), left: Val::Px(16.0),
            padding: UiRect::all(Val::Px(8.0)), ..default() },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)),
        Text::new("Top Layer"), TextFont { font_size: 20.0, ..default() }, TextColor(Color::WHITE),
    ));
}

fn update_hotbar(
    hotbar: Res<Hotbar>,
    registry: Res<ModelRegistry>,
    mut slots: Query<(&HotbarSlotUi, &mut BorderColor, &mut BackgroundColor)>,
    mut label: Query<&mut Text, With<HotbarLabel>>,
) {
    for (slot, mut border, mut bg) in &mut slots {
        let i = slot.0 as usize;
        let is_active = i == hotbar.active;

        // Update the slot's color based on what's in it
        match &hotbar.slots[i] {
            HotbarItem::Block(bt) => {
                bg.0 = bt.color();
            }
            HotbarItem::SavedModel(idx) => {
                // Use a representative color from the model
                let color = registry.models.get(*idx)
                    .and_then(|m| m.blocks.iter().flatten().flatten().find_map(|b| *b))
                    .map(|bt| bt.color())
                    .unwrap_or(Color::srgb(0.4, 0.4, 0.4));
                bg.0 = color;
            }
        }

        *border = if is_active {
            BorderColor::all(Color::WHITE)
        } else {
            BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.15))
        };
    }

    // Update label with active item name
    if let Ok(mut t) = label.single_mut() {
        let name = match hotbar.active_item() {
            HotbarItem::Block(bt) => format!("{:?}", bt),
            HotbarItem::SavedModel(idx) => {
                registry.models.get(*idx)
                    .map(|m| m.name.clone())
                    .unwrap_or("Unknown".into())
            }
        };
        *t = Text::new(name);
    }
}

fn update_mode_indicator(
    state: Res<crate::world::WorldState>,
    mut q: Query<&mut Text, With<ModeIndicator>>,
) {
    let Ok(mut t) = q.single_mut() else { return };
    if state.is_top_layer() {
        *t = Text::new("Top Layer (F to drill in)");
    } else {
        *t = Text::new(format!("Depth {} (Q to exit)", state.depth()));
    }
}
