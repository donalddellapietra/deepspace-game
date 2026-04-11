use bevy::prelude::*;

use crate::block::BlockType;
use crate::editor::{Hotbar, HotbarItem};
use crate::world::CameraZoom;

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
                Text::new("E: inventory | F: reset | LClick: break | RClick: place | 1-0: hotbar"),
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
        Text::new("Layer 0"), TextFont { font_size: 20.0, ..default() }, TextColor(Color::WHITE),
    ));
}

fn update_hotbar(
    hotbar: Res<Hotbar>,
    mut slots: Query<(&HotbarSlotUi, &mut BorderColor, &mut BackgroundColor)>,
    mut label: Query<&mut Text, With<HotbarLabel>>,
) {
    for (slot, mut border, mut bg) in &mut slots {
        let i = slot.0 as usize;
        let is_active = i == hotbar.active;

        match &hotbar.slots[i] {
            HotbarItem::Block(bt) => {
                bg.0 = bt.color();
            }
        }

        *border = if is_active {
            BorderColor::all(Color::WHITE)
        } else {
            BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.15))
        };
    }

    if let Ok(mut t) = label.single_mut() {
        let name = match hotbar.active_item() {
            HotbarItem::Block(bt) => format!("{:?}", bt),
        };
        *t = Text::new(name);
    }
}

fn update_mode_indicator(
    zoom: Res<CameraZoom>,
    mut q: Query<&mut Text, With<ModeIndicator>>,
) {
    let Ok(mut t) = q.single_mut() else { return };
    *t = Text::new(format!("Layer {}", zoom.layer));
}
