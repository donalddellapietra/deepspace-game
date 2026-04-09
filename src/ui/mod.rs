use bevy::prelude::*;

use crate::block::BlockType;
use crate::editor::EditorState;
use crate::layer::GameLayer;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, spawn_hotbar)
            .add_systems(Update, update_hotbar);
    }
}

#[derive(Component)]
struct HotbarSlot(u8);

#[derive(Component)]
struct HotbarLabel;

fn spawn_hotbar(mut commands: Commands) {
    // Root container at bottom center
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            bottom: Val::Px(20.0),
            left: Val::Percent(50.0),
            margin: UiRect {
                left: Val::Px(-220.0),
                ..default()
            },
            flex_direction: FlexDirection::Column,
            align_items: AlignItems::Center,
            row_gap: Val::Px(6.0),
            ..default()
        })
        .with_children(|parent| {
            // Block name label
            parent.spawn((
                HotbarLabel,
                Text::new("Stone"),
                TextFont {
                    font_size: 16.0,
                    ..default()
                },
                TextColor(Color::WHITE),
            ));

            // Slot row
            parent
                .spawn(Node {
                    flex_direction: FlexDirection::Row,
                    column_gap: Val::Px(4.0),
                    ..default()
                })
                .with_children(|row| {
                    for (i, bt) in BlockType::ALL.iter().enumerate() {
                        row.spawn((
                            HotbarSlot(i as u8),
                            Node {
                                width: Val::Px(40.0),
                                height: Val::Px(40.0),
                                border: UiRect::all(Val::Px(2.0)),
                                ..default()
                            },
                            BackgroundColor(bt.color().into()),
                            BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.2)),
                        ));
                    }
                });

            // Key hint
            parent.spawn((
                Text::new("1-0: select  |  E: edit cell  |  LMB: place  |  RMB: remove"),
                TextFont {
                    font_size: 12.0,
                    ..default()
                },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.5)),
            ));
        });
}

fn update_hotbar(
    editor: Res<EditorState>,
    mut slots: Query<(&HotbarSlot, &mut BorderColor)>,
    mut label: Query<&mut Text, With<HotbarLabel>>,
) {
    let selected = editor.selected_block as u8;

    for (slot, mut border) in &mut slots {
        if slot.0 == selected {
            *border = BorderColor::all(Color::WHITE);
        } else {
            *border = BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.15));
        }
    }

    if let Ok(mut text) = label.single_mut() {
        *text = Text::new(format!("{:?}", editor.selected_block));
    }
}
