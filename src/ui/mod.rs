use bevy::prelude::*;

use crate::block::BlockType;
use crate::editor::EditorState;
use crate::layer::GameLayer;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (spawn_hotbar, spawn_mode_indicator))
            .add_systems(Update, (update_hotbar, update_mode_indicator));
    }
}

#[derive(Component)]
struct HotbarSlot(u8);

#[derive(Component)]
struct HotbarLabel;

#[derive(Component)]
struct ModeIndicator;

fn spawn_hotbar(mut commands: Commands) {
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
            parent.spawn((
                HotbarLabel,
                Text::new("Stone"),
                TextFont { font_size: 16.0, ..default() },
                TextColor(Color::WHITE),
            ));

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

            parent.spawn((
                Text::new("E: edit cell | Q: exit edit | LClick: break | RClick: place | 1-0: select"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.5)),
            ));
        });
}

fn spawn_mode_indicator(mut commands: Commands) {
    commands.spawn((
        ModeIndicator,
        Node {
            position_type: PositionType::Absolute,
            top: Val::Px(16.0),
            left: Val::Px(16.0),
            padding: UiRect::all(Val::Px(8.0)),
            ..default()
        },
        BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.5)),
        Text::new("WORLD MODE"),
        TextFont { font_size: 20.0, ..default() },
        TextColor(Color::WHITE),
    ));
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

fn update_mode_indicator(
    state: Res<State<GameLayer>>,
    mut indicator: Query<(&mut Text, &mut TextColor), With<ModeIndicator>>,
) {
    let Ok((mut text, mut color)) = indicator.single_mut() else { return };

    match state.get() {
        GameLayer::World => {
            *text = Text::new("WORLD MODE");
            color.0 = Color::WHITE;
        }
        GameLayer::Editing => {
            *text = Text::new("EDITING MODE");
            color.0 = Color::srgb(0.3, 1.0, 0.3);
        }
    }
}
