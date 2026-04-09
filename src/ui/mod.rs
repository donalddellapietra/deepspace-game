use bevy::prelude::*;

use crate::block::BlockType;
use crate::editor::EditorState;
use crate::layer::ActiveLayer;
use crate::model::ModelRegistry;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, (spawn_hotbar, spawn_mode_indicator, spawn_saved_panel))
            .add_systems(Update, (update_hotbar, update_mode_indicator, update_saved_panel));
    }
}

#[derive(Component)]
struct HotbarSlot(u8);
#[derive(Component)]
struct HotbarLabel;
#[derive(Component)]
struct ModeIndicator;
#[derive(Component)]
struct SavedPanel;
#[derive(Component)]
struct SavedCount;

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
                    for (i, bt) in BlockType::ALL.iter().enumerate() {
                        row.spawn((
                            HotbarSlot(i as u8),
                            Node { width: Val::Px(40.0), height: Val::Px(40.0),
                                border: UiRect::all(Val::Px(2.0)), ..default() },
                            BackgroundColor(bt.color().into()),
                            BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.2)),
                        ));
                    }
                });
            p.spawn((
                Text::new("E: drill in | Q: drill out | LClick: break | RClick: place | P: save | 1-0: select"),
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

fn spawn_saved_panel(mut commands: Commands) {
    commands
        .spawn((
            SavedPanel,
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(16.0), right: Val::Px(16.0),
                padding: UiRect::all(Val::Px(10.0)),
                flex_direction: FlexDirection::Column,
                row_gap: Val::Px(4.0),
                ..default()
            },
            BackgroundColor(Color::srgba(0.0, 0.0, 0.0, 0.4)),
        ))
        .with_children(|p| {
            p.spawn((
                Text::new("Saved Models"),
                TextFont { font_size: 14.0, ..default() },
                TextColor(Color::srgba(1.0, 1.0, 0.7, 0.9)),
            ));
            p.spawn((
                SavedCount,
                Text::new("(none yet — press P to save)"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(Color::srgba(1.0, 1.0, 1.0, 0.6)),
            ));
        });
}

fn update_hotbar(
    editor: Res<EditorState>,
    mut slots: Query<(&HotbarSlot, &mut BorderColor)>,
    mut label: Query<&mut Text, With<HotbarLabel>>,
) {
    let sel = editor.selected_block as u8;
    for (slot, mut b) in &mut slots {
        *b = if slot.0 == sel { BorderColor::all(Color::WHITE) }
             else { BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.15)) };
    }
    if let Ok(mut t) = label.single_mut() { *t = Text::new(format!("{:?}", editor.selected_block)); }
}

fn update_mode_indicator(
    active: Res<ActiveLayer>,
    mut q: Query<&mut Text, With<ModeIndicator>>,
) {
    let Ok(mut t) = q.single_mut() else { return };
    if active.is_top_layer() {
        *t = Text::new("Top Layer (E to drill in)");
    } else {
        *t = Text::new(format!("Depth {} (Q to exit)", active.nav_stack.len()));
    }
}

fn update_saved_panel(
    registry: Res<ModelRegistry>,
    mut q: Query<&mut Text, With<SavedCount>>,
) {
    let Ok(mut t) = q.single_mut() else { return };
    if registry.models.is_empty() {
        *t = Text::new("(none yet - press P to save)");
    } else {
        let names: Vec<String> = registry.models.iter().map(|m| m.name.clone()).collect();
        *t = Text::new(format!("{} saved: {}", names.len(), names.join(", ")));
    }
}
