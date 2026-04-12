//! Inventory panel — press E to open/close.
//!
//! Shows all available blocks (built-in block types), saved meshes
//! for the current layer, and a "New Block" button that opens the
//! color picker.
//!
//! Palette-ready: when the Palette resource lands, replace
//! `BlockType::ALL` iteration with `palette.iter()` and split
//! entries into "Built-in" vs "Custom" sections.

use bevy::prelude::*;

use crate::block::{BlockType, Palette};
use crate::editor::save_mode::SavedMeshes;
use crate::editor::{Hotbar, HotbarItem};
use crate::world::view::target_layer_for;
use crate::world::CameraZoom;

// ── Design tokens ──────────────────────────────────────────────────

const PANEL_BG: Color = Color::srgba(0.06, 0.06, 0.10, 0.94);
const PANEL_BORDER: Color = Color::srgba(0.30, 0.55, 0.85, 0.25);
const ACCENT: Color = Color::srgba(0.50, 0.78, 1.0, 1.0);
const TEXT_PRIMARY: Color = Color::srgba(1.0, 1.0, 1.0, 1.0);
const TEXT_SECONDARY: Color = Color::srgba(1.0, 1.0, 1.0, 0.6);
const TEXT_HINT: Color = Color::srgba(1.0, 1.0, 1.0, 0.35);
const SLOT_BG_HOVER: Color = Color::srgba(1.0, 1.0, 1.0, 0.08);
const SLOT_BORDER: Color = Color::srgba(1.0, 1.0, 1.0, 0.12);
const SLOT_BORDER_HOVER: Color = Color::srgba(0.55, 0.82, 1.0, 0.8);
const MESH_ACCENT: Color = Color::srgba(0.30, 0.95, 0.80, 0.9);
const MESH_BORDER: Color = Color::srgba(0.35, 0.90, 0.80, 0.5);
const MESH_BG: Color = Color::srgba(0.15, 0.40, 0.35, 0.35);
const SECTION_LINE: Color = Color::srgba(1.0, 1.0, 1.0, 0.08);

// ── Plugin ─────────────────────────────────────────────────────────

pub struct InventoryPlugin;

impl Plugin for InventoryPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<InventoryState>().add_systems(
            Update,
            (
                toggle_inventory,
                handle_block_clicks,
                handle_mesh_clicks,
                refresh_saved_meshes,
                refresh_custom_blocks,
            ),
        );
    }
}

// ── State ──────────────────────────────────────────────────────────

#[derive(Resource, Default)]
pub struct InventoryState {
    pub open: bool,
    root: Option<Entity>,
}

// ── Marker components ──────────────────────────────────────────────

#[derive(Component)]
struct InventoryRoot;

/// A block type slot. Index maps to `BlockType::ALL`.
#[derive(Component)]
struct BlockSlot(u8);

/// A saved-mesh slot. The `usize` indexes into `SavedMeshes.items`.
#[derive(Component)]
struct MeshSlot(usize);

/// Parent node of the saved-mesh tiles; rebuilt on changes.
#[derive(Component)]
struct SavedMeshRow;

/// Text label above the saved-mesh row.
#[derive(Component)]
struct SavedMeshLabel;

/// Parent node of the custom-block tiles; rebuilt when Palette changes.
#[derive(Component)]
struct CustomBlockRow;

// ── Toggle ─────────────────────────────────────────────────────────

fn toggle_inventory(
    keyboard: Res<ButtonInput<KeyCode>>,
    mut state: ResMut<InventoryState>,
    mut commands: Commands,
    root_q: Query<Entity, With<InventoryRoot>>,
) {
    if !keyboard.just_pressed(KeyCode::KeyE) {
        return;
    }

    state.open = !state.open;

    if state.open {
        if state.root.is_none() {
            let root = spawn_panel(&mut commands);
            state.root = Some(root);
        } else if let Ok(entity) = root_q.single() {
            commands.entity(entity).insert(Visibility::Inherited);
        }
    } else if let Ok(entity) = root_q.single() {
        commands.entity(entity).insert(Visibility::Hidden);
    }
}

// ── Click handling ─────────────────────────────────────────────────

fn handle_block_clicks(
    state: Res<InventoryState>,
    zoom: Res<CameraZoom>,
    mut hotbar: ResMut<Hotbar>,
    mut block_slots: Query<
        (&Interaction, &BlockSlot, &mut BorderColor, &mut BackgroundColor),
        (Changed<Interaction>, Without<MeshSlot>),
    >,
) {
    if !state.open {
        return;
    }

    let active_slot = hotbar.active;
    let layer = zoom.layer;

    for (interaction, slot, mut border, mut bg) in &mut block_slots {
        match interaction {
            Interaction::Pressed => {
                // slot.0 is already a 1-based voxel index
                hotbar.slots_mut(layer)[active_slot] = HotbarItem::Block(slot.0);
            }
            Interaction::Hovered => {
                *border = BorderColor::all(SLOT_BORDER_HOVER);
                bg.0 = SLOT_BG_HOVER;
            }
            Interaction::None => {
                *border = BorderColor::all(SLOT_BORDER);
                bg.0 = Color::NONE;
            }
        }
    }
}

fn handle_mesh_clicks(
    state: Res<InventoryState>,
    zoom: Res<CameraZoom>,
    mut hotbar: ResMut<Hotbar>,
    mut mesh_slots: Query<
        (&Interaction, &MeshSlot, &mut BorderColor, &mut BackgroundColor),
        (Changed<Interaction>, Without<BlockSlot>),
    >,
) {
    if !state.open {
        return;
    }

    let active_slot = hotbar.active;
    let layer = zoom.layer;

    for (interaction, slot, mut border, mut bg) in &mut mesh_slots {
        match interaction {
            Interaction::Pressed => {
                hotbar.slots_mut(layer)[active_slot] = HotbarItem::Model(slot.0);
            }
            Interaction::Hovered => {
                *border = BorderColor::all(Color::srgba(0.40, 1.0, 0.90, 1.0));
                bg.0 = Color::srgba(0.15, 0.40, 0.35, 0.55);
            }
            Interaction::None => {
                *border = BorderColor::all(MESH_BORDER);
                bg.0 = MESH_BG;
            }
        }
    }
}

// ── Panel layout ───────────────────────────────────────────────────

fn spawn_panel(commands: &mut Commands) -> Entity {
    commands
        .spawn((
            InventoryRoot,
            Node {
                position_type: PositionType::Absolute,
                left: Val::Percent(50.0),
                top: Val::Percent(50.0),
                margin: UiRect {
                    left: Val::Px(-230.0),
                    top: Val::Px(-260.0),
                    ..default()
                },
                width: Val::Px(460.0),
                max_height: Val::Px(520.0),
                padding: UiRect::axes(Val::Px(24.0), Val::Px(20.0)),
                flex_direction: FlexDirection::Column,
                align_items: AlignItems::Center,
                row_gap: Val::Px(12.0),
                border: UiRect::all(Val::Px(1.0)),
                border_radius: BorderRadius::all(Val::Px(12.0)),
                overflow: Overflow::scroll_y(),
                ..default()
            },
            BackgroundColor(PANEL_BG),
            BorderColor::all(PANEL_BORDER),
            Visibility::Inherited,
        ))
        .with_children(|panel| {
            // ── Title ──
            panel.spawn((
                Text::new("INVENTORY"),
                TextFont { font_size: 22.0, ..default() },
                TextColor(ACCENT),
            ));
            panel.spawn((
                Text::new("Click a block to assign to active hotbar slot"),
                TextFont { font_size: 12.0, ..default() },
                TextColor(TEXT_HINT),
            ));

            // ── Section: Built-in Blocks ──
            spawn_section_header(panel, "Built-in Blocks");

            // Palette-ready: when Palette lands, iterate palette.iter()
            // and split into "Built-in" and "Custom" sections based on
            // index ranges or a flag on PaletteEntry.
            // Block grid: rows of 5
            let block_count = BlockType::ALL.len();
            let rows = (block_count + 4) / 5;
            for row_idx in 0..rows {
                let start = row_idx * 5;
                let end = (start + 5).min(block_count);

                panel
                    .spawn(Node {
                        flex_direction: FlexDirection::Row,
                        column_gap: Val::Px(6.0),
                        ..default()
                    })
                    .with_children(|row| {
                        for i in start..end {
                            let bt = BlockType::ALL[i];
                            let voxel = (i as u8) + 1; // 1-based
                            spawn_block_tile(row, voxel, bt.color(), &format!("{:?}", bt));
                        }
                    });
            }

            // ── Section: Custom Blocks ──
            spawn_section_divider(panel);
            spawn_section_header(panel, "Custom Blocks (C to create)");

            panel.spawn((
                CustomBlockRow,
                Node {
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: Val::Px(6.0),
                    row_gap: Val::Px(6.0),
                    max_width: Val::Px(412.0),
                    ..default()
                },
            ));

            // ── Section: Saved Meshes ──
            spawn_section_divider(panel);

            panel.spawn((
                SavedMeshLabel,
                Text::new("Saved Meshes (0)"),
                TextFont { font_size: 14.0, ..default() },
                TextColor(MESH_ACCENT),
            ));
            panel.spawn((
                SavedMeshRow,
                Node {
                    flex_direction: FlexDirection::Row,
                    flex_wrap: FlexWrap::Wrap,
                    column_gap: Val::Px(6.0),
                    row_gap: Val::Px(6.0),
                    max_width: Val::Px(412.0),
                    ..default()
                },
            ));

            // ── Footer hints ──
            spawn_section_divider(panel);
            panel.spawn((
                Text::new("E: close  |  1-0: select slot  |  Q/F: zoom  |  V: save mode"),
                TextFont { font_size: 10.0, ..default() },
                TextColor(TEXT_HINT),
            ));
        })
        .id()
}

/// Spawn a clickable block tile. `voxel` is the 1-based palette index.
fn spawn_block_tile(
    parent: &mut ChildSpawnerCommands,
    voxel: u8,
    color: Color,
    name: &str,
) {
    parent
        .spawn((
            BlockSlot(voxel),
            Button,
            Node {
                width: Val::Px(72.0),
                height: Val::Px(72.0),
                border: UiRect::all(Val::Px(2.0)),
                border_radius: BorderRadius::all(Val::Px(8.0)),
                padding: UiRect::all(Val::Px(4.0)),
                flex_direction: FlexDirection::Column,
                justify_content: JustifyContent::End,
                align_items: AlignItems::Center,
                ..default()
            },
            BackgroundColor(Color::NONE),
            BorderColor::all(SLOT_BORDER),
        ))
        .with_children(|tile| {
            // Color swatch
            tile.spawn((
                Node {
                    width: Val::Px(44.0),
                    height: Val::Px(32.0),
                    margin: UiRect::bottom(Val::Px(4.0)),
                    border: UiRect::all(Val::Px(1.0)),
                    border_radius: BorderRadius::all(Val::Px(4.0)),
                    ..default()
                },
                BackgroundColor(color),
                BorderColor::all(Color::srgba(1.0, 1.0, 1.0, 0.08)),
            ));
            // Name label
            tile.spawn((
                Text::new(name.to_string()),
                TextFont { font_size: 10.0, ..default() },
                TextColor(TEXT_SECONDARY),
            ));
        });
}

fn spawn_section_header(parent: &mut ChildSpawnerCommands, title: &str) {
    parent.spawn((
        Text::new(title.to_uppercase()),
        TextFont { font_size: 11.0, ..default() },
        TextColor(TEXT_SECONDARY),
    ));
}

fn spawn_section_divider(parent: &mut ChildSpawnerCommands) {
    parent.spawn((
        Node {
            width: Val::Percent(90.0),
            height: Val::Px(1.0),
            margin: UiRect::axes(Val::ZERO, Val::Px(4.0)),
            ..default()
        },
        BackgroundColor(SECTION_LINE),
    ));
}

// ── Custom block refresh ──────────────────────────────────────────

/// Rebuild the custom blocks row whenever the Palette changes.
/// Custom entries are those beyond the 10 built-in blocks (voxel > 10).
fn refresh_custom_blocks(
    palette: Option<Res<Palette>>,
    mut commands: Commands,
    row_q: Query<Entity, With<CustomBlockRow>>,
    added_row: Query<(), Added<CustomBlockRow>>,
) {
    let Some(palette) = palette else { return };
    if !palette.is_changed() && added_row.is_empty() {
        return;
    }
    let Ok(row) = row_q.single() else { return };

    commands.entity(row).despawn_related::<Children>();
    let custom_entries: Vec<_> = palette
        .iter()
        .filter(|(voxel, _)| *voxel > BlockType::ALL.len() as u8)
        .collect();

    if custom_entries.is_empty() {
        commands.entity(row).with_children(|parent| {
            parent.spawn((
                Text::new("Press C to create custom blocks"),
                TextFont { font_size: 11.0, ..default() },
                TextColor(TEXT_HINT),
            ));
        });
    } else {
        commands.entity(row).with_children(|parent| {
            for (voxel, entry) in custom_entries {
                spawn_block_tile(parent, voxel, entry.color, &entry.name);
            }
        });
    }
}

// ── Saved mesh refresh ─────────────────────────────────────────────

fn refresh_saved_meshes(
    saved: Res<SavedMeshes>,
    zoom: Res<CameraZoom>,
    mut commands: Commands,
    row_q: Query<Entity, With<SavedMeshRow>>,
    added_row: Query<(), Added<SavedMeshRow>>,
    mut label_q: Query<&mut Text, With<SavedMeshLabel>>,
) {
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
            "Saved Meshes — Layer {} ({})",
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
                        width: Val::Px(54.0),
                        height: Val::Px(54.0),
                        border: UiRect::all(Val::Px(2.0)),
                        border_radius: BorderRadius::all(Val::Px(6.0)),
                        flex_direction: FlexDirection::Column,
                        justify_content: JustifyContent::Center,
                        align_items: AlignItems::Center,
                        row_gap: Val::Px(2.0),
                        ..default()
                    },
                    BackgroundColor(MESH_BG),
                    BorderColor::all(MESH_BORDER),
                ))
                .with_children(|tile| {
                    tile.spawn((
                        Text::new(format!("#{}", i)),
                        TextFont { font_size: 11.0, ..default() },
                        TextColor(MESH_ACCENT),
                    ));
                    tile.spawn((
                        Text::new(format!("L{}", mesh.layer)),
                        TextFont { font_size: 9.0, ..default() },
                        TextColor(TEXT_HINT),
                    ));
                });
        }
    });
}
