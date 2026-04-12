//! Color picker panel for creating custom-colored blocks.
//!
//! Press C to open/close. Uses `bevy_egui` for immediate-mode UI with
//! R/G/B sliders, a live preview swatch, and a "Create Block" button
//! that registers a new [`Palette`] entry.

use bevy::prelude::*;
use bevy_egui::egui;
use bevy_egui::EguiContexts;

use crate::block::{Palette, PaletteEntry};
use crate::inventory::InventoryState;

// ── Plugin ─────────────────────────────────────────────────────────

pub struct ColorPickerPlugin;

impl Plugin for ColorPickerPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ColorPickerState>()
            .add_systems(Update, (toggle_color_picker, draw_color_picker));
    }
}

// ── State ──────────────────────────────────────────────────────────

#[derive(Resource)]
pub struct ColorPickerState {
    pub open: bool,
    pub r: f32,
    pub g: f32,
    pub b: f32,
    custom_count: u32,
    /// True when egui wants pointer input (sliders being dragged, etc.)
    pub egui_wants_pointer: bool,
}

impl Default for ColorPickerState {
    fn default() -> Self {
        Self {
            open: false,
            r: 0.5,
            g: 0.5,
            b: 0.5,
            custom_count: 0,
            egui_wants_pointer: false,
        }
    }
}

impl ColorPickerState {
    pub fn current_color(&self) -> Color {
        Color::srgb(self.r, self.g, self.b)
    }

    fn next_name(&mut self) -> String {
        self.custom_count += 1;
        format!("Custom #{}", self.custom_count)
    }
}

// ── Toggle ─────────────────────────────────────────────────────────

fn toggle_color_picker(
    keyboard: Res<ButtonInput<KeyCode>>,
    inv: Res<InventoryState>,
    mut state: ResMut<ColorPickerState>,
) {
    if keyboard.just_pressed(KeyCode::KeyC) && !inv.open {
        state.open = !state.open;
    }
}

// ── egui panel ────────────────────────────────────────────────────

fn draw_color_picker(
    mut contexts: EguiContexts,
    mut state: ResMut<ColorPickerState>,
    mut palette: ResMut<Palette>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    if !state.open {
        return;
    }

    let Ok(ctx) = contexts.ctx_mut() else { return };

    // Update pointer capture state so game systems can yield to egui.
    state.egui_wants_pointer = ctx.wants_pointer_input();

    egui::Window::new("CREATE BLOCK")
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
        .default_width(320.0)
        .show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.label(
                    egui::RichText::new("Drag sliders to pick a color")
                        .color(egui::Color32::from_white_alpha(100))
                        .size(12.0),
                );
                ui.add_space(8.0);

                // ── Preview swatch ──
                let preview = egui::Color32::from_rgb(
                    (state.r * 255.0) as u8,
                    (state.g * 255.0) as u8,
                    (state.b * 255.0) as u8,
                );
                let (rect, _) =
                    ui.allocate_exact_size(egui::vec2(80.0, 80.0), egui::Sense::hover());
                ui.painter().rect_filled(rect, 8.0, preview);
                ui.painter().rect_stroke(
                    rect,
                    8.0,
                    egui::Stroke::new(2.0, egui::Color32::from_white_alpha(60)),
                    egui::StrokeKind::Outside,
                );

                let hex = format!(
                    "#{:02X}{:02X}{:02X}",
                    (state.r * 255.0).round() as u8,
                    (state.g * 255.0).round() as u8,
                    (state.b * 255.0).round() as u8,
                );
                ui.label(
                    egui::RichText::new(hex)
                        .color(egui::Color32::from_white_alpha(150))
                        .size(12.0),
                );
            });

            ui.add_space(8.0);

            // ── RGB Sliders ──
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("R").color(egui::Color32::from_rgb(255, 80, 80)));
                ui.add(egui::Slider::new(&mut state.r, 0.0..=1.0).show_value(false));
                ui.label(format!("{:.0}", state.r * 255.0));
            });
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("G").color(egui::Color32::from_rgb(80, 255, 80)));
                ui.add(egui::Slider::new(&mut state.g, 0.0..=1.0).show_value(false));
                ui.label(format!("{:.0}", state.g * 255.0));
            });
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new("B").color(egui::Color32::from_rgb(100, 150, 255)));
                ui.add(egui::Slider::new(&mut state.b, 0.0..=1.0).show_value(false));
                ui.label(format!("{:.0}", state.b * 255.0));
            });

            ui.add_space(8.0);

            // ── Create button ──
            ui.vertical_centered(|ui| {
                if ui
                    .add_sized(
                        [200.0, 36.0],
                        egui::Button::new(egui::RichText::new("Create Block").size(15.0)),
                    )
                    .clicked()
                {
                    let name = state.next_name();
                    let color = state.current_color();
                    palette.register(
                        PaletteEntry {
                            name: name.clone(),
                            color,
                            roughness: 0.9,
                            metallic: 0.0,
                            alpha_mode: AlphaMode::Opaque,
                        },
                        &mut materials,
                    );
                    info!("Created custom block: {name}");
                    state.open = false;
                }

                ui.add_space(4.0);
                ui.label(
                    egui::RichText::new("C: close")
                        .color(egui::Color32::from_white_alpha(100))
                        .size(11.0),
                );
            });
        });
}
