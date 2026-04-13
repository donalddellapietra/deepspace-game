use std::f32::consts::FRAC_PI_2;

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    input::mouse::AccumulatedMouseMotion,
    light::ShadowFilteringMethod,
    pbr::{Atmosphere, AtmosphereSettings, ScreenSpaceAmbientOcclusion, ScatteringMedium},
    post_process::bloom::Bloom,
    prelude::*,
    render::view::{ColorGrading, ColorGradingGlobal, ColorGradingSection},
};

use crate::player::{Player, PLAYER_HEIGHT};
use crate::world::view::cell_size_at_layer;
use crate::world::CameraZoom;

// ------------------------------------------------- zoom transition

/// Duration of the camera height animation when switching layers.
const ZOOM_TRANSITION_SECS: f32 = 0.3;

/// Smooth camera animation state for layer transitions. When a zoom
/// happens, the layer switch and entity rebuild are instant, but the
/// camera height interpolates from the old cell size to the new one
/// over `ZOOM_TRANSITION_SECS` so the jump feels like a glide.
#[derive(Resource, Default)]
pub struct ZoomTransition {
    active: Option<AnimatingZoom>,
}

struct AnimatingZoom {
    from_cell_size: f32,
    to_cell_size: f32,
    t: f32,
}

impl ZoomTransition {
    /// Start a new transition. `from_layer` is the layer *before* the
    /// zoom; `to_layer` is the layer *after*.
    pub fn start(&mut self, from_layer: u8, to_layer: u8) {
        self.active = Some(AnimatingZoom {
            from_cell_size: cell_size_at_layer(from_layer),
            to_cell_size: cell_size_at_layer(to_layer),
            t: 0.0,
        });
    }

    /// The interpolated cell size, or the steady-state value when no
    /// transition is active.
    pub fn effective_cell_size(&self, current_layer: u8) -> f32 {
        match &self.active {
            Some(anim) => {
                // Smooth-step easing: 3t² - 2t³
                let t = anim.t.clamp(0.0, 1.0);
                let ease = t * t * (3.0 - 2.0 * t);
                anim.from_cell_size + (anim.to_cell_size - anim.from_cell_size) * ease
            }
            None => cell_size_at_layer(current_layer),
        }
    }
}

const SENSITIVITY: f32 = 0.003;

pub struct CameraPlugin;

impl Plugin for CameraPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(CursorLocked(false))
            .init_resource::<ZoomTransition>()
            .add_systems(Startup, (spawn_camera, spawn_crosshair))
            .add_systems(
                Update,
                (
                    tick_zoom_transition,
                    first_person_camera
                        .after(tick_zoom_transition)
                        .after(crate::player::derive_transforms)
                        .after(crate::ui::sync_cursor),
                    sync_atmosphere_scale,
                ),
            );
    }
}

#[derive(Component)]
pub struct FpsCam {
    pub yaw: f32,
    pub pitch: f32,
}

/// Whether the cursor is currently locked. Block interaction only fires when true.
#[derive(Resource)]
pub struct CursorLocked(pub bool);

fn spawn_camera(
    mut commands: Commands,
    mut scattering_mediums: ResMut<Assets<ScatteringMedium>>,
) {
    let medium = scattering_mediums.add(ScatteringMedium::earthlike(256, 256));

    commands.spawn((
        Camera3d::default(),
        FpsCam { yaw: 0.0, pitch: 0.0 },
        Transform::default(),
        Atmosphere {
            ground_albedo: Vec3::new(0.3, 0.6, 0.2),
            ..Atmosphere::earthlike(medium)
        },
        // AcesFitted: filmic S-curve that rolls off highlights and
        // lifts shadows, matching BSL's cinematic look. Previously
        // caused shadow acne with high lighting values — fixed by
        // rebalancing to HDR-appropriate intensities in setup_environment.
        Tonemapping::AcesFitted,
        ShadowFilteringMethod::Gaussian,
        ScreenSpaceAmbientOcclusion {
            quality_level: bevy::pbr::ScreenSpaceAmbientOcclusionQualityLevel::High,
            // Higher thickness = samples stay closer to the surface,
            // reducing noise on thin voxel geometry.
            constant_object_thickness: 4.0,
        },
        // Bloom only on bright highlights (threshold 0.8) — sky, sun
        // reflections, emissive blocks. Energy-conserving mode adds
        // glow without washing out. Reduced max_mip_dimension cuts
        // GPU passes for zoomed-out layers.
        Bloom {
            intensity: 0.2,
            low_frequency_boost: 0.5,
            low_frequency_boost_curvature: 0.95,
            high_pass_frequency: 1.0,
            prefilter: bevy::post_process::bloom::BloomPrefilter {
                threshold: 0.8,
                threshold_softness: 0.3,
            },
            composite_mode: bevy::post_process::bloom::BloomCompositeMode::EnergyConserving,
            max_mip_dimension: 256,
            ..default()
        },
        // BSL-style color grading: cool shadows, warm highlights.
        // Pushes shadowed regions toward blue and lit regions toward
        // golden amber — the signature BSL look.
        ColorGrading {
            global: ColorGradingGlobal {
                exposure: 0.0,
                temperature: 0.0,
                tint: 0.0,
                hue: 0.0,
                post_saturation: 1.0,
                midtones_range: 0.2..0.7,
            },
            shadows: ColorGradingSection {
                saturation: 1.1,
                contrast: 1.05,
                gamma: 1.0,
                gain: 1.0,
                lift: 0.01,
            },
            midtones: ColorGradingSection {
                saturation: 1.0,
                contrast: 1.0,
                gamma: 1.0,
                gain: 1.0,
                lift: 0.0,
            },
            highlights: ColorGradingSection {
                saturation: 1.0,
                contrast: 1.0,
                gamma: 1.0,
                gain: 1.0,
                lift: 0.0,
            },
        },
        Msaa::Off,
    ));
}

fn spawn_crosshair(mut commands: Commands) {
    commands.spawn((
        Node {
            position_type: PositionType::Absolute,
            left: Val::Percent(50.0), top: Val::Percent(50.0),
            width: Val::Px(4.0), height: Val::Px(4.0),
            margin: UiRect { left: Val::Px(-2.0), top: Val::Px(-2.0), ..default() },
            ..default()
        },
        BackgroundColor(Color::srgba(1.0, 1.0, 1.0, 0.8)),
    ));
}

/// Advance the zoom transition each frame.
fn tick_zoom_transition(time: Res<Time>, mut transition: ResMut<ZoomTransition>) {
    if let Some(anim) = &mut transition.active {
        anim.t += time.delta_secs() / ZOOM_TRANSITION_SECS;
        if anim.t >= 1.0 {
            transition.active = None;
        }
    }
}

fn first_person_camera(
    motion: Res<AccumulatedMouseMotion>,
    locked: Res<CursorLocked>,
    zoom: Res<CameraZoom>,
    transition: Res<ZoomTransition>,
    player_q: Query<&Transform, (With<Player>, Without<FpsCam>)>,
    mut cam_q: Query<(&mut Transform, &mut FpsCam), Without<Player>>,
) {
    let Ok(player_tf) = player_q.single() else { return };
    let Ok((mut cam_tf, mut cam)) = cam_q.single_mut() else { return };

    // Only rotate when cursor is locked
    if locked.0 {
        cam.yaw -= motion.delta.x * SENSITIVITY;
        cam.pitch = (cam.pitch + motion.delta.y * SENSITIVITY)
            .clamp(-FRAC_PI_2 + 0.05, FRAC_PI_2 - 0.05);
    }

    // Player.y = feet. Camera sits at the eye, but the eye height
    // scales with the view layer's cell size. During a zoom transition
    // the cell size is interpolated so the camera glides smoothly
    // between layers instead of snapping.
    let cell = transition.effective_cell_size(zoom.layer);
    cam_tf.translation = player_tf.translation + Vec3::Y * (PLAYER_HEIGHT * cell);
    cam_tf.rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, -cam.pitch, 0.0);
}

/// Keep the aerial-view LUT range in sync with the actual view
/// distance so atmospheric fog distributes evenly across all visible
/// chunks (avoiding banding at chunk boundaries).
fn sync_atmosphere_scale(
    zoom: Res<CameraZoom>,
    mut cam_q: Query<&mut AtmosphereSettings, With<FpsCam>>,
) {
    if !zoom.is_changed() {
        return;
    }
    let cell = cell_size_at_layer(zoom.layer);
    let view_radius = crate::world::render::RADIUS_VIEW_CELLS * cell;
    for mut settings in &mut cam_q {
        // At lower layers the camera is at PLAYER_HEIGHT * cell Bevy
        // units. Dividing by cell keeps the atmosphere seeing a
        // consistent ~2m altitude regardless of zoom, preventing
        // heavy blue scattering at zoomed-out layers.
        settings.scene_units_to_m = 1.0 / cell;
        settings.aerial_view_lut_max_distance = view_radius / cell;
    }
}
