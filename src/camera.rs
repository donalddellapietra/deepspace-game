use std::f32::consts::FRAC_PI_2;

use bevy::{
    core_pipeline::tonemapping::Tonemapping,
    input::mouse::AccumulatedMouseMotion,
    light::ShadowFilteringMethod,
    pbr::ScreenSpaceAmbientOcclusion,
    post_process::bloom::Bloom,
    prelude::*,
    render::view::{ColorGrading, ColorGradingGlobal, ColorGradingSection},
};

use bevy::pbr::{Atmosphere, AtmosphereSettings, ScatteringMedium};
use bevy::post_process::dof::{DepthOfField, DepthOfFieldMode};

use crate::player::{Player, PLAYER_HEIGHT};
use crate::world::view::{cell_size_at_layer, scale_for_layer, target_layer_for, WorldAnchor};
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
    /// Start a new transition. Each layer's cell size is computed with
    /// its OWN target norm, not the current anchor.norm, so the
    /// animation endpoint matches the post-transition steady state.
    pub fn start(&mut self, from_layer: u8, to_layer: u8) {
        let from_norm = scale_for_layer(target_layer_for(from_layer));
        let to_norm = scale_for_layer(target_layer_for(to_layer));
        self.active = Some(AnimatingZoom {
            from_cell_size: scale_for_layer(from_layer) / from_norm,
            to_cell_size: scale_for_layer(to_layer) / to_norm,
            t: 0.0,
        });
    }

    /// The interpolated cell size in normalized Bevy units.
    pub fn effective_cell_size(&self, current_layer: u8, anchor: &WorldAnchor) -> f32 {
        match &self.active {
            Some(anim) => {
                let t = anim.t.clamp(0.0, 1.0);
                let ease = t * t * (3.0 - 2.0 * t);
                anim.from_cell_size + (anim.to_cell_size - anim.from_cell_size) * ease
            }
            None => anchor.cell_bevy(current_layer),
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
                    sync_horizon_dof,
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
    let mut cam = commands.spawn((
        Camera3d::default(),
        FpsCam { yaw: 0.0, pitch: 0.0 },
        Transform::default(),
        Tonemapping::AcesFitted,
        ShadowFilteringMethod::Gaussian,
        // SSAO re-enabled. Creates a subtle dark halo at the terrain
        // clip boundary (depth discontinuity artifact). This is the
        // tradeoff: SSAO quality on terrain vs clean horizon edge.
        // The halo is smooth (thanks to shader clip) but visible.
        ScreenSpaceAmbientOcclusion {
            quality_level: bevy::pbr::ScreenSpaceAmbientOcclusionQualityLevel::High,
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

    let medium = scattering_mediums.add(ScatteringMedium::earthlike(256, 256));
    cam.insert((
        Atmosphere {
            ground_albedo: Vec3::new(0.3, 0.6, 0.2),
            ..Atmosphere::earthlike(medium)
        },
        // DOF blurs the horizon where imposters meet sky.
        DepthOfField {
            mode: DepthOfFieldMode::Gaussian,
            focal_distance: 200.0,
            sensor_height: 10.0,
            aperture_f_stops: 1.0,
            max_circle_of_confusion_diameter: 16.0,
            max_depth: 1000.0,
        },
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
    anchor: Res<WorldAnchor>,
    transition: Res<ZoomTransition>,
    player_q: Query<&Transform, (With<Player>, Without<FpsCam>)>,
    mut cam_q: Query<(&mut Transform, &mut FpsCam), Without<Player>>,
) {
    let Ok(player_tf) = player_q.single() else { return };
    let Ok((mut cam_tf, mut cam)) = cam_q.single_mut() else { return };

    if locked.0 {
        cam.yaw -= motion.delta.x * SENSITIVITY;
        cam.pitch = (cam.pitch + motion.delta.y * SENSITIVITY)
            .clamp(-FRAC_PI_2 + 0.05, FRAC_PI_2 - 0.05);
    }

    let cell = transition.effective_cell_size(zoom.layer, &anchor);
    cam_tf.translation = player_tf.translation + Vec3::Y * (PLAYER_HEIGHT * cell);
    cam_tf.rotation = Quat::from_euler(EulerRot::YXZ, cam.yaw, -cam.pitch, 0.0);
}

/// Keep the aerial-view LUT range in sync with the actual view
/// distance so atmospheric fog distributes evenly across all visible
/// chunks (avoiding banding at chunk boundaries).
fn sync_atmosphere_scale(
    zoom: Res<CameraZoom>,
    anchor: Res<WorldAnchor>,
    mut cam_q: Query<&mut AtmosphereSettings, With<FpsCam>>,
) {
    if !zoom.is_changed() {
        return;
    }
    let cell_bevy = anchor.cell_bevy(zoom.layer);
    let view_radius = crate::world::render::RADIUS_VIEW_CELLS * cell_bevy;
    for mut settings in &mut cam_q {
        // With coordinate normalization, Bevy-space distances are
        // bounded (~800 units) at all zoom levels. 1 Bevy unit ≈
        // 1 target-layer voxel. scene_units_to_m converts to meters
        // for the atmosphere — divide by cell_bevy so the atmosphere
        // sees a consistent ~2m camera altitude regardless of zoom.
        settings.scene_units_to_m = 1.0 / cell_bevy;
        settings.aerial_view_lut_max_distance = view_radius / cell_bevy;
    }
}

/// Sync DOF so the far edge of the imposter ring blurs into the sky.
fn sync_horizon_dof(
    zoom: Res<CameraZoom>,
    anchor: Res<WorldAnchor>,
    mut cam_q: Query<&mut DepthOfField, With<FpsCam>>,
) {
    if !zoom.is_changed() {
        return;
    }
    let cell_bevy = anchor.cell_bevy(zoom.layer);
    let view_radius = crate::world::render::RADIUS_VIEW_CELLS * cell_bevy;
    // Focus at the inner terrain edge; imposters beyond blur into sky.
    for mut dof in &mut cam_q {
        dof.focal_distance = view_radius * 0.5;
        dof.max_depth = view_radius * 3.5;
    }
}
