//! Transient on-screen notifications ("toasts").
//!
//! Bevy has no built-in toast primitive, so we roll a minimal one: a
//! `Toast` component wraps a `Timer`, a system ticks it each frame
//! and despawns the entity when the timer finishes, fading background
//! and text alpha on the way out. Call [`show_toast`] from any system
//! that has `Commands` to pop one up.

use bevy::prelude::*;

const TOAST_LIFETIME_SECS: f32 = 1.5;
const TOAST_FADE_START: f32 = 0.4; // fraction of lifetime at which fade begins

/// Attached to the toast's root node. The child `Text` inherits the
/// lifetime via a second query in [`tick_toasts`].
#[derive(Component)]
pub struct Toast {
    remaining: Timer,
}

/// Marker on the text child so we can fade its colour independently
/// of the parent `BackgroundColor`.
#[derive(Component)]
pub struct ToastText;

/// Spawn a toast with the given message. Position is fixed to the
/// top-right of the window; the toast plugin ticks and despawns it
/// automatically.
pub fn show_toast(commands: &mut Commands, text: impl Into<String>) {
    let text = text.into();
    commands
        .spawn((
            Toast {
                remaining: Timer::from_seconds(TOAST_LIFETIME_SECS, TimerMode::Once),
            },
            Node {
                position_type: PositionType::Absolute,
                top: Val::Px(80.0),
                right: Val::Px(24.0),
                padding: UiRect::all(Val::Px(14.0)),
                ..default()
            },
            BackgroundColor(Color::srgba(0.05, 0.15, 0.12, 0.9)),
            BorderColor::all(Color::srgba(0.4, 1.0, 0.9, 0.9)),
        ))
        .with_children(|parent| {
            parent.spawn((
                ToastText,
                Text::new(text),
                TextFont { font_size: 18.0, ..default() },
                TextColor(Color::srgba(0.6, 1.0, 0.95, 1.0)),
            ));
        });
}

pub fn tick_toasts(
    time: Res<Time>,
    mut commands: Commands,
    mut toasts: Query<(Entity, &mut Toast, &mut BackgroundColor, &Children)>,
    mut text_q: Query<&mut TextColor, With<ToastText>>,
) {
    for (entity, mut toast, mut bg, children) in &mut toasts {
        toast.remaining.tick(time.delta());
        if toast.remaining.is_finished() {
            commands.entity(entity).despawn();
            continue;
        }
        // Hold full opacity until `TOAST_FADE_START` of the way
        // through, then linearly fade to zero by the end. This keeps
        // the toast readable for most of its lifetime.
        let elapsed = toast.remaining.fraction();
        let alpha = if elapsed < TOAST_FADE_START {
            1.0
        } else {
            1.0 - ((elapsed - TOAST_FADE_START) / (1.0 - TOAST_FADE_START))
        };
        bg.0 = bg.0.with_alpha(0.9 * alpha);
        for child in children.iter() {
            if let Ok(mut tc) = text_q.get_mut(child) {
                tc.0 = tc.0.with_alpha(alpha);
            }
        }
    }
}
