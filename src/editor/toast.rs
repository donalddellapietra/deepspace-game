//! Toast notifications — forwarded to the React overlay.
//!
//! Call [`show_toast`] from any system that has access to a
//! `MessageWriter<ToastEvent>`. The overlay plugin picks up the message
//! and pushes it to the React UI.

use bevy::prelude::*;

use crate::overlay::ToastEvent;

/// Fire a toast via the React overlay.
pub fn show_toast(writer: &mut MessageWriter<ToastEvent>, text: impl Into<String>) {
    writer.write(ToastEvent { text: text.into() });
}
