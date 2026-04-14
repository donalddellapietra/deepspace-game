//! Keyboard state for the camera/player.

use winit::keyboard::KeyCode;

#[derive(Default)]
pub struct Keys {
    pub w: bool, pub a: bool, pub s: bool, pub d: bool,
    pub space: bool, pub shift: bool,
}

impl Keys {
    pub fn apply(&mut self, code: KeyCode, pressed: bool) {
        match code {
            KeyCode::KeyW => self.w = pressed,
            KeyCode::KeyA => self.a = pressed,
            KeyCode::KeyS => self.s = pressed,
            KeyCode::KeyD => self.d = pressed,
            KeyCode::Space => self.space = pressed,
            KeyCode::ShiftLeft => self.shift = pressed,
            _ => {}
        }
    }

    pub fn clear(&mut self) {
        *self = Keys::default();
    }
}
