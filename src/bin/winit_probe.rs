use std::sync::Arc;
use std::time::Duration;

use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
#[cfg(target_os = "macos")]
use winit::platform::macos::{ActivationPolicy, EventLoopBuilderExtMacOS};
use winit::platform::pump_events::{EventLoopExtPumpEvents, PumpStatus};
use winit::window::{Window, WindowAttributes, WindowId};

struct ProbeApp {
    window: Option<Arc<Window>>,
}

impl ApplicationHandler for ProbeApp {
    fn new_events(&mut self, _event_loop: &ActiveEventLoop, cause: winit::event::StartCause) {
        eprintln!("probe: new_events cause={cause:?}");
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        eprintln!("probe: resumed");
        if self.window.is_none() {
            let attrs = WindowAttributes::default()
                .with_title("Probe")
                .with_inner_size(winit::dpi::LogicalSize::new(640, 480));
            let window = Arc::new(event_loop.create_window(attrs).unwrap());
            self.window = Some(window);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        eprintln!("probe: about_to_wait");
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        eprintln!("probe: window_event {event:?}");
        if matches!(event, WindowEvent::RedrawRequested) {
            event_loop.exit();
        }
    }
}

fn main() {
    eprintln!("probe: begin");
    let event_loop = {
        let mut builder = EventLoop::builder();
        #[cfg(target_os = "macos")]
        {
            builder.with_activation_policy(ActivationPolicy::Regular);
            builder.with_activate_ignoring_other_apps(false);
        }
        builder.build().unwrap()
    };
    eprintln!("probe: event_loop_created");
    let mut event_loop = event_loop;
    let mut app = ProbeApp { window: None };
    for tick in 0..300u32 {
        eprintln!("probe: pump tick={tick}");
        match event_loop.pump_app_events(Some(Duration::ZERO), &mut app) {
            PumpStatus::Continue => {}
            PumpStatus::Exit(code) => {
                eprintln!("probe: pump_exit code={code}");
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(16));
    }
    eprintln!("probe: pump_loop_returned");
}
