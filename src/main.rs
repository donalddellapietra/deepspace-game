//! Deep Space — ray-marched voxel engine.

use winit::event_loop::EventLoop;

use deepspace_game::app::{test_runner, App, TestConfig};

fn main() {
    env_logger::init();
    let test_cfg = TestConfig::from_args();
    if test_cfg.use_render_harness() {
        test_runner::run_render_harness(test_cfg).unwrap();
        return;
    }
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::with_test_config(test_cfg);
    event_loop.run_app(&mut app).unwrap();
}
