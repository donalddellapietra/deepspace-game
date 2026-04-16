//! Deep Space — ray-marched voxel engine.

use winit::event_loop::EventLoop;

use deepspace_game::app::{App, TestConfig, test_runner};

fn main() {
    env_logger::init();
    eprintln!("startup_perf main: begin");
    let test_cfg = TestConfig::from_args();
    eprintln!("startup_perf main: parsed_args");
    if test_cfg.use_render_harness() {
        test_runner::run_render_harness(test_cfg).unwrap();
        return;
    }
    let event_loop = EventLoop::new().unwrap();
    eprintln!("startup_perf main: event_loop_created");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    eprintln!("startup_perf main: control_flow_set");
    let mut app = App::with_test_config(test_cfg);
    eprintln!("startup_perf main: app_constructed");
    event_loop.run_app(&mut app).unwrap();
    eprintln!("startup_perf main: run_app_returned");
}
