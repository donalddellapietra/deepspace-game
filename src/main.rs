//! Deep Space — ray-marched voxel engine.

use winit::event_loop::EventLoop;

use deepspace_game::app::{test_runner, App, TestConfig, UserEvent};

fn main() {
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        let _ = console_log::init_with_level(log::Level::Info);
    }
    #[cfg(not(target_arch = "wasm32"))]
    env_logger::init();

    eprintln!("startup_perf main: begin");
    let test_cfg = TestConfig::from_args();
    eprintln!("startup_perf main: parsed_args");

    #[cfg(not(target_arch = "wasm32"))]
    if test_cfg.use_render_harness() {
        test_runner::run_render_harness(test_cfg).unwrap();
        return;
    }

    let event_loop = EventLoop::<UserEvent>::with_user_event().build().unwrap();
    eprintln!("startup_perf main: event_loop_created");
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    eprintln!("startup_perf main: control_flow_set");
    let mut app = App::with_test_config(test_cfg, event_loop.create_proxy());
    eprintln!("startup_perf main: app_constructed");
    event_loop.run_app(&mut app).unwrap();
    eprintln!("startup_perf main: run_app_returned");
}
