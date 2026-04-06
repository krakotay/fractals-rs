mod app;
mod gpu;
mod math;
mod render;

use app::FractalApp;
use winit::event_loop::EventLoop;

fn main() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let mut app = FractalApp::default();
    event_loop.run_app(&mut app).expect("event loop failed");
}
