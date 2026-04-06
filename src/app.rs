use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
    mpsc::{Receiver, Sender},
};

use pollster::block_on;
use winit::{
    application::ApplicationHandler,
    dpi::{PhysicalPosition, PhysicalSize},
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    event_loop::{ActiveEventLoop, ControlFlow},
    window::{WindowAttributes, WindowId},
};

use crate::{
    gpu::{RenderOutcome, RendererState},
    math::ViewportState,
    render::{RenderMessage, RenderRequest, build_preview_frame, spawn_render_worker},
};

const INITIAL_WIDTH: u32 = 1280;
const INITIAL_HEIGHT: u32 = 720;

#[derive(Default)]
pub struct FractalApp {
    renderer: Option<RendererState>,
    viewport: Option<ViewportState>,
    cursor_position: Option<PhysicalPosition<f64>>,
    render_tx: Option<Sender<RenderRequest>>,
    render_rx: Option<Receiver<RenderMessage>>,
    latest_requested_generation: Option<Arc<AtomicU64>>,
    next_generation: u64,
    displayed_generation: u64,
    displayed_viewport: Option<ViewportState>,
    displayed_pixels: Vec<u8>,
    is_dragging: bool,
    drag_last_cursor: Option<PhysicalPosition<f64>>,
}

impl FractalApp {
    fn request_render(&mut self) {
        let Some(viewport) = self.viewport.as_ref() else {
            return;
        };
        let Some(render_tx) = self.render_tx.as_ref() else {
            return;
        };
        let Some(latest_requested_generation) = self.latest_requested_generation.as_ref() else {
            return;
        };

        let generation = self.next_generation;
        latest_requested_generation.store(generation, Ordering::Relaxed);
        if render_tx
            .send(RenderRequest::from_viewport(viewport, generation))
            .is_ok()
        {
            self.next_generation += 1;
        }
    }

    fn start_preview_and_render(&mut self, previous_viewport: Option<ViewportState>) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        let Some(viewport) = self.viewport.as_ref() else {
            return;
        };

        let preview = build_preview_frame(
            &self.displayed_pixels,
            previous_viewport.as_ref(),
            viewport,
        );
        self.displayed_pixels = preview;
        self.displayed_viewport = Some(viewport.clone());
        self.displayed_generation = self.next_generation;
        renderer.upload_full(viewport.width, viewport.height, &self.displayed_pixels);
        renderer.window.request_redraw();
        self.request_render();
    }

    fn drain_render_messages(&mut self) -> bool {
        let Some(render_rx) = self.render_rx.as_ref() else {
            return false;
        };
        let Some(renderer) = self.renderer.as_mut() else {
            return false;
        };

        let mut has_updates = false;
        while let Ok(message) = render_rx.try_recv() {
            match message {
                RenderMessage::Tile {
                    generation,
                    x,
                    y,
                    width,
                    height,
                    pixels,
                } => {
                    if generation == self.displayed_generation {
                        blit_tile_into_framebuffer(
                            &mut self.displayed_pixels,
                            self.displayed_viewport
                                .as_ref()
                                .map(|viewport| viewport.width)
                                .unwrap_or(width),
                            x,
                            y,
                            width,
                            height,
                            &pixels,
                        );
                        renderer.upload_region(x, y, width, height, &pixels);
                        has_updates = true;
                    }
                }
            }
        }

        has_updates
    }
}

impl ApplicationHandler for FractalApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        event_loop.set_control_flow(ControlFlow::Wait);

        if self.renderer.is_some() {
            return;
        }

        let window = Arc::new(
            event_loop
                .create_window(
                    WindowAttributes::default()
                        .with_title("fractals-rs | Mandelbrot")
                        .with_inner_size(PhysicalSize::new(INITIAL_WIDTH, INITIAL_HEIGHT)),
                )
                .expect("failed to create window"),
        );

        let renderer = block_on(RendererState::new(window));
        let viewport = ViewportState::new(renderer.config.width, renderer.config.height);
        renderer.window.set_title(&viewport.describe());

        let (render_tx, render_rx, latest_requested_generation) = spawn_render_worker();
        self.cursor_position = Some(PhysicalPosition::new(
            renderer.config.width as f64 * 0.5,
            renderer.config.height as f64 * 0.5,
        ));
        self.next_generation = 1;
        self.displayed_generation = 0;
        self.renderer = Some(renderer);
        self.viewport = Some(viewport);
        self.render_tx = Some(render_tx);
        self.render_rx = Some(render_rx);
        self.latest_requested_generation = Some(latest_requested_generation);
        self.start_preview_and_render(None);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(renderer) = self.renderer.as_mut() else {
            return;
        };
        if window_id != renderer.window.id() {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if size.width > 0 && size.height > 0 {
                    let previous_viewport = self.viewport.clone();
                    renderer.resize(size);
                    if let Some(viewport) = self.viewport.as_mut() {
                        viewport.update_size(size.width, size.height);
                        renderer.window.set_title(&viewport.describe());
                    }
                    self.start_preview_and_render(previous_viewport);
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if self.is_dragging {
                    if let Some(last) = self.drag_last_cursor {
                        let delta_x = position.x - last.x;
                        let delta_y = position.y - last.y;
                        let previous_viewport = self.viewport.clone();
                        if let Some(viewport) = self.viewport.as_mut() {
                            viewport.pan_by_pixels(delta_x, delta_y);
                            renderer.window.set_title(&viewport.describe());
                        }
                        self.start_preview_and_render(previous_viewport);
                    }
                    self.drag_last_cursor = Some(position);
                }
                self.cursor_position = Some(position);
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => match state {
                ElementState::Pressed => {
                    self.is_dragging = true;
                    self.drag_last_cursor = self.cursor_position;
                }
                ElementState::Released => {
                    self.is_dragging = false;
                    self.drag_last_cursor = None;
                }
            },
            WindowEvent::MouseWheel { delta, .. } => {
                let steps = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y as f64,
                    MouseScrollDelta::PixelDelta(pos) => pos.y / 60.0,
                };

                if steps.abs() > f64::EPSILON {
                    let cursor = self.cursor_position.unwrap_or(PhysicalPosition::new(
                        renderer.config.width as f64 * 0.5,
                        renderer.config.height as f64 * 0.5,
                    ));
                    let previous_viewport = self.viewport.clone();
                    if let Some(viewport) = self.viewport.as_mut() {
                        viewport.zoom_at_cursor(0.85_f64.powf(steps), cursor);
                        renderer.window.set_title(&viewport.describe());
                    }
                    self.start_preview_and_render(previous_viewport);
                }
            }
            WindowEvent::RedrawRequested => match renderer.render() {
                RenderOutcome::Ok | RenderOutcome::Skipped => {}
                RenderOutcome::NeedsReconfigure => renderer.resize(renderer.window.inner_size()),
            },
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if self.drain_render_messages() {
            if let Some(renderer) = self.renderer.as_ref() {
                renderer.window.request_redraw();
            }
        }
    }
}

fn blit_tile_into_framebuffer(
    framebuffer: &mut [u8],
    framebuffer_width: u32,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    tile_pixels: &[u8],
) {
    for row in 0..height as usize {
        let dst_start = (((y as usize + row) * framebuffer_width as usize + x as usize) * 4) as usize;
        let src_start = row * width as usize * 4;
        let byte_len = width as usize * 4;
        framebuffer[dst_start..dst_start + byte_len]
            .copy_from_slice(&tile_pixels[src_start..src_start + byte_len]);
    }
}
