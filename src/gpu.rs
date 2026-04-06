use std::{borrow::Cow, sync::{Arc, mpsc}};

use winit::{dpi::PhysicalSize, window::Window};

use crate::render::GpuRenderParams;

const PRESENT_SHADER: &str = r#"
@group(0) @binding(0)
var fractal_tex: texture_2d<f32>;

@group(0) @binding(1)
var fractal_sampler: sampler;

struct VertexOut {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOut {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0),
    );

    var uvs = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(1.0, 0.0),
    );

    var out: VertexOut;
    out.position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
    out.uv = uvs[vertex_index];
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return textureSample(fractal_tex, fractal_sampler, in.uv);
}
"#;

const COMPUTE_SHADER: &str = r#"
struct Params {
    width: u32,
    height: u32,
    max_iterations: u32,
    strategy: u32,
    orbit_len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    center_x: f64,
    center_y: f64,
    scale: f64,
    _pad3: f64,
};

struct Seed {
    iteration: u32,
    dz_real: f64,
    dz_imag: f64,
};

@group(0) @binding(0)
var<uniform> params: Params;

@group(0) @binding(1)
var<storage, read_write> pixels: array<u32>;

@group(0) @binding(2)
var<storage, read_write> fallback: array<u32>;

@group(0) @binding(3)
var<storage, read> orbit_z_real: array<f64>;

@group(0) @binding(4)
var<storage, read> orbit_z_imag: array<f64>;

@group(0) @binding(5)
var<storage, read> orbit_z_norm_sqr: array<f64>;

@group(0) @binding(6)
var<storage, read> orbit_series_a_real: array<f64>;

@group(0) @binding(7)
var<storage, read> orbit_series_a_imag: array<f64>;

@group(0) @binding(8)
var<storage, read> orbit_series_b_real: array<f64>;

@group(0) @binding(9)
var<storage, read> orbit_series_b_imag: array<f64>;

fn complex_mul(a_real: f64, a_imag: f64, b_real: f64, b_imag: f64) -> vec2<f64> {
    return vec2<f64>(
        a_real * b_real - a_imag * b_imag,
        a_real * b_imag + a_imag * b_real,
    );
}

fn escape_color(iteration: u32, radius: f64) -> vec4<u32> {
    if iteration >= params.max_iterations {
        return vec4<u32>(7u, 10u, 18u, 255u);
    }

    let safe_radius = max(radius, 1.0000001);
    let smooth_iter = f64(iteration) + 1.0 - log(log(safe_radius)) / log(2.0);
    let t = clamp(smooth_iter / f64(params.max_iterations), 0.0, 1.0);

    let r = u32(round(9.0 * (1.0 - t) * t * t * t * 255.0));
    let g = u32(round(15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0));
    let b = u32(round(8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0));
    return vec4<u32>(r, g, b, 255u);
}

fn pack_rgba(color: vec4<u32>) -> u32 {
    return color.x | (color.y << 8u) | (color.z << 16u) | (color.w << 24u);
}

fn is_finite_f64(value: f64) -> bool {
    return value == value && abs(value) <= 1.7976931348623157e308;
}

fn render_fast(pixel_x: u32, pixel_y: u32) -> u32 {
    let half_width = f64(params.width) * 0.5;
    let half_height = f64(params.height) * 0.5;
    let cx = params.center_x + (f64(pixel_x) - half_width) * params.scale;
    let cy = params.center_y + (f64(pixel_y) - half_height) * params.scale;

    var zx: f64 = 0.0;
    var zy: f64 = 0.0;
    var zx2: f64 = 0.0;
    var zy2: f64 = 0.0;
    var iteration = 0u;

    loop {
        if zx2 + zy2 > 4.0 || iteration >= params.max_iterations {
            break;
        }

        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        zx2 = zx * zx;
        zy2 = zy * zy;
        iteration += 1u;
    }

    return pack_rgba(escape_color(iteration, sqrt(zx2 + zy2)));
}

fn perturbation_seed(delta_c_real: f64, delta_c_imag: f64) -> Seed {
    let delta_norm_sqr = delta_c_real * delta_c_real + delta_c_imag * delta_c_imag;
    if delta_norm_sqr == 0.0 {
        return Seed(0u, 0.0, 0.0);
    }

    let dc_sq = complex_mul(delta_c_real, delta_c_imag, delta_c_real, delta_c_imag);
    var best = Seed(0u, 0.0, 0.0);

    for (var iteration = 1u; iteration < params.orbit_len; iteration += 1u) {
        let a_dc = complex_mul(
            orbit_series_a_real[iteration],
            orbit_series_a_imag[iteration],
            delta_c_real,
            delta_c_imag,
        );
        let b_dc2 = complex_mul(
            orbit_series_b_real[iteration],
            orbit_series_b_imag[iteration],
            dc_sq.x,
            dc_sq.y,
        );
        let dz_real = a_dc.x + b_dc2.x;
        let dz_imag = a_dc.y + b_dc2.y;
        let dz_norm_sqr = dz_real * dz_real + dz_imag * dz_imag;
        let reference_norm_sqr = orbit_z_norm_sqr[iteration];

        if !is_finite_f64(dz_real)
            || !is_finite_f64(dz_imag)
            || (reference_norm_sqr > 0.0 && dz_norm_sqr > reference_norm_sqr * 1.0e-3)
        {
            break;
        }

        best = Seed(iteration, dz_real, dz_imag);
    }

    return best;
}

fn render_perturbation(pixel_x: u32, pixel_y: u32) -> u32 {
    let half_width = f64(params.width) * 0.5;
    let half_height = f64(params.height) * 0.5;
    let delta_c_real = (f64(pixel_x) - half_width) * params.scale;
    let delta_c_imag = (f64(pixel_y) - half_height) * params.scale;
    let seed = perturbation_seed(delta_c_real, delta_c_imag);
    var dz_real: f64 = seed.dz_real;
    var dz_imag: f64 = seed.dz_imag;

    for (var iteration = seed.iteration; iteration < params.max_iterations; iteration += 1u) {
        if iteration >= params.orbit_len {
            return 0xffffffffu;
        }

        let zr = orbit_z_real[iteration];
        let zi = orbit_z_imag[iteration];
        let real = zr + dz_real;
        let imag = zi + dz_imag;
        let magnitude = real * real + imag * imag;
        if magnitude > 4.0 {
            return pack_rgba(escape_color(iteration, sqrt(magnitude)));
        }

        let reference_norm_sqr = orbit_z_norm_sqr[iteration];
        let delta_norm_sqr = dz_real * dz_real + dz_imag * dz_imag;
        if iteration > 0u
            && reference_norm_sqr > 0.0
            && (magnitude < reference_norm_sqr * 1.0e-8
                || delta_norm_sqr > reference_norm_sqr * 1.0e8)
        {
            return 0xffffffffu;
        }

        let next_dz_real = 2.0 * (zr * dz_real - zi * dz_imag)
            + (dz_real * dz_real - dz_imag * dz_imag)
            + delta_c_real;
        let next_dz_imag = 2.0 * (zr * dz_imag + zi * dz_real)
            + (2.0 * dz_real * dz_imag)
            + delta_c_imag;

        if !is_finite_f64(next_dz_real)
            || !is_finite_f64(next_dz_imag)
            || max(abs(next_dz_real), abs(next_dz_imag)) > 1.0e6
        {
            return 0xffffffffu;
        }

        dz_real = next_dz_real;
        dz_imag = next_dz_imag;
    }

    return pack_rgba(vec4<u32>(7u, 10u, 18u, 255u));
}

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }

    let index = gid.y * params.width + gid.x;
    var pixel = 0u;
    var needs_fallback = 0u;

    if params.strategy == 0u {
        pixel = render_fast(gid.x, gid.y);
    } else {
        pixel = render_perturbation(gid.x, gid.y);
        if pixel == 0xffffffffu {
            pixel = pack_rgba(vec4<u32>(7u, 10u, 18u, 255u));
            needs_fallback = 1u;
        }
    }

    pixels[index] = pixel;
    fallback[index] = needs_fallback;
}
"#;

const WORKGROUP_SIZE: u32 = 8;
const GPU_RENDER_STRATEGY_FAST_F64: u32 = 0;

pub struct RendererState {
    pub surface: wgpu::Surface<'static>,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub config: wgpu::SurfaceConfiguration,
    pub window: Arc<Window>,
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    texture: wgpu::Texture,
    texture_view: wgpu::TextureView,
    texture_size: PhysicalSize<u32>,
    fractal_compute: Option<FractalComputeState>,
}

struct FractalComputeState {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    dummy_buffer: wgpu::Buffer,
}

pub struct GpuRenderOutput {
    pub pixels: Vec<u8>,
    pub fallback_mask: Vec<u8>,
}

impl RendererState {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create wgpu surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("failed to find GPU adapter");

        let supported_features = adapter.features();
        let requested_features = if supported_features.contains(wgpu::Features::SHADER_F64) {
            wgpu::Features::SHADER_F64
        } else {
            wgpu::Features::empty()
        };

        let required_limits = wgpu::Limits {
            max_storage_buffers_per_shader_stage: 9,
            ..wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("fractals-rs device"),
                required_features: requested_features,
                required_limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to create device");

        let caps = surface.get_capabilities(&adapter);
        let surface_format = caps
            .formats
            .iter()
            .copied()
            .find(|format| format.is_srgb())
            .unwrap_or(caps.formats[0]);

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: if caps.present_modes.contains(&wgpu::PresentMode::Mailbox) {
                wgpu::PresentMode::Mailbox
            } else {
                wgpu::PresentMode::Fifo
            },
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("fractal sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        let texture = create_texture(&device, size);
        let texture_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fractal present shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(PRESENT_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fractal bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = create_bind_group(&device, &bind_group_layout, &texture_view, &sampler);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fractal present pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("fractal present pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let fractal_compute = if requested_features.contains(wgpu::Features::SHADER_F64) {
            Some(FractalComputeState::new(&device))
        } else {
            None
        };

        Self {
            surface,
            device,
            queue,
            config,
            window,
            pipeline,
            bind_group_layout,
            bind_group,
            sampler,
            texture,
            texture_view,
            texture_size: size,
            fractal_compute,
        }
    }

    pub fn supports_fractal_compute(&self) -> bool {
        self.fractal_compute.is_some()
    }

    pub fn render_fractal_to_pixels(&self, params: &GpuRenderParams) -> Option<GpuRenderOutput> {
        let compute = self.fractal_compute.as_ref()?;
        compute.render(&self.device, &self.queue, params)
    }

    pub fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        self.config.width = new_size.width;
        self.config.height = new_size.height;
        self.surface.configure(&self.device, &self.config);
        self.recreate_texture(new_size);
    }

    pub fn upload_full(&mut self, width: u32, height: u32, pixels: &[u8]) {
        let size = PhysicalSize::new(width, height);
        if self.texture_size != size {
            self.recreate_texture(size);
        }

        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn upload_region(&mut self, x: u32, y: u32, width: u32, height: u32, pixels: &[u8]) {
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.texture,
                mip_level: 0,
                origin: wgpu::Origin3d { x, y, z: 0 },
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
    }

    pub fn render(&mut self) -> RenderOutcome {
        let output = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(output) => output,
            wgpu::CurrentSurfaceTexture::Suboptimal(output) => {
                self.render_to_surface(output);
                return RenderOutcome::NeedsReconfigure;
            }
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                return RenderOutcome::NeedsReconfigure;
            }
            wgpu::CurrentSurfaceTexture::Timeout
            | wgpu::CurrentSurfaceTexture::Occluded
            | wgpu::CurrentSurfaceTexture::Validation => return RenderOutcome::Skipped,
        };

        self.render_to_surface(output);
        RenderOutcome::Ok
    }

    fn recreate_texture(&mut self, size: PhysicalSize<u32>) {
        self.texture = create_texture(&self.device, size);
        self.texture_view = self
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.bind_group = create_bind_group(
            &self.device,
            &self.bind_group_layout,
            &self.texture_view,
            &self.sampler,
        );
        self.texture_size = size;
    }

    fn render_to_surface(&mut self, output: wgpu::SurfaceTexture) {
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fractal present encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("fractal present pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.draw(0..6, 0..1);
        }

        self.queue.submit(Some(encoder.finish()));
        output.present();
    }
}

impl FractalComputeState {
    fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fractal compute shader"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(COMPUTE_SHADER)),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("fractal compute bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fractal compute pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fractal compute pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let dummy_buffer = create_storage_buffer(device, 8, Some("fractal dummy storage"));

        Self {
            pipeline,
            bind_group_layout,
            dummy_buffer,
        }
    }

    fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        params: &GpuRenderParams,
    ) -> Option<GpuRenderOutput> {
        let GpuRenderParams::FastF64 {
            width,
            height,
            center_x,
            center_y,
            scale,
            max_iterations,
        } = params;

        let pixel_count = *width as u64 * *height as u64;
        let byte_len = pixel_count * 4;
        let params_buffer = create_buffer_with_data(
            device,
            queue,
            &encode_gpu_params(
                *width,
                *height,
                *max_iterations,
                GPU_RENDER_STRATEGY_FAST_F64,
                0,
                *center_x,
                *center_y,
                *scale,
            ),
            wgpu::BufferUsages::UNIFORM,
            Some("fractal compute params"),
        );
        let output_buffer =
            create_storage_buffer(device, byte_len.max(4), Some("fractal compute pixels"));
        let fallback_buffer =
            create_storage_buffer(device, byte_len.max(4), Some("fractal compute fallback"));
        let readback_pixels = create_readback_buffer(device, byte_len.max(4), "fractal pixels readback");
        let readback_fallback =
            create_readback_buffer(device, byte_len.max(4), "fractal fallback readback");

        let orbit_buffers = create_orbit_buffers(device, queue, None, &self.dummy_buffer);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fractal compute bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                binding_entire_buffer(0, &params_buffer),
                binding_entire_buffer(1, &output_buffer),
                binding_entire_buffer(2, &fallback_buffer),
                binding_entire_buffer(3, &orbit_buffers.z_real),
                binding_entire_buffer(4, &orbit_buffers.z_imag),
                binding_entire_buffer(5, &orbit_buffers.z_norm_sqr),
                binding_entire_buffer(6, &orbit_buffers.series_a_real),
                binding_entire_buffer(7, &orbit_buffers.series_a_imag),
                binding_entire_buffer(8, &orbit_buffers.series_b_real),
                binding_entire_buffer(9, &orbit_buffers.series_b_imag),
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("fractal compute encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("fractal compute pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(
                width.div_ceil(WORKGROUP_SIZE),
                height.div_ceil(WORKGROUP_SIZE),
                1,
            );
        }
        encoder.copy_buffer_to_buffer(&output_buffer, 0, &readback_pixels, 0, byte_len);
        encoder.copy_buffer_to_buffer(&fallback_buffer, 0, &readback_fallback, 0, byte_len);
        queue.submit(Some(encoder.finish()));

        let pixels = read_buffer(device, queue, &readback_pixels, byte_len as usize)?;
        let fallback_raw = read_buffer(device, queue, &readback_fallback, byte_len as usize)?;
        let fallback_mask = fallback_raw
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u8)
            .collect();

        Some(GpuRenderOutput {
            pixels,
            fallback_mask,
        })
    }
}

pub enum RenderOutcome {
    Ok,
    NeedsReconfigure,
    Skipped,
}

struct OrbitBuffers {
    z_real: wgpu::Buffer,
    z_imag: wgpu::Buffer,
    z_norm_sqr: wgpu::Buffer,
    series_a_real: wgpu::Buffer,
    series_a_imag: wgpu::Buffer,
    series_b_real: wgpu::Buffer,
    series_b_imag: wgpu::Buffer,
}

fn create_texture(device: &wgpu::Device, size: PhysicalSize<u32>) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some("fractal texture"),
        size: wgpu::Extent3d {
            width: size.width.max(1),
            height: size.height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn create_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    texture_view: &wgpu::TextureView,
    sampler: &wgpu::Sampler,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("fractal present bind group"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(texture_view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(sampler),
            },
        ],
    })
}

fn create_storage_buffer(device: &wgpu::Device, size: u64, label: Option<&str>) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label,
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    })
}

fn create_readback_buffer(device: &wgpu::Device, size: u64, label: &str) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    })
}

fn create_buffer_with_data(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    data: &[u8],
    usage: wgpu::BufferUsages,
    label: Option<&str>,
) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label,
        size: data.len().max(8) as u64,
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !data.is_empty() {
        queue.write_buffer(&buffer, 0, data);
    }
    buffer
}

fn create_orbit_buffers(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    orbit: Option<&()>,
    dummy: &wgpu::Buffer,
) -> OrbitBuffers {
    let make = |label: &str, values: Option<&[f64]>| -> wgpu::Buffer {
        match values {
            Some(values) if !values.is_empty() => {
                create_buffer_with_data(
                    device,
                    queue,
                    encode_f64_slice(values),
                    wgpu::BufferUsages::STORAGE,
                    Some(label),
                )
            }
            _ => dummy.clone(),
        }
    };
    let _ = orbit;

    OrbitBuffers {
        z_real: make("orbit z_real", None),
        z_imag: make("orbit z_imag", None),
        z_norm_sqr: make("orbit z_norm_sqr", None),
        series_a_real: make("orbit series_a_real", None),
        series_a_imag: make("orbit series_a_imag", None),
        series_b_real: make("orbit series_b_real", None),
        series_b_imag: make("orbit series_b_imag", None),
    }
}

fn encode_gpu_params(
    width: u32,
    height: u32,
    max_iterations: u32,
    strategy: u32,
    orbit_len: u32,
    center_x: f64,
    center_y: f64,
    scale: f64,
) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(64);
    push_u32(&mut bytes, width);
    push_u32(&mut bytes, height);
    push_u32(&mut bytes, max_iterations);
    push_u32(&mut bytes, strategy);
    push_u32(&mut bytes, orbit_len);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_u32(&mut bytes, 0);
    push_f64(&mut bytes, center_x);
    push_f64(&mut bytes, center_y);
    push_f64(&mut bytes, scale);
    push_f64(&mut bytes, 0.0);
    bytes
}

fn encode_f64_slice(values: &[f64]) -> &[u8] {
    unsafe {
        std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), std::mem::size_of_val(values))
    }
}

fn binding_entire_buffer(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn read_buffer(
    device: &wgpu::Device,
    _queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    expected_len: usize,
) -> Option<Vec<u8>> {
    let (tx, rx) = mpsc::channel();
    let slice = buffer.slice(..);
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
    rx.recv().ok()?.ok()?;

    let mapped = slice.get_mapped_range();
    let bytes = mapped.iter().copied().take(expected_len).collect();
    drop(mapped);
    buffer.unmap();
    Some(bytes)
}

fn push_u32(bytes: &mut Vec<u8>, value: u32) {
    bytes.extend_from_slice(&value.to_le_bytes());
}

fn push_f64(bytes: &mut Vec<u8>, value: f64) {
    bytes.extend_from_slice(&value.to_le_bytes());
}
