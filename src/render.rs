use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc::{self, Receiver, Sender},
    },
    thread,
};

use num_bigint::BigInt;
use num_traits::Zero;
use rayon::prelude::*;

use crate::math::{BigFixed, ViewportState, mul_fixed_raw};

const TILE_SIZE: u32 = 96;
const FAST_PATH_SCALE_THRESHOLD: f64 = 1.0e-18;
const PERTURBATION_SCALE_THRESHOLD: f64 = 1.0e-280;
const PREVIEW_MAGNIFICATION_LIMIT: f64 = 8.0;
const PREVIEW_BG: [u8; 4] = [10, 12, 18, 255];
const PREVIEW_GRID: [u8; 4] = [210, 45, 45, 255];
const PERTURBATION_GROWTH_LIMIT: f64 = 1.0e6;
const PERTURBATION_GLITCH_RATIO: f64 = 1.0e-8;
const PERTURBATION_REBASE_RATIO: f64 = 1.0e8;
const SERIES_DELTA_RATIO: f64 = 1.0e-3;

#[derive(Clone)]
pub struct RenderRequest {
    pub generation: u64,
    pub width: u32,
    pub height: u32,
    pub frac_bits: u32,
    pub center_x: BigFixed,
    pub center_y: BigFixed,
    pub scale: BigFixed,
    pub max_iterations: u32,
}

pub enum RenderMessage {
    Started {
        generation: u64,
        pending_tiles: Vec<(u32, u32, u32, u32)>,
    },
    Tile {
        generation: u64,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    },
}

#[derive(Clone, Copy)]
struct TileRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

struct ReferenceOrbit {
    z_real: Vec<f64>,
    z_imag: Vec<f64>,
    z_norm_sqr: Vec<f64>,
    series_a_real: Vec<f64>,
    series_a_imag: Vec<f64>,
    series_b_real: Vec<f64>,
    series_b_imag: Vec<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderStrategy {
    FastF64,
    Perturbation,
    Exact,
}

impl RenderRequest {
    pub fn from_viewport(viewport: &ViewportState, generation: u64) -> Self {
        Self {
            generation,
            width: viewport.width,
            height: viewport.height,
            frac_bits: viewport.frac_bits,
            center_x: viewport.center_x.clone(),
            center_y: viewport.center_y.clone(),
            scale: viewport.scale.clone(),
            max_iterations: recommended_iterations(&viewport.scale),
        }
    }
}

pub fn spawn_render_worker() -> (
    Sender<RenderRequest>,
    Receiver<RenderMessage>,
    Arc<AtomicU64>,
) {
    let (request_tx, request_rx) = mpsc::channel::<RenderRequest>();
    let (result_tx, result_rx) = mpsc::channel::<RenderMessage>();
    let latest_generation = Arc::new(AtomicU64::new(0));
    let latest_generation_for_thread = latest_generation.clone();

    thread::spawn(move || {
        while let Ok(mut request) = request_rx.recv() {
            while let Ok(newer_request) = request_rx.try_recv() {
                request = newer_request;
            }

            render_progressive(&request, &latest_generation_for_thread, &result_tx);
        }
    });

    (request_tx, result_rx, latest_generation)
}

pub fn recommended_iterations(scale: &BigFixed) -> u32 {
    let log10_scale = scale.abs_log10_estimate().unwrap_or(-200.0);
    let extra = (-log10_scale).max(0.0) * 96.0;
    (384.0 + extra).round().clamp(384.0, 24_000.0) as u32
}

fn base_render_strategy_for_scale(scale: &BigFixed) -> RenderStrategy {
    let scale_f64 = scale.to_f64().abs();
    if scale_f64.is_finite() && scale_f64 >= FAST_PATH_SCALE_THRESHOLD {
        RenderStrategy::FastF64
    } else if scale_f64.is_finite() && scale_f64 >= PERTURBATION_SCALE_THRESHOLD {
        RenderStrategy::Perturbation
    } else {
        RenderStrategy::Exact
    }
}

pub fn render_strategy_for_viewport(viewport: &ViewportState) -> RenderStrategy {
    render_strategy_for_geometry(
        &viewport.center_x,
        &viewport.center_y,
        &viewport.scale,
        viewport.width,
        viewport.height,
    )
}

fn render_strategy_for_request(request: &RenderRequest) -> RenderStrategy {
    render_strategy_for_geometry(
        &request.center_x,
        &request.center_y,
        &request.scale,
        request.width,
        request.height,
    )
}

fn render_strategy_for_geometry(
    center_x: &BigFixed,
    center_y: &BigFixed,
    scale: &BigFixed,
    width: u32,
    height: u32,
) -> RenderStrategy {
    match base_render_strategy_for_scale(scale) {
        RenderStrategy::FastF64 if f64_fast_path_is_safe(center_x, center_y, scale, width, height) => {
            RenderStrategy::FastF64
        }
        RenderStrategy::FastF64 => RenderStrategy::Perturbation,
        strategy => strategy,
    }
}

fn render_progressive(
    request: &RenderRequest,
    latest_generation: &AtomicU64,
    result_tx: &Sender<RenderMessage>,
) {
    let reference = if matches!(
        render_strategy_for_request(request),
        RenderStrategy::Perturbation
    ) {
        Some(Arc::new(build_reference_orbit(request)))
    } else {
        None
    };

    for phase in 0..4 {
        if latest_generation.load(Ordering::Relaxed) != request.generation {
            return;
        }

        let tiles = tiles_for_phase(request.width, request.height, phase);
        let _ = result_tx.send(RenderMessage::Started {
            generation: request.generation,
            pending_tiles: tiles
                .iter()
                .map(|tile| (tile.x, tile.y, tile.width, tile.height))
                .collect(),
        });
        tiles
            .into_par_iter()
            .for_each_with(result_tx.clone(), |tx, tile| {
                if latest_generation.load(Ordering::Relaxed) != request.generation {
                    return;
                }

                let pixels = render_tile(request, tile, reference.as_deref());
                let _ = tx.send(RenderMessage::Tile {
                    generation: request.generation,
                    x: tile.x,
                    y: tile.y,
                    width: tile.width,
                    height: tile.height,
                    pixels,
                });
            });
    }
}

fn tiles_for_phase(width: u32, height: u32, phase: u32) -> Vec<TileRect> {
    let tiles_x = width.div_ceil(TILE_SIZE);
    let tiles_y = height.div_ceil(TILE_SIZE);
    let parity_x = phase % 2;
    let parity_y = phase / 2;
    let mut tiles = Vec::new();

    for tile_y in 0..tiles_y {
        for tile_x in 0..tiles_x {
            if tile_x % 2 != parity_x || tile_y % 2 != parity_y {
                continue;
            }

            let x = tile_x * TILE_SIZE;
            let y = tile_y * TILE_SIZE;
            tiles.push(TileRect {
                x,
                y,
                width: (width - x).min(TILE_SIZE),
                height: (height - y).min(TILE_SIZE),
            });
        }
    }

    tiles
}

fn render_tile(
    request: &RenderRequest,
    tile: TileRect,
    reference: Option<&ReferenceOrbit>,
) -> Vec<u8> {
    match render_strategy_for_request(request) {
        RenderStrategy::FastF64 => render_tile_f64(request, tile),
        RenderStrategy::Perturbation => {
            let reference = reference.expect("perturbation render requires a reference orbit");
            render_tile_perturbation(request, tile, reference)
        }
        RenderStrategy::Exact => render_tile_fixed(request, tile),
    }
}

fn render_tile_f64(request: &RenderRequest, tile: TileRect) -> Vec<u8> {
    let mut pixels = vec![0; tile.width as usize * tile.height as usize * 4];
    let center_x = request.center_x.to_f64();
    let center_y = request.center_y.to_f64();
    let scale = request.scale.to_f64();
    let half_width = request.width as f64 * 0.5;
    let half_height = request.height as f64 * 0.5;

    for local_y in 0..tile.height as usize {
        let global_y = tile.y as usize + local_y;
        let imag = center_y + (global_y as f64 - half_height) * scale;
        let row_offset = local_y * tile.width as usize * 4;

        for local_x in 0..tile.width as usize {
            let global_x = tile.x as usize + local_x;
            let real = center_x + (global_x as f64 - half_width) * scale;
            let color = mandelbrot_color_f64(real, imag, request.max_iterations);
            let offset = row_offset + local_x * 4;
            pixels[offset..offset + 4].copy_from_slice(&color);
        }
    }

    pixels
}

pub fn build_preview_frame(
    source_pixels: &[u8],
    source_viewport: Option<&ViewportState>,
    target_viewport: &ViewportState,
) -> Vec<u8> {
    let width = target_viewport.width;
    let height = target_viewport.height;
    let mut pixels = vec![0; width as usize * height as usize * 4];

    if let Some(source_viewport) = source_viewport {
        if !source_pixels.is_empty()
            && source_viewport.width > 0
            && source_viewport.height > 0
            && source_pixels.len() == (source_viewport.width * source_viewport.height * 4) as usize
            && preview_reprojection_is_useful(source_viewport, target_viewport)
        {
            let src_center_x = source_viewport.center_x.to_f64();
            let src_center_y = source_viewport.center_y.to_f64();
            let src_scale = source_viewport.scale.to_f64();
            let dst_center_x = target_viewport.center_x.to_f64();
            let dst_center_y = target_viewport.center_y.to_f64();
            let dst_scale = target_viewport.scale.to_f64();
            if !src_center_x.is_finite()
                || !src_center_y.is_finite()
                || !src_scale.is_finite()
                || src_scale == 0.0
                || !dst_center_x.is_finite()
                || !dst_center_y.is_finite()
                || !dst_scale.is_finite()
                || dst_scale == 0.0
            {
                return fill_preview_background(pixels);
            }
            let src_half_width = source_viewport.width as f64 * 0.5;
            let src_half_height = source_viewport.height as f64 * 0.5;
            let dst_half_width = width as f64 * 0.5;
            let dst_half_height = height as f64 * 0.5;

            for y in 0..height {
                for x in 0..width {
                    let world_x = dst_center_x + (x as f64 - dst_half_width) * dst_scale;
                    let world_y = dst_center_y + (y as f64 - dst_half_height) * dst_scale;
                    let src_x = ((world_x - src_center_x) / src_scale + src_half_width).round();
                    let src_y = ((world_y - src_center_y) / src_scale + src_half_height).round();
                    let dst_offset = ((y * width + x) * 4) as usize;

                    if src_x >= 0.0
                        && src_y >= 0.0
                        && src_x < source_viewport.width as f64
                        && src_y < source_viewport.height as f64
                    {
                        let sx = src_x as u32;
                        let sy = src_y as u32;
                        let src_offset = ((sy * source_viewport.width + sx) * 4) as usize;
                        pixels[dst_offset..dst_offset + 4]
                            .copy_from_slice(&source_pixels[src_offset..src_offset + 4]);
                    } else {
                        pixels[dst_offset..dst_offset + 4].copy_from_slice(&PREVIEW_BG);
                    }
                }
            }
        }
    }

    fill_preview_background(pixels)
}

pub fn overlay_tile_grid(
    pixels: &mut [u8],
    width: u32,
    height: u32,
    tiles: &[(u32, u32, u32, u32)],
) {
    for &(tile_x, tile_y, tile_width, tile_height) in tiles {
        let max_y = (tile_y + tile_height).min(height);
        let max_x = (tile_x + tile_width).min(width);
        for y in tile_y..max_y {
            for x in tile_x..max_x {
                if ((x - tile_x) / 8 + (y - tile_y) / 8) % 2 != 0 {
                    continue;
                }

                let offset = ((y * width + x) * 4) as usize;
                pixels[offset] = ((pixels[offset] as u16 + PREVIEW_GRID[0] as u16) / 2) as u8;
                pixels[offset + 1] =
                    ((pixels[offset + 1] as u16 + PREVIEW_GRID[1] as u16) / 2) as u8;
                pixels[offset + 2] =
                    ((pixels[offset + 2] as u16 + PREVIEW_GRID[2] as u16) / 2) as u8;
                pixels[offset + 3] = 255;
            }
        }
    }
}

fn render_tile_fixed(request: &RenderRequest, tile: TileRect) -> Vec<u8> {
    let mut pixels = vec![0; tile.width as usize * tile.height as usize * 4];
    let frac_bits = request.frac_bits;
    let half_width = request.width as f64 * 0.5;
    let half_height = request.height as f64 * 0.5;
    let start_x = BigFixed::from_f64(tile.x as f64 - half_width, frac_bits);
    let start_y = BigFixed::from_f64(tile.y as f64 - half_height, frac_bits);
    let start_real =
        &request.center_x.raw + mul_fixed_raw(&request.scale.raw, &start_x.raw, frac_bits);
    let mut imag =
        &request.center_y.raw + mul_fixed_raw(&request.scale.raw, &start_y.raw, frac_bits);

    for local_y in 0..tile.height as usize {
        let row_offset = local_y * tile.width as usize * 4;
        let mut real = start_real.clone();

        for local_x in 0..tile.width as usize {
            let color = mandelbrot_color_fixed(&real, &imag, request.max_iterations, frac_bits);
            let offset = row_offset + local_x * 4;
            pixels[offset..offset + 4].copy_from_slice(&color);
            real += &request.scale.raw;
        }

        imag += &request.scale.raw;
    }

    pixels
}

fn render_tile_perturbation(
    request: &RenderRequest,
    tile: TileRect,
    reference: &ReferenceOrbit,
) -> Vec<u8> {
    let mut pixels = vec![0; tile.width as usize * tile.height as usize * 4];
    let frac_bits = request.frac_bits;
    let half_width = request.width as f64 * 0.5;
    let half_height = request.height as f64 * 0.5;
    let start_dx = BigFixed::from_f64(tile.x as f64 - half_width, frac_bits);
    let start_dy = BigFixed::from_f64(tile.y as f64 - half_height, frac_bits);
    let start_offset_real = mul_fixed_raw(&request.scale.raw, &start_dx.raw, frac_bits);
    let mut offset_imag = mul_fixed_raw(&request.scale.raw, &start_dy.raw, frac_bits);
    let tile_reference_offset_real = pixel_offset_real_raw(
        request,
        tile.x as f64 + tile.width as f64 * 0.5 - 0.5,
    );
    let tile_reference_offset_imag = pixel_offset_imag_raw(
        request,
        tile.y as f64 + tile.height as f64 * 0.5 - 0.5,
    );
    let mut tile_reference = None;

    for local_y in 0..tile.height as usize {
        let row_offset = local_y * tile.width as usize * 4;
        let mut offset_real = start_offset_real.clone();

        for local_x in 0..tile.width as usize {
            let tile_reference = tile_reference.get_or_insert_with(|| {
                build_reference_orbit_at_offset(
                    request,
                    &tile_reference_offset_real,
                    &tile_reference_offset_imag,
                )
            });
            let rebased_real = &offset_real - &tile_reference_offset_real;
            let rebased_imag = &offset_imag - &tile_reference_offset_imag;
            let color = mandelbrot_color_perturbation(
                request,
                tile_reference,
                &rebased_real,
                &rebased_imag,
            )
            .or_else(|| {
                mandelbrot_color_perturbation(
                    request,
                    reference,
                    &offset_real,
                    &offset_imag,
                )
            })
            .unwrap_or_else(|| {
                let c_real = &request.center_x.raw + &offset_real;
                let c_imag = &request.center_y.raw + &offset_imag;
                mandelbrot_color_fixed(
                    &c_real,
                    &c_imag,
                    request.max_iterations,
                    request.frac_bits,
                )
            });
            let offset = row_offset + local_x * 4;
            pixels[offset..offset + 4].copy_from_slice(&color);
            offset_real += &request.scale.raw;
        }

        offset_imag += &request.scale.raw;
    }

    pixels
}

fn build_reference_orbit(request: &RenderRequest) -> ReferenceOrbit {
    build_reference_orbit_for_point(
        request.center_x.raw.clone(),
        request.center_y.raw.clone(),
        request.frac_bits,
        request.max_iterations,
    )
}

fn build_reference_orbit_at_offset(
    request: &RenderRequest,
    offset_real: &BigInt,
    offset_imag: &BigInt,
) -> ReferenceOrbit {
    build_reference_orbit_for_point(
        &request.center_x.raw + offset_real,
        &request.center_y.raw + offset_imag,
        request.frac_bits,
        request.max_iterations,
    )
}

fn build_reference_orbit_for_point(
    c_real_raw: BigInt,
    c_imag_raw: BigInt,
    frac_bits: u32,
    max_iterations: u32,
) -> ReferenceOrbit {
    let mut z_real = BigInt::zero();
    let mut z_imag = BigInt::zero();
    let escape_limit = BigInt::from(4_u8) << frac_bits;

    let mut orbit_real = Vec::with_capacity(max_iterations as usize + 1);
    let mut orbit_imag = Vec::with_capacity(max_iterations as usize + 1);
    let mut orbit_norm_sqr = Vec::with_capacity(max_iterations as usize + 1);
    let mut series_a_real = Vec::with_capacity(max_iterations as usize + 1);
    let mut series_a_imag = Vec::with_capacity(max_iterations as usize + 1);
    let mut series_b_real = Vec::with_capacity(max_iterations as usize + 1);
    let mut series_b_imag = Vec::with_capacity(max_iterations as usize + 1);
    orbit_real.push(0.0);
    orbit_imag.push(0.0);
    orbit_norm_sqr.push(0.0);
    series_a_real.push(0.0);
    series_a_imag.push(0.0);
    series_b_real.push(0.0);
    series_b_imag.push(0.0);

    for _ in 0..max_iterations {
        let current_real = *orbit_real.last().unwrap_or(&0.0);
        let current_imag = *orbit_imag.last().unwrap_or(&0.0);
        let zr2 = mul_fixed_raw(&z_real, &z_real, frac_bits);
        let zi2 = mul_fixed_raw(&z_imag, &z_imag, frac_bits);
        let zri = mul_fixed_raw(&z_real, &z_imag, frac_bits);

        let next_real = zr2 - zi2 + &c_real_raw;
        let next_imag = (&zri << 1) + &c_imag_raw;
        let next_real_f64 = raw_fixed_to_f64(&next_real, frac_bits);
        let next_imag_f64 = raw_fixed_to_f64(&next_imag, frac_bits);
        let next_norm_sqr = next_real_f64 * next_real_f64 + next_imag_f64 * next_imag_f64;

        orbit_real.push(next_real_f64);
        orbit_imag.push(next_imag_f64);
        orbit_norm_sqr.push(next_norm_sqr);

        let current_a_real = *series_a_real.last().unwrap_or(&0.0);
        let current_a_imag = *series_a_imag.last().unwrap_or(&0.0);
        let current_b_real = *series_b_real.last().unwrap_or(&0.0);
        let current_b_imag = *series_b_imag.last().unwrap_or(&0.0);
        let (z_times_a_real, z_times_a_imag) =
            complex_mul(current_real, current_imag, current_a_real, current_a_imag);
        let (a_sq_real, a_sq_imag) =
            complex_mul(current_a_real, current_a_imag, current_a_real, current_a_imag);
        let (z_times_b_real, z_times_b_imag) =
            complex_mul(current_real, current_imag, current_b_real, current_b_imag);
        series_a_real.push(2.0 * z_times_a_real + 1.0);
        series_a_imag.push(2.0 * z_times_a_imag);
        series_b_real.push(2.0 * z_times_b_real + a_sq_real);
        series_b_imag.push(2.0 * z_times_b_imag + a_sq_imag);

        let magnitude = mul_fixed_raw(&next_real, &next_real, frac_bits)
            + mul_fixed_raw(&next_imag, &next_imag, frac_bits);

        z_real = next_real;
        z_imag = next_imag;

        if magnitude > escape_limit {
            break;
        }
    }

    ReferenceOrbit {
        z_real: orbit_real,
        z_imag: orbit_imag,
        z_norm_sqr: orbit_norm_sqr,
        series_a_real,
        series_a_imag,
        series_b_real,
        series_b_imag,
    }
}

fn mandelbrot_color_f64(cx: f64, cy: f64, max_iterations: u32) -> [u8; 4] {
    let mut zx = 0.0;
    let mut zy = 0.0;
    let mut zx2 = 0.0;
    let mut zy2 = 0.0;
    let mut iteration = 0;

    while zx2 + zy2 <= 4.0 && iteration < max_iterations {
        zy = 2.0 * zx * zy + cy;
        zx = zx2 - zy2 + cx;
        zx2 = zx * zx;
        zy2 = zy * zy;
        iteration += 1;
    }

    escape_color(iteration, max_iterations, (zx2 + zy2).sqrt())
}

fn mandelbrot_color_perturbation(
    request: &RenderRequest,
    reference: &ReferenceOrbit,
    delta_c_real_raw: &BigInt,
    delta_c_imag_raw: &BigInt,
) -> Option<[u8; 4]> {
    let delta_c_real = raw_fixed_to_f64(delta_c_real_raw, request.frac_bits);
    let delta_c_imag = raw_fixed_to_f64(delta_c_imag_raw, request.frac_bits);

    if (delta_c_real == 0.0 && !delta_c_real_raw.is_zero())
        || (delta_c_imag == 0.0 && !delta_c_imag_raw.is_zero())
    {
        return None;
    }

    let (start_iteration, mut dz_real, mut dz_imag) =
        perturbation_series_seed(reference, delta_c_real, delta_c_imag);

    for iteration in start_iteration..request.max_iterations as usize {
        if iteration >= reference.z_real.len() {
            return None;
        }

        let zr = reference.z_real[iteration];
        let zi = reference.z_imag[iteration];
        let real = zr + dz_real;
        let imag = zi + dz_imag;
        let magnitude = real * real + imag * imag;
        if magnitude > 4.0 {
            return Some(escape_color(
                iteration as u32,
                request.max_iterations,
                magnitude.sqrt(),
            ));
        }

        let reference_norm_sqr = reference.z_norm_sqr[iteration];
        let delta_norm_sqr = dz_real * dz_real + dz_imag * dz_imag;
        if iteration > 0
            && reference_norm_sqr > 0.0
            && (magnitude < reference_norm_sqr * PERTURBATION_GLITCH_RATIO
                || delta_norm_sqr > reference_norm_sqr * PERTURBATION_REBASE_RATIO)
        {
            return None;
        }

        let next_dz_real = 2.0 * (zr * dz_real - zi * dz_imag)
            + (dz_real * dz_real - dz_imag * dz_imag)
            + delta_c_real;
        let next_dz_imag = 2.0 * (zr * dz_imag + zi * dz_real)
            + (2.0 * dz_real * dz_imag)
            + delta_c_imag;

        if !next_dz_real.is_finite()
            || !next_dz_imag.is_finite()
            || next_dz_real.abs().max(next_dz_imag.abs()) > PERTURBATION_GROWTH_LIMIT
        {
            return None;
        }

        dz_real = next_dz_real;
        dz_imag = next_dz_imag;
    }

    Some([7, 10, 18, 255])
}

fn perturbation_series_seed(
    reference: &ReferenceOrbit,
    delta_c_real: f64,
    delta_c_imag: f64,
) -> (usize, f64, f64) {
    let delta_norm_sqr = delta_c_real * delta_c_real + delta_c_imag * delta_c_imag;
    if delta_norm_sqr == 0.0 {
        return (0, 0.0, 0.0);
    }

    let (dc_sq_real, dc_sq_imag) =
        complex_mul(delta_c_real, delta_c_imag, delta_c_real, delta_c_imag);
    let mut best_iteration = 0;
    let mut best_dz_real = 0.0;
    let mut best_dz_imag = 0.0;

    for iteration in 1..reference.z_real.len() {
        let (a_dc_real, a_dc_imag) = complex_mul(
            reference.series_a_real[iteration],
            reference.series_a_imag[iteration],
            delta_c_real,
            delta_c_imag,
        );
        let (b_dc2_real, b_dc2_imag) = complex_mul(
            reference.series_b_real[iteration],
            reference.series_b_imag[iteration],
            dc_sq_real,
            dc_sq_imag,
        );
        let dz_real = a_dc_real + b_dc2_real;
        let dz_imag = a_dc_imag + b_dc2_imag;
        let dz_norm_sqr = dz_real * dz_real + dz_imag * dz_imag;
        let reference_norm_sqr = reference.z_norm_sqr[iteration];

        if !dz_real.is_finite()
            || !dz_imag.is_finite()
            || (reference_norm_sqr > 0.0 && dz_norm_sqr > reference_norm_sqr * SERIES_DELTA_RATIO)
        {
            break;
        }

        best_iteration = iteration;
        best_dz_real = dz_real;
        best_dz_imag = dz_imag;
    }

    (best_iteration, best_dz_real, best_dz_imag)
}

fn pixel_offset_real_raw(request: &RenderRequest, pixel_x: f64) -> BigInt {
    let half_width = request.width as f64 * 0.5;
    let dx = BigFixed::from_f64(pixel_x - half_width, request.frac_bits);
    mul_fixed_raw(&request.scale.raw, &dx.raw, request.frac_bits)
}

fn pixel_offset_imag_raw(request: &RenderRequest, pixel_y: f64) -> BigInt {
    let half_height = request.height as f64 * 0.5;
    let dy = BigFixed::from_f64(pixel_y - half_height, request.frac_bits);
    mul_fixed_raw(&request.scale.raw, &dy.raw, request.frac_bits)
}

fn mandelbrot_color_fixed(
    cx: &BigInt,
    cy: &BigInt,
    max_iterations: u32,
    frac_bits: u32,
) -> [u8; 4] {
    let mut zx = BigInt::zero();
    let mut zy = BigInt::zero();
    let escape_limit = BigInt::from(4_u8) << frac_bits;
    let mut iteration = 0;

    while iteration < max_iterations {
        let zx2 = mul_fixed_raw(&zx, &zx, frac_bits);
        let zy2 = mul_fixed_raw(&zy, &zy, frac_bits);
        let magnitude = &zx2 + &zy2;
        if magnitude > escape_limit {
            let radius = raw_fixed_to_f64(&magnitude, frac_bits);
            return escape_color(iteration, max_iterations, radius.sqrt());
        }

        let zxy = mul_fixed_raw(&zx, &zy, frac_bits);
        let new_zy = (&zxy << 1) + cy;
        let new_zx = zx2 - zy2 + cx;
        zx = new_zx;
        zy = new_zy;
        iteration += 1;
    }

    [7, 10, 18, 255]
}

fn raw_fixed_to_f64(raw: &BigInt, frac_bits: u32) -> f64 {
    BigFixed {
        raw: raw.clone(),
        frac_bits,
    }
    .to_f64()
}

fn complex_mul(a_real: f64, a_imag: f64, b_real: f64, b_imag: f64) -> (f64, f64) {
    (
        a_real * b_real - a_imag * b_imag,
        a_real * b_imag + a_imag * b_real,
    )
}

fn fill_preview_background(mut pixels: Vec<u8>) -> Vec<u8> {
    if pixels.iter().all(|value| *value == 0) {
        for chunk in pixels.chunks_exact_mut(4) {
            chunk.copy_from_slice(&PREVIEW_BG);
        }
    }

    pixels
}

fn preview_reprojection_is_useful(
    source_viewport: &ViewportState,
    target_viewport: &ViewportState,
) -> bool {
    if matches!(
        render_strategy_for_viewport(source_viewport),
        RenderStrategy::Exact
    ) || matches!(
        render_strategy_for_viewport(target_viewport),
        RenderStrategy::Exact
    ) {
        return false;
    }

    let src_scale = source_viewport.scale.to_f64().abs();
    let dst_scale = target_viewport.scale.to_f64().abs();
    if !src_scale.is_finite() || !dst_scale.is_finite() || src_scale == 0.0 || dst_scale == 0.0 {
        return false;
    }

    (src_scale / dst_scale).max(dst_scale / src_scale) <= PREVIEW_MAGNIFICATION_LIMIT
}

fn f64_fast_path_is_safe(
    center_x: &BigFixed,
    center_y: &BigFixed,
    scale: &BigFixed,
    width: u32,
    height: u32,
) -> bool {
    let scale_f64 = scale.to_f64().abs();
    if !scale_f64.is_finite() || scale_f64 == 0.0 {
        return false;
    }

    let span_x = width.max(2);
    let span_y = height.max(2);
    f64_axis_has_pixel_resolution(center_x, scale_f64, span_x)
        && f64_axis_has_pixel_resolution(center_y, scale_f64, span_y)
}

fn f64_axis_has_pixel_resolution(center: &BigFixed, scale_f64: f64, span_pixels: u32) -> bool {
    let center_f64 = center.to_f64();
    if !center_f64.is_finite() {
        return false;
    }

    let half_span = span_pixels as f64 * 0.5;
    let edge_magnitude = center_f64.abs() + scale_f64 * half_span;
    let ulp = f64_ulp(edge_magnitude.max(center_f64.abs()));
    scale_f64 > ulp * 4.0
}

fn f64_ulp(value: f64) -> f64 {
    let value = value.abs();
    if !value.is_finite() {
        return f64::INFINITY;
    }
    if value == 0.0 {
        return f64::from_bits(1);
    }

    let bits = value.to_bits();
    let next = f64::from_bits(bits + 1);
    next - value
}

fn escape_color(iteration: u32, max_iterations: u32, radius: f64) -> [u8; 4] {
    if iteration >= max_iterations {
        return [7, 10, 18, 255];
    }

    let radius = radius.max(1.000_000_1);
    let smooth = iteration as f64 + 1.0 - radius.ln().ln() / std::f64::consts::LN_2;
    let t = (smooth / max_iterations as f64).clamp(0.0, 1.0);

    let r = (9.0 * (1.0 - t) * t * t * t * 255.0).round() as u8;
    let g = (15.0 * (1.0 - t) * (1.0 - t) * t * t * 255.0).round() as u8;
    let b = (8.5 * (1.0 - t) * (1.0 - t) * (1.0 - t) * t * 255.0).round() as u8;
    [r, g, b, 255]
}

#[cfg(test)]
mod tests {
    use super::{RenderStrategy, f64_axis_has_pixel_resolution, render_strategy_for_viewport};
    use crate::math::{BigFixed, INITIAL_FRAC_BITS, ViewportState};

    #[test]
    fn moderate_zoom_keeps_f64_fast_path() {
        let viewport = ViewportState::new(1280, 720);
        assert_eq!(render_strategy_for_viewport(&viewport), RenderStrategy::FastF64);
    }

    #[test]
    fn deep_zoom_near_half_disables_f64_fast_path() {
        let frac_bits = INITIAL_FRAC_BITS;
        let viewport = ViewportState {
            width: 1280,
            height: 720,
            frac_bits,
            center_x: BigFixed::from_f64(-0.5, frac_bits),
            center_y: BigFixed::from_f64(0.0, frac_bits),
            scale: BigFixed::from_f64(4.708e-18, frac_bits),
        };

        assert_eq!(
            render_strategy_for_viewport(&viewport),
            RenderStrategy::Perturbation
        );
    }

    #[test]
    fn f64_resolution_test_catches_collapsed_adjacent_pixels() {
        let frac_bits = INITIAL_FRAC_BITS;
        let center = BigFixed::from_f64(-0.5, frac_bits);
        assert!(!f64_axis_has_pixel_resolution(&center, 4.708e-18, 1280));
        assert!(f64_axis_has_pixel_resolution(&center, 1.0e-12, 1280));
    }
}
