use dashu_int::{IBig, ops::BitTest};
use num_traits::{Signed, ToPrimitive, Zero};
use winit::dpi::PhysicalPosition;

pub const INITIAL_FRAC_BITS: u32 = 160;
pub const MIN_FRAC_BITS: u32 = 128;
const PIXEL_PRECISION_GUARD_BITS: u32 = 96;

#[derive(Clone, Debug)]
pub struct BigFixed {
    pub raw: IBig,
    pub frac_bits: u32,
}

impl BigFixed {
    pub fn from_f64(value: f64, frac_bits: u32) -> Self {
        if !value.is_finite() || value == 0.0 {
            return Self {
                raw: IBig::zero(),
                frac_bits,
            };
        }

        let bits = value.to_bits();
        let negative = (bits >> 63) != 0;
        let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
        let mantissa_bits = bits & ((1_u64 << 52) - 1);

        let (mantissa, exponent) = if exponent_bits == 0 {
            (mantissa_bits, -1022 - 52)
        } else {
            ((1_u64 << 52) | mantissa_bits, exponent_bits - 1023 - 52)
        };

        let mut raw = IBig::from(mantissa);
        let shift = exponent as i64 + frac_bits as i64;
        if shift >= 0 {
            raw <<= shift as usize;
        } else {
            raw >>= (-shift) as usize;
        }

        if negative {
            raw = -raw;
        }

        Self { raw, frac_bits }
    }

    pub fn with_frac_bits(&self, new_frac_bits: u32) -> Self {
        match new_frac_bits.cmp(&self.frac_bits) {
            std::cmp::Ordering::Equal => self.clone(),
            std::cmp::Ordering::Greater => Self {
                raw: &self.raw << (new_frac_bits - self.frac_bits) as usize,
                frac_bits: new_frac_bits,
            },
            std::cmp::Ordering::Less => Self {
                raw: &self.raw >> (self.frac_bits - new_frac_bits) as usize,
                frac_bits: new_frac_bits,
            },
        }
    }

    pub fn add(&self, other: &Self) -> Self {
        debug_assert_eq!(self.frac_bits, other.frac_bits);
        Self {
            raw: &self.raw + &other.raw,
            frac_bits: self.frac_bits,
        }
    }

    pub fn sub(&self, other: &Self) -> Self {
        debug_assert_eq!(self.frac_bits, other.frac_bits);
        Self {
            raw: &self.raw - &other.raw,
            frac_bits: self.frac_bits,
        }
    }

    pub fn mul(&self, other: &Self) -> Self {
        debug_assert_eq!(self.frac_bits, other.frac_bits);
        Self {
            raw: mul_fixed_raw(&self.raw, &other.raw, self.frac_bits),
            frac_bits: self.frac_bits,
        }
    }

    pub fn abs_log10_estimate(&self) -> Option<f64> {
        let value = self.to_f64().abs();
        if value > 0.0 {
            return Some(value.log10());
        }

        let bits = self.raw.bit_len();
        if bits == 0 {
            return None;
        }

        Some((bits as f64 - self.frac_bits as f64) * std::f64::consts::LOG10_2)
    }

    pub fn abs_log2_estimate(&self) -> Option<f64> {
        let bits = self.raw.bit_len();
        if bits == 0 {
            return None;
        }

        Some(bits as f64 - self.frac_bits as f64)
    }

    pub fn to_f64(&self) -> f64 {
        raw_to_f64(&self.raw, self.frac_bits)
    }
}

pub fn mul_fixed_raw(lhs: &IBig, rhs: &IBig, frac_bits: u32) -> IBig {
    (lhs * rhs) >> frac_bits as usize
}

pub fn raw_to_f64(raw: &IBig, frac_bits: u32) -> f64 {
    if raw.is_zero() {
        return 0.0;
    }

    let negative = raw.is_negative();
    let abs = raw.abs();
    let bits = abs.bit_len() as i64;
    let mantissa_bits = 53_i64;
    let shift = (bits - mantissa_bits).max(0);
    let mantissa = (&abs >> shift as usize).to_u64().unwrap_or(u64::MAX);
    let exponent = shift - frac_bits as i64;
    let value = if exponent > i32::MAX as i64 {
        f64::INFINITY
    } else if exponent < i32::MIN as i64 {
        0.0
    } else {
        (mantissa as f64) * 2_f64.powi(exponent as i32)
    };

    if negative { -value } else { value }
}

#[derive(Clone)]
pub struct ViewportState {
    pub width: u32,
    pub height: u32,
    pub frac_bits: u32,
    pub center_x: BigFixed,
    pub center_y: BigFixed,
    pub scale: BigFixed,
}

impl ViewportState {
    pub fn new(width: u32, height: u32) -> Self {
        let frac_bits = INITIAL_FRAC_BITS;
        Self {
            width,
            height,
            frac_bits,
            center_x: BigFixed::from_f64(-0.5, frac_bits),
            center_y: BigFixed::from_f64(0.0, frac_bits),
            scale: BigFixed::from_f64(4.0 / width.max(1) as f64, frac_bits),
        }
    }

    pub fn update_size(&mut self, width: u32, height: u32) {
        self.width = width.max(1);
        self.height = height.max(1);
    }

    pub fn zoom_at_cursor(&mut self, zoom_factor: f64, cursor: PhysicalPosition<f64>) {
        if zoom_factor <= 0.0 {
            return;
        }

        let target_frac_bits = self.target_frac_bits_for_scale(zoom_factor);
        self.requantize(target_frac_bits);

        let factor = BigFixed::from_f64(zoom_factor, self.frac_bits);
        let new_scale = self.scale.mul(&factor);

        let dx = BigFixed::from_f64(cursor.x - self.width as f64 * 0.5, self.frac_bits);
        let dy = BigFixed::from_f64(cursor.y - self.height as f64 * 0.5, self.frac_bits);

        let old_dx = self.scale.mul(&dx);
        let new_dx = new_scale.mul(&dx);
        let old_dy = self.scale.mul(&dy);
        let new_dy = new_scale.mul(&dy);

        self.center_x = self.center_x.add(&old_dx).sub(&new_dx);
        self.center_y = self.center_y.add(&old_dy).sub(&new_dy);
        self.scale = new_scale;
    }

    pub fn pan_by_pixels(&mut self, delta_x: f64, delta_y: f64) {
        let dx = BigFixed::from_f64(delta_x, self.frac_bits);
        let dy = BigFixed::from_f64(delta_y, self.frac_bits);
        let shift_x = self.scale.mul(&dx);
        let shift_y = self.scale.mul(&dy);

        self.center_x = self.center_x.sub(&shift_x);
        self.center_y = self.center_y.sub(&shift_y);
    }

    fn target_frac_bits_for_scale(&self, zoom_factor: f64) -> u32 {
        let next_scale = self.scale.to_f64().abs() * zoom_factor.abs();
        let next_log2_scale = if next_scale.is_finite() && next_scale > 0.0 {
            next_scale.log2()
        } else {
            self.scale.abs_log2_estimate().unwrap_or(0.0) + zoom_factor.abs().log2()
        };
        let viewport_bits = self.width.max(self.height).max(1).ilog2() + 1;
        let required =
            (-next_log2_scale).ceil().max(0.0) as u32 + viewport_bits + PIXEL_PRECISION_GUARD_BITS;

        required.max(MIN_FRAC_BITS)
    }

    pub fn requantize(&mut self, frac_bits: u32) {
        if frac_bits == self.frac_bits {
            return;
        }

        self.center_x = self.center_x.with_frac_bits(frac_bits);
        self.center_y = self.center_y.with_frac_bits(frac_bits);
        self.scale = self.scale.with_frac_bits(frac_bits);
        self.frac_bits = frac_bits;
    }

    pub fn describe(&self) -> String {
        let scale_text = if let Some(log10_scale) = self.scale.abs_log10_estimate() {
            let scale = self.scale.to_f64().abs();
            if scale.is_finite() && scale > 0.0 {
                format!("{scale:.3e}")
            } else {
                format!("1e{log10_scale:.1}")
            }
        } else {
            "0".to_string()
        };

        format!(
            "fractals-rs | Mandelbrot | scale {scale_text} | frac {}",
            self.frac_bits,
        )
    }
}
