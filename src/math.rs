use num_bigint::BigInt;
use num_traits::{Signed, ToPrimitive, Zero};
use winit::dpi::PhysicalPosition;

pub const INITIAL_FRAC_BITS: u32 = 160;
pub const MIN_FRAC_BITS: u32 = 128;
pub const MAX_FRAC_BITS: u32 = 16_384;

#[derive(Clone, Debug)]
pub struct BigFixed {
    pub raw: BigInt,
    pub frac_bits: u32,
}

impl BigFixed {
    pub fn from_f64(value: f64, frac_bits: u32) -> Self {
        if !value.is_finite() || value == 0.0 {
            return Self {
                raw: BigInt::zero(),
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

        let mut raw = BigInt::from(mantissa);
        let shift = exponent + frac_bits as i32;
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
                raw: &self.raw << (new_frac_bits - self.frac_bits),
                frac_bits: new_frac_bits,
            },
            std::cmp::Ordering::Less => Self {
                raw: &self.raw >> (self.frac_bits - new_frac_bits),
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

        let bits = self.raw.bits();
        if bits == 0 {
            return None;
        }

        Some((bits as f64 - self.frac_bits as f64) * std::f64::consts::LOG10_2)
    }

    pub fn to_f64(&self) -> f64 {
        if self.raw.is_zero() {
            return 0.0;
        }

        let negative = self.raw.is_negative();
        let abs = self.raw.abs();
        let bits = abs.bits() as i32;
        let mantissa_bits = 53_i32;
        let shift = (bits - mantissa_bits).max(0);
        let mantissa = (&abs >> shift as usize).to_u64().unwrap_or(u64::MAX);
        let exponent = shift - self.frac_bits as i32;
        let value = (mantissa as f64) * 2_f64.powi(exponent);

        if negative { -value } else { value }
    }
}

pub fn mul_fixed_raw(lhs: &BigInt, rhs: &BigInt, frac_bits: u32) -> BigInt {
    (lhs * rhs) >> frac_bits
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

        let target_frac_bits = if zoom_factor < 1.0 {
            (self.frac_bits + 28).min(MAX_FRAC_BITS)
        } else {
            self.frac_bits.saturating_sub(10).max(MIN_FRAC_BITS)
        };
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
        format!(
            "fractals-rs | Mandelbrot | scale {:.3e} | frac {}",
            self.scale.to_f64().abs(),
            self.frac_bits
        )
    }
}
