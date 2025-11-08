//! Functions for unifying images with different numeric types.
//!
//! This module provides functionality to take two `ImageBuffers` with potentially different
//! subpixel types and return a unified `ImageBuffer` where both images are converted to
//! the larger numeric type with proper value normalization.
//!
//! # Value Normalization
//!
//! When converting between different numeric types, values are properly normalized:
//! - `u8` (0-255) to `u16` (0-65535): value * 65535 / 255
//! - `u8` (0-255) to `f32/f64` (0.0-1.0): value / 255.0
//! - `u16` (0-65535) to `f32/f64` (0.0-1.0): value / 65535.0
//! - And so on for all supported conversions

use image::{Luma, Pixel, Primitive, Rgb};
use imageproc::definitions::Image;
use imageproc::map::{WithChannel, map_colors};

type UnifiedRgbImages<T, U> = (
    Image<Rgb<<T as LargerType<U>>::Output>>,
    Image<Rgb<<T as LargerType<U>>::Output>>,
);

type UnifiedGrayImages<T, U> = (
    Image<Luma<<T as LargerType<U>>::Output>>,
    Image<Luma<<T as LargerType<U>>::Output>>,
);

/// Trait for determining the larger of two numeric types.
/// This is used to unify two images with different subpixel types.
pub trait LargerType<T> {
    /// The larger of the two types.
    type Output;
}

/// Trait for normalized conversion between image value types.
pub trait NormalizedFrom<T> {
    /// Convert from type T with proper normalization.
    fn normalized_from(value: T) -> Self;
}

/// Get the maximum value for a primitive type.
fn max_value_impl<T: Primitive>() -> f64 {
    match core::any::type_name::<T>() {
        "u8" => 255.0,
        "u16" => 65535.0,
        "u32" => f64::from(u32::MAX),
        "u64" => u64::MAX as f64,
        "i8" => f64::from(i8::MAX),
        "i16" => f64::from(i16::MAX),
        "i32" => f64::from(i32::MAX),
        "i64" => i64::MAX as f64,
        "f32" | "f64" => 1.0,
        _ => 1.0,
    }
}

/// Implement normalized conversion from one type to another.
macro_rules! impl_normalized_from {
    ($from:ty, $to:ty) => {
        impl NormalizedFrom<$from> for $to {
            fn normalized_from(value: $from) -> Self {
                let from_max = max_value_impl::<$from>();
                let to_max = max_value_impl::<$to>();

                if from_max == 1.0 || to_max == 1.0 {
                    // One of the types is floating point
                    if from_max == 1.0 {
                        // From float to int/float
                        let normalized = value as f64;
                        (normalized * to_max) as $to
                    } else {
                        // From int to float
                        (value as f64 / from_max) as $to
                    }
                } else {
                    // Both are integers
                    ((value as f64 / from_max) * to_max).round() as $to
                }
            }
        }
    };
    ($t:ty) => {
        impl NormalizedFrom<$t> for $t {
            fn normalized_from(value: $t) -> Self {
                value
            }
        }
    };
}

// Implement normalized conversions for all type combinations
// Self conversions
impl_normalized_from!(u8);
impl_normalized_from!(u16);
impl_normalized_from!(u32);
impl_normalized_from!(u64);
impl_normalized_from!(i8);
impl_normalized_from!(i16);
impl_normalized_from!(i32);
impl_normalized_from!(i64);
impl_normalized_from!(f32);
impl_normalized_from!(f64);

// u8 conversions
impl_normalized_from!(u8, u16);
impl_normalized_from!(u8, u32);
impl_normalized_from!(u8, u64);
impl_normalized_from!(u8, i16);
impl_normalized_from!(u8, i32);
impl_normalized_from!(u8, i64);
impl_normalized_from!(u8, f32);
impl_normalized_from!(u8, f64);

// u16 conversions
impl_normalized_from!(u16, u8);
impl_normalized_from!(u16, u32);
impl_normalized_from!(u16, u64);
impl_normalized_from!(u16, i32);
impl_normalized_from!(u16, i64);
impl_normalized_from!(u16, f32);
impl_normalized_from!(u16, f64);

// u32 conversions
impl_normalized_from!(u32, u8);
impl_normalized_from!(u32, u16);
impl_normalized_from!(u32, u64);
impl_normalized_from!(u32, i64);
impl_normalized_from!(u32, f32);
impl_normalized_from!(u32, f64);

// u64 conversions
impl_normalized_from!(u64, u8);
impl_normalized_from!(u64, u16);
impl_normalized_from!(u64, u32);
impl_normalized_from!(u64, f32);
impl_normalized_from!(u64, f64);

// i8 conversions
impl_normalized_from!(i8, i16);
impl_normalized_from!(i8, i32);
impl_normalized_from!(i8, i64);
impl_normalized_from!(i8, f32);
impl_normalized_from!(i8, f64);

// i16 conversions
impl_normalized_from!(i16, i8);
impl_normalized_from!(i16, i32);
impl_normalized_from!(i16, i64);
impl_normalized_from!(i16, f32);
impl_normalized_from!(i16, f64);

// i32 conversions
impl_normalized_from!(i32, i8);
impl_normalized_from!(i32, i16);
impl_normalized_from!(i32, i64);
impl_normalized_from!(i32, f32);
impl_normalized_from!(i32, f64);

// i64 conversions
impl_normalized_from!(i64, i8);
impl_normalized_from!(i64, i16);
impl_normalized_from!(i64, i32);
impl_normalized_from!(i64, f32);
impl_normalized_from!(i64, f64);

// f32 conversions
impl_normalized_from!(f32, u8);
impl_normalized_from!(f32, u16);
impl_normalized_from!(f32, u32);
impl_normalized_from!(f32, i8);
impl_normalized_from!(f32, i16);
impl_normalized_from!(f32, i32);
impl_normalized_from!(f32, f64);

// f64 conversions
impl_normalized_from!(f64, u8);
impl_normalized_from!(f64, u16);
impl_normalized_from!(f64, u32);
impl_normalized_from!(f64, i8);
impl_normalized_from!(f64, i16);
impl_normalized_from!(f64, i32);
impl_normalized_from!(f64, f32);

/// Macro to implement `LargerType` for two types, where the second type is larger.
macro_rules! impl_larger_type {
    ($smaller:ty, $larger:ty) => {
        impl LargerType<$larger> for $smaller {
            type Output = $larger;
        }
        impl LargerType<$smaller> for $larger {
            type Output = $larger;
        }
    };
}

/// Macro to implement `LargerType` for a type with itself.
macro_rules! impl_larger_type_self {
    ($type:ty) => {
        impl LargerType<$type> for $type {
            type Output = $type;
        }
    };
}

// Implement LargerType for all combinations
impl_larger_type_self!(u8);
impl_larger_type_self!(u16);
impl_larger_type_self!(u32);
impl_larger_type_self!(u64);
impl_larger_type_self!(i8);
impl_larger_type_self!(i16);
impl_larger_type_self!(i32);
impl_larger_type_self!(i64);
impl_larger_type_self!(f32);
impl_larger_type_self!(f64);

// Unsigned integer hierarchy: u8 < u16 < u32 < u64
impl_larger_type!(u8, u16);
impl_larger_type!(u8, u32);
impl_larger_type!(u8, u64);
impl_larger_type!(u16, u32);
impl_larger_type!(u16, u64);
impl_larger_type!(u32, u64);

// Signed integer hierarchy: i8 < i16 < i32 < i64
impl_larger_type!(i8, i16);
impl_larger_type!(i8, i32);
impl_larger_type!(i8, i64);
impl_larger_type!(i16, i32);
impl_larger_type!(i16, i64);
impl_larger_type!(i32, i64);

// Float hierarchy: f32 < f64
impl_larger_type!(f32, f64);

// Mixed type promotions: integers promote to larger floats
impl_larger_type!(u8, f32);
impl_larger_type!(u16, f32);
impl_larger_type!(u8, f64);
impl_larger_type!(u16, f64);
impl_larger_type!(u32, f64);
impl_larger_type!(u64, f64);
impl_larger_type!(i8, f32);
impl_larger_type!(i16, f32);
impl_larger_type!(i8, f64);
impl_larger_type!(i16, f64);
impl_larger_type!(i32, f64);
impl_larger_type!(i64, f64);

// Cross-sign promotions: promote to larger signed type or float
impl_larger_type!(u8, i16);
impl_larger_type!(u8, i32);
impl_larger_type!(u8, i64);
impl_larger_type!(u16, i32);
impl_larger_type!(u16, i64);
impl_larger_type!(u32, i64);
impl_larger_type!(i8, u16);
impl_larger_type!(i8, u32);
impl_larger_type!(i8, u64);
impl_larger_type!(i16, u32);
impl_larger_type!(i16, u64);
impl_larger_type!(i32, u64);

/// Unifies two RGB images with different subpixel types to a common type.
///
/// # Examples
/// ```no_run
/// use image::{ImageBuffer, Rgb};
/// use imageops_kit::unify_rgb_images;
///
/// let image1: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::from_raw(2, 2, vec![
///     10, 20, 30, 40, 50, 60,
///     70, 80, 90, 100, 110, 120
/// ]).unwrap();
///
/// let image2: ImageBuffer<Rgb<u16>, Vec<u16>> = ImageBuffer::from_raw(2, 2, vec![
///     1000, 2000, 3000, 4000, 5000, 6000,
///     7000, 8000, 9000, 10000, 11000, 12000
/// ]).unwrap();
///
/// let (unified1, unified2) = unify_rgb_images(&image1, &image2);
/// // Both are now RGB<u16> images
/// ```
#[must_use]
pub fn unify_rgb_images<T, U>(
    first_image: &Image<Rgb<T>>,
    second_image: &Image<Rgb<U>>,
) -> UnifiedRgbImages<T, U>
where
    T: LargerType<U> + Primitive,
    U: Primitive,
    <T as LargerType<U>>::Output: Primitive + NormalizedFrom<T> + NormalizedFrom<U>,
    Rgb<T>: WithChannel<<T as LargerType<U>>::Output>,
    Rgb<U>: WithChannel<<T as LargerType<U>>::Output>,
    Rgb<<T as LargerType<U>>::Output>: Pixel<Subpixel = <T as LargerType<U>>::Output>,
{
    let unified1 = map_colors(first_image, |x| {
        Rgb([
            <T as LargerType<U>>::Output::normalized_from(x[0]),
            <T as LargerType<U>>::Output::normalized_from(x[1]),
            <T as LargerType<U>>::Output::normalized_from(x[2]),
        ])
    });
    let unified2 = map_colors(second_image, |x| {
        Rgb([
            <T as LargerType<U>>::Output::normalized_from(x[0]),
            <T as LargerType<U>>::Output::normalized_from(x[1]),
            <T as LargerType<U>>::Output::normalized_from(x[2]),
        ])
    });
    (unified1, unified2)
}

/// Unifies two grayscale images with different subpixel types to a common type.
///
/// # Examples
/// ```no_run
/// use image::{ImageBuffer, Luma};
/// use imageops_kit::unify_gray_images;
///
/// let image1: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::from_raw(2, 2, vec![
///     10, 20, 30, 40
/// ]).unwrap();
///
/// let image2: ImageBuffer<Luma<u16>, Vec<u16>> = ImageBuffer::from_raw(2, 2, vec![
///     1000, 2000, 3000, 4000
/// ]).unwrap();
///
/// let (unified1, unified2) = unify_gray_images(&image1, &image2);
/// // Both are now Luma<u16> images
/// ```
#[must_use]
pub fn unify_gray_images<T, U>(
    first_image: &Image<Luma<T>>,
    second_image: &Image<Luma<U>>,
) -> UnifiedGrayImages<T, U>
where
    T: LargerType<U> + Primitive,
    U: Primitive,
    <T as LargerType<U>>::Output: Primitive + NormalizedFrom<T> + NormalizedFrom<U>,
    Luma<T>: WithChannel<<T as LargerType<U>>::Output>,
    Luma<U>: WithChannel<<T as LargerType<U>>::Output>,
{
    let unified1 = map_colors(first_image, |x| {
        Luma([<T as LargerType<U>>::Output::normalized_from(x[0])])
    });
    let unified2 = map_colors(second_image, |x| {
        Luma([<T as LargerType<U>>::Output::normalized_from(x[0])])
    });
    (unified1, unified2)
}

#[cfg(test)]
mod tests {
    use super::*;

    use imageproc::{gray_image, rgb_image};

    #[test]
    fn unify_gray_images_with_u8_to_u16_converts_correctly() {
        let image_u8 = gray_image!(
            10, 20;
            30, 40);

        let image_u16 = gray_image!(type: u16,
            1000, 2000;
            3000, 4000);

        let (unified1, unified2) = unify_gray_images(&image_u8, &image_u16);

        // Check that conversion worked correctly with normalization
        // u8 to u16: value * 65535 / 255
        assert_eq!(unified1.get_pixel(0, 0).0[0], (10u32 * 65535 / 255) as u16);
        assert_eq!(unified1.get_pixel(1, 0).0[0], (20u32 * 65535 / 255) as u16);
        assert_eq!(unified2.get_pixel(0, 0).0[0], 1000u16);
        assert_eq!(unified2.get_pixel(1, 0).0[0], 2000u16);
    }

    #[test]
    fn unify_rgb_images_with_u8_to_u16_converts_correctly() {
        let image_u8 = rgb_image!([10, 20, 30], [40, 50, 60]);

        let image_u16 = rgb_image!(type: u16,
            [1000, 2000, 3000], [4000, 5000, 6000]);

        let (unified1, unified2) = unify_rgb_images(&image_u8, &image_u16);

        // Check that conversion worked correctly with normalization
        // u8 to u16: value * 65535 / 255
        assert_eq!(
            unified1.get_pixel(0, 0).0,
            [
                (10u32 * 65535 / 255) as u16,
                (20u32 * 65535 / 255) as u16,
                (30u32 * 65535 / 255) as u16
            ]
        );
        assert_eq!(
            unified1.get_pixel(1, 0).0,
            [
                (40u32 * 65535 / 255) as u16,
                (50u32 * 65535 / 255) as u16,
                (60u32 * 65535 / 255) as u16
            ]
        );
        assert_eq!(unified2.get_pixel(0, 0).0, [1000u16, 2000u16, 3000u16]);
        assert_eq!(unified2.get_pixel(1, 0).0, [4000u16, 5000u16, 6000u16]);
    }

    #[test]
    fn unify_gray_images_with_same_types_preserves_types() {
        let image1 = gray_image!(10, 20);
        let image2 = gray_image!(30, 40);

        let (unified1, unified2) = unify_gray_images(&image1, &image2);

        // Should work even with same types
        assert_eq!(unified1.get_pixel(0, 0).0[0], 10u8);
        assert_eq!(unified2.get_pixel(0, 0).0[0], 30u8);
    }

    #[test]
    fn unify_gray_images_with_u8_to_f32_converts_correctly() {
        let image_u8 = gray_image!(100, 200);

        let image_f32 = gray_image!(type: f32,
            0.5, 0.8);

        let (unified1, unified2) = unify_gray_images(&image_u8, &image_f32);

        // u8 should be converted to f32 with normalization
        // u8 to f32: value / 255.0
        assert_eq!(unified1.get_pixel(0, 0).0[0], 100.0f32 / 255.0);
        assert_eq!(unified1.get_pixel(1, 0).0[0], 200.0f32 / 255.0);
        assert_eq!(unified2.get_pixel(0, 0).0[0], 0.5f32);
        assert_eq!(unified2.get_pixel(1, 0).0[0], 0.8f32);
    }

    #[test]
    fn normalized_from_with_various_types_converts_correctly() {
        // Test u8 to u16
        assert_eq!(u16::normalized_from(0u8), 0u16);
        assert_eq!(u16::normalized_from(255u8), 65535u16);
        assert_eq!(u16::normalized_from(127u8), 32639u16); // Close to 127*65535/255

        // Test u8 to f32
        assert_eq!(f32::normalized_from(0u8), 0.0f32);
        assert_eq!(f32::normalized_from(255u8), 1.0f32);
        assert!((f32::normalized_from(127u8) - 127.0f32 / 255.0).abs() < 0.001);

        // Test u16 to f32
        assert_eq!(f32::normalized_from(0u16), 0.0f32);
        assert_eq!(f32::normalized_from(65535u16), 1.0f32);
        assert!((f32::normalized_from(32768u16) - 0.5).abs() < 0.001);

        // Test f32 to u8
        assert_eq!(u8::normalized_from(0.0f32), 0u8);
        assert_eq!(u8::normalized_from(1.0f32), 255u8);
        assert_eq!(u8::normalized_from(0.5f32), 127u8);

        // Test f32 to u16
        assert_eq!(u16::normalized_from(0.0f32), 0u16);
        assert_eq!(u16::normalized_from(1.0f32), 65535u16);
        assert_eq!(u16::normalized_from(0.5f32), 32767u16);
    }
}
