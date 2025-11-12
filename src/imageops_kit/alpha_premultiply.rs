use image::{GenericImageView, ImageBuffer, Luma, LumaA, Pixel, Primitive, Rgb, Rgba};
use imageproc::{
    definitions::{Clamp, Image},
    map::map_colors,
};
use itertools::Itertools;

use crate::{error::ColorConversionError, utils::normalize_alpha_with_max};

/// Trait for merging (premultiplying) alpha channel into color channels and discarding alpha.
///
/// This operation multiplies each color channel by the alpha value,
/// effectively creating a premultiplied alpha image. The alpha channel
/// is discarded in the output.
///
/// # Alpha Premultiplication
///
/// Alpha premultiplication is the process of multiplying the color channels
/// by the alpha value, resulting in:
/// - Red' = Red × Alpha
/// - Green' = Green × Alpha
/// - Blue' = Blue × Alpha
/// - Luminance' = Luminance × Alpha
///
/// This is commonly used in compositing operations.
///
/// Note: This trait performs type conversion (e.g., Rgba -> Rgb). For in-place
/// premultiplication while keeping the alpha channel, use the `PremultiplyAlphaAndKeepExt` trait.
pub trait PremultiplyAlphaAndDropExt {
    type Output;

    /// Premultiplies color channels by alpha and returns an image without alpha channel.
    ///
    /// This consumes the original image.
    ///
    /// # Returns
    /// * `Ok(Self::Output)` - Successfully premultiplied image
    /// * `Err(ColorConversionError)` - If conversion fails
    ///
    /// # Panics
    /// This function does not panic. All error conditions are handled through `Result`.
    ///
    /// # Examples
    /// ```no_run
    /// use imageops_kit::PremultiplyAlphaAndDropExt;
    /// use imageproc::definitions::Image;
    /// use image::Rgba;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
    /// let rgb_image = rgba_image.premultiply_alpha_and_drop()?;
    /// # Ok(())
    /// # }
    /// ```
    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError>;
}

/// Trait for alpha premultiplication that keeps the alpha channel.
///
/// This trait provides functionality to premultiply color channels with alpha
/// while preserving the alpha channel in the output.
pub trait PremultiplyAlphaAndKeepExt {
    /// Premultiplies color channels by alpha, keeping the alpha channel.
    ///
    /// This consumes the original image and returns a premultiplied version
    /// with the same pixel type.
    ///
    /// # Returns
    /// * `Ok(Self)` - Successfully premultiplied image with alpha
    /// * `Err(ColorConversionError)` - If conversion fails
    ///
    /// # Examples
    /// ```no_run
    /// use imageops_kit::PremultiplyAlphaAndKeepExt;
    /// use imageproc::definitions::Image;
    /// use image::Rgba;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
    /// let premultiplied = rgba_image.premultiply_alpha_and_keep()?;
    /// # Ok(())
    /// # }
    /// ```
    fn premultiply_alpha_and_keep(self) -> Result<Self, ColorConversionError>
    where
        Self: Sized;

    /// Premultiplies color channels by alpha in-place, modifying the image.
    ///
    /// # Returns
    /// * `Ok(&mut Self)` - Successfully modified image
    /// * `Err(ColorConversionError)` - If conversion fails
    ///
    /// # Examples
    /// ```no_run
    /// use imageops_kit::PremultiplyAlphaAndKeepExt;
    /// use imageproc::definitions::Image;
    /// use image::Rgba;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let mut rgba_image: Image<Rgba<u8>> = Image::new(100, 100);
    /// rgba_image.premultiply_alpha_and_keep_mut()?;
    /// # Ok(())
    /// # }
    /// ```
    fn premultiply_alpha_and_keep_mut(&mut self) -> Result<&mut Self, ColorConversionError>
    where
        Self: Sized;
}

/// Generic fallback implementation for `LumaA` -> Luma conversion with alpha premultiplication
#[inline]
fn premultiply_lumaa_impl<S>(
    image: &Image<LumaA<S>>,
) -> Result<Image<Luma<S>>, ColorConversionError>
where
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    validate_image_dimensions(image)?;

    let max_value = f32::from(S::DEFAULT_MAX_VALUE);

    Ok(map_colors(image, |pixel| {
        let LumaA([luminance, alpha]) = pixel;
        let alpha_normalized = normalize_alpha_with_max(alpha, max_value);
        let luminance = f32::from(luminance);

        let premultiplied = luminance * alpha_normalized;
        let clamped = S::clamp(premultiplied);

        Luma([clamped])
    }))
}

/// Generic fallback implementation for Rgba -> Rgb conversion with alpha premultiplication
#[inline]
fn premultiply_rgba_impl<S>(image: &Image<Rgba<S>>) -> Result<Image<Rgb<S>>, ColorConversionError>
where
    Rgba<S>: Pixel<Subpixel = S>,
    Rgb<S>: Pixel<Subpixel = S>,
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    validate_image_dimensions(image)?;

    let max_value = f32::from(S::DEFAULT_MAX_VALUE);

    Ok(map_colors(image, |pixel| {
        let Rgba([red, green, blue, alpha]) = pixel;
        let alpha_normalized = normalize_alpha_with_max(alpha, max_value);

        compute_premultiplied_rgb_impl([red, green, blue], alpha_normalized)
    }))
}

/// Implementation for f32 `LumaA` -> Luma conversion
impl PremultiplyAlphaAndDropExt for Image<LumaA<f32>> {
    type Output = Image<Luma<f32>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        premultiply_lumaa_impl(&self)
    }
}

/// Implementation for u16 `LumaA` -> Luma conversion
impl PremultiplyAlphaAndDropExt for Image<LumaA<u16>> {
    type Output = Image<Luma<u16>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        map_pixels_to_new_image(&self, |&LumaA([luminance, alpha])| {
            Luma([premultiply_u16(luminance, alpha)])
        })
    }
}

/// Implementation for u8 `LumaA` -> Luma conversion using LUT
impl PremultiplyAlphaAndDropExt for Image<LumaA<u8>> {
    type Output = Image<Luma<u8>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        map_pixels_to_new_image(&self, |&LumaA([luminance, alpha])| {
            Luma([premultiply_u8(luminance, alpha)])
        })
    }
}

/// Implementation for f32 Rgba -> Rgb conversion
impl PremultiplyAlphaAndDropExt for Image<Rgba<f32>> {
    type Output = Image<Rgb<f32>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        premultiply_rgba_impl(&self)
    }
}

/// Implementation for u8 Rgba -> Rgb conversion using LUT
impl PremultiplyAlphaAndDropExt for Image<Rgba<u8>> {
    type Output = Image<Rgb<u8>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        map_pixels_to_new_image(&self, |&Rgba([red, green, blue, alpha])| {
            Rgb(premultiply_rgb_u8([red, green, blue], alpha))
        })
    }
}

/// Implementation for u16 Rgba -> Rgb conversion using integer arithmetic
impl PremultiplyAlphaAndDropExt for Image<Rgba<u16>> {
    type Output = Image<Rgb<u16>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        map_pixels_to_new_image(&self, |&Rgba([red, green, blue, alpha])| {
            Rgb([
                premultiply_u16(red, alpha),
                premultiply_u16(green, alpha),
                premultiply_u16(blue, alpha),
            ])
        })
    }
}

/// Implementation for u32 Rgba -> Rgb conversion using 64-bit integer arithmetic
impl PremultiplyAlphaAndDropExt for Image<Rgba<u32>> {
    type Output = Image<Rgb<u32>>;

    fn premultiply_alpha_and_drop(self) -> Result<Self::Output, ColorConversionError> {
        map_pixels_to_new_image(&self, |&Rgba([red, green, blue, alpha])| {
            Rgb([
                premultiply_u32(red, alpha),
                premultiply_u32(green, alpha),
                premultiply_u32(blue, alpha),
            ])
        })
    }
}

/// Implementation for f32 `LumaA` images to premultiply while keeping alpha
impl PremultiplyAlphaAndKeepExt for Image<LumaA<f32>> {
    fn premultiply_alpha_and_keep(self) -> Result<Self, ColorConversionError> {
        validate_image_dimensions(&self)?;

        // Constant prevents repeated runtime computation of max value in per-pixel loop
        const MAX_VALUE: f32 = 1.0;

        Ok(map_colors(&self, |pixel| {
            let LumaA([luminance, alpha]) = pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, MAX_VALUE);
            let luminance: f32 = luminance;

            let premultiplied: f32 = luminance * alpha_normalized;
            let clamped = premultiplied.clamp(0.0, MAX_VALUE);

            LumaA([clamped, alpha])
        }))
    }

    fn premultiply_alpha_and_keep_mut(&mut self) -> Result<&mut Self, ColorConversionError> {
        validate_image_dimensions(self)?;

        const MAX_VALUE: f32 = 1.0;

        self.pixels_mut().for_each(|pixel| {
            let LumaA([luminance, alpha]) = *pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, MAX_VALUE);
            let premultiplied = (luminance * alpha_normalized).clamp(0.0, MAX_VALUE);
            *pixel = LumaA([premultiplied, alpha]);
        });

        Ok(self)
    }
}

/// Implementation for u8 `LumaA` images using LUT
impl PremultiplyAlphaAndKeepExt for Image<LumaA<u8>> {
    fn premultiply_alpha_and_keep(self) -> Result<Self, ColorConversionError> {
        map_pixels_to_new_image(&self, |&LumaA([luminance, alpha])| {
            LumaA([premultiply_u8(luminance, alpha), alpha])
        })
    }

    fn premultiply_alpha_and_keep_mut(&mut self) -> Result<&mut Self, ColorConversionError> {
        validate_image_dimensions(self)?;

        self.pixels_mut().for_each(|pixel| {
            let LumaA([luminance, alpha]) = *pixel;
            *pixel = LumaA([premultiply_u8(luminance, alpha), alpha]);
        });

        Ok(self)
    }
}

/// Implementation for f32 Rgba images to premultiply while keeping alpha
impl PremultiplyAlphaAndKeepExt for Image<Rgba<f32>> {
    fn premultiply_alpha_and_keep(self) -> Result<Self, ColorConversionError> {
        validate_image_dimensions(&self)?;

        const MAX_VALUE: f32 = 1.0;

        Ok(map_colors(&self, |pixel| {
            let Rgba([red, green, blue, alpha]) = pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, MAX_VALUE);

            let premultiplied =
                compute_premultiplied_rgb_impl([red, green, blue], alpha_normalized);
            let Rgb([r_pre, g_pre, b_pre]) = premultiplied;

            Rgba([r_pre, g_pre, b_pre, alpha])
        }))
    }

    fn premultiply_alpha_and_keep_mut(&mut self) -> Result<&mut Self, ColorConversionError> {
        validate_image_dimensions(self)?;

        const MAX_VALUE: f32 = 1.0;

        self.pixels_mut().for_each(|pixel| {
            let Rgba([red, green, blue, alpha]) = *pixel;
            let alpha_normalized = normalize_alpha_with_max(alpha, MAX_VALUE);

            let Rgb([r_pre, g_pre, b_pre]) =
                compute_premultiplied_rgb_impl([red, green, blue], alpha_normalized);
            *pixel = Rgba([r_pre, g_pre, b_pre, alpha]);
        });

        Ok(self)
    }
}

/// Implementation for u8 Rgba images using LUT
impl PremultiplyAlphaAndKeepExt for Image<Rgba<u8>> {
    fn premultiply_alpha_and_keep(self) -> Result<Self, ColorConversionError> {
        map_pixels_to_new_image(&self, |&Rgba([red, green, blue, alpha])| {
            let [r, g, b] = premultiply_rgb_u8([red, green, blue], alpha);
            Rgba([r, g, b, alpha])
        })
    }

    fn premultiply_alpha_and_keep_mut(&mut self) -> Result<&mut Self, ColorConversionError> {
        validate_image_dimensions(self)?;

        self.pixels_mut().for_each(|pixel| {
            let Rgba([red, green, blue, alpha]) = *pixel;
            let [r, g, b] = premultiply_rgb_u8([red, green, blue], alpha);
            *pixel = Rgba([r, g, b, alpha]);
        });

        Ok(self)
    }
}

/// Compile-time Look-Up Table generator for u8 alpha premultiplication.
///
/// Integer approximation avoids expensive division in per-pixel loops.
/// Formula: ((x * a + 2^(N-1)) * (2^N + 1)) >> (2N) approximates (x * a) / (2^N - 1)
/// For N=8: ((x * a + 128) * 257) >> 16 approximates (x * a) / 255 with rounding
const fn generate_alpha_lut() -> [[u8; 256]; 256] {
    let mut lut = [[0u8; 256]; 256];
    let mut alpha = 0;
    while alpha < 256 {
        let mut color = 0;
        while color < 256 {
            let n = color * alpha;
            lut[alpha][color] = (((n + 128) * 257) >> 16) as u8;
            color += 1;
        }
        alpha += 1;
    }
    lut
}

/// Precomputed LUT eliminates division from hot path for u8 images
static ALPHA_LUT: [[u8; 256]; 256] = generate_alpha_lut();

/// u8 alpha premultiplication using LUT for O(1) lookup instead of division
#[inline]
const fn premultiply_u8(color: u8, alpha: u8) -> u8 {
    ALPHA_LUT[alpha as usize][color as usize]
}

/// u8 RGB premultiplication by applying LUT to each channel independently
#[inline]
const fn premultiply_rgb_u8(channels: [u8; 3], alpha: u8) -> [u8; 3] {
    [
        premultiply_u8(channels[0], alpha),
        premultiply_u8(channels[1], alpha),
        premultiply_u8(channels[2], alpha),
    ]
}

/// Computes premultiplied RGB pixel and clamps to valid range
#[inline]
fn compute_premultiplied_rgb_impl<S>(channels: [S; 3], alpha_normalized: f32) -> Rgb<S>
where
    S: Clamp<f32> + Primitive,
    f32: From<S>,
{
    let [r, g, b] = channels;
    let r = f32::from(r) * alpha_normalized;
    let g = f32::from(g) * alpha_normalized;
    let b = f32::from(b) * alpha_normalized;

    Rgb([S::clamp(r), S::clamp(g), S::clamp(b)])
}

/// Integer premultiplication for u16 using approximation to avoid division.
///
/// Integer shifts are faster than division on most architectures.
/// Formula: ((x * a + 2^(N-1)) * (2^N + 1)) >> (2N) approximates (x * a) / (2^N - 1)
/// For N=16: ((x * a + 32768) * 65537) >> 32 approximates (x * a) / 65535 with rounding
#[inline]
const fn premultiply_u16(color: u16, alpha: u16) -> u16 {
    let n = color as u32 * alpha as u32;
    let result = ((n + 32768) as u64 * 65537) >> 32;
    result as u16
}

/// Integer premultiplication for u32 using division instead of approximation.
///
/// Integer approximation would require u128 or lose precision for u32.
/// Division is acceptable here because u32 images are rare and precision
/// is critical for image quality.
#[inline]
const fn premultiply_u32(color: u32, alpha: u32) -> u32 {
    let product = color as u64 * alpha as u64;
    let rounded = product + (u32::MAX as u64 >> 1);
    (rounded / u32::MAX as u64) as u32
}

/// Creates a new image by processing each pixel with a mapping function.
///
/// This function applies function fusion by inlining the pixel iteration,
/// allowing LLVM to better optimize the complete pipeline. The sequential
/// iteration provides cache-friendly memory access patterns.
#[inline]
fn map_pixels_to_new_image<SP, DP, F>(
    src_image: &Image<SP>,
    mapper: F,
) -> Result<Image<DP>, ColorConversionError>
where
    SP: Pixel,
    DP: Pixel,
    F: Fn(&SP) -> DP,
{
    validate_image_dimensions(src_image)?;

    let (width, height) = src_image.dimensions();
    let mut out = ImageBuffer::new(width, height);

    // Function fusion: inline the iteration for better optimization
    src_image
        .pixels()
        .zip_eq(out.pixels_mut())
        .for_each(|(src_pixel, dst_pixel)| {
            *dst_pixel = mapper(src_pixel);
        });

    Ok(out)
}

/// Validates image dimensions for processing
#[inline]
fn validate_image_dimensions<I>(image: &I) -> Result<(), ColorConversionError>
where
    I: GenericImageView,
{
    let (width, height) = image.dimensions();

    if width == 0 || height == 0 {
        Err(ColorConversionError::EmptyImage)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_premultiplied_rgb_impl_with_valid_input_returns_correct_values() {
        let channels = [200u8, 150u8, 100u8];
        let alpha = 0.5;
        let result = compute_premultiplied_rgb_impl(channels, alpha);

        assert_eq!(result[0], 100); // 200 * 0.5
        assert_eq!(result[1], 75); // 150 * 0.5
        assert_eq!(result[2], 50); // 100 * 0.5
    }

    #[test]
    fn validate_image_dimensions_with_empty_images_rejects() {
        let valid_image: Image<Rgb<u8>> = Image::new(10, 10);
        validate_image_dimensions(&valid_image).unwrap();

        let empty_image: Image<Rgb<u8>> = Image::new(0, 0);
        assert!(validate_image_dimensions(&empty_image).is_err());

        let invalid_width: Image<Rgb<u8>> = Image::new(0, 10);
        assert!(validate_image_dimensions(&invalid_width).is_err());

        let invalid_height: Image<Rgb<u8>> = Image::new(10, 0);
        assert!(validate_image_dimensions(&invalid_height).is_err());
    }

    #[test]
    fn premultiply_alpha_and_drop_for_luma_drops_alpha_channel() {
        let mut image: Image<LumaA<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, LumaA([200, 255])); // Full opacity
        image.put_pixel(1, 0, LumaA([200, 127])); // Half opacity
        image.put_pixel(0, 1, LumaA([200, 0])); // Transparent
        image.put_pixel(1, 1, LumaA([100, 255])); // Full opacity, different value

        let result = image.premultiply_alpha_and_drop().unwrap();

        assert_eq!(result.get_pixel(0, 0)[0], 200); // 200 * 1.0
        assert_eq!(result.get_pixel(1, 0)[0], 100); // 200 * 127/255
        assert_eq!(result.get_pixel(0, 1)[0], 0); // 200 * 0.0
        assert_eq!(result.get_pixel(1, 1)[0], 100); // 100 * 1.0
    }

    #[test]
    fn premultiply_alpha_and_drop_for_rgba_drops_alpha_channel() {
        let mut image: Image<Rgba<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, Rgba([200, 150, 100, 255])); // Full opacity
        image.put_pixel(1, 0, Rgba([200, 150, 100, 127])); // Half opacity
        image.put_pixel(0, 1, Rgba([200, 150, 100, 0])); // Transparent
        image.put_pixel(1, 1, Rgba([100, 50, 25, 255])); // Full opacity, different values

        let result = image.premultiply_alpha_and_drop().unwrap();

        // Full opacity case
        let pixel_00 = result.get_pixel(0, 0);
        assert_eq!(pixel_00[0], 200);
        assert_eq!(pixel_00[1], 150);
        assert_eq!(pixel_00[2], 100);

        // Half opacity case
        let pixel_10 = result.get_pixel(1, 0);
        assert_eq!(pixel_10[0], 100); // 200 * 127/255
        assert_eq!(pixel_10[1], 75); // 150 * 127/255
        assert_eq!(pixel_10[2], 50); // 100 * 127/255

        // Transparent case
        let pixel_01 = result.get_pixel(0, 1);
        assert_eq!(pixel_01[0], 0);
        assert_eq!(pixel_01[1], 0);
        assert_eq!(pixel_01[2], 0);

        // Full opacity, different values
        let pixel_11 = result.get_pixel(1, 1);
        assert_eq!(pixel_11[0], 100);
        assert_eq!(pixel_11[1], 50);
        assert_eq!(pixel_11[2], 25);
    }

    #[test]
    fn premultiply_alpha_and_keep_for_luma_preserves_alpha() {
        let mut image: Image<LumaA<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, LumaA([200, 255])); // Full opacity
        image.put_pixel(1, 0, LumaA([200, 127])); // Half opacity

        let result = image.clone().premultiply_alpha_and_keep().unwrap();

        // Check that luminance is premultiplied but alpha is preserved
        assert_eq!(result.get_pixel(0, 0).0, [200, 255]); // 200 * 1.0, alpha preserved
        assert_eq!(result.get_pixel(1, 0).0, [100, 127]); // 200 * 127/255, alpha preserved
    }

    #[test]
    fn premultiply_alpha_and_keep_mut_for_rgba_preserves_alpha() {
        let mut image: Image<Rgba<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, Rgba([200, 150, 100, 255])); // Full opacity
        image.put_pixel(1, 0, Rgba([200, 150, 100, 127])); // Half opacity

        let mut image_copy = image.clone();
        image_copy.premultiply_alpha_and_keep_mut().unwrap();

        // Check that colors are premultiplied but alpha is preserved
        assert_eq!(image_copy.get_pixel(0, 0).0, [200, 150, 100, 255]); // Full opacity unchanged
        assert_eq!(image_copy.get_pixel(1, 0).0, [100, 75, 50, 127]); // Premultiplied, alpha preserved
    }

    // Tests for u16 data type premultiplication
    #[test]
    fn premultiply_alpha_and_drop_for_lumaa_u16_drops_alpha_channel() {
        let mut image: Image<LumaA<u16>> = Image::new(2, 2);
        image.put_pixel(0, 0, LumaA([32768, 65535])); // Full opacity, half luminance
        image.put_pixel(1, 0, LumaA([65535, 32768])); // Half opacity, full luminance

        let result = image.premultiply_alpha_and_drop().unwrap();

        assert_eq!(result.get_pixel(0, 0)[0], 32768); // 32768 * 1.0
        assert_eq!(result.get_pixel(1, 0)[0], 32768); // 65535 * 0.5 ≈ 32768
    }

    #[test]
    fn premultiply_alpha_and_drop_for_rgba_u16_drops_alpha_channel() {
        let mut image: Image<Rgba<u16>> = Image::new(2, 2);
        image.put_pixel(0, 0, Rgba([32768, 16384, 8192, 65535])); // Full opacity
        image.put_pixel(1, 0, Rgba([65535, 32768, 16384, 32768])); // Half opacity

        let result = image.premultiply_alpha_and_drop().unwrap();

        let pixel_00 = result.get_pixel(0, 0);
        assert_eq!(pixel_00[0], 32768);
        assert_eq!(pixel_00[1], 16384);
        assert_eq!(pixel_00[2], 8192);

        let pixel_10 = result.get_pixel(1, 0);
        assert_eq!(pixel_10[0], 32768); // 65535 * 0.5 ≈ 32768
        assert_eq!(pixel_10[1], 16384); // 32768 * 0.5 ≈ 16384
        assert_eq!(pixel_10[2], 8192); // 16384 * 0.5 ≈ 8192
    }

    // Tests for u32 data type premultiplication
    #[test]
    fn premultiply_alpha_and_drop_for_rgba_u32_drops_alpha_channel() {
        let mut image: Image<Rgba<u32>> = Image::new(2, 2);
        image.put_pixel(0, 0, Rgba([2147483648, 1073741824, 536870912, u32::MAX])); // Full opacity
        image.put_pixel(1, 0, Rgba([u32::MAX, 2147483648, 1073741824, 2147483648])); // Half opacity

        let result = image.premultiply_alpha_and_drop().unwrap();

        let pixel_00 = result.get_pixel(0, 0);
        assert_eq!(pixel_00[0], 2147483648);
        assert_eq!(pixel_00[1], 1073741824);
        assert_eq!(pixel_00[2], 536870912);

        let pixel_10 = result.get_pixel(1, 0);
        assert_eq!(pixel_10[0], 2147483648); // u32::MAX * 0.5 ≈ 2147483648
        assert_eq!(pixel_10[1], 1073741824); // 2147483648 * 0.5 ≈ 1073741824
        assert_eq!(pixel_10[2], 536870912); // 1073741824 * 0.5 ≈ 536870912
    }

    // Tests for f32 data type premultiplication
    #[test]
    fn premultiply_alpha_and_drop_for_lumaa_f32_drops_alpha_channel() {
        let mut image: Image<LumaA<f32>> = Image::new(2, 2);
        image.put_pixel(0, 0, LumaA([0.8, 1.0])); // Full opacity
        image.put_pixel(1, 0, LumaA([0.6, 0.5])); // Half opacity

        let result = image.premultiply_alpha_and_drop().unwrap();

        assert!((result.get_pixel(0, 0)[0] - 0.8).abs() < f32::EPSILON);
        assert!((result.get_pixel(1, 0)[0] - 0.3).abs() < f32::EPSILON); // 0.6 * 0.5
    }

    #[test]
    fn premultiply_alpha_and_drop_for_rgba_f32_drops_alpha_channel() {
        let mut image: Image<Rgba<f32>> = Image::new(2, 2);
        image.put_pixel(0, 0, Rgba([0.8, 0.6, 0.4, 1.0])); // Full opacity
        image.put_pixel(1, 0, Rgba([0.8, 0.6, 0.4, 0.5])); // Half opacity

        let result = image.premultiply_alpha_and_drop().unwrap();

        let pixel_00 = result.get_pixel(0, 0);
        assert!((pixel_00[0] - 0.8).abs() < f32::EPSILON);
        assert!((pixel_00[1] - 0.6).abs() < f32::EPSILON);
        assert!((pixel_00[2] - 0.4).abs() < f32::EPSILON);

        let pixel_10 = result.get_pixel(1, 0);
        assert!((pixel_10[0] - 0.4).abs() < f32::EPSILON); // 0.8 * 0.5
        assert!((pixel_10[1] - 0.3).abs() < f32::EPSILON); // 0.6 * 0.5
        assert!((pixel_10[2] - 0.2).abs() < f32::EPSILON); // 0.4 * 0.5
    }

    // Boundary value tests
    #[test]
    fn premultiply_alpha_and_drop_with_zero_alpha_returns_black() {
        let mut image: Image<Rgba<u8>> = Image::new(1, 1);
        image.put_pixel(0, 0, Rgba([255, 128, 64, 0])); // Zero alpha

        let result = image.premultiply_alpha_and_drop().unwrap();

        let pixel = result.get_pixel(0, 0);
        assert_eq!(pixel[0], 0);
        assert_eq!(pixel[1], 0);
        assert_eq!(pixel[2], 0);
    }

    #[test]
    fn premultiply_alpha_and_drop_with_max_values_preserves_color() {
        let mut image: Image<Rgba<u8>> = Image::new(1, 1);
        image.put_pixel(0, 0, Rgba([255, 255, 255, 255])); // Max values

        let result = image.premultiply_alpha_and_drop().unwrap();

        let pixel = result.get_pixel(0, 0);
        assert_eq!(pixel[0], 255);
        assert_eq!(pixel[1], 255);
        assert_eq!(pixel[2], 255);
    }

    #[test]
    fn premultiply_alpha_and_drop_with_zero_alpha_f32_returns_zero() {
        let mut image: Image<Rgba<f32>> = Image::new(1, 1);
        image.put_pixel(0, 0, Rgba([1.0, 0.8, 0.6, 0.0])); // Zero alpha

        let result = image.premultiply_alpha_and_drop().unwrap();

        let pixel = result.get_pixel(0, 0);
        assert!((pixel[0] - 0.0).abs() < f32::EPSILON);
        assert!((pixel[1] - 0.0).abs() < f32::EPSILON);
        assert!((pixel[2] - 0.0).abs() < f32::EPSILON);
    }

    // Tests for LUT accuracy (u8 optimization)
    #[test]
    fn premultiply_u8_with_known_values_matches_expected() {
        assert_eq!(premultiply_u8(255, 255), 255); // Max * Max = Max
        assert_eq!(premultiply_u8(255, 0), 0); // Max * 0 = 0
        assert_eq!(premultiply_u8(0, 255), 0); // 0 * Max = 0
        assert_eq!(premultiply_u8(255, 127), 127); // Max * Half ≈ Half
        assert_eq!(premultiply_u8(128, 255), 128); // Half * Max = Half
    }

    #[test]
    fn premultiply_rgb_u8_with_known_values_matches_expected() {
        let result = premultiply_rgb_u8([255, 128, 64], 127);
        assert_eq!(result[0], 127); // 255 * 127/255 ≈ 127
        assert_eq!(result[1], 64); // 128 * 127/255 ≈ 64
        assert_eq!(result[2], 32); // 64 * 127/255 ≈ 32
    }

    // Tests for integer arithmetic accuracy (u16/u32)
    #[test]
    fn premultiply_u16_with_known_values_matches_expected() {
        assert_eq!(premultiply_u16(65535, 65535), 65535); // Max * Max = Max
        assert_eq!(premultiply_u16(65535, 0), 0); // Max * 0 = 0
        assert_eq!(premultiply_u16(0, 65535), 0); // 0 * Max = 0
        assert_eq!(premultiply_u16(65535, 32768), 32768); // Max * Half ≈ Half
    }

    #[test]
    fn premultiply_u32_with_known_values_matches_expected() {
        assert_eq!(premultiply_u32(u32::MAX, u32::MAX), u32::MAX); // Max * Max = Max
        assert_eq!(premultiply_u32(u32::MAX, 0), 0); // Max * 0 = 0
        assert_eq!(premultiply_u32(0, u32::MAX), 0); // 0 * Max = 0

        let half_max = u32::MAX / 2;
        let result = premultiply_u32(u32::MAX, half_max);
        assert!((i64::from(result) - i64::from(half_max)).abs() <= 1); // Allow for rounding
    }

    // Missing in-place operation tests
    #[test]
    fn premultiply_alpha_and_keep_mut_for_lumaa_u8_preserves_alpha() {
        let mut image: Image<LumaA<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, LumaA([200, 255])); // Full opacity
        image.put_pixel(1, 0, LumaA([200, 127])); // Half opacity

        image.premultiply_alpha_and_keep_mut().unwrap();

        assert_eq!(image.get_pixel(0, 0).0, [200, 255]); // 200 * 1.0, alpha preserved
        assert_eq!(image.get_pixel(1, 0).0, [100, 127]); // 200 * 127/255, alpha preserved
    }

    #[test]
    fn premultiply_alpha_and_keep_for_rgba_f32_preserves_alpha() {
        let mut image: Image<Rgba<f32>> = Image::new(2, 2);
        image.put_pixel(0, 0, Rgba([0.8, 0.6, 0.4, 1.0])); // Full opacity
        image.put_pixel(1, 0, Rgba([0.8, 0.6, 0.4, 0.5])); // Half opacity

        let result = image.premultiply_alpha_and_keep().unwrap();

        let pixel_00 = result.get_pixel(0, 0);
        assert!((pixel_00[0] - 0.8).abs() < f32::EPSILON);
        assert!((pixel_00[1] - 0.6).abs() < f32::EPSILON);
        assert!((pixel_00[2] - 0.4).abs() < f32::EPSILON);
        assert!((pixel_00[3] - 1.0).abs() < f32::EPSILON); // Alpha preserved

        let pixel_10 = result.get_pixel(1, 0);
        assert!((pixel_10[0] - 0.4).abs() < f32::EPSILON); // 0.8 * 0.5
        assert!((pixel_10[1] - 0.3).abs() < f32::EPSILON); // 0.6 * 0.5
        assert!((pixel_10[2] - 0.2).abs() < f32::EPSILON); // 0.4 * 0.5
        assert!((pixel_10[3] - 0.5).abs() < f32::EPSILON); // Alpha preserved
    }

    #[test]
    fn premultiply_alpha_and_keep_mut_for_lumaa_f32_preserves_alpha() {
        let mut image: Image<LumaA<f32>> = Image::new(2, 2);
        image.put_pixel(0, 0, LumaA([0.8, 1.0])); // Full opacity
        image.put_pixel(1, 0, LumaA([0.6, 0.5])); // Half opacity

        image.premultiply_alpha_and_keep_mut().unwrap();

        let pixel_00 = image.get_pixel(0, 0);
        assert!((pixel_00[0] - 0.8).abs() < f32::EPSILON);
        assert!((pixel_00[1] - 1.0).abs() < f32::EPSILON); // Alpha preserved

        let pixel_10 = image.get_pixel(1, 0);
        assert!((pixel_10[0] - 0.3).abs() < f32::EPSILON); // 0.6 * 0.5
        assert!((pixel_10[1] - 0.5).abs() < f32::EPSILON); // Alpha preserved
    }

    // Error case tests
    #[test]
    fn premultiply_alpha_and_drop_with_empty_image_returns_error() {
        let empty_image: Image<Rgba<u8>> = Image::new(0, 0);
        let result = empty_image.premultiply_alpha_and_drop();
        result.unwrap_err();
    }

    #[test]
    fn premultiply_alpha_and_keep_with_empty_image_returns_error() {
        let empty_image: Image<LumaA<u8>> = Image::new(0, 0);
        let result = empty_image.premultiply_alpha_and_keep();
        result.unwrap_err();
    }

    #[test]
    fn premultiply_alpha_and_keep_mut_with_empty_image_returns_error() {
        let mut empty_image: Image<Rgba<f32>> = Image::new(0, 0);
        let result = empty_image.premultiply_alpha_and_keep_mut();
        result.unwrap_err();
    }
}
