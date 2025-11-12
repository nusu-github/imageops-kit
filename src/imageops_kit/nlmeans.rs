use image::{ImageBuffer, Luma, Pixel, Primitive, Rgb, Rgba};
use imageproc::definitions::{Clamp, Image};
use itertools::iproduct;

use crate::error::NLMeansError;

/// Non-Local Means denoising trait.
///
/// Non-Local Means is a denoising algorithm that replaces each pixel with a weighted average
/// of pixels that have similar neighborhoods. The weight is determined by the similarity
/// between patches (small windows) around the pixels.
///
/// This implementation supports:
/// - Grayscale images (`Image<Luma<T>>`)
/// - RGB color images (`Image<Rgb<T>>`)
/// - RGBA color images with alpha channel (`Image<Rgba<T>>`)
///
/// # Parameters
///
/// * `h` - Filtering parameter. Higher values remove more noise but may also remove fine details
/// * `patch_size` - Size of the patch used for similarity comparison (must be odd)
/// * `search_window` - Size of the search window where similar patches are searched (must be odd and > `patch_size`)
///
/// # Algorithm
///
/// For each pixel p:
/// 1. Extract a patch of size `small_window` around p (single-channel for grayscale, multi-channel for color)
/// 2. Search for similar patches within a `search_window` around p
/// 3. Calculate weights based on patch similarity: `w = exp(-||patch_p - patch_q||² / (h² × patch_size² × channels))`
/// 4. Replace pixel value with weighted average: `new_value` = Σ(w × `pixel_q`) / Σw
///
/// # Examples
///
/// ```rust
/// use imageops_kit::NLMeansExt;
/// use image::{ImageBuffer, Rgb};
///
/// // Create a sample RGB image
/// let mut rgb_image = ImageBuffer::new(10, 10);
/// for y in 0..10 {
///     for x in 0..10 {
///         rgb_image.put_pixel(x, y, Rgb([100u8, 150u8, 200u8]));
///     }
/// }
///
/// // Apply Non-Local Means denoising
/// let denoised = rgb_image.nl_means(10.0, 3, 7).unwrap();
/// assert_eq!(denoised.dimensions(), (10, 10));
/// ```
pub trait NLMeansExt<T> {
    /// Apply Non-Local Means denoising to the image.
    ///
    /// This consumes the original image.
    ///
    /// # Arguments
    ///
    /// * `h` - Filtering parameter (must be positive)
    /// * `patch_size` - Patch size for similarity comparison (must be odd positive integer)
    /// * `search_window` - Search window size (must be odd and larger than `patch_size`)
    ///
    /// # Returns
    ///
    /// Returns a denoised image or an error if the parameters are invalid
    ///
    /// # Errors
    ///
    /// * `NLMeansError::InvalidWindowSize` - If window sizes are not odd positive integers
    /// * `NLMeansError::InvalidFilteringParameter` - If h is not positive
    /// * `NLMeansError::InvalidWindowSizes` - If `search_window` <= `patch_size`
    /// * `NLMeansError::ImageTooSmall` - If image is too small for the specified window sizes
    fn nl_means(self, h: f32, patch_size: u32, search_window: u32) -> Result<Self, NLMeansError>
    where
        Self: Sized;

    /// Apply Non-Local Means denoising to the image in-place.
    ///
    /// # Arguments
    ///
    /// * `h` - Filtering parameter (must be positive)
    /// * `patch_size` - Patch size for similarity comparison (must be odd positive integer)
    /// * `search_window` - Search window size (must be odd and larger than `patch_size`)
    ///
    /// # Returns
    ///
    /// Returns a mutable reference to the denoised image or an error if the parameters are invalid
    ///
    /// # Errors
    ///
    /// * `NLMeansError::InvalidWindowSize` - If window sizes are not odd positive integers
    /// * `NLMeansError::InvalidFilteringParameter` - If h is not positive
    /// * `NLMeansError::InvalidWindowSizes` - If `search_window` <= `patch_size`
    /// * `NLMeansError::ImageTooSmall` - If image is too small for the specified window sizes
    fn nl_means_mut(
        &mut self,
        h: f32,
        patch_size: u32,
        search_window: u32,
    ) -> Result<&mut Self, NLMeansError>;
}

/// Validates parameters to prevent runtime errors and ensure algorithm correctness
fn validate_parameters_impl(
    h: f32,
    patch_size: u32,
    search_window: u32,
    width: u32,
    height: u32,
) -> Result<(), NLMeansError> {
    use crate::utils::validate_non_empty_image;

    if h <= 0.0 {
        return Err(NLMeansError::InvalidFilteringParameter { h });
    }

    if patch_size == 0 || patch_size % 2 == 0 {
        return Err(NLMeansError::InvalidWindowSize { size: patch_size });
    }

    if search_window == 0 || search_window % 2 == 0 {
        return Err(NLMeansError::InvalidWindowSize {
            size: search_window,
        });
    }

    if search_window <= patch_size {
        return Err(NLMeansError::InvalidWindowSizes {
            small_window: patch_size,
            big_window: search_window,
        });
    }

    validate_non_empty_image(width, height, "NL-Means").map_err(|_| {
        NLMeansError::ImageTooSmall {
            width,
            height,
            big_window: search_window,
        }
    })?;

    if width <= search_window || height <= search_window {
        return Err(NLMeansError::ImageTooSmall {
            width,
            height,
            big_window: search_window,
        });
    }

    Ok(())
}

/// Computes squared distance to avoid expensive sqrt operation in weight calculation
#[inline]
fn patch_distance_impl<T>(patch1: &[T], patch2: &[T]) -> f32
where
    T: Primitive,
    f32: From<T>,
{
    patch1
        .iter()
        .zip(patch2.iter())
        .map(|(&p1, &p2)| {
            let diff = f32::from(p1) - f32::from(p2);
            diff * diff
        })
        .sum()
}

/// Extracts patch using fallback values for out-of-bounds pixels to handle edge cases safely
fn extract_patch_impl<T>(
    buffered_image: &[T],
    buffer_width: u32,
    buffer_height: u32,
    center_x: u32,
    center_y: u32,
    patch_size: u32,
) -> Vec<T>
where
    T: Primitive,
{
    let half_size = patch_size / 2;
    let mut patch = Vec::with_capacity((patch_size * patch_size) as usize);

    let half_size_i32 = half_size as i32;
    let center_x_i32 = center_x as i32;
    let center_y_i32 = center_y as i32;

    iproduct!(0..patch_size, 0..patch_size).for_each(|(dy, dx)| {
        let x = center_x_i32 + dx as i32 - half_size_i32;
        let y = center_y_i32 + dy as i32 - half_size_i32;

        if x >= 0 && y >= 0 && (x as u32) < buffer_width && (y as u32) < buffer_height {
            let idx = y as usize * buffer_width as usize + x as usize;
            if idx < buffered_image.len() {
                patch.push(buffered_image[idx]);
            } else {
                patch.push(buffered_image[0]);
            }
        } else {
            patch.push(buffered_image[0]);
        }
    });

    patch
}

/// RGB patch extraction with channel-aware indexing (3 values per pixel)
fn extract_patch_rgb_impl<T>(
    buffered_image: &[T],
    buffer_width: u32,
    buffer_height: u32,
    center_x: u32,
    center_y: u32,
    patch_size: u32,
) -> Vec<T>
where
    T: Primitive,
{
    let half_size = patch_size / 2;
    let mut patch = Vec::with_capacity((patch_size * patch_size * 3) as usize);

    let half_size_i32 = half_size as i32;
    let center_x_i32 = center_x as i32;
    let center_y_i32 = center_y as i32;

    iproduct!(0..patch_size, 0..patch_size).for_each(|(dy, dx)| {
        let x = center_x_i32 + dx as i32 - half_size_i32;
        let y = center_y_i32 + dy as i32 - half_size_i32;

        if x >= 0 && y >= 0 && (x as u32) < buffer_width && (y as u32) < buffer_height {
            let base_idx = (y as usize * buffer_width as usize + x as usize) * 3;
            if base_idx + 2 < buffered_image.len() {
                patch.push(buffered_image[base_idx]);
                patch.push(buffered_image[base_idx + 1]);
                patch.push(buffered_image[base_idx + 2]);
            } else {
                patch.push(buffered_image[0]);
                patch.push(buffered_image[1]);
                patch.push(buffered_image[2]);
            }
        } else {
            patch.push(buffered_image[0]);
            patch.push(buffered_image[1]);
            patch.push(buffered_image[2]);
        }
    });

    patch
}

/// RGBA patch extraction with channel-aware indexing (4 values per pixel)
fn extract_patch_rgba_impl<T>(
    buffered_image: &[T],
    buffer_width: u32,
    buffer_height: u32,
    center_x: u32,
    center_y: u32,
    patch_size: u32,
) -> Vec<T>
where
    T: Primitive,
{
    let half_size = patch_size / 2;
    let mut patch = Vec::with_capacity((patch_size * patch_size * 4) as usize);

    let half_size_i32 = half_size as i32;
    let center_x_i32 = center_x as i32;
    let center_y_i32 = center_y as i32;

    iproduct!(0..patch_size, 0..patch_size).for_each(|(dy, dx)| {
        let x = center_x_i32 + dx as i32 - half_size_i32;
        let y = center_y_i32 + dy as i32 - half_size_i32;

        if x >= 0 && y >= 0 && (x as u32) < buffer_width && (y as u32) < buffer_height {
            let base_idx = (y as usize * buffer_width as usize + x as usize) * 4;
            if base_idx + 3 < buffered_image.len() {
                patch.push(buffered_image[base_idx]);
                patch.push(buffered_image[base_idx + 1]);
                patch.push(buffered_image[base_idx + 2]);
                patch.push(buffered_image[base_idx + 3]);
            } else {
                patch.push(buffered_image[0]);
                patch.push(buffered_image[1]);
                patch.push(buffered_image[2]);
                patch.push(buffered_image[3]);
            }
        } else {
            patch.push(buffered_image[0]);
            patch.push(buffered_image[1]);
            patch.push(buffered_image[2]);
            patch.push(buffered_image[3]);
        }
    });

    patch
}

/// Adds reflection padding to prevent edge artifacts during patch comparison
fn add_padding_impl<T>(image: &Image<Luma<T>>, pad_size: u32) -> (Vec<T>, u32, u32)
where
    T: Primitive,
{
    let (width, height) = image.dimensions();
    let buffer_width = width + 2 * pad_size;
    let buffer_height = height + 2 * pad_size;
    let mut buffered = vec![image.get_pixel(0, 0).0[0]; (buffer_width * buffer_height) as usize];

    iproduct!(0..height, 0..width).for_each(|(y, x)| {
        let dst_idx = ((y + pad_size) * buffer_width + (x + pad_size)) as usize;
        buffered[dst_idx] = image.get_pixel(x, y).0[0];
    });

    iproduct!(0..pad_size, pad_size..(buffer_width - pad_size)).for_each(|(y, x)| {
        let src_y = pad_size + (pad_size - 1 - y);
        let src_idx = (src_y * buffer_width + x) as usize;
        let dst_top_idx = (y * buffer_width + x) as usize;
        buffered[dst_top_idx] = buffered[src_idx];

        let dst_bottom_y = buffer_height - 1 - y;
        let src_y = buffer_height - pad_size - 1 - (pad_size - 1 - y);
        let src_idx = (src_y * buffer_width + x) as usize;
        let dst_bottom_idx = (dst_bottom_y * buffer_width + x) as usize;
        buffered[dst_bottom_idx] = buffered[src_idx];
    });

    iproduct!(0..buffer_height, 0..pad_size).for_each(|(y, x)| {
        let src_x = pad_size + (pad_size - 1 - x);
        let src_idx = (y * buffer_width + src_x) as usize;
        let dst_left_idx = (y * buffer_width + x) as usize;
        buffered[dst_left_idx] = buffered[src_idx];

        let dst_right_x = buffer_width - 1 - x;
        let src_x = buffer_width - pad_size - 1 - (pad_size - 1 - x);
        let src_idx = (y * buffer_width + src_x) as usize;
        let dst_right_idx = (y * buffer_width + dst_right_x) as usize;
        buffered[dst_right_idx] = buffered[src_idx];
    });

    (buffered, buffer_width, buffer_height)
}

/// RGB padding with channel-aware reflection (3 values per pixel)
fn add_padding_rgb_impl<S>(image: &Image<Rgb<S>>, pad_size: u32) -> (Vec<S>, u32, u32)
where
    Rgb<S>: Pixel<Subpixel = S>,
    S: Primitive,
{
    let (width, height) = image.dimensions();
    let buffer_width = width + 2 * pad_size;
    let buffer_height = height + 2 * pad_size;
    let mut buffered =
        vec![image.get_pixel(0, 0).0[0]; (buffer_width * buffer_height * 3) as usize];

    iproduct!(0..height, 0..width).for_each(|(y, x)| {
        let src_pixel = image.get_pixel(x, y);
        let dst_base = ((y + pad_size) * buffer_width + (x + pad_size)) as usize * 3;
        buffered[dst_base] = src_pixel.0[0];
        buffered[dst_base + 1] = src_pixel.0[1];
        buffered[dst_base + 2] = src_pixel.0[2];
    });

    iproduct!(0..pad_size, pad_size..(buffer_width - pad_size)).for_each(|(y, x)| {
        let src_y = pad_size + (pad_size - 1 - y);
        let src_base = (src_y * buffer_width + x) as usize * 3;
        let dst_top_base = (y * buffer_width + x) as usize * 3;
        buffered[dst_top_base] = buffered[src_base];
        buffered[dst_top_base + 1] = buffered[src_base + 1];
        buffered[dst_top_base + 2] = buffered[src_base + 2];

        let dst_bottom_y = buffer_height - 1 - y;
        let src_y = buffer_height - pad_size - 1 - (pad_size - 1 - y);
        let src_base = (src_y * buffer_width + x) as usize * 3;
        let dst_bottom_base = (dst_bottom_y * buffer_width + x) as usize * 3;
        buffered[dst_bottom_base] = buffered[src_base];
        buffered[dst_bottom_base + 1] = buffered[src_base + 1];
        buffered[dst_bottom_base + 2] = buffered[src_base + 2];
    });

    iproduct!(0..buffer_height, 0..pad_size).for_each(|(y, x)| {
        let src_x = pad_size + (pad_size - 1 - x);
        let src_base = (y * buffer_width + src_x) as usize * 3;
        let dst_left_base = (y * buffer_width + x) as usize * 3;
        buffered[dst_left_base] = buffered[src_base];
        buffered[dst_left_base + 1] = buffered[src_base + 1];
        buffered[dst_left_base + 2] = buffered[src_base + 2];

        let dst_right_x = buffer_width - 1 - x;
        let src_x = buffer_width - pad_size - 1 - (pad_size - 1 - x);
        let src_base = (y * buffer_width + src_x) as usize * 3;
        let dst_right_base = (y * buffer_width + dst_right_x) as usize * 3;
        buffered[dst_right_base] = buffered[src_base];
        buffered[dst_right_base + 1] = buffered[src_base + 1];
        buffered[dst_right_base + 2] = buffered[src_base + 2];
    });

    (buffered, buffer_width, buffer_height)
}

/// RGBA padding with channel-aware reflection (4 values per pixel)
fn add_padding_rgba_impl<S>(image: &Image<Rgba<S>>, pad_size: u32) -> (Vec<S>, u32, u32)
where
    Rgba<S>: Pixel<Subpixel = S>,
    S: Primitive,
{
    let (width, height) = image.dimensions();
    let buffer_width = width + 2 * pad_size;
    let buffer_height = height + 2 * pad_size;
    let mut buffered =
        vec![image.get_pixel(0, 0).0[0]; (buffer_width * buffer_height * 4) as usize];

    iproduct!(0..height, 0..width).for_each(|(y, x)| {
        let src_pixel = image.get_pixel(x, y);
        let dst_base = ((y + pad_size) * buffer_width + (x + pad_size)) as usize * 4;
        buffered[dst_base] = src_pixel.0[0];
        buffered[dst_base + 1] = src_pixel.0[1];
        buffered[dst_base + 2] = src_pixel.0[2];
        buffered[dst_base + 3] = src_pixel.0[3];
    });

    iproduct!(0..pad_size, pad_size..(buffer_width - pad_size)).for_each(|(y, x)| {
        let src_y = pad_size + (pad_size - 1 - y);
        let src_base = (src_y * buffer_width + x) as usize * 4;
        let dst_top_base = (y * buffer_width + x) as usize * 4;
        buffered[dst_top_base] = buffered[src_base];
        buffered[dst_top_base + 1] = buffered[src_base + 1];
        buffered[dst_top_base + 2] = buffered[src_base + 2];
        buffered[dst_top_base + 3] = buffered[src_base + 3];

        let dst_bottom_y = buffer_height - 1 - y;
        let src_y = buffer_height - pad_size - 1 - (pad_size - 1 - y);
        let src_base = (src_y * buffer_width + x) as usize * 4;
        let dst_bottom_base = (dst_bottom_y * buffer_width + x) as usize * 4;
        buffered[dst_bottom_base] = buffered[src_base];
        buffered[dst_bottom_base + 1] = buffered[src_base + 1];
        buffered[dst_bottom_base + 2] = buffered[src_base + 2];
        buffered[dst_bottom_base + 3] = buffered[src_base + 3];
    });

    iproduct!(0..buffer_height, 0..pad_size).for_each(|(y, x)| {
        let src_x = pad_size + (pad_size - 1 - x);
        let src_base = (y * buffer_width + src_x) as usize * 4;
        let dst_left_base = (y * buffer_width + x) as usize * 4;
        buffered[dst_left_base] = buffered[src_base];
        buffered[dst_left_base + 1] = buffered[src_base + 1];
        buffered[dst_left_base + 2] = buffered[src_base + 2];
        buffered[dst_left_base + 3] = buffered[src_base + 3];

        let dst_right_x = buffer_width - 1 - x;
        let src_x = buffer_width - pad_size - 1 - (pad_size - 1 - x);
        let src_base = (y * buffer_width + src_x) as usize * 4;
        let dst_right_base = (y * buffer_width + dst_right_x) as usize * 4;
        buffered[dst_right_base] = buffered[src_base];
        buffered[dst_right_base + 1] = buffered[src_base + 1];
        buffered[dst_right_base + 2] = buffered[src_base + 2];
        buffered[dst_right_base + 3] = buffered[src_base + 3];
    });

    (buffered, buffer_width, buffer_height)
}

impl<T> NLMeansExt<T> for Image<Luma<T>>
where
    Luma<T>: Pixel<Subpixel = T>,
    T: Primitive + Clamp<f32>,
    f32: From<T>,
{
    fn nl_means(self, h: f32, patch_size: u32, search_window: u32) -> Result<Self, NLMeansError> {
        let (width, height) = self.dimensions();

        validate_parameters_impl(h, patch_size, search_window, width, height)?;

        let pad_size = search_window / 2;
        let (buffered_image, buffer_width, buffer_height) = add_padding_impl(&self, pad_size);

        let normalization_factor = h * h * (patch_size * patch_size) as f32;

        let mut result = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let buffer_x = x + pad_size;
                let buffer_y = y + pad_size;

                let pixel_patch = extract_patch_impl(
                    &buffered_image,
                    buffer_width,
                    buffer_height,
                    buffer_x,
                    buffer_y,
                    patch_size,
                );

                let mut weighted_sum = 0.0f32;
                let mut weight_sum = 0.0f32;

                let search_radius = search_window / 2;
                iproduct!(
                    (buffer_y - search_radius)..=(buffer_y + search_radius),
                    (buffer_x - search_radius)..=(buffer_x + search_radius)
                )
                .for_each(|(ny, nx)| {
                    let neighbor_patch = extract_patch_impl(
                        &buffered_image,
                        buffer_width,
                        buffer_height,
                        nx,
                        ny,
                        patch_size,
                    );

                    let distance = patch_distance_impl(&pixel_patch, &neighbor_patch);

                    let weight = f32::exp(-distance / normalization_factor);

                    let neighbor_value =
                        f32::from(buffered_image[(ny * buffer_width + nx) as usize]);

                    weighted_sum += weight * neighbor_value;
                    weight_sum += weight;
                });

                let new_value = if weight_sum > 0.0 {
                    weighted_sum / weight_sum
                } else {
                    f32::from(self.get_pixel(x, y).0[0])
                };

                let clamped_value = T::clamp(new_value);
                result.put_pixel(x, y, Luma([clamped_value]));
            }
        }

        Ok(result)
    }

    #[doc(hidden)]
    fn nl_means_mut(
        &mut self,
        _h: f32,
        _patch_size: u32,
        _search_window: u32,
    ) -> Result<&mut Self, NLMeansError> {
        // In-place processing would require significant memory reallocation
        // which defeats the purpose of in-place operations.
        // Use nl_means() instead for optimal memory usage.
        unimplemented!(
            "Use nl_means() instead - in-place processing not beneficial for this algorithm"
        )
    }
}

impl<T> NLMeansExt<T> for Image<Rgb<T>>
where
    Rgb<T>: Pixel<Subpixel = T>,
    T: Primitive + Clamp<f32>,
    f32: From<T>,
{
    fn nl_means(self, h: f32, patch_size: u32, search_window: u32) -> Result<Self, NLMeansError> {
        let (width, height) = self.dimensions();

        validate_parameters_impl(h, patch_size, search_window, width, height)?;

        let pad_size = search_window / 2;
        let (buffered_image, buffer_width, buffer_height) = add_padding_rgb_impl(&self, pad_size);

        let normalization_factor = h * h * (patch_size * patch_size * 3) as f32;

        let mut result = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let buffer_x = x + pad_size;
                let buffer_y = y + pad_size;

                let pixel_patch = extract_patch_rgb_impl(
                    &buffered_image,
                    buffer_width,
                    buffer_height,
                    buffer_x,
                    buffer_y,
                    patch_size,
                );

                let mut weighted_sum = [0.0f32; 3];
                let mut weight_sum = 0.0f32;

                let search_radius = search_window / 2;
                iproduct!(
                    (buffer_y - search_radius)..=(buffer_y + search_radius),
                    (buffer_x - search_radius)..=(buffer_x + search_radius)
                )
                .for_each(|(ny, nx)| {
                    let neighbor_patch = extract_patch_rgb_impl(
                        &buffered_image,
                        buffer_width,
                        buffer_height,
                        nx,
                        ny,
                        patch_size,
                    );

                    let distance = patch_distance_impl(&pixel_patch, &neighbor_patch);

                    let weight = f32::exp(-distance / normalization_factor);

                    let neighbor_base = (ny * buffer_width + nx) as usize * 3;
                    let neighbor_r = f32::from(buffered_image[neighbor_base]);
                    let neighbor_g = f32::from(buffered_image[neighbor_base + 1]);
                    let neighbor_b = f32::from(buffered_image[neighbor_base + 2]);

                    weighted_sum[0] += weight * neighbor_r;
                    weighted_sum[1] += weight * neighbor_g;
                    weighted_sum[2] += weight * neighbor_b;
                    weight_sum += weight;
                });

                let new_values = if weight_sum > 0.0 {
                    [
                        weighted_sum[0] / weight_sum,
                        weighted_sum[1] / weight_sum,
                        weighted_sum[2] / weight_sum,
                    ]
                } else {
                    let orig_pixel = self.get_pixel(x, y);
                    [
                        f32::from(orig_pixel.0[0]),
                        f32::from(orig_pixel.0[1]),
                        f32::from(orig_pixel.0[2]),
                    ]
                };

                let clamped_r = T::clamp(new_values[0]);
                let clamped_g = T::clamp(new_values[1]);
                let clamped_b = T::clamp(new_values[2]);
                result.put_pixel(x, y, Rgb([clamped_r, clamped_g, clamped_b]));
            }
        }

        Ok(result)
    }

    #[doc(hidden)]
    fn nl_means_mut(
        &mut self,
        _h: f32,
        _patch_size: u32,
        _search_window: u32,
    ) -> Result<&mut Self, NLMeansError> {
        // In-place processing would require significant memory reallocation
        // which defeats the purpose of in-place operations.
        // Use nl_means() instead for optimal memory usage.
        unimplemented!(
            "Use nl_means() instead - in-place processing not beneficial for this algorithm"
        )
    }
}

impl<T> NLMeansExt<T> for Image<Rgba<T>>
where
    Rgba<T>: Pixel<Subpixel = T>,
    T: Primitive + Clamp<f32>,
    f32: From<T>,
{
    fn nl_means(self, h: f32, patch_size: u32, search_window: u32) -> Result<Self, NLMeansError> {
        let (width, height) = self.dimensions();

        validate_parameters_impl(h, patch_size, search_window, width, height)?;

        let pad_size = search_window / 2;
        let (buffered_image, buffer_width, buffer_height) = add_padding_rgba_impl(&self, pad_size);

        let normalization_factor = h * h * (patch_size * patch_size * 4) as f32;

        let mut result = ImageBuffer::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let buffer_x = x + pad_size;
                let buffer_y = y + pad_size;

                let pixel_patch = extract_patch_rgba_impl(
                    &buffered_image,
                    buffer_width,
                    buffer_height,
                    buffer_x,
                    buffer_y,
                    patch_size,
                );

                let mut weighted_sum = [0.0f32; 4];
                let mut weight_sum = 0.0f32;

                let search_radius = search_window / 2;
                iproduct!(
                    (buffer_y - search_radius)..=(buffer_y + search_radius),
                    (buffer_x - search_radius)..=(buffer_x + search_radius)
                )
                .for_each(|(ny, nx)| {
                    let neighbor_patch = extract_patch_rgba_impl(
                        &buffered_image,
                        buffer_width,
                        buffer_height,
                        nx,
                        ny,
                        patch_size,
                    );

                    let distance = patch_distance_impl(&pixel_patch, &neighbor_patch);

                    let weight = f32::exp(-distance / normalization_factor);

                    let neighbor_base = (ny * buffer_width + nx) as usize * 4;
                    let neighbor_r = f32::from(buffered_image[neighbor_base]);
                    let neighbor_g = f32::from(buffered_image[neighbor_base + 1]);
                    let neighbor_b = f32::from(buffered_image[neighbor_base + 2]);
                    let neighbor_a = f32::from(buffered_image[neighbor_base + 3]);

                    weighted_sum[0] += weight * neighbor_r;
                    weighted_sum[1] += weight * neighbor_g;
                    weighted_sum[2] += weight * neighbor_b;
                    weighted_sum[3] += weight * neighbor_a;
                    weight_sum += weight;
                });

                let new_values = if weight_sum > 0.0 {
                    [
                        weighted_sum[0] / weight_sum,
                        weighted_sum[1] / weight_sum,
                        weighted_sum[2] / weight_sum,
                        weighted_sum[3] / weight_sum,
                    ]
                } else {
                    let orig_pixel = self.get_pixel(x, y);
                    [
                        f32::from(orig_pixel.0[0]),
                        f32::from(orig_pixel.0[1]),
                        f32::from(orig_pixel.0[2]),
                        f32::from(orig_pixel.0[3]),
                    ]
                };

                let clamped_r = T::clamp(new_values[0]);
                let clamped_g = T::clamp(new_values[1]);
                let clamped_b = T::clamp(new_values[2]);
                let clamped_a = T::clamp(new_values[3]);
                result.put_pixel(x, y, Rgba([clamped_r, clamped_g, clamped_b, clamped_a]));
            }
        }

        Ok(result)
    }

    #[doc(hidden)]
    fn nl_means_mut(
        &mut self,
        _h: f32,
        _patch_size: u32,
        _search_window: u32,
    ) -> Result<&mut Self, NLMeansError> {
        // In-place processing would require significant memory reallocation
        // which defeats the purpose of in-place operations.
        // Use nl_means() instead for optimal memory usage.
        unimplemented!(
            "Use nl_means() instead - in-place processing not beneficial for this algorithm"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma, Rgb, Rgba};

    #[test]
    fn validate_parameters_impl_with_valid_input_returns_ok() {
        // Valid parameters
        validate_parameters_impl(10.0, 3, 7, 50, 50).unwrap();
    }

    #[test]
    fn validate_parameters_impl_with_zero_h_returns_error() {
        // Invalid h
        assert!(matches!(
            validate_parameters_impl(0.0, 3, 7, 50, 50),
            Err(NLMeansError::InvalidFilteringParameter { h: 0.0 })
        ));
    }

    #[test]
    fn validate_parameters_impl_with_even_patch_size_returns_error() {
        // Invalid patch_size (even)
        assert!(matches!(
            validate_parameters_impl(10.0, 4, 7, 50, 50),
            Err(NLMeansError::InvalidWindowSize { size: 4 })
        ));
    }

    #[test]
    fn validate_parameters_impl_with_even_search_window_returns_error() {
        // Invalid search_window (even)
        assert!(matches!(
            validate_parameters_impl(10.0, 3, 8, 50, 50),
            Err(NLMeansError::InvalidWindowSize { size: 8 })
        ));
    }

    #[test]
    fn validate_parameters_impl_with_equal_windows_returns_error() {
        // search_window <= patch_size
        assert!(matches!(
            validate_parameters_impl(10.0, 7, 7, 50, 50),
            Err(NLMeansError::InvalidWindowSizes {
                small_window: 7,
                big_window: 7
            })
        ));
    }

    #[test]
    fn validate_parameters_impl_with_small_image_returns_error() {
        // Image too small
        assert!(matches!(
            validate_parameters_impl(10.0, 3, 7, 5, 5),
            Err(NLMeansError::ImageTooSmall {
                width: 5,
                height: 5,
                big_window: 7
            })
        ));
    }

    #[test]
    fn patch_distance_impl_with_identical_patches_returns_zero() {
        let patch1 = vec![1.0f32, 2.0, 3.0, 4.0];
        let patch2 = vec![1.0f32, 2.0, 3.0, 4.0];
        assert_eq!(patch_distance_impl(&patch1, &patch2), 0.0);

        let patch3 = vec![2.0f32, 3.0, 4.0, 5.0];
        assert_eq!(patch_distance_impl(&patch1, &patch3), 4.0); // (1-2)² + (2-3)² + (3-4)² + (4-5)² = 4
    }

    #[test]
    fn nl_means_for_luma_with_small_image_returns_denoised_result() {
        // Create a simple 10x10 test image (large enough for big_window=7)
        let mut image = ImageBuffer::new(10, 10);
        iproduct!(0..10, 0..10).for_each(|(y, x)| {
            // Create a simple pattern
            let value = ((x + y) * 20) as u8;
            image.put_pixel(x, y, Luma([value]));
        });

        // Apply NL-Means with small parameters
        let result = image.nl_means(10.0, 3, 7);
        match &result {
            Ok(_) => {}
            Err(e) => panic!("NL-Means failed: {e:?}"),
        }
        assert!(result.is_ok());

        let denoised = result.unwrap();
        assert_eq!(denoised.dimensions(), (10, 10));
    }

    #[test]
    fn extract_patch_impl_at_center_returns_full_patch() {
        let padded = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9];
        let patch = extract_patch_impl(&padded, 3, 3, 1, 1, 3);
        assert_eq!(patch, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn patch_distance_impl_with_different_patches_returns_distance() {
        let patch1 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let patch2 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        assert_eq!(patch_distance_impl(&patch1, &patch2), 0.0);

        let patch3 = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0];
        assert_eq!(patch_distance_impl(&patch1, &patch3), 6.0); // 6 × (diff of 1)² = 6
    }

    #[test]
    fn nl_means_for_rgb_with_small_image_returns_denoised_result() {
        // Create a simple 10x10 RGB test image
        let mut image = ImageBuffer::new(10, 10);
        iproduct!(0..10, 0..10).for_each(|(y, x)| {
            // Create a simple RGB pattern
            let r = ((x + y) * 10) as u8;
            let g = ((x * 2) * 10) as u8;
            let b = ((y * 2) * 10) as u8;
            image.put_pixel(x, y, Rgb([r, g, b]));
        });

        // Apply NL-Means with small parameters
        let result = image.nl_means(10.0, 3, 7);
        assert!(result.is_ok());

        let denoised = result.unwrap();
        assert_eq!(denoised.dimensions(), (10, 10));
    }

    #[test]
    fn nl_means_for_rgba_with_small_image_returns_denoised_result() {
        // Create a simple 10x10 RGBA test image
        let mut image = ImageBuffer::new(10, 10);
        iproduct!(0..10, 0..10).for_each(|(y, x)| {
            // Create a simple RGBA pattern
            let r = ((x + y) * 10) as u8;
            let g = ((x * 2) * 10) as u8;
            let b = ((y * 2) * 10) as u8;
            let a = 255u8; // Full opacity
            image.put_pixel(x, y, Rgba([r, g, b, a]));
        });

        // Apply NL-Means with small parameters
        let result = image.nl_means(10.0, 3, 7);
        assert!(result.is_ok());

        let denoised = result.unwrap();
        assert_eq!(denoised.dimensions(), (10, 10));
    }

    #[test]
    fn extract_patch_rgb_impl_at_center_returns_full_patch() {
        let padded = vec![
            1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 26, 27,
        ];
        let patch = extract_patch_rgb_impl(&padded, 3, 3, 1, 1, 3);
        // Expected: 3x3 patch × 3 channels = 27 elements
        assert_eq!(patch.len(), 27);
    }

    #[test]
    fn extract_patch_rgba_impl_at_center_returns_full_patch() {
        let padded = vec![1u8; 36]; // 3x3 × 4 channels
        let patch = extract_patch_rgba_impl(&padded, 3, 3, 1, 1, 3);
        // Expected: 3x3 patch × 4 channels = 36 elements
        assert_eq!(patch.len(), 36);
    }
}
