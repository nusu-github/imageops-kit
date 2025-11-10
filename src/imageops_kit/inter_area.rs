use image::{ImageBuffer, Pixel, Primitive};
use imageproc::definitions::{Clamp, Image};
use itertools::Itertools;

use crate::error::InterAreaError;

/// Element of the weight table for area interpolation.
#[derive(Debug, Clone, Copy)]
pub struct InterpolationWeight {
    /// Destination index
    pub destination_index: u32,
    /// Source index
    pub source_index: u32,
    /// Weight value
    pub weight: f32,
}

/// OpenCV INTER_AREA interpolation implementation.
pub struct InterAreaResize {
    /// New width
    pub new_width: u32,
    /// New height
    pub new_height: u32,
}

impl InterAreaResize {
    /// Create a new INTER_AREA resizer.
    pub const fn new(new_width: u32, new_height: u32) -> Result<Self, InterAreaError> {
        if new_width == 0 || new_height == 0 {
            return Err(InterAreaError::InvalidTargetDimensions {
                width: new_width,
                height: new_height,
            });
        }
        Ok(Self {
            new_width,
            new_height,
        })
    }
}

/// Compute resize area decimation table.
///
/// This function computes the weight table for area interpolation based on the
/// source size, destination size, and scale factor.
fn compute_interpolation_weights_impl(
    src_size: u32,
    dst_size: u32,
    scale: f32,
) -> Vec<InterpolationWeight> {
    (0..dst_size)
        .flat_map(|dx| compute_weights_for_destination_pixel(dx, src_size, scale))
        .collect()
}

/// Compute interpolation weights for a single destination pixel.
fn compute_weights_for_destination_pixel(
    dx: u32,
    src_size: u32,
    scale: f32,
) -> impl Iterator<Item = InterpolationWeight> {
    let src_x_start = dx as f32 * scale;
    let src_x_end = src_x_start + scale;

    let src_x_start_int = (src_x_start.ceil() as u32).min(src_size);
    let src_x_end_int = (src_x_end.floor() as u32).min(src_size);

    let cell_width = compute_cell_width(
        src_x_start,
        src_x_end,
        src_x_start_int,
        src_x_end_int,
        src_size,
        scale,
    );

    // Create iterators for each type of weight contribution
    let left_partial = create_left_partial_weight(dx, src_x_start, src_x_start_int, cell_width);
    let full_overlaps = create_full_overlap_weights(dx, src_x_start_int, src_x_end_int, cell_width);
    let right_partial =
        create_right_partial_weight(dx, src_x_end, src_x_end_int, src_size, cell_width);

    left_partial
        .into_iter()
        .chain(full_overlaps)
        .chain(right_partial)
}

/// Compute the effective cell width for interpolation.
fn compute_cell_width(
    src_x_start: f32,
    src_x_end: f32,
    src_x_start_int: u32,
    src_x_end_int: u32,
    src_size: u32,
    scale: f32,
) -> f32 {
    if (src_x_end - src_x_start - scale).abs() < f32::EPSILON {
        scale
    } else if src_x_start_int == 0 {
        src_x_end_int as f32
    } else if src_x_end_int == src_size {
        src_size as f32 - src_x_start
    } else {
        scale
    }
}

/// Create left partial overlap weight if applicable.
fn create_left_partial_weight(
    dx: u32,
    src_x_start: f32,
    src_x_start_int: u32,
    cell_width: f32,
) -> Option<InterpolationWeight> {
    let overlap = src_x_start_int as f32 - src_x_start;
    (src_x_start_int > 0 && overlap > 1e-3).then(|| InterpolationWeight {
        destination_index: dx,
        source_index: src_x_start_int - 1,
        weight: overlap / cell_width,
    })
}

/// Create full overlap weights using iterators.
fn create_full_overlap_weights(
    dx: u32,
    src_x_start_int: u32,
    src_x_end_int: u32,
    cell_width: f32,
) -> impl Iterator<Item = InterpolationWeight> {
    (src_x_start_int..src_x_end_int).map(move |sx| InterpolationWeight {
        destination_index: dx,
        source_index: sx,
        weight: 1.0 / cell_width,
    })
}

/// Create right partial overlap weight if applicable.
fn create_right_partial_weight(
    dx: u32,
    src_x_end: f32,
    src_x_end_int: u32,
    src_size: u32,
    cell_width: f32,
) -> Option<InterpolationWeight> {
    let overlap = src_x_end - src_x_end_int as f32;
    (src_x_end_int < src_size && overlap > 1e-3).then(|| InterpolationWeight {
        destination_index: dx,
        source_index: src_x_end_int,
        weight: overlap / cell_width,
    })
}

/// Check if we can use the integer scale optimization.
fn can_use_integer_scale_impl(src_size: u32, dst_size: u32) -> bool {
    if dst_size >= src_size {
        return false;
    }

    let scale = src_size as f32 / dst_size as f32;
    let int_scale = scale.round() as u32;

    // Check if the scale is close to an integer
    (scale - int_scale as f32).abs() < f32::EPSILON && int_scale >= 2
}

/// Integer scale implementation for optimized performance.
fn resize_area_integer_scale_impl<P>(
    src: &Image<P>,
    dst_width: u32,
    dst_height: u32,
) -> Result<Image<P>, InterAreaError>
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Primitive,
    f32: From<P::Subpixel>,
{
    let (src_width, src_height) = src.dimensions();
    let scale_x = src_width / dst_width;
    let scale_y = src_height / dst_height;
    let inv_area = 1.0 / (scale_x * scale_y) as f32;

    let src_buffer = src.as_raw();
    let channels = P::CHANNEL_COUNT as usize;

    let result = ImageBuffer::from_fn(dst_width, dst_height, |dx, dy| {
        let start_x = dx * scale_x;
        let start_y = dy * scale_y;
        let end_x = start_x + scale_x;
        let end_y = start_y + scale_y;

        // Hoist constant cast outside the loop
        let channels_u32 = channels as u32;

        // Initialize accumulator once, not in fold
        let mut pixel_sum = vec![0.0f32; channels];

        for sy in start_y..end_y {
            let row_base_idx = (sy * src_width * channels_u32) as usize;
            for sx in start_x..end_x {
                let pixel_base_idx = row_base_idx + (sx * channels_u32) as usize;

                for c in 0..channels {
                    pixel_sum[c] += f32::from(src_buffer[pixel_base_idx + c]);
                }
            }
        }

        // Convert accumulated values to output pixel
        let output_channels = pixel_sum
            .into_iter()
            .map(|sum| P::Subpixel::clamp(sum * inv_area))
            .collect_vec();

        *P::from_slice(&output_channels)
    });

    Ok(result)
}

/// Fractional scale implementation for arbitrary scale factors.
fn resize_area_fractional_scale_impl<P>(
    src: &Image<P>,
    dst_width: u32,
    dst_height: u32,
) -> Result<Image<P>, InterAreaError>
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Primitive,
    f32: From<P::Subpixel>,
{
    let (src_width, src_height) = src.dimensions();
    let scale_x = src_width as f32 / dst_width as f32;
    let scale_y = src_height as f32 / dst_height as f32;

    // Compute X and Y tables
    let x_weights = compute_interpolation_weights_impl(src_width, dst_width, scale_x);
    let y_weights = compute_interpolation_weights_impl(src_height, dst_height, scale_y);

    let channels = P::CHANNEL_COUNT as usize;
    let mut output = ImageBuffer::new(dst_width, dst_height);
    let src_buffer = src.as_raw();

    // Hoist constant computation outside the loop
    let channels_u32 = channels as u32;

    // Process each destination row by grouping y_weights by destination index
    for (dy, y_group) in &y_weights.iter().chunk_by(|w| w.destination_index) {
        let mut row_accumulator = vec![0.0f32; dst_width as usize * channels];

        // Process each source row that contributes to this destination row
        for y_entry in y_group {
            let sy = y_entry.source_index;
            let beta = y_entry.weight;

            // Fused horizontal pass: accumulate directly into row_accumulator
            // This eliminates intermediate allocation of horizontal_result
            let row_base_idx = (sy * src_width * channels_u32) as usize;

            for x_entry in &x_weights {
                let dx = x_entry.destination_index;
                let sx = x_entry.source_index;
                let alpha = x_entry.weight;

                let src_pixel_base_idx = row_base_idx + (sx * channels_u32) as usize;
                let dst_pixel_base_idx = dx as usize * channels;

                for c in 0..channels {
                    row_accumulator[dst_pixel_base_idx + c] +=
                        f32::from(src_buffer[src_pixel_base_idx + c]) * alpha * beta;
                }
            }
        }

        // Write the completed row to output
        write_output_row(&mut output, dy, &row_accumulator, dst_width, channels);
    }

    Ok(output)
}

/// Write accumulated row data to output buffer.
fn write_output_row<P>(
    output: &mut Image<P>,
    dy: u32,
    row_data: &[f32],
    dst_width: u32,
    channels: usize,
) where
    P: Pixel,
    P::Subpixel: Clamp<f32>,
{
    let output_buffer = output.as_mut();

    // Hoist invariant computations outside the loop
    let channels_u32 = channels as u32;
    let row_base_idx = (dy * dst_width * channels_u32) as usize;

    for dx in 0..dst_width {
        let dst_pixel_base_idx = row_base_idx + (dx * channels_u32) as usize;
        let src_pixel_base_idx = dx as usize * channels;

        for c in 0..channels {
            output_buffer[dst_pixel_base_idx + c] =
                P::Subpixel::clamp(row_data[src_pixel_base_idx + c]);
        }
    }
}

impl InterAreaResize {
    /// Resize image using INTER_AREA interpolation.
    pub fn resize<P>(&self, src: &Image<P>) -> Result<Image<P>, InterAreaError>
    where
        P: Pixel,
        P::Subpixel: Clamp<f32> + Primitive,
        f32: From<P::Subpixel>,
    {
        let (src_width, src_height) = src.dimensions();

        if src_width == 0 || src_height == 0 {
            return Err(InterAreaError::EmptyImage {
                width: src_width,
                height: src_height,
            });
        }

        // Handle upscaling (use bilinear interpolation)
        if self.new_width > src_width || self.new_height > src_height {
            // For upscaling, INTER_AREA behaves like INTER_LINEAR
            // For simplicity, we'll return an error for now
            return Err(InterAreaError::UpscalingNotSupported {
                src_width,
                src_height,
                target_width: self.new_width,
                target_height: self.new_height,
            });
        }

        // Check if we can use the integer scale optimization
        if can_use_integer_scale_impl(src_width, self.new_width)
            && can_use_integer_scale_impl(src_height, self.new_height)
        {
            resize_area_integer_scale_impl(src, self.new_width, self.new_height)
        } else {
            resize_area_fractional_scale_impl(src, self.new_width, self.new_height)
        }
    }
}

/// Extension trait for `ImageBuffer` to provide INTER_AREA resize methods.
pub trait InterAreaResizeExt<P>
where
    P: Pixel,
{
    /// Resize image using INTER_AREA interpolation.
    fn resize_area(self, new_width: u32, new_height: u32) -> Result<Self, InterAreaError>
    where
        Self: Sized;

    /// Resize image using INTER_AREA interpolation in-place.
    fn resize_area_mut(
        &mut self,
        new_width: u32,
        new_height: u32,
    ) -> Result<&mut Self, InterAreaError>;
}

impl<P> InterAreaResizeExt<P> for Image<P>
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Primitive,
    f32: From<P::Subpixel>,
{
    fn resize_area(self, new_width: u32, new_height: u32) -> Result<Self, InterAreaError> {
        let resizer = InterAreaResize::new(new_width, new_height)?;
        resizer.resize(&self)
    }

    /// Hidden _mut variant that is not available for this operation.
    #[doc(hidden)]
    fn resize_area_mut(
        &mut self,
        _new_width: u32,
        _new_height: u32,
    ) -> Result<&mut Self, InterAreaError> {
        unimplemented!(
            "resize_area_mut is not available because the operation requires creating a new image with different dimensions"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    #[test]
    fn can_use_integer_scale_impl_with_valid_cases_identifies_optimizable_scales() {
        assert!(can_use_integer_scale_impl(100, 50)); // 2x downscale
        assert!(can_use_integer_scale_impl(150, 50)); // 3x downscale
        assert!(!can_use_integer_scale_impl(100, 67)); // 1.5x downscale (not integer)
        assert!(!can_use_integer_scale_impl(50, 100)); // upscale
    }

    #[test]
    fn compute_interpolation_weights_impl_with_valid_input_produces_normalized_weights() {
        let tab = compute_interpolation_weights_impl(4, 2, 2.0);
        assert!(!tab.is_empty());

        // Check that weights sum to 1.0 for each destination pixel
        let mut weights_sum = [0.0; 2];
        for entry in &tab {
            weights_sum[entry.destination_index as usize] += entry.weight;
        }

        for sum in &weights_sum {
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn resize_area_integer_scale_impl_with_valid_input_preserves_image_structure() {
        let src = ImageBuffer::from_fn(4, 4, |x, y| Rgb([((x + y) * 50) as u8, 100, 150]));

        let result = resize_area_integer_scale_impl(&src, 2, 2).unwrap();
        assert_eq!(result.dimensions(), (2, 2));

        // Check that result is not empty
        assert!(result.get_pixel(0, 0)[0] > 0);
    }

    #[test]
    fn resize_area_fractional_scale_impl_with_valid_input_preserves_image_structure() {
        let src = ImageBuffer::from_fn(6, 6, |x, y| Rgb([((x + y) * 20) as u8, 100, 150]));

        let result = resize_area_fractional_scale_impl(&src, 4, 4).unwrap();
        assert_eq!(result.dimensions(), (4, 4));

        // Check that result is not empty
        assert!(result.get_pixel(0, 0)[0] > 0);
    }

    #[test]
    fn resize_with_valid_input_produces_correct_dimensions() {
        let src = ImageBuffer::from_fn(8, 8, |x, y| Rgb([((x + y) * 16) as u8, 100, 150]));

        let resizer = InterAreaResize::new(4, 4).unwrap();
        let result = resizer.resize(&src).unwrap();

        assert_eq!(result.dimensions(), (4, 4));

        // Check that result is not empty
        assert!(result.get_pixel(0, 0)[0] > 0);
    }

    #[test]
    fn resize_area_ext_with_valid_input_produces_correct_dimensions() {
        let src = ImageBuffer::from_fn(8, 8, |x, y| Rgb([((x + y) * 16) as u8, 100, 150]));

        let result = src.resize_area(4, 4).unwrap();
        assert_eq!(result.dimensions(), (4, 4));
    }

    #[test]
    fn resize_area_with_chained_resizes_produces_correct_dimensions() {
        // Create a gradient image
        let src = ImageBuffer::from_fn(100, 100, |x, y| {
            Rgb([((x + y) as f32 / 200.0 * 255.0) as u8, 128, 192])
        });

        println!("Source image size: {:?}", src.dimensions());

        // Test resizing using the extension trait
        let resized = src.resize_area(50, 50).unwrap();
        assert_eq!(resized.dimensions(), (50, 50));

        // Test resizing using the struct directly
        let resizer = InterAreaResize::new(25, 25).unwrap();
        let resized2 = resizer.resize(&resized).unwrap();
        assert_eq!(resized2.dimensions(), (25, 25));

        // Verify the pixel values are reasonable
        let pixel = resized2.get_pixel(12, 12);
        assert!(pixel[0] > 0 && pixel[0] < 255);
        assert_eq!(pixel[1], 128);
        assert_eq!(pixel[2], 192);
    }

    // Additional tests for error cases and edge conditions

    #[test]
    fn inter_area_resize_new_zero_width_returns_error() {
        let result = InterAreaResize::new(0, 100);
        assert!(result.is_err());
    }

    #[test]
    fn inter_area_resize_new_zero_height_returns_error() {
        let result = InterAreaResize::new(100, 0);
        assert!(result.is_err());
    }

    #[test]
    fn resize_empty_image_returns_error() {
        let src: ImageBuffer<Rgb<u8>, Vec<u8>> = ImageBuffer::new(0, 100);
        let resizer = InterAreaResize::new(50, 50).unwrap();
        let result = resizer.resize(&src);
        result.unwrap_err();
    }

    #[test]
    fn resize_upscaling_returns_error() {
        let src = ImageBuffer::from_fn(10, 10, |_, _| Rgb([100u8, 100, 100]));
        let resizer = InterAreaResize::new(20, 15).unwrap();
        let result = resizer.resize(&src);
        result.unwrap_err();
    }

    #[test]
    #[should_panic(expected = "resize_area_mut is not available")]
    fn resize_area_mut_implementation_panics() {
        let mut src = ImageBuffer::from_fn(10, 10, |_, _| Rgb([100u8, 100, 100]));
        let _ = src.resize_area_mut(5, 5);
    }

    #[test]
    fn compute_weights_boundary_case_produces_normalized_weights() {
        // Test edge case with very small scale
        let weights = compute_interpolation_weights_impl(100, 1, 100.0);
        assert!(!weights.is_empty());

        // Sum should be close to 1.0
        let total_weight: f32 = weights.iter().map(|w| w.weight).sum();
        assert!((total_weight - 1.0).abs() < 1e-5);
    }

    #[test]
    fn integer_scale_optimization_large_image_preserves_precision() {
        // Test with larger image to check for overflow issues
        let src = ImageBuffer::from_fn(1000, 1000, |x, y| Rgb([((x + y) % 256) as u8, 128, 200]));

        let result = resize_area_integer_scale_impl(&src, 250, 250).unwrap();
        assert_eq!(result.dimensions(), (250, 250));

        // Verify some pixel values are reasonable
        let pixel = result.get_pixel(125, 125);
        assert!(pixel[0] > 0); // Check it's not completely black
        assert_eq!(pixel[1], 128);
        assert_eq!(pixel[2], 200);
    }

    #[test]
    fn fractional_scale_single_pixel_image_produces_result() {
        let src = ImageBuffer::from_fn(1, 1, |_, _| Rgb([255u8, 128, 64]));
        let result = resize_area_fractional_scale_impl(&src, 1, 1).unwrap();
        assert_eq!(result.dimensions(), (1, 1));
        assert_eq!(result.get_pixel(0, 0)[0], 255);
    }

    #[test]
    fn weight_calculation_precision_consistent_with_epsilon() {
        // Test that weight calculations handle floating point precision consistently
        let weights = compute_interpolation_weights_impl(7, 3, 7.0 / 3.0);

        // Verify all weights are positive and reasonable
        for weight in &weights {
            assert!(weight.weight > 0.0);
            assert!(weight.weight <= 1.0);
        }

        // Check normalization per destination pixel
        for dst_idx in 0..3 {
            let sum: f32 = weights
                .iter()
                .filter(|w| w.destination_index == dst_idx)
                .map(|w| w.weight)
                .sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Destination {dst_idx} weight sum: {sum}"
            );
        }
    }
}
