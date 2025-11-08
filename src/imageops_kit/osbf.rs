use image::{ImageBuffer, Pixel, Primitive};
use imageproc::definitions::{Clamp, Image};
use itertools::iproduct;

use crate::error::OSBFilterError;

/// Trait for One-Sided Box Filter implementations.
pub trait OneSidedBoxFilterApplicator<P>
where
    P: Pixel,
{
    /// Apply One-Sided Box Filter to the image.
    fn one_sided_box_filter(
        &self,
        image: &Image<P>,
        iterations: u32,
    ) -> Result<Image<P>, OSBFilterError>;
}

/// One-Sided Box Filter (OSBF).
///
/// This filter selects the mean value from 8 adjacent regions (4 quarter windows and 4 half windows)
/// that is closest to the current pixel value, preserving edges during smoothing.
pub struct OneSidedBoxFilter {
    radius: u32,
}

impl OneSidedBoxFilter {
    /// Create a new One-Sided Box Filter.
    pub const fn new(radius: u32) -> Result<Self, OSBFilterError> {
        if radius == 0 {
            return Err(OSBFilterError::InvalidRadius { radius });
        }
        Ok(Self { radius })
    }

    /// Get the kernel size (2 * radius + 1).
    #[inline]
    #[must_use]
    pub const fn kernel_size(&self) -> u32 {
        2 * self.radius + 1
    }
}

/// Pads image with edge replication.
fn pad_image_impl<P>(image: &Image<P>, padding: u32) -> Image<P>
where
    P: Pixel,
    P::Subpixel: Primitive,
{
    let (width, height) = image.dimensions();
    let new_width = width + 2 * padding;
    let new_height = height + 2 * padding;

    // Use from_fn with clamping for boundary handling
    ImageBuffer::from_fn(new_width, new_height, |x, y| {
        let orig_x = x.saturating_sub(padding).min(width - 1);
        let orig_y = y.saturating_sub(padding).min(height - 1);
        *image.get_pixel(orig_x, orig_y)
    })
}

/// Calculates box sum using integral image.
#[inline]
fn box_sum_impl(
    integral: &[f32],
    integral_width: usize,
    y1: usize,
    x1: usize,
    y2: usize,
    x2: usize,
) -> f32 {
    integral[y2 * integral_width + x2]
        - integral[y1 * integral_width + x2]
        - integral[y2 * integral_width + x1]
        + integral[y1 * integral_width + x1]
}

/// Pre-computed region coordinates for One-Sided Box Filter.
#[derive(Debug, Clone)]
struct OneSidedBoxFilterRegions {
    /// Quarter window coordinates: (y1, x1, y2, x2)
    quarters: [(usize, usize, usize, usize); 4],
    /// Half window coordinates: (y1, x1, y2, x2)
    halves: [(usize, usize, usize, usize); 4],
    quarter_area: f32,
    half_area: f32,
}

impl OneSidedBoxFilterRegions {
    const fn new(padded_y: usize, padded_x: usize, radius: usize) -> Self {
        let padded_y_sub_radius = padded_y.saturating_sub(radius);
        let padded_x_sub_radius = padded_x.saturating_sub(radius);

        let quarters = [
            (
                padded_y,
                padded_x_sub_radius,
                padded_y + radius + 1,
                padded_x + 1,
            ), // q1
            (
                padded_y,
                padded_x,
                padded_y + radius + 1,
                padded_x + radius + 1,
            ), // q2
            (
                padded_y_sub_radius,
                padded_x,
                padded_y + 1,
                padded_x + radius + 1,
            ), // q3
            (
                padded_y_sub_radius,
                padded_x_sub_radius,
                padded_y + 1,
                padded_x + 1,
            ), // q4
        ];

        let halves = [
            (
                padded_y_sub_radius,
                padded_x_sub_radius,
                padded_y + radius + 1,
                padded_x + 1,
            ), // h1
            (
                padded_y_sub_radius,
                padded_x,
                padded_y + radius + 1,
                padded_x + radius + 1,
            ), // h2
            (
                padded_y,
                padded_x_sub_radius,
                padded_y + radius + 1,
                padded_x + radius + 1,
            ), // h3
            (
                padded_y_sub_radius,
                padded_x_sub_radius,
                padded_y + 1,
                padded_x + radius + 1,
            ), // h4
        ];

        let quarter_area = ((radius + 1) * (radius + 1)) as f32;
        let half_area = ((radius + 1) * (2 * radius + 1)) as f32;

        Self {
            quarters,
            halves,
            quarter_area,
            half_area,
        }
    }
}

/// Perform one One-Sided Box Filter iteration on the entire image.
fn one_sided_box_filter_impl<P>(image: &Image<P>, radius: u32) -> Result<Image<P>, OSBFilterError>
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Primitive,
    f32: From<P::Subpixel>,
{
    let (width, height) = image.dimensions();
    let channels = P::CHANNEL_COUNT as usize;

    // Pad the image
    let padded = pad_image_impl(image, 2);
    let (padded_width, padded_height) = padded.dimensions();
    let padding = 2;

    // Create integral images for each channel using optimized 1D layout
    let integral_width = (padded_width + 1) as usize;
    let integral_height = (padded_height + 1) as usize;
    let integral_size = integral_width * integral_height;
    let mut channel_integrals = vec![0.0f32; channels * integral_size];

    // Build integral images using coordinate pairs for better cache efficiency
    iproduct!(0..padded_height, 0..padded_width).for_each(|(y, x)| {
        let pixel = padded.get_pixel(x, y);
        let pixel_channels = pixel.channels();

        let current_idx = ((y + 1) as usize) * integral_width + ((x + 1) as usize);
        let top_idx = (y as usize) * integral_width + ((x + 1) as usize);
        let left_idx = ((y + 1) as usize) * integral_width + (x as usize);
        let diag_idx = (y as usize) * integral_width + (x as usize);

        for (channel_idx, &channel_value) in pixel_channels.iter().enumerate().take(channels) {
            let value = f32::from(channel_value);
            let base_offset = channel_idx * integral_size;

            channel_integrals[base_offset + current_idx] = value
                + channel_integrals[base_offset + top_idx]
                + channel_integrals[base_offset + left_idx]
                - channel_integrals[base_offset + diag_idx];
        }
    });

    // Pre-compute radius and area constants
    let radius_usize = radius as usize;

    // Reusable pixel data buffer
    let mut pixel_data = Vec::with_capacity(channels);

    // Process each pixel using region-based filtering
    let output = ImageBuffer::from_fn(width, height, |x, y| {
        // Coordinates in padded space
        let padded_y = (y + padding as u32) as usize;
        let padded_x = (x + padding as u32) as usize;

        // Get current pixel value
        let current_pixel = image.get_pixel(x, y);
        let current_channels = current_pixel.channels();

        // Pre-compute regions for this pixel
        let regions = OneSidedBoxFilterRegions::new(padded_y, padded_x, radius_usize);

        pixel_data.clear();

        for channel_idx in 0..channels {
            let current_value = f32::from(current_channels[channel_idx]);
            let mut min_diff = f32::INFINITY;
            let mut best_value = current_value;

            let integral_base = &channel_integrals[channel_idx * integral_size..];

            // Calculate quarter windows
            for &(y1, x1, y2, x2) in &regions.quarters {
                let value = box_sum_impl(integral_base, integral_width, y1, x1, y2, x2)
                    / regions.quarter_area;
                let diff = (value - current_value).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best_value = value;
                }
            }

            // Calculate half windows
            for &(y1, x1, y2, x2) in &regions.halves {
                let value =
                    box_sum_impl(integral_base, integral_width, y1, x1, y2, x2) / regions.half_area;
                let diff = (value - current_value).abs();
                if diff < min_diff {
                    min_diff = diff;
                    best_value = value;
                }
            }

            pixel_data.push(P::Subpixel::clamp(best_value));
        }

        *P::from_slice(&pixel_data)
    });

    Ok(output)
}

impl<P> OneSidedBoxFilterApplicator<P> for OneSidedBoxFilter
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Primitive,
    f32: From<P::Subpixel>,
{
    fn one_sided_box_filter(
        &self,
        image: &Image<P>,
        iterations: u32,
    ) -> Result<Image<P>, OSBFilterError> {
        let (width, height) = image.dimensions();

        if width == 0 || height == 0 {
            return Err(OSBFilterError::EmptyImage { width, height });
        }

        if iterations == 0 {
            return Err(OSBFilterError::InvalidIterations { iterations });
        }

        // Check if image is large enough for the filter radius
        let min_dimension = width.min(height);
        if min_dimension < 2 * self.radius + 1 {
            return Err(OSBFilterError::ImageTooSmall {
                width,
                height,
                radius: self.radius,
            });
        }

        let mut result = image.clone();

        // Apply filter iterations
        for _ in 0..iterations {
            result = one_sided_box_filter_impl(&result, self.radius)?;
        }

        Ok(result)
    }
}

/// Extension trait for `ImageBuffer` to provide fluent One-Sided Box Filter methods.
pub trait OneSidedBoxFilterExt<P>
where
    P: Pixel,
{
    /// Apply One-Sided Box Filter.
    fn one_sided_box_filter(self, radius: u32, iterations: u32) -> Result<Self, OSBFilterError>
    where
        Self: Sized;

    /// Apply One-Sided Box Filter in-place (not available - requires reallocation).
    #[doc(hidden)]
    fn one_sided_box_filter_mut(
        &mut self,
        radius: u32,
        iterations: u32,
    ) -> Result<&mut Self, OSBFilterError>;
}

impl<P> OneSidedBoxFilterExt<P> for Image<P>
where
    P: Pixel,
    P::Subpixel: Clamp<f32> + Primitive,
    f32: From<P::Subpixel>,
{
    fn one_sided_box_filter(self, radius: u32, iterations: u32) -> Result<Self, OSBFilterError> {
        let filter = OneSidedBoxFilter::new(radius)?;
        filter.one_sided_box_filter(&self, iterations)
    }

    #[doc(hidden)]
    fn one_sided_box_filter_mut(
        &mut self,
        _radius: u32,
        _iterations: u32,
    ) -> Result<&mut Self, OSBFilterError> {
        unimplemented!(
            "one_sided_box_filter_mut is not available because the operation requires additional memory allocations equivalent to the owning version"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Luma, Rgb};

    #[test]
    fn one_sided_box_filter_with_center_pixel_smoothes_pixel() {
        let mut image = ImageBuffer::from_pixel(5, 5, Luma([100u8]));
        image.put_pixel(2, 2, Luma([255]));

        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result = filter.one_sided_box_filter(&image, 1).unwrap();

        // Center pixel should be smoothed
        let center = result.get_pixel(2, 2);
        assert!(center[0] > 100 && center[0] < 255);
    }

    #[test]
    fn one_sided_box_filter_with_rgb_image_smoothes_channels() {
        let mut image = ImageBuffer::from_pixel(5, 5, Rgb([100u8, 100, 100]));
        image.put_pixel(2, 2, Rgb([255, 255, 255]));

        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result = filter.one_sided_box_filter(&image, 1).unwrap();

        let center = result.get_pixel(2, 2);
        assert!(center[0] > 100 && center[0] < 255);
        assert!(center[1] > 100 && center[1] < 255);
        assert!(center[2] > 100 && center[2] < 255);
    }

    #[test]
    fn one_sided_box_filter_with_more_iterations_produces_more_smoothing() {
        let mut image = ImageBuffer::from_pixel(7, 7, Luma([100u8]));
        image.put_pixel(3, 3, Luma([255]));

        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result1 = filter.one_sided_box_filter(&image, 1).unwrap();
        let result5 = filter.one_sided_box_filter(&image, 5).unwrap();

        // More iterations should produce more smoothing
        let center1 = result1.get_pixel(3, 3)[0];
        let center5 = result5.get_pixel(3, 3)[0];
        assert!(center5 < center1);
    }

    #[test]
    fn one_sided_box_filter_with_sharp_edge_preserves_edge() {
        // Create an image with a sharp edge
        let mut image = ImageBuffer::new(10, 10);
        iproduct!(0..10, 0..10).for_each(|(y, x)| {
            if x < 5 {
                image.put_pixel(x, y, Luma([50u8]));
            } else {
                image.put_pixel(x, y, Luma([200u8]));
            }
        });

        let filter = OneSidedBoxFilter::new(2).unwrap();
        let result = filter.one_sided_box_filter(&image, 3).unwrap();

        // Edge should be somewhat preserved
        let left_side = result.get_pixel(2, 5)[0];
        let right_side = result.get_pixel(7, 5)[0];
        assert!(left_side < 100);
        assert!(right_side > 150);
    }

    #[test]
    fn one_sided_box_filter_ext_with_method_chaining_enables_fluent_interface() {
        let image = ImageBuffer::from_pixel(5, 5, Rgb([100u8, 100, 100]));

        // Test chaining
        let result = image.one_sided_box_filter(1, 2).unwrap();
        assert_eq!(result.dimensions(), (5, 5));
    }

    #[test]
    fn new_with_zero_radius_returns_error() {
        let result = OneSidedBoxFilter::new(0);
        assert!(matches!(
            result,
            Err(OSBFilterError::InvalidRadius { radius: 0 })
        ));
    }

    #[test]
    fn one_sided_box_filter_with_zero_iterations_returns_error() {
        let image = ImageBuffer::from_pixel(5, 5, Luma([100u8]));
        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result = filter.one_sided_box_filter(&image, 0);
        assert!(matches!(
            result,
            Err(OSBFilterError::InvalidIterations { iterations: 0 })
        ));
    }

    #[test]
    fn one_sided_box_filter_with_empty_image_returns_error() {
        let image: ImageBuffer<Luma<u8>, Vec<u8>> = ImageBuffer::new(0, 0);
        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result = filter.one_sided_box_filter(&image, 1);
        assert!(matches!(
            result,
            Err(OSBFilterError::EmptyImage {
                width: 0,
                height: 0
            })
        ));
    }

    #[test]
    fn one_sided_box_filter_with_image_too_small_returns_error() {
        let image = ImageBuffer::from_pixel(2, 2, Luma([100u8]));
        let filter = OneSidedBoxFilter::new(2).unwrap();
        let result = filter.one_sided_box_filter(&image, 1);
        assert!(matches!(
            result,
            Err(OSBFilterError::ImageTooSmall {
                width: 2,
                height: 2,
                radius: 2
            })
        ));
    }

    #[test]
    fn one_sided_box_filter_with_u16_image_produces_expected_result() {
        let mut image = ImageBuffer::from_pixel(5, 5, Luma([1000u16]));
        image.put_pixel(2, 2, Luma([5000]));

        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result = filter.one_sided_box_filter(&image, 1).unwrap();

        let center = result.get_pixel(2, 2);
        assert!(center[0] > 1000 && center[0] < 5000);
    }

    #[test]
    fn one_sided_box_filter_with_f32_image_produces_expected_result() {
        let mut image = ImageBuffer::from_pixel(5, 5, Luma([0.5f32]));
        image.put_pixel(2, 2, Luma([1.0]));

        let filter = OneSidedBoxFilter::new(1).unwrap();
        let result = filter.one_sided_box_filter(&image, 1).unwrap();

        let center = result.get_pixel(2, 2);
        assert!(center[0] > 0.5 && center[0] < 1.0);
    }
}
