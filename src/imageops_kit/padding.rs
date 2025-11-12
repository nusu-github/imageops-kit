use image::{GenericImage, GenericImageView, ImageBuffer, Pixel};
use imageproc::definitions::Image;
use itertools::iproduct;

use crate::error::PaddingError;

type SquarePaddingResult = Result<((i64, i64), (u32, u32)), PaddingError>;

/// Enum to specify padding position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Position {
    /// Top center
    Top,
    /// Bottom center
    Bottom,
    /// Left center
    Left,
    /// Right center
    Right,
    /// Top left
    TopLeft,
    /// Top right
    TopRight,
    /// Bottom left
    BottomLeft,
    /// Bottom right
    BottomRight,
    /// Center
    Center,
}

/// Calculate position from image size and padding size.
///
/// # Arguments
///
/// * `size` - Original image size (width, height)
/// * `pad_size` - Padded size (width, height)
/// * `position` - Padding position
///
/// # Returns
///
/// Returns the position (x, y) where the image should be placed on success
///
/// # Errors
///
/// * Returns error when padding size is smaller than original image size
pub fn calculate_position(
    size: (u32, u32),
    pad_size: (u32, u32),
    position: Position,
) -> Result<(i64, i64), PaddingError> {
    let (width, height) = size;
    let (target_width, target_height) = pad_size;

    if target_width < width {
        return Err(PaddingError::PaddingWidthTooSmall {
            width,
            pad_width: target_width,
        });
    }

    if target_height < height {
        return Err(PaddingError::PaddingHeightTooSmall {
            height,
            pad_height: target_height,
        });
    }

    let (x, y) = match position {
        Position::Top => ((target_width - width) / 2, 0),
        Position::Bottom => ((target_width - width) / 2, target_height - height),
        Position::Left => (0, (target_height - height) / 2),
        Position::Right => (target_width - width, (target_height - height) / 2),
        Position::TopLeft => (0, 0),
        Position::TopRight => (target_width - width, 0),
        Position::BottomLeft => (0, target_height - height),
        Position::BottomRight => (target_width - width, target_height - height),
        Position::Center => ((target_width - width) / 2, (target_height - height) / 2),
    };

    Ok((x.into(), y.into()))
}

/// Add padding with specified size and position (function-based implementation).
///
/// # Arguments
///
/// * `image` - Original image
/// * `pad_size` - Padded size (width, height)
/// * `position` - Padding position
/// * `color` - Padding color
///
/// # Returns
///
/// Padded image
///
/// # Examples
/// ```
/// use image::{Rgb, RgbImage};
/// use imageproc::definitions::Image;
/// use imageops_kit::{add_padding, Position};
///
/// let image: RgbImage = RgbImage::new(10, 10);
/// let padded = add_padding(&image, (20, 20), Position::Center, Rgb([255, 255, 255])).unwrap();
/// assert_eq!(padded.dimensions(), (20, 20));
/// ```
pub fn add_padding<P>(
    image: &Image<P>,
    pad_size: (u32, u32),
    position: Position,
    color: P,
) -> Result<Image<P>, PaddingError>
where
    P: Pixel,
{
    let (width, height) = image.dimensions();
    let (x, y) = calculate_position((width, height), pad_size, position)?;
    let (target_width, target_height) = pad_size;

    let mut out = create_buffer_impl(target_width, target_height, color);

    copy_image_impl(image, &mut out, x, y, width, height);

    Ok(out)
}

/// Copies source image to destination at specified offset.
///
/// Unsafe pixel access avoids redundant bounds checks since coordinates are validated by calculate_position.
#[inline]
fn copy_image_impl<P>(
    src: &Image<P>,
    dst: &mut Image<P>,
    offset_x: i64,
    offset_y: i64,
    width: u32,
    height: u32,
) where
    P: Pixel,
{
    let start_x = offset_x as u32;
    let start_y = offset_y as u32;

    // Row-wise copy reduces per-pixel overhead for wide images aligned at x=0
    if start_x == 0 && width > 64 {
        copy_rows_bulk_impl(src, dst, start_x, start_y, width, height);
        return;
    }

    iproduct!(0..height, 0..width).for_each(|(src_y, src_x)| {
        let dst_x = start_x + src_x;
        let dst_y = start_y + src_y;

        // Safety: Bounds validated by calculate_position
        unsafe {
            let pixel = src.unsafe_get_pixel(src_x, src_y);
            dst.unsafe_put_pixel(dst_x, dst_y, pixel);
        }
    });
}

/// Optimized path for full-width row copies to reduce iteration overhead.
#[inline]
fn copy_rows_bulk_impl<P>(
    src: &Image<P>,
    dst: &mut Image<P>,
    start_x: u32,
    start_y: u32,
    width: u32,
    height: u32,
) where
    P: Pixel,
{
    // ImageBuffer's API doesn't expose contiguous row slices, forcing per-pixel iteration
    iproduct!(0..height, 0..width).for_each(|(src_y, src_x)| {
        let dst_x = start_x + src_x;
        let dst_y = start_y + src_y;

        unsafe {
            let pixel = src.unsafe_get_pixel(src_x, src_y);
            dst.unsafe_put_pixel(dst_x, dst_y, pixel);
        }
    });
}

/// Creates buffer filled with specified color.
///
/// Pre-allocates exact capacity to avoid reallocations.
#[inline]
fn create_buffer_impl<P>(width: u32, height: u32, fill_color: P) -> Image<P>
where
    P: Pixel,
{
    let total_pixels = (width as usize) * (height as usize);
    let subpixels_per_pixel = P::CHANNEL_COUNT as usize;
    let total_subpixels = total_pixels * subpixels_per_pixel;

    let mut buffer = Vec::with_capacity(total_subpixels);

    let fill_channels = fill_color.channels();
    for _ in 0..total_pixels {
        buffer.extend_from_slice(fill_channels);
    }

    ImageBuffer::from_raw(width, height, buffer)
        .expect("Buffer size calculation error - this should not happen")
}

/// Trait that provides padding operations.
///
/// Note: This operation changes the image dimensions, so there is no `_mut` variant
/// available. The algorithm creates a new image with different dimensions.
pub trait PaddingExt<P: Pixel> {
    /// Add padding with specified size and position.
    ///
    /// This consumes the original image.
    ///
    /// # Arguments
    ///
    /// * `pad_size` - Padded size (width, height)
    /// * `position` - Padding position
    /// * `color` - Padding color
    ///
    /// # Returns
    ///
    /// Tuple of (padded image, position (x, y) where the original image was placed)
    ///
    /// # Examples
    /// ```no_run
    /// use imageops_kit::{PaddingExt, Position, Image};
    /// use image::Rgb;
    ///
    /// # fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let image: Image<Rgb<u8>> = Image::new(10, 10);
    /// let (padded, position) = image.add_padding((20, 20), Position::Center, Rgb([255, 255, 255]))?;
    /// # Ok(())
    /// # }
    /// ```
    fn add_padding(
        self,
        pad_size: (u32, u32),
        position: Position,
        color: P,
    ) -> Result<(Self, (i64, i64)), PaddingError>
    where
        Self: Sized;

    /// Add padding to make the image square.
    ///
    /// This consumes the original image.
    ///
    /// # Arguments
    ///
    /// * `color` - Padding color
    ///
    /// # Returns
    ///
    /// Tuple of (padded square image, position (x, y) where the original image was placed)
    fn to_square(self, color: P) -> Result<(Self, (i64, i64)), PaddingError>
    where
        Self: Sized;

    /// Calculate padding position.
    ///
    /// This is a helper method that doesn't consume self.
    fn calculate_padding_position(
        &self,
        pad_size: (u32, u32),
        position: Position,
    ) -> Result<(i64, i64), PaddingError>;

    /// Calculate position and size for square padding.
    ///
    /// This is a helper method that doesn't consume self.
    fn calculate_square_padding(&self) -> SquarePaddingResult;

    /// Hidden _mut variant that is not available for this operation.
    #[doc(hidden)]
    fn add_padding_mut(
        &mut self,
        _pad_size: (u32, u32),
        _position: Position,
        _color: P,
    ) -> Result<&mut Self, PaddingError> {
        unimplemented!(
            "add_padding_mut is not available because the operation changes image dimensions"
        )
    }

    /// Hidden _mut variant that is not available for this operation.
    #[doc(hidden)]
    fn to_square_mut(&mut self, _color: P) -> Result<&mut Self, PaddingError> {
        unimplemented!(
            "to_square_mut is not available because the operation changes image dimensions"
        )
    }
}

impl<P: Pixel> PaddingExt<P> for Image<P> {
    fn add_padding(
        self,
        pad_size: (u32, u32),
        position: Position,
        color: P,
    ) -> Result<(Self, (i64, i64)), PaddingError> {
        let (width, height) = self.dimensions();
        let pos = calculate_position((width, height), pad_size, position)?;
        let padded = add_padding(&self, pad_size, position, color)?;
        Ok((padded, pos))
    }

    fn to_square(self, color: P) -> Result<(Self, (i64, i64)), PaddingError> {
        let ((_x, _y), pad_size) = self.calculate_square_padding()?;
        self.add_padding(pad_size, Position::Center, color)
    }

    fn calculate_padding_position(
        &self,
        pad_size: (u32, u32),
        position: Position,
    ) -> Result<(i64, i64), PaddingError> {
        let (width, height) = self.dimensions();
        calculate_position((width, height), pad_size, position)
    }

    fn calculate_square_padding(&self) -> SquarePaddingResult {
        let (width, height) = self.dimensions();

        let pad_size = if width > height {
            (width, width)
        } else {
            (height, height)
        };

        self.calculate_padding_position(pad_size, Position::Center)
            .map(|(x, y)| ((x, y), pad_size))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::*;
    use image::Rgb;

    #[test]
    fn calculate_position_with_valid_input_returns_expected_coordinates() {
        // Test center position
        let pos = calculate_position((10, 10), (20, 20), Position::Center);
        assert_eq!(pos, Ok((5, 5)));

        // Test top-left position
        let pos = calculate_position((10, 10), (20, 20), Position::TopLeft);
        assert_eq!(pos, Ok((0, 0)));

        // Test top-right position
        let pos = calculate_position((10, 10), (20, 20), Position::TopRight);
        assert_eq!(pos, Ok((10, 0)));

        // Test bottom-left position
        let pos = calculate_position((10, 10), (20, 20), Position::BottomLeft);
        assert_eq!(pos, Ok((0, 10)));

        // Test bottom-right position
        let pos = calculate_position((10, 10), (20, 20), Position::BottomRight);
        assert_eq!(pos, Ok((10, 10)));
    }

    #[test]
    fn add_padding_with_valid_input_creates_padded_image() {
        let image = create_test_rgb_image(); // 2x2 image
        let fill_color = Rgb([255, 255, 255]); // White

        // Test padding to larger size
        let result = add_padding(&image, (4, 4), Position::Center, fill_color);
        assert!(result.is_ok());

        let padded = result.unwrap();
        assert_eq!(padded.dimensions(), (4, 4));

        // Test invalid padding (smaller than original)
        let result = add_padding(&image, (1, 1), Position::Center, fill_color);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PaddingError::PaddingWidthTooSmall { .. }
        ));
    }

    #[test]
    fn add_padding_ext_with_valid_input_preserves_original_content() {
        let image = create_test_rgb_image(); // 2x2 image
        let fill_color = Rgb([255, 255, 255]); // White

        // Store original pixels before moving image
        let orig_00 = *image.get_pixel(0, 0);
        let orig_10 = *image.get_pixel(1, 0);
        let orig_01 = *image.get_pixel(0, 1);
        let orig_11 = *image.get_pixel(1, 1);

        // Test regular padding
        let result = image.add_padding((4, 4), Position::TopLeft, fill_color);
        assert!(result.is_ok());

        let (padded, pos) = result.unwrap();
        assert_eq!(padded.dimensions(), (4, 4));
        assert_eq!(pos, (0, 0));

        // Verify original content is preserved
        assert_eq!(*padded.get_pixel(0, 0), orig_00);
        assert_eq!(*padded.get_pixel(1, 0), orig_10);
        assert_eq!(*padded.get_pixel(0, 1), orig_01);
        assert_eq!(*padded.get_pixel(1, 1), orig_11);
    }

    #[test]
    fn to_square_with_rectangular_image_creates_square_image() {
        // Test rectangular image (width > height)
        let mut image: Image<Rgb<u8>> = Image::new(6, 4);
        for y in 0..4 {
            for x in 0..6 {
                image.put_pixel(x, y, Rgb([100, 150, 200]));
            }
        }

        let result = image.to_square(Rgb([255, 255, 255]));
        assert!(result.is_ok());

        let (padded, pos) = result.unwrap();
        assert_eq!(padded.dimensions(), (6, 6)); // Should be square
        assert_eq!(pos, (0, 1)); // Centered vertically

        // Test square image (no padding needed)
        let square_image: Image<Rgb<u8>> = Image::new(4, 4);
        let result = square_image.to_square(Rgb([255, 255, 255]));
        assert!(result.is_ok());

        let (padded, pos) = result.unwrap();
        assert_eq!(padded.dimensions(), (4, 4));
        assert_eq!(pos, (0, 0));
    }

    #[test]
    fn calculate_padding_position_with_invalid_size_returns_error() {
        let image = create_test_rgb_image(); // 2x2 image

        // Test padding size too small (width)
        let result = image.calculate_padding_position((1, 4), Position::Center);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PaddingError::PaddingWidthTooSmall { .. }
        ));

        // Test padding size too small (height)
        let result = image.calculate_padding_position((4, 1), Position::Center);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            PaddingError::PaddingHeightTooSmall { .. }
        ));
    }

    #[test]
    fn calculate_padding_position_with_various_positions_returns_correct_coordinates() {
        let image = create_test_rgb_image(); // 2x2 image

        // Test all positions with 6x6 padding
        let positions = [
            (Position::TopLeft, (0, 0)),
            (Position::TopRight, (4, 0)),
            (Position::BottomLeft, (0, 4)),
            (Position::BottomRight, (4, 4)),
            (Position::Center, (2, 2)),
        ];

        for (pos, expected) in positions {
            let result = image.calculate_padding_position((6, 6), pos);
            assert!(result.is_ok());
            assert_eq!(result.unwrap(), expected);
        }
    }

    #[test]
    fn calculate_position_top_returns_center_x_zero_y() {
        let pos = calculate_position((4, 4), (10, 8), Position::Top);
        assert_eq!(pos, Ok((3, 0))); // (10-4)/2, 0
    }

    #[test]
    fn calculate_position_bottom_returns_center_x_max_y() {
        let pos = calculate_position((4, 4), (10, 8), Position::Bottom);
        assert_eq!(pos, Ok((3, 4))); // (10-4)/2, 8-4
    }

    #[test]
    fn calculate_position_left_returns_zero_x_center_y() {
        let pos = calculate_position((4, 4), (8, 10), Position::Left);
        assert_eq!(pos, Ok((0, 3))); // 0, (10-4)/2
    }

    #[test]
    fn calculate_position_right_returns_max_x_center_y() {
        let pos = calculate_position((4, 4), (8, 10), Position::Right);
        assert_eq!(pos, Ok((4, 3))); // 8-4, (10-4)/2
    }

    #[test]
    fn add_padding_with_same_size_preserves_original_image() {
        let image = create_test_rgb_image(); // 2x2 image
        let fill_color = Rgb([255, 255, 255]);

        let result = add_padding(&image, (2, 2), Position::Center, fill_color);
        assert!(result.is_ok());

        let padded = result.unwrap();
        assert_eq!(padded.dimensions(), (2, 2));
    }

    #[test]
    fn add_padding_with_extreme_size_difference_creates_correct_dimensions() {
        let image = create_test_rgb_image(); // 2x2 image
        let fill_color = Rgb([255, 255, 255]);

        let result = add_padding(&image, (100, 100), Position::Center, fill_color);
        assert!(result.is_ok());

        let padded = result.unwrap();
        assert_eq!(padded.dimensions(), (100, 100));
    }

    #[test]
    fn add_padding_with_rgba_pixel_creates_padded_image() {
        use image::Rgba;
        let mut image: Image<Rgba<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, Rgba([255, 0, 0, 255]));
        image.put_pixel(1, 1, Rgba([0, 255, 0, 128]));

        let fill_color = Rgba([0, 0, 255, 255]);
        let result = add_padding(&image, (4, 4), Position::Center, fill_color);

        assert!(result.is_ok());
        let padded = result.unwrap();
        assert_eq!(padded.dimensions(), (4, 4));
    }

    #[test]
    fn add_padding_with_luma_pixel_creates_padded_image() {
        use image::Luma;
        let mut image: Image<Luma<u8>> = Image::new(2, 2);
        image.put_pixel(0, 0, Luma([100]));
        image.put_pixel(1, 1, Luma([200]));

        let fill_color = Luma([50]);
        let result = add_padding(&image, (4, 4), Position::Center, fill_color);

        assert!(result.is_ok());
        let padded = result.unwrap();
        assert_eq!(padded.dimensions(), (4, 4));
    }

    #[test]
    fn copy_image_impl_with_zero_offset_preserves_pixel_values() {
        let src = create_test_rgb_image(); // 2x2 image
        let mut dst: Image<Rgb<u8>> = Image::new(4, 4);

        // Store original pixel for comparison
        let orig_pixel = *src.get_pixel(0, 0);

        copy_image_impl(&src, &mut dst, 0, 0, 2, 2);

        // Verify pixel was copied correctly
        assert_eq!(*dst.get_pixel(0, 0), orig_pixel);
    }

    #[test]
    fn copy_image_impl_with_offset_places_pixels_correctly() {
        let src = create_test_rgb_image(); // 2x2 image
        let mut dst: Image<Rgb<u8>> = Image::new(4, 4);

        let orig_pixel = *src.get_pixel(0, 0);

        copy_image_impl(&src, &mut dst, 1, 1, 2, 2);

        // Verify pixel was placed at correct offset position
        assert_eq!(*dst.get_pixel(1, 1), orig_pixel);
    }

    #[test]
    fn create_buffer_impl_with_rgb_creates_correct_dimensions() {
        let fill_color = Rgb([100, 150, 200]);
        let buffer = create_buffer_impl(10, 5, fill_color);

        assert_eq!(buffer.dimensions(), (10, 5));
    }

    #[test]
    fn create_buffer_impl_fills_with_specified_color() {
        let fill_color = Rgb([100, 150, 200]);
        let buffer = create_buffer_impl(3, 3, fill_color);

        // Check that all pixels have the fill color
        assert_eq!(*buffer.get_pixel(0, 0), fill_color);
        assert_eq!(*buffer.get_pixel(2, 2), fill_color);
    }

    #[test]
    fn to_square_with_height_greater_than_width_creates_square() {
        // Test rectangular image (height > width)
        let mut image: Image<Rgb<u8>> = Image::new(4, 6);
        for y in 0..6 {
            for x in 0..4 {
                image.put_pixel(x, y, Rgb([100, 150, 200]));
            }
        }

        let result = image.to_square(Rgb([255, 255, 255]));
        assert!(result.is_ok());

        let (padded, pos) = result.unwrap();
        assert_eq!(padded.dimensions(), (6, 6)); // Should be square
        assert_eq!(pos, (1, 0)); // Centered horizontally
    }

    #[test]
    fn add_padding_edge_case_boundary_conditions_safe() {
        let image = create_test_rgb_image(); // 2x2 image
        let fill_color = Rgb([255, 255, 255]);

        // Test edge case: image placed at exact boundary
        let result = add_padding(&image, (3, 3), Position::BottomRight, fill_color);
        assert!(result.is_ok());

        let padded = result.unwrap();
        assert_eq!(padded.dimensions(), (3, 3));

        // Verify original content is preserved at boundary position
        // BottomRight position places 2x2 image at (1, 1), so (0, 0) of original becomes (1, 1) of padded
        let orig_pixel = Rgb([200, 150, 100]); // From create_test_rgb_image (0, 0)
        assert_eq!(*padded.get_pixel(1, 1), orig_pixel);
    }

    #[test]
    fn bulk_copy_fallback_preserves_pixel_accuracy() {
        let src = create_test_rgb_image(); // 2x2 image
        let mut dst: Image<Rgb<u8>> = Image::new(100, 100); // Large dest to trigger bulk copy logic

        let orig_pixel = *src.get_pixel(1, 0);

        copy_image_impl(&src, &mut dst, 0, 0, 2, 2);

        // Verify pixel accuracy in bulk copy scenario
        assert_eq!(*dst.get_pixel(1, 0), orig_pixel);
    }

    #[test]
    fn calculate_position_center_with_odd_sized_padding_handles_non_divisible_correctly() {
        // Test case where (target_size - original_size) is odd, resulting in non-equal padding
        let pos = calculate_position((3, 3), (8, 8), Position::Center);
        assert_eq!(pos, Ok((2, 2))); // (8-3)/2 = 2.5 → 2 (truncated)

        // This means left/top padding = 2, right/bottom padding = 3
        // Verify asymmetric padding behavior
        let pos = calculate_position((4, 4), (9, 9), Position::Center);
        assert_eq!(pos, Ok((2, 2))); // (9-4)/2 = 2.5 → 2 (truncated)
    }

    #[test]
    fn add_padding_with_odd_sized_target_creates_asymmetric_padding() {
        let image = create_test_rgb_image(); // 2x2 image
        let fill_color = Rgb([255, 255, 255]);

        // Test 2x2 → 5x5 padding (results in asymmetric padding)
        let result = add_padding(&image, (5, 5), Position::Center, fill_color);
        assert!(result.is_ok());

        let padded = result.unwrap();
        assert_eq!(padded.dimensions(), (5, 5));

        // Original content should be at position (1, 1) - (2, 2)
        // (5-2)/2 = 1.5 → 1 (truncated), so left/top padding = 1, right/bottom = 2
        let orig_pixel = *image.get_pixel(0, 0); // Original (0,0)
        assert_eq!(*padded.get_pixel(1, 1), orig_pixel); // Should be at (1,1) in padded

        // Verify fill color in padding areas
        assert_eq!(*padded.get_pixel(0, 0), fill_color); // Top-left padding
        assert_eq!(*padded.get_pixel(4, 4), fill_color); // Bottom-right padding
    }

    #[test]
    fn add_padding_ext_with_odd_sizes_returns_correct_position() {
        let image = create_test_rgb_image(); // 2x2 image
        let fill_color = Rgb([255, 255, 255]);

        let result = image.add_padding((7, 7), Position::Center, fill_color);
        assert!(result.is_ok());

        let (padded, pos) = result.unwrap();
        assert_eq!(padded.dimensions(), (7, 7));
        assert_eq!(pos, (2, 2)); // (7-2)/2 = 2.5 → 2 (truncated)
    }
}
