//! Internal utility functions for imageops-ai.
//!
//! This module contains common functionality used across different image operations.

mod unify;
pub use unify::{LargerType, NormalizedFrom, unify_gray_images, unify_rgb_images};

use image::Primitive;

/// Normalizes an alpha value using a pre-computed max value.
///
/// # Arguments
///
/// * `alpha` - The alpha value to normalize
/// * `max_value` - The pre-computed maximum value for the type
///
/// # Returns
///
/// The normalized alpha value as a floating-point number between 0 and 1
#[inline]
pub fn normalize_alpha_with_max<S>(alpha: S, max_value: f32) -> f32
where
    S: Into<f32> + Primitive,
{
    alpha.into() / max_value
}

/// Validates that an image has non-zero dimensions.
///
/// # Arguments
///
/// * `width` - The width of the image
/// * `height` - The height of the image
/// * `context` - A description of the context for error messages
///
/// # Returns
///
/// `Ok(())` if the dimensions are valid, otherwise an error
pub fn validate_non_empty_image(width: u32, height: u32, context: &str) -> Result<(), String> {
    if width == 0 || height == 0 {
        Err(format!("{context}: Image dimensions must be non-zero"))
    } else {
        Ok(())
    }
}

/// Validates that two images have matching dimensions.
///
/// # Arguments
///
/// * `width1` - The width of the first image
/// * `height1` - The height of the first image
/// * `width2` - The width of the second image
/// * `height2` - The height of the second image
/// * `context` - A description of the context for error messages
///
/// # Returns
///
/// `Ok(())` if the dimensions match, otherwise an error
pub fn validate_matching_dimensions(
    width1: u32,
    height1: u32,
    width2: u32,
    height2: u32,
    context: &str,
) -> Result<(), String> {
    if width1 != width2 || height1 != height2 {
        Err(format!(
            "{context}: Image dimensions must match. Got {width1}x{height1} and {width2}x{height2}"
        ))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_alpha_with_max_with_valid_input_returns_correct_ratios() {
        assert_eq!(normalize_alpha_with_max(0u8, 255.0), 0.0);
        assert_eq!(normalize_alpha_with_max(127u8, 255.0), 127.0 / 255.0);
        assert_eq!(normalize_alpha_with_max(255u8, 255.0), 1.0);
    }

    #[test]
    fn validate_non_empty_image_with_valid_dimensions_accepts() {
        validate_non_empty_image(100, 100, "test").unwrap();
        validate_non_empty_image(1, 1, "test").unwrap();
        assert!(validate_non_empty_image(0, 100, "test").is_err());
        assert!(validate_non_empty_image(100, 0, "test").is_err());
        assert!(validate_non_empty_image(0, 0, "test").is_err());
    }

    #[test]
    fn validate_matching_dimensions_with_matching_sizes_accepts() {
        validate_matching_dimensions(100, 100, 100, 100, "test").unwrap();
        validate_matching_dimensions(50, 75, 50, 75, "test").unwrap();
        assert!(validate_matching_dimensions(100, 100, 100, 50, "test").is_err());
        assert!(validate_matching_dimensions(100, 100, 50, 100, "test").is_err());
        assert!(validate_matching_dimensions(100, 100, 50, 50, "test").is_err());
    }
}
