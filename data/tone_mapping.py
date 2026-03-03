"""
Tone mapping strategies as described in LightLab Section 3.3.

Two strategies:
  "separate": Tone-map each image independently (auto-exposure per image).
  "together": Tone-map a sequence using a shared fixed exposure computed from
              a "deciding" image irelit(γ_d, α_d), so that perceived light
              intensity changes are intuitive and physically consistent.

Both are exposed as input conditions to the diffusion model during training.
"""

import numpy as np
import cv2
from typing import Union


def _clip_outliers(image: np.ndarray, top_percentile: float = 99.95) -> np.ndarray:
    """Clip pixel values at the given top percentile to suppress unbounded outliers."""
    emax = np.percentile(image, top_percentile)
    if emax < 1e-8:
        return image
    return np.clip(image, 0.0, emax)


def _linear_to_srgb(image: np.ndarray) -> np.ndarray:
    """Apply sRGB gamma correction (approximate)."""
    image = np.clip(image, 0.0, 1.0)
    # Standard sRGB transfer function
    low_mask = image <= 0.0031308
    result = np.where(
        low_mask,
        12.92 * image,
        1.055 * np.power(np.maximum(image, 1e-10), 1.0 / 2.4) - 0.055,
    )
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def tone_map_separate(
    image_linear: np.ndarray,
    percentile: float = 99.5,
    apply_srgb: bool = True,
) -> np.ndarray:
    """
    'Separate' tone mapping: auto-expose each image independently.

    Normalizes by the given percentile of pixel values, then applies
    optional sRGB gamma correction. This ensures each image is well-exposed
    regardless of absolute brightness.

    Args:
        image_linear: Linear HDR image, float32 (H, W, 3).
        percentile:   Clipping percentile for exposure estimation.
        apply_srgb:   Whether to apply sRGB gamma correction.

    Returns:
        SDR image in [0, 1], float32 (H, W, 3).
    """
    image_clipped = _clip_outliers(image_linear, top_percentile=percentile)
    emax = np.percentile(image_clipped, percentile)
    if emax < 1e-8:
        return np.zeros_like(image_linear, dtype=np.float32)

    normalized = np.clip(image_clipped / emax, 0.0, 1.0)
    if apply_srgb:
        return _linear_to_srgb(normalized)
    return normalized.astype(np.float32)


def tone_map_together(
    images_linear: list,
    deciding_idx: int = 0,
    apply_srgb: bool = True,
) -> list:
    """
    'Together' tone mapping: all images in a sequence share the same
    fixed exposure, computed from a reference 'deciding' image.

    This ensures that perceived changes in light intensity are intuitive —
    turning up a light makes it brighter relative to the ambient, rather
    than appearing constant due to auto-exposure normalization.

    Uses Mertens exposure fusion (Mertens et al. 2007) on the deciding image
    to compute a shared exposure, then applies the same normalization to all images.

    Args:
        images_linear:  List of linear HDR images, each float32 (H, W, 3).
        deciding_idx:   Index of the reference image for computing exposure.
        apply_srgb:     Apply sRGB gamma correction after tone mapping.

    Returns:
        List of SDR images in [0, 1], same length as input.
    """
    if not images_linear:
        return []

    # Compute shared exposure from the deciding image using Mertens fusion
    deciding_img = images_linear[deciding_idx]
    exposure = _compute_mertens_exposure(deciding_img)

    results = []
    for img_linear in images_linear:
        # Apply the same shared exposure to all images
        normalized = np.clip(img_linear * exposure, 0.0, 1.0).astype(np.float32)
        if apply_srgb:
            normalized = _linear_to_srgb(normalized)
        results.append(normalized)

    return results


def _compute_mertens_exposure(image_linear: np.ndarray) -> float:
    """
    Use Mertens exposure fusion to determine the optimal exposure multiplier
    for a given HDR image.

    Creates a multi-exposure bracket from the single HDR image, fuses them
    with Mertens to find the well-exposed range, then derives a scalar
    exposure multiplier.

    Args:
        image_linear: Linear HDR image, float32 (H, W, 3).

    Returns:
        Scalar exposure multiplier to apply to all images in the sequence.
    """
    # Create a multi-exposure bracket from the single HDR image
    # Simulate different exposure levels by scaling
    exposure_scales = [0.25, 0.5, 1.0, 2.0, 4.0]
    bracket = []
    for scale in exposure_scales:
        exposed = np.clip(image_linear * scale, 0.0, None)
        # Apply simple gamma for Mertens input (it expects LDR uint8)
        ldr = np.clip(np.power(exposed / (exposed.max() + 1e-8), 1.0 / 2.2), 0.0, 1.0)
        bracket.append((ldr * 255).astype(np.uint8))

    # Run Mertens exposure fusion
    merge_mertens = cv2.createMergeMertens(
        contrast_weight=1.0,
        saturation_weight=1.0,
        exposure_weight=1.0,
    )
    fused = merge_mertens.process(bracket)  # float32 (H, W, 3) in [0, 1+]
    fused = np.clip(fused, 0.0, 1.0)

    # Derive exposure multiplier: find the scale that maps the deciding image
    # so that its 99.5th percentile matches the fused result's brightness
    fused_brightness = np.percentile(fused, 99.5)
    original_brightness = np.percentile(image_linear, 99.5)

    if original_brightness < 1e-8:
        return 1.0

    exposure = fused_brightness / original_brightness
    return max(exposure, 1e-8)


def tone_map_image(
    image_linear: np.ndarray,
    strategy: str = "separate",
    apply_srgb: bool = True,
) -> np.ndarray:
    """
    Convenience function to tone-map a single image.

    Args:
        image_linear: Linear HDR image, float32 (H, W, 3).
        strategy: "separate" or "together" (for single image, both are equivalent).
        apply_srgb: Apply sRGB gamma correction.

    Returns:
        SDR image in [0, 1], float32 (H, W, 3).
    """
    return tone_map_separate(image_linear, apply_srgb=apply_srgb)


def pil_to_linear(image_pil) -> np.ndarray:
    """Convert a PIL SDR image (uint8 sRGB) to linear float32."""
    arr = np.array(image_pil).astype(np.float32) / 255.0
    # Inverse sRGB gamma
    low_mask = arr <= 0.04045
    linear = np.where(
        low_mask,
        arr / 12.92,
        np.power((arr + 0.055) / 1.055, 2.4),
    )
    return linear.astype(np.float32)


def linear_to_pil(image_linear: np.ndarray):
    """Convert a linear float32 image to PIL (sRGB uint8)."""
    from PIL import Image
    sdr = _linear_to_srgb(np.clip(image_linear, 0.0, 1.0))
    return Image.fromarray((sdr * 255).astype(np.uint8))
