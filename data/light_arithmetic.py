"""
Light arithmetic operations as described in LightLab (Section 3.2).

Core formulas:
  ichange = clip(ion - ioff, 0)                          (Eq. from Sec 3.1)
  c = ct ⊙ co^{-1}                                       (color coefficient)
  irelit(α, γ, ct) = α·iamb + γ·ichange⊙c               (Eq. 1)

All operations are in linear RGB color space (float32).
"""

import numpy as np
import torch
import torch.nn.functional as F


def extract_light_change(
    ion: np.ndarray,
    ioff: np.ndarray,
    clip_negative: bool = True,
) -> np.ndarray:
    """
    Extract the contribution of the target light source.

    Args:
        ion:  Image with light ON, linear HDR float32 (H, W, 3).
        ioff: Image with light OFF (ambient only), linear HDR float32 (H, W, 3).
        clip_negative: Whether to clip negative values to 0 (paper default).

    Returns:
        ichange: Light contribution image, shape (H, W, 3), float32.
    """
    diff = ion.astype(np.float64) - ioff.astype(np.float64)
    if clip_negative:
        diff = np.clip(diff, 0.0, None)
    return diff.astype(np.float32)


def estimate_light_color(
    ichange: np.ndarray,
    mask: np.ndarray,
    percentile: float = 90.0,
) -> np.ndarray:
    """
    Estimate the original RGB color of a light source from its ichange image.

    Takes the brightest percentile of pixels within the light mask region
    to robustly estimate the light color without noise interference.

    Args:
        ichange: Light contribution image, float32 (H, W, 3).
        mask:    Binary mask of light source region, bool/float (H, W).
        percentile: Percentile for robust estimation (default 90).

    Returns:
        co: Estimated light color, shape (3,), normalized to sum to 1.
    """
    mask_bool = mask.astype(bool)
    # Extract pixels within the mask
    pixels = ichange[mask_bool]  # (N, 3)
    if len(pixels) == 0:
        return np.array([1.0, 1.0, 1.0], dtype=np.float32)

    # Use the luminance channel to find the brightest pixels
    luminance = 0.2126 * pixels[:, 0] + 0.7152 * pixels[:, 1] + 0.0722 * pixels[:, 2]
    threshold = np.percentile(luminance, percentile)
    bright_pixels = pixels[luminance >= threshold]

    co = bright_pixels.mean(axis=0)
    # Normalize so that the max channel is 1.0
    max_val = co.max()
    if max_val > 1e-8:
        co = co / max_val
    else:
        co = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    return co.astype(np.float32)


def compute_color_coefficient(
    co: np.ndarray,
    ct: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Compute the color change coefficient c = ct ⊙ co^{-1}.

    Args:
        co: Original light color, shape (3,).
        ct: Target light color, shape (3,).
        eps: Small value to avoid division by zero.

    Returns:
        c: Color coefficient, shape (3,).
    """
    inv_co = np.where(co > eps, 1.0 / co, 0.0)
    return (ct * inv_co).astype(np.float32)


def relit_image(
    iamb: np.ndarray,
    ichange: np.ndarray,
    alpha: float,
    gamma: float,
    c: np.ndarray,
) -> np.ndarray:
    """
    Synthesize a relit image using the LightLab light arithmetic formula (Eq. 1):
        irelit(α, γ, ct; iamb, ichange) = α·iamb + γ·ichange⊙c

    Args:
        iamb:    Ambient (light off) image, linear HDR float32 (H, W, 3).
        ichange: Light contribution image, linear HDR float32 (H, W, 3).
        alpha:   Relative ambient illumination intensity ∈ [0, 1].
        gamma:   Relative target light intensity ∈ [0, 1].
        c:       Color change coefficient, shape (3,).

    Returns:
        irelit: Relit image in linear color space, float32 (H, W, 3).
    """
    c_broadcast = c[np.newaxis, np.newaxis, :]  # (1, 1, 3)
    result = alpha * iamb + gamma * ichange * c_broadcast
    return result.astype(np.float32)


def generate_training_pairs(
    iamb: np.ndarray,
    ichange: np.ndarray,
    mask: np.ndarray,
    n_intensity: int = 6,
    n_alpha: int = 5,
    n_colors: int = 2,
    color_temps: list = None,
) -> list:
    """
    Generate a set of (input, target, condition) tuples from a single
    (iamb, ichange) pair by sampling the intensity/color parameter space.

    Implements the ×60 inflation strategy from the paper (Section 3.2):
    - 6 intensity values × 5 ambient values × 2 color settings = 60 pairs

    Args:
        iamb:        Ambient image (H, W, 3), linear float32.
        ichange:     Light contribution (H, W, 3), linear float32.
        mask:        Binary light source mask (H, W).
        n_intensity: Number of target light intensity samples.
        n_alpha:     Number of ambient intensity samples.
        n_colors:    Number of color temperature samples per pair.
        color_temps: List of (R, G, B) tuples for target colors. If None,
                     uses default set of blackbody temperatures + random colors.

    Returns:
        List of dicts with keys:
            'input':      Source image (linear float32, H×W×3)
            'target':     Target relit image (linear float32, H×W×3)
            'gamma':      Target light intensity used
            'alpha':      Ambient intensity used
            'ct':         Target color (3,)
            'c':          Color coefficient (3,)
            'gamma_src':  Source image gamma
            'alpha_src':  Source image alpha
    """
    if color_temps is None:
        # Approximate blackbody temperatures as normalized RGB
        color_temps = [
            np.array([1.00, 0.56, 0.16]),  # 2700K warm white
            np.array([1.00, 0.70, 0.40]),  # 3000K
            np.array([1.00, 0.85, 0.70]),  # 4000K neutral
            np.array([1.00, 0.95, 0.90]),  # 5500K daylight
            np.array([0.90, 0.95, 1.00]),  # 6500K cool
            np.array([0.60, 0.80, 1.00]),  # blue
            np.array([1.00, 0.20, 0.10]),  # red
            np.array([0.20, 1.00, 0.20]),  # green
        ]

    # Estimate original light color
    co = estimate_light_color(ichange, mask)

    # Intensity grids (Section 3.2: 6 intensity × 5 ambient × 2 colors = 60 pairs)
    gamma_values = np.linspace(0.0, 1.0, n_intensity)
    alpha_values = np.linspace(0.5, 1.2, n_alpha)

    # Source image uses gamma_src=1.0 (light fully on) as the reference
    gamma_src = 1.0

    pairs = []
    for ct in color_temps[:n_colors]:
        c = compute_color_coefficient(co, ct)
        input_img = relit_image(iamb, ichange, alpha=1.0, gamma=gamma_src, c=c)
        for gamma_tgt in gamma_values:
            for alpha_tgt in alpha_values:
                target_img = relit_image(iamb, ichange, alpha=alpha_tgt, gamma=gamma_tgt, c=c)
                pairs.append({
                    "input": input_img,
                    "target": target_img,
                    "gamma": float(gamma_tgt),
                    "alpha": float(alpha_tgt),
                    "ct": ct.astype(np.float32),
                    "c": c.astype(np.float32),
                    "gamma_src": float(gamma_src),
                    "alpha_src": 1.0,
                })

    return pairs
