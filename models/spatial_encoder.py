"""
Spatial Condition Encoder for LightLab.

Implements the 1×1 convolution that fuses the 9-channel spatial condition
stack into 4 channels before concatenation with the noise latents.

Per Section 3.4 and Supplementary C.1 of LightLab:
  Spatial conditions:
    - Input image encoded by VAE:  4 channels
    - Depth map:                   1 channel
    - Intensity mask × γ:          1 channel
    - Color mask × ct (3-ch):      3 channels
  Total:                           9 channels → 4 channels (1×1 conv, zero-init)

Zero-initialization (from ControlNet, Nichol et al. 2021):
  At training start, the spatial condition contributes nothing to the noise
  prediction, preserving the pretrained diffusion prior. The network learns
  to use these signals progressively.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialConditionEncoder(nn.Module):
    """
    Reduces the stacked 9-channel spatial conditions to 4 channels via
    a zero-initialized 1×1 convolution, then these 4 channels are
    concatenated to the input noise latents (4 channels) before the UNet.

    The full input to the UNet first conv layer is therefore 8 channels:
        [noise_latents (4) | spatial_encoded (4)] → UNet conv_in (8→320)

    Args:
        in_channels:  Number of spatial condition channels (default 9).
        out_channels: Output channels, must match noise latent channels (default 4).
    """

    def __init__(self, in_channels: int = 9, out_channels: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )
        # Zero-initialization: at training start, no contribution from spatial conditions
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(
        self,
        image_latent: torch.Tensor,    # (B, 4, H, W) — VAE-encoded input image
        depth_map: torch.Tensor,        # (B, 1, H, W) — normalized depth ∈ [0, 1]
        intensity_mask: torch.Tensor,   # (B, 1, H, W) — light mask × γ scalar
        color_mask: torch.Tensor,       # (B, 3, H, W) — light mask × ct RGB
    ) -> torch.Tensor:
        """
        Args:
            image_latent:    VAE-encoded input image, (B, 4, lH, lW).
            depth_map:       Depth map at latent resolution, (B, 1, lH, lW).
            intensity_mask:  Light source mask scaled by γ, (B, 1, lH, lW).
            color_mask:      Light source mask scaled by ct RGB, (B, 3, lH, lW).

        Returns:
            Encoded spatial condition, (B, 4, lH, lW).
        """
        # Ensure all tensors are at the same spatial resolution
        target_h, target_w = image_latent.shape[-2:]
        if depth_map.shape[-2:] != (target_h, target_w):
            depth_map = F.interpolate(depth_map, size=(target_h, target_w), mode="bilinear", align_corners=False)
        if intensity_mask.shape[-2:] != (target_h, target_w):
            intensity_mask = F.interpolate(intensity_mask, size=(target_h, target_w), mode="nearest")
        if color_mask.shape[-2:] != (target_h, target_w):
            color_mask = F.interpolate(color_mask, size=(target_h, target_w), mode="nearest")

        # Stack all 9 spatial conditions along channel dimension
        spatial_stack = torch.cat(
            [image_latent, depth_map, intensity_mask, color_mask], dim=1
        )  # (B, 9, lH, lW)

        # 1×1 conv → (B, 4, lH, lW)
        return self.conv(spatial_stack)

    def extra_repr(self) -> str:
        return (
            f"in_channels=9 (image_latent:4 + depth:1 + intensity_mask:1 + color_mask:3), "
            f"out_channels=4, zero_initialized=True"
        )
