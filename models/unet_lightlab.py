"""
LightLab UNet: Modified SDXL UNet that accepts 8 input channels.

Per Section 3.4 and Supplementary C.1 of LightLab:
  The noise latents (4 channels) are concatenated with the encoded spatial
  conditions (4 channels from SpatialConditionEncoder) to form an 8-channel
  tensor. The first conv layer of the SDXL UNet is expanded from 4→8 channels.

Weight initialization strategy:
  - First 4 weight slices (for noise channels): copied from pretrained SDXL
  - Last 4 weight slices (for spatial conditions): zero-initialized
  - This ensures the network starts identical to the pretrained model at step 0.
"""

import torch
import torch.nn as nn
from typing import Optional


def build_lightlab_unet(
    pretrained_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    new_in_channels: int = 8,
    torch_dtype: torch.dtype = torch.float32,
    enable_gradient_checkpointing: bool = False,
) -> "UNet2DConditionModel":
    """
    Load SDXL UNet and expand conv_in from 4→8 input channels.

    Strategy:
        1. Load original 4-channel UNet to capture pretrained weights.
        2. Load a second copy with ignore_mismatched_sizes=True (8 channels).
        3. Copy pretrained weights into first 4 slices of new conv_in.
        4. Zero-initialize the remaining 4 slices.

    Args:
        pretrained_model_id:           HuggingFace model ID or local path.
        new_in_channels:               Target input channels (default 8).
        torch_dtype:                   dtype for model weights.
        enable_gradient_checkpointing: Saves VRAM during training (recommended).

    Returns:
        Modified UNet2DConditionModel with 8-channel conv_in.
    """
    from diffusers import UNet2DConditionModel

    print(f"Loading pretrained SDXL UNet from {pretrained_model_id}...")

    # Step 1: Load original UNet to get pretrained weights
    original_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_id,
        subfolder="unet",
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
    )
    original_conv_weight = original_unet.conv_in.weight.clone().detach()  # (320, 4, 3, 3)
    original_conv_bias = original_unet.conv_in.bias.clone().detach()      # (320,)
    del original_unet

    # Step 2: Load UNet with expanded conv_in
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_id,
        subfolder="unet",
        in_channels=new_in_channels,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    )

    # Step 3: Initialize the expanded conv_in weights
    with torch.no_grad():
        out_channels, _, kH, kW = original_conv_weight.shape  # (320, 4, 3, 3)

        # New weight tensor: (320, 8, 3, 3)
        new_weight = torch.zeros(
            out_channels, new_in_channels, kH, kW,
            dtype=original_conv_weight.dtype,
        )

        # Copy pretrained weights for the first 4 input channels (noise latents)
        new_weight[:, :4, :, :] = original_conv_weight

        # Channels 4–7 (spatial conditions) remain zero:
        # → At training start, spatial conditions have no effect → preserves pretrained prior

        unet.conv_in.weight.copy_(new_weight)
        unet.conv_in.bias.copy_(original_conv_bias)

    print(
        f"UNet conv_in expanded: 4→{new_in_channels} channels. "
        f"Pretrained weights in [:4], zero-init in [4:]."
    )

    if enable_gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        print("Gradient checkpointing enabled.")

    return unet


def save_lightlab_checkpoint(
    unet,
    spatial_encoder,
    global_embedder,
    step: int,
    output_dir: str,
    accelerator=None,
):
    """
    Save all trainable components to a single checkpoint file.

    Args:
        unet:             Modified LightLab UNet.
        spatial_encoder:  SpatialConditionEncoder.
        global_embedder:  GlobalConditionEmbedder.
        step:             Current training step.
        output_dir:       Directory to save checkpoint.
        accelerator:      HuggingFace Accelerator (for unwrapping on multi-GPU).
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Unwrap DDP-wrapped models if needed
    def unwrap(model):
        if accelerator is not None:
            return accelerator.unwrap_model(model)
        return model

    checkpoint = {
        "unet": unwrap(unet).state_dict(),
        "spatial_encoder": unwrap(spatial_encoder).state_dict(),
        "global_embedder": unwrap(global_embedder).state_dict(),
        "step": step,
    }

    save_path = os.path.join(output_dir, f"checkpoint_{step:06d}.pt")
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")
    return save_path


def load_lightlab_checkpoint(
    checkpoint_path: str,
    pretrained_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
    device: str = "cuda",
    torch_dtype: torch.dtype = torch.float16,
) -> dict:
    """
    Load a LightLab checkpoint and reconstruct all model components.

    Args:
        checkpoint_path:   Path to a .pt checkpoint file.
        pretrained_model_id: Base SDXL model for reconstructing UNet architecture.
        device:            Target device.
        torch_dtype:       dtype for inference (float16 recommended).

    Returns:
        Dict with keys: 'unet', 'spatial_encoder', 'global_embedder', 'step'.
    """
    from .spatial_encoder import SpatialConditionEncoder
    from .global_conditioning import GlobalConditionEmbedder

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    unet = build_lightlab_unet(
        pretrained_model_id=pretrained_model_id,
        torch_dtype=torch_dtype,
    )
    unet.load_state_dict(checkpoint["unet"])
    unet = unet.to(device, dtype=torch_dtype)

    spatial_encoder = SpatialConditionEncoder(in_channels=9, out_channels=4)
    spatial_encoder.load_state_dict(checkpoint["spatial_encoder"])
    spatial_encoder = spatial_encoder.to(device, dtype=torch_dtype)

    global_embedder = GlobalConditionEmbedder(output_dim=2048)
    global_embedder.load_state_dict(checkpoint["global_embedder"])
    global_embedder = global_embedder.to(device, dtype=torch_dtype)

    return {
        "unet": unet,
        "spatial_encoder": spatial_encoder,
        "global_embedder": global_embedder,
        "step": checkpoint.get("step", 0),
    }
