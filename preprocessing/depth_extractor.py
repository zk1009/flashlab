"""
Depth Anything V2 wrapper for LightLab.

Per Section 3.4 / Section B.2 of LightLab:
  "We use [Yang et al. 2024] to create depth maps."
  [Yang et al. 2024] = Depth Anything V2 (arXiv:2406.09414)

The depth map is used as a spatial condition to help the model understand
scene geometry and generate physically plausible shadows/reflections.

Output: Normalized depth in [0, 1] at the VAE latent resolution (image_size // 8).
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple, Optional


class DepthExtractor:
    """
    Wraps Depth Anything V2 (HuggingFace Transformers) to produce
    depth maps at the VAE latent resolution.

    Uses the Large variant for best quality. Falls back to Small variant
    if GPU memory is limited.

    Args:
        model_size:  "large", "base", or "small".
        device:      Computation device (default "cuda").
        dtype:       Model dtype (float16 for efficiency, float32 for accuracy).
    """

    MODEL_IDS = {
        "large": "depth-anything/Depth-Anything-V2-Large-hf",
        "base": "depth-anything/Depth-Anything-V2-Base-hf",
        "small": "depth-anything/Depth-Anything-V2-Small-hf",
    }

    def __init__(
        self,
        model_size: str = "large",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        model_id = self.MODEL_IDS.get(model_size, self.MODEL_IDS["large"])
        print(f"Loading Depth Anything V2 ({model_size}) from {model_id}...")

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_id, torch_dtype=dtype
        ).to(device)
        self.model.eval()
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """
        Compute a depth map for the given PIL image and resize to target dimensions.

        Args:
            image:    PIL Image (RGB).
            target_h: Target height (typically image_height // 8 for VAE latent).
            target_w: Target width (typically image_width // 8 for VAE latent).

        Returns:
            Normalized depth map, shape (1, 1, target_h, target_w), ∈ [0, 1].
            1.0 = farthest, 0.0 = closest (or vice versa depending on convention;
            Depth Anything V2 outputs relative inverse depth, so higher = closer).
        """
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device, dtype=self.dtype) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        # predicted_depth: (1, H, W) — raw inverse depth
        depth = outputs.predicted_depth.float()  # cast to float32 for interpolation
        depth = depth.unsqueeze(1)  # (1, 1, H, W)

        # Resize to latent spatial dimensions
        depth = F.interpolate(
            depth,
            size=(target_h, target_w),
            mode="bilinear",
            align_corners=False,
        )

        # Normalize to [0, 1] per image
        d_min = depth.amin(dim=(-2, -1), keepdim=True)
        d_max = depth.amax(dim=(-2, -1), keepdim=True)
        depth = (depth - d_min) / (d_max - d_min + 1e-8)

        return depth  # (1, 1, target_h, target_w)

    def extract_and_save(
        self,
        image_path: str,
        save_path: str,
        target_h: int = 128,
        target_w: int = 128,
    ):
        """
        Extract depth from an image file and save as .npy for preprocessing caching.

        Args:
            image_path: Path to input image.
            save_path:  Path to save the .npy depth map.
            target_h:   Target height (default 128 = 1024 // 8).
            target_w:   Target width.
        """
        import os
        image = Image.open(image_path).convert("RGB")
        depth = self(image, target_h, target_w)
        depth_np = depth.squeeze().cpu().numpy()  # (H, W)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, depth_np)

    def batch_process_directory(
        self,
        input_dir: str,
        output_dir: str,
        image_size: int = 1024,
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".tiff"),
        skip_existing: bool = True,
    ):
        """
        Batch-process all images in a directory and save depth maps.
        Used for offline preprocessing to cache depth maps before training.

        Args:
            input_dir:     Directory containing input images.
            output_dir:    Directory to save .npy depth files.
            image_size:    Expected image size (used to compute latent size).
            extensions:    Image file extensions to process.
            skip_existing: Skip if .npy already exists.
        """
        import os
        from pathlib import Path
        from tqdm import tqdm

        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        lat_size = image_size // 8
        image_files = [
            f for f in sorted(input_path.rglob("*"))
            if f.suffix.lower() in extensions
        ]

        print(f"Processing {len(image_files)} images from {input_dir}...")

        for img_path in tqdm(image_files):
            # Preserve relative directory structure
            rel_path = img_path.relative_to(input_path)
            save_path = output_path / rel_path.with_suffix(".npy")

            if skip_existing and save_path.exists():
                continue

            try:
                image = Image.open(img_path).convert("RGB")
                depth = self(image, lat_size, lat_size)
                depth_np = depth.squeeze().cpu().numpy()

                save_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(str(save_path), depth_np)
            except Exception as e:
                print(f"Warning: Failed to process {img_path}: {e}")

        print(f"Depth maps saved to {output_dir}")
