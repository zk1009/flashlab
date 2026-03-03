"""
Flash Photography Dataset for LightLab.

Simplified dataset for flash on/off pairs where:
  - The "light source" is the camera flash (not a visible lamp in the scene)
  - The mask covers the entire image (flash illuminates globally)
  - No bounding box or SAM 2 segmentation needed

Expected directory structure:
    flash_pairs/
        scene_001/
            flash.jpg    (or flash.png — image with flash ON)
            noflash.jpg  (or noflash.png — image with flash OFF, ambient only)
        scene_002/
            ...

Each (flash, noflash) pair produces multiple training samples via
light arithmetic inflation: varying γ (flash intensity), α (ambient),
and ct (flash color).
"""

import os
import random
from pathlib import Path
from typing import Optional, Callable, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

from .light_arithmetic import (
    extract_light_change,
    estimate_light_color,
    compute_color_coefficient,
    relit_image,
)
from .tone_mapping import tone_map_separate, tone_map_together, pil_to_linear


# Flash color temperatures (normalized RGB)
FLASH_COLOR_TEMPS = [
    np.array([1.00, 1.00, 1.00], dtype=np.float32),  # neutral white (standard flash)
    np.array([1.00, 0.95, 0.90], dtype=np.float32),  # 5500K daylight flash
    np.array([1.00, 0.85, 0.70], dtype=np.float32),  # 4000K warm flash
    np.array([1.00, 0.70, 0.40], dtype=np.float32),  # 3000K very warm
    np.array([0.90, 0.95, 1.00], dtype=np.float32),  # 6500K cool flash
    np.array([0.70, 0.85, 1.00], dtype=np.float32),  # blue-tinted flash
]


class FlashPairDataset(Dataset):
    """
    Dataset of flash on/off photograph pairs.

    Key simplification vs LightLab's RealPairDataset:
      - mask = full image (1.0 everywhere) since flash illuminates globally
      - No bbox.json needed
      - No SAM 2 segmentation needed
      - Color estimation uses the full ichange image

    Args:
        root:            Path to flash pairs directory.
        image_size:      Spatial resolution for training (default 1024).
        gamma_values:    Flash intensity values to sample.
        alpha_values:    Ambient intensity values to sample.
        color_temps:     List of target flash color temperatures.
        depth_cache_dir: Directory with pre-computed depth maps (.npy).
        dropout_prob:    Probability of dropping depth/color conditions.
    """

    def __init__(
        self,
        root: str,
        image_size: int = 1024,
        gamma_values: list = None,
        alpha_values: list = None,
        color_temps: list = None,
        depth_cache_dir: Optional[str] = None,
        dropout_prob: float = 0.10,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.gamma_values = gamma_values or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.alpha_values = alpha_values or [0.7, 0.85, 1.0, 1.1, 1.2]
        self.color_temps = color_temps or FLASH_COLOR_TEMPS
        self.depth_cache_dir = Path(depth_cache_dir) if depth_cache_dir else None
        self.dropout_prob = dropout_prob

        self._discover_scenes()

    def _discover_scenes(self):
        """Find all valid flash on/off scene directories."""
        self.scenes = []
        if not self.root.exists():
            return

        for scene_dir in sorted(self.root.iterdir()):
            if not scene_dir.is_dir():
                continue
            flash_path = self._find_image(scene_dir, ["flash", "on", "lit"])
            noflash_path = self._find_image(scene_dir, ["noflash", "off", "ambient", "amb", "no_flash"])
            if flash_path and noflash_path:
                self.scenes.append({
                    "flash": flash_path,
                    "noflash": noflash_path,
                    "name": scene_dir.name,
                })

        # Build flat index for inflation:
        # Each scene × each gamma_tgt × each alpha_tgt × random color
        self.index = []
        for si in range(len(self.scenes)):
            for gamma_tgt in self.gamma_values:
                for alpha_tgt in self.alpha_values:
                    ct_idx = random.randint(0, len(self.color_temps) - 1)
                    self.index.append((si, gamma_tgt, alpha_tgt, ct_idx))

        print(f"FlashPairDataset: {len(self.scenes)} scenes, {len(self.index)} inflated samples")

    def _find_image(self, directory: Path, stems: list) -> Optional[Path]:
        for stem in stems:
            for ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif"]:
                p = directory / f"{stem}{ext}"
                if p.exists():
                    return p
        return None

    def _load_image_linear(self, path: Path) -> np.ndarray:
        """Load image and convert to linear float32 (H, W, 3)."""
        pil = Image.open(path).convert("RGB")
        pil = pil.resize((self.image_size, self.image_size), Image.LANCZOS)
        img = pil_to_linear(pil)
        return img.astype(np.float32)

    def _get_depth(self, scene_name: str) -> np.ndarray:
        """Load cached depth map or return uniform fallback."""
        lat_size = self.image_size // 8
        if self.depth_cache_dir:
            cache_path = self.depth_cache_dir / f"{scene_name}.npy"
            if cache_path.exists():
                depth = np.load(str(cache_path))
                if depth.shape != (lat_size, lat_size):
                    depth = cv2.resize(depth, (lat_size, lat_size), interpolation=cv2.INTER_LINEAR)
                d_min, d_max = depth.min(), depth.max()
                if d_max > d_min:
                    depth = (depth - d_min) / (d_max - d_min)
                return depth.astype(np.float32)

        # Uniform depth fallback — still works due to 10% depth dropout
        return np.ones((lat_size, lat_size), dtype=np.float32) * 0.5

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        scene_idx, gamma_tgt, alpha_tgt, ct_idx = self.index[idx]
        scene = self.scenes[scene_idx]

        # Load image pair in linear color space
        iamb = self._load_image_linear(scene["noflash"])    # flash OFF = ambient
        ion = self._load_image_linear(scene["flash"])        # flash ON
        ichange = extract_light_change(ion, iamb)

        # Full-image mask (flash illuminates everything)
        mask = np.ones((self.image_size, self.image_size), dtype=np.float32)

        # Color estimation and coefficient
        ct = self.color_temps[ct_idx]
        co = estimate_light_color(ichange, mask)
        c = compute_color_coefficient(co, ct)

        # Build source image (gamma_src=1.0 = original flash on)
        gamma_src = 1.0
        input_linear = relit_image(iamb, ichange, 1.0, gamma_src, c)
        target_linear = relit_image(iamb, ichange, alpha_tgt, gamma_tgt, c)

        # Tone mapping: randomly pick strategy (50/50)
        use_together = random.random() > 0.5
        if use_together:
            [input_sdr, target_sdr] = tone_map_together([input_linear, target_linear])
            tonemap_flag = 1.0
        else:
            input_sdr = tone_map_separate(input_linear)
            target_sdr = tone_map_separate(target_linear)
            tonemap_flag = 0.0

        # Ambient delta normalized to [-1, 1]
        ambient_alpha = float(np.clip(alpha_tgt - 1.0, -1.0, 1.0))

        # Build tensors at latent resolution
        lat_size = self.image_size // 8

        # Depth map
        depth = self._get_depth(scene["name"])

        # Mask at latent resolution — full image = all ones
        mask_lat = np.ones((lat_size, lat_size), dtype=np.float32)

        # Condition dropout (10%)
        drop_depth = random.random() < self.dropout_prob
        drop_color = random.random() < self.dropout_prob

        def to_tensor(arr):
            return torch.from_numpy(arr).permute(2, 0, 1).float()

        input_t = to_tensor(np.clip(input_sdr, 0, 1) * 2.0 - 1.0)    # [-1, 1] for VAE
        target_t = to_tensor(np.clip(target_sdr, 0, 1) * 2.0 - 1.0)

        depth_t = torch.from_numpy(depth).unsqueeze(0).float()
        if drop_depth:
            depth_t = torch.zeros_like(depth_t)

        mask_t = torch.from_numpy(mask_lat).unsqueeze(0).float()
        intensity_mask_t = mask_t * gamma_tgt

        ct_t = torch.from_numpy(ct).float()
        color_mask_t = mask_t.expand(3, -1, -1) * ct_t.view(3, 1, 1)
        if drop_color:
            color_mask_t = torch.zeros_like(color_mask_t)

        return {
            "input_image": input_t,
            "target_image": target_t,
            "depth_map": depth_t,
            "intensity_mask": intensity_mask_t,
            "color_mask": color_mask_t,
            "ambient_alpha": torch.tensor(ambient_alpha, dtype=torch.float32),
            "tonemap_flag": torch.tensor(tonemap_flag, dtype=torch.float32),
            "gamma": torch.tensor(gamma_tgt, dtype=torch.float32),
        }


class FlashLabDataset(Dataset):
    """
    Unified flash dataset with configurable total sample count.

    Wraps FlashPairDataset and loops over it to provide the requested
    number of training samples.

    Args:
        flash_root:      Path to flash pairs directory.
        image_size:      Spatial resolution for training.
        depth_cache_dir: Directory with pre-computed depth maps.
        dropout_prob:    Condition dropout probability.
        total_samples:   Total number of training samples to serve.
    """

    def __init__(
        self,
        flash_root: str,
        image_size: int = 1024,
        depth_cache_dir: Optional[str] = None,
        dropout_prob: float = 0.10,
        total_samples: int = 200000,
    ):
        self.total_samples = total_samples

        if not flash_root or not Path(flash_root).exists():
            raise ValueError(f"Flash data root not found: {flash_root}")

        self.flash_dataset = FlashPairDataset(
            root=flash_root,
            image_size=image_size,
            depth_cache_dir=depth_cache_dir,
            dropout_prob=dropout_prob,
        )

        if len(self.flash_dataset) == 0:
            raise ValueError(
                f"No valid flash pairs found in {flash_root}. "
                "Expected subdirectories with flash.jpg + noflash.jpg"
            )

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> dict:
        real_idx = idx % len(self.flash_dataset)
        return self.flash_dataset[real_idx]
