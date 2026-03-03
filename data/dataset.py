"""
Unified LightLab training Dataset.

Combines real photograph pairs and synthetic dataset (Multi-Illumination)
with the light arithmetic inflation strategy from Section 3.2.

Each item returned contains all the inputs needed for training:
  - input_image:     Source image (SDR, normalized [0,1])  (3, H, W)
  - target_image:    Target relit image (SDR, normalized)  (3, H, W)
  - depth_map:       Normalized depth map                  (1, H, W)
  - intensity_mask:  Light source mask × γ                 (1, H, W)
  - color_mask:      Light source mask expanded × ct_rgb   (3, H, W)
  - ambient_alpha:   Ambient intensity change scalar        float ∈ [-1, 1]
  - tonemap_flag:    0.0 (separate) or 1.0 (together)      float
  - gamma:           Target light intensity                 float ∈ [0, 1]
"""

import os
import random
import json
from pathlib import Path
from typing import Optional, Callable

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


# Default color temperatures (normalized RGB approximations of blackbody)
DEFAULT_COLOR_TEMPS = [
    np.array([1.00, 0.56, 0.16], dtype=np.float32),  # 2700K warm
    np.array([1.00, 0.70, 0.40], dtype=np.float32),  # 3000K
    np.array([1.00, 0.85, 0.70], dtype=np.float32),  # 4000K neutral
    np.array([1.00, 0.95, 0.90], dtype=np.float32),  # 5500K daylight
    np.array([0.90, 0.95, 1.00], dtype=np.float32),  # 6500K cool
    np.array([0.60, 0.80, 1.00], dtype=np.float32),  # blue
    np.array([1.00, 0.20, 0.10], dtype=np.float32),  # red
    np.array([0.20, 1.00, 0.20], dtype=np.float32),  # green
]


class RealPairDataset(Dataset):
    """
    Dataset of real photograph pairs (light on / off).

    Expected directory structure:
        real_pairs/
            scene_001/
                off.png  (or off.jpg, off.tiff)
                on.png
                mask.png  (optional, binary mask of light source)
                depth.npy (optional, pre-computed depth map)
            scene_002/
                ...

    If mask.png is not provided, a simple luminance-based mask is estimated.
    If depth.npy is not provided, it will be computed on-the-fly using Depth
    Anything V2 (requires preprocessing/depth_extractor.py).
    """

    def __init__(
        self,
        root: str,
        image_size: int = 1024,
        gamma_values: list = None,
        alpha_values: list = None,
        color_temps: list = None,
        depth_cache_dir: Optional[str] = None,
        mask_cache_dir: Optional[str] = None,
        dropout_prob: float = 0.10,
        transform: Optional[Callable] = None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.gamma_values = gamma_values or [0.0, 0.25, 0.5, 0.75, 1.0]
        self.alpha_values = alpha_values or [0.7, 0.85, 1.0, 1.1, 1.2]
        self.color_temps = color_temps or DEFAULT_COLOR_TEMPS
        self.depth_cache_dir = Path(depth_cache_dir) if depth_cache_dir else None
        self.mask_cache_dir = Path(mask_cache_dir) if mask_cache_dir else None
        self.dropout_prob = dropout_prob
        self.transform = transform

        self._discover_scenes()

    def _discover_scenes(self):
        """Find all valid scene directories."""
        self.scenes = []
        if not self.root.exists():
            return

        for scene_dir in sorted(self.root.iterdir()):
            if not scene_dir.is_dir():
                continue
            # Look for on/off pairs in various formats
            on_path = self._find_image(scene_dir, ["on"])
            off_path = self._find_image(scene_dir, ["off", "amb", "ambient"])
            if on_path and off_path:
                self.scenes.append({
                    "on": on_path,
                    "off": off_path,
                    "mask": self._find_image(scene_dir, ["mask", "light_mask"]),
                    "depth": self._find_npy(scene_dir, ["depth"]),
                    "name": scene_dir.name,
                })

        # Build flat index for inflation
        self.index = []
        for si, scene in enumerate(self.scenes):
            for gamma_tgt in self.gamma_values:
                for alpha_tgt in self.alpha_values:
                    ct_idx = random.randint(0, len(self.color_temps) - 1)
                    self.index.append((si, gamma_tgt, alpha_tgt, ct_idx))

    def _find_image(self, directory: Path, stems: list) -> Optional[Path]:
        for stem in stems:
            for ext in [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".exr"]:
                p = directory / f"{stem}{ext}"
                if p.exists():
                    return p
        return None

    def _find_npy(self, directory: Path, stems: list) -> Optional[Path]:
        for stem in stems:
            for ext in [".npy", ".pt"]:
                p = directory / f"{stem}{ext}"
                if p.exists():
                    return p
        return None

    def _load_image_linear(self, path: Path) -> np.ndarray:
        """Load image and convert to linear float32 (H, W, 3)."""
        if path.suffix == ".exr":
            from .multi_illumination import load_exr_as_float
            img = load_exr_as_float(str(path))
        else:
            pil = Image.open(path).convert("RGB")
            pil = pil.resize((self.image_size, self.image_size), Image.LANCZOS)
            img = pil_to_linear(pil)
        return img.astype(np.float32)

    def _get_depth(self, scene: dict, image_pil: Image.Image) -> np.ndarray:
        """Load or compute depth map, normalized to [0, 1], shape (H, H)."""
        name = scene["name"]
        # Try cached depth first
        if self.depth_cache_dir:
            cache_path = self.depth_cache_dir / f"{name}.npy"
            if cache_path.exists():
                depth = np.load(str(cache_path))
                return self._resize_depth(depth)

        if scene["depth"] is not None:
            if scene["depth"].suffix == ".npy":
                depth = np.load(str(scene["depth"]))
            else:
                depth = torch.load(str(scene["depth"])).numpy()
            return self._resize_depth(depth)

        # Fallback: uniform depth (degraded but functional)
        lat_size = self.image_size // 8
        return np.ones((lat_size, lat_size), dtype=np.float32) * 0.5

    def _resize_depth(self, depth: np.ndarray) -> np.ndarray:
        lat_size = self.image_size // 8
        if depth.shape != (lat_size, lat_size):
            import cv2
            depth = cv2.resize(depth, (lat_size, lat_size), interpolation=cv2.INTER_LINEAR)
        # Normalize to [0, 1]
        d_min, d_max = depth.min(), depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        return depth.astype(np.float32)

    def _get_mask(self, scene: dict, ichange: np.ndarray) -> np.ndarray:
        """Load or estimate light source mask, shape (H, W)."""
        name = scene["name"]
        if self.mask_cache_dir:
            cache_path = self.mask_cache_dir / f"{name}.npy"
            if cache_path.exists():
                return np.load(str(cache_path))

        if scene["mask"] is not None:
            pil = Image.open(scene["mask"]).convert("L")
            pil = pil.resize((self.image_size, self.image_size), Image.NEAREST)
            mask = np.array(pil).astype(np.float32) / 255.0
            return mask

        # Estimate from ichange: bright luminance regions
        lum = (
            0.2126 * ichange[:, :, 0]
            + 0.7152 * ichange[:, :, 1]
            + 0.0722 * ichange[:, :, 2]
        )
        threshold = np.percentile(lum, 85)
        return (lum >= threshold).astype(np.float32)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        scene_idx, gamma_tgt, alpha_tgt, ct_idx = self.index[idx]
        scene = self.scenes[scene_idx]

        # Load image pair in linear color space
        iamb = self._load_image_linear(scene["off"])
        ion = self._load_image_linear(scene["on"])
        ichange = extract_light_change(ion, iamb)

        # Get mask and color
        mask = self._get_mask(scene, ichange)
        ct = self.color_temps[ct_idx]
        co = estimate_light_color(ichange, mask)
        c = compute_color_coefficient(co, ct)

        # Build source image (random gamma_src)
        gamma_src = random.choice(self.gamma_values)
        alpha_src = 1.0
        input_linear = relit_image(iamb, ichange, alpha_src, gamma_src, c)
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

        # Ambient delta: difference between alpha values, normalized to [-1, 1]
        ambient_alpha = float(np.clip(alpha_tgt - 1.0, -1.0, 1.0))

        # Get depth map at latent resolution
        depth = self._get_depth(scene, None)  # (lat_H, lat_W)
        lat_size = self.image_size // 8

        # Build mask at latent resolution
        mask_pil = Image.fromarray((mask * 255).astype(np.uint8)).resize(
            (lat_size, lat_size), Image.NEAREST
        )
        mask_lat = np.array(mask_pil).astype(np.float32) / 255.0  # (lat_H, lat_W)

        # Condition dropout (10%)
        drop_depth = random.random() < self.dropout_prob
        drop_color = random.random() < self.dropout_prob

        # Build tensors
        def to_tensor(arr):
            return torch.from_numpy(arr).permute(2, 0, 1).float()  # (3, H, W)

        input_t = to_tensor(np.clip(input_sdr, 0, 1) * 2.0 - 1.0)   # [-1, 1] for VAE
        target_t = to_tensor(np.clip(target_sdr, 0, 1) * 2.0 - 1.0)

        depth_t = torch.from_numpy(depth).unsqueeze(0).float()  # (1, lat_H, lat_W)
        if drop_depth:
            depth_t = torch.zeros_like(depth_t)

        mask_t = torch.from_numpy(mask_lat).unsqueeze(0).float()  # (1, lat_H, lat_W)
        intensity_mask_t = mask_t * gamma_tgt                      # scale by γ

        ct_t = torch.from_numpy(ct).float()  # (3,)
        color_mask_t = mask_t.expand(3, -1, -1) * ct_t.view(3, 1, 1)  # (3, lat_H, lat_W)
        if drop_color:
            color_mask_t = torch.zeros_like(color_mask_t)

        return {
            "input_image": input_t,           # (3, H, W)  ∈ [-1, 1]
            "target_image": target_t,          # (3, H, W)  ∈ [-1, 1]
            "depth_map": depth_t,              # (1, lat_H, lat_W)  ∈ [0, 1]
            "intensity_mask": intensity_mask_t,# (1, lat_H, lat_W)  ∈ [0, γ]
            "color_mask": color_mask_t,        # (3, lat_H, lat_W)
            "ambient_alpha": torch.tensor(ambient_alpha, dtype=torch.float32),
            "tonemap_flag": torch.tensor(tonemap_flag, dtype=torch.float32),
            "gamma": torch.tensor(gamma_tgt, dtype=torch.float32),
        }


class LightLabDataset(Dataset):
    """
    Unified dataset that combines real pairs and synthetic data
    with configurable sampling weights.

    Args:
        real_root:       Path to real photograph pairs directory.
        synthetic_root:  Path to Multi-Illumination (or Hypersim) dataset.
        real_weight:     Probability of sampling from real data [0, 1].
        image_size:      Spatial resolution for training (default 1024).
        depth_cache_dir: Directory with pre-computed depth maps (.npy files).
        mask_cache_dir:  Directory with pre-computed masks (.npy files).
        dropout_prob:    Probability of dropping depth+color conditions.
        synthetic_type:  "multi_illumination" or "hypersim".
    """

    def __init__(
        self,
        real_root: Optional[str] = None,
        synthetic_root: Optional[str] = None,
        real_weight: float = 0.05,
        image_size: int = 1024,
        depth_cache_dir: Optional[str] = None,
        mask_cache_dir: Optional[str] = None,
        dropout_prob: float = 0.10,
        synthetic_type: str = "multi_illumination",
        total_samples: int = 200000,
    ):
        self.real_weight = real_weight
        self.total_samples = total_samples

        # Build constituent datasets
        self.real_dataset = None
        self.synthetic_dataset = None

        if real_root and Path(real_root).exists():
            self.real_dataset = RealPairDataset(
                root=real_root,
                image_size=image_size,
                depth_cache_dir=depth_cache_dir,
                mask_cache_dir=mask_cache_dir,
                dropout_prob=dropout_prob,
            )
            print(f"Real dataset: {len(self.real_dataset)} samples")

        if synthetic_root and Path(synthetic_root).exists():
            if synthetic_type == "multi_illumination":
                from .multi_illumination import MultiIlluminationDataset
                self.synthetic_base = MultiIlluminationDataset(
                    root=synthetic_root,
                    image_size=image_size,
                )
                self.synthetic_dataset = InflatedSyntheticDataset(
                    base_dataset=self.synthetic_base,
                    image_size=image_size,
                    depth_cache_dir=depth_cache_dir,
                    dropout_prob=dropout_prob,
                )
                print(f"Synthetic dataset: {len(self.synthetic_dataset)} samples")

        if self.real_dataset is None and self.synthetic_dataset is None:
            raise ValueError(
                "At least one of real_root or synthetic_root must be provided "
                "and point to an existing directory."
            )

    def __len__(self) -> int:
        return self.total_samples

    def __getitem__(self, idx: int) -> dict:
        # Sample from real or synthetic based on weight
        if self.real_dataset is not None and self.synthetic_dataset is not None:
            use_real = random.random() < self.real_weight
        elif self.real_dataset is not None:
            use_real = True
        else:
            use_real = False

        if use_real:
            real_idx = idx % len(self.real_dataset)
            return self.real_dataset[real_idx]
        else:
            syn_idx = idx % len(self.synthetic_dataset)
            return self.synthetic_dataset[syn_idx]


class InflatedSyntheticDataset(Dataset):
    """
    Wraps a base dataset (e.g. MultiIlluminationDataset) and applies the
    light arithmetic inflation strategy (×36 from the paper).

    For each (iamb, ichange) pair, samples 36 different (γ, α, ct) combinations.
    """

    def __init__(
        self,
        base_dataset,
        image_size: int = 1024,
        depth_cache_dir: Optional[str] = None,
        dropout_prob: float = 0.10,
        inflation: int = 36,
        gamma_values: list = None,
        alpha_values: list = None,
        color_temps: list = None,
    ):
        self.base = base_dataset
        self.image_size = image_size
        self.depth_cache_dir = Path(depth_cache_dir) if depth_cache_dir else None
        self.dropout_prob = dropout_prob
        self.inflation = inflation

        self.gamma_values = gamma_values or [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        self.alpha_values = alpha_values or [0.7, 0.85, 1.0, 1.1, 1.2]
        self.color_temps = color_temps or DEFAULT_COLOR_TEMPS[:3]

    def __len__(self) -> int:
        return len(self.base) * self.inflation

    def __getitem__(self, idx: int) -> dict:
        base_idx = idx // self.inflation
        var_idx = idx % self.inflation

        item = self.base[base_idx]
        iamb = item["iamb"]      # (H, W, 3)
        ichange = item["ichange"]
        mask = item["mask"]      # (H, W)

        # Sample parameters based on var_idx
        gamma_tgt = self.gamma_values[var_idx % len(self.gamma_values)]
        alpha_tgt = self.alpha_values[(var_idx // len(self.gamma_values)) % len(self.alpha_values)]
        ct_idx = (var_idx // (len(self.gamma_values) * len(self.alpha_values))) % len(self.color_temps)
        ct = self.color_temps[ct_idx]

        # Compute color coefficient
        from .light_arithmetic import estimate_light_color, compute_color_coefficient
        co = estimate_light_color(ichange, mask)
        c = compute_color_coefficient(co, ct)

        # Build input (γ=1.0 as source = light on) and target
        gamma_src = 1.0
        input_linear = relit_image(iamb, ichange, 1.0, gamma_src, c)
        target_linear = relit_image(iamb, ichange, alpha_tgt, gamma_tgt, c)

        use_together = random.random() > 0.5
        if use_together:
            [input_sdr, target_sdr] = tone_map_together([input_linear, target_linear])
            tonemap_flag = 1.0
        else:
            input_sdr = tone_map_separate(input_linear)
            target_sdr = tone_map_separate(target_linear)
            tonemap_flag = 0.0

        ambient_alpha = float(np.clip(alpha_tgt - 1.0, -1.0, 1.0))

        lat_size = self.image_size // 8

        # Get depth
        depth = self._get_depth(item, lat_size)

        # Build mask at latent resolution
        import cv2
        mask_lat = cv2.resize(mask, (lat_size, lat_size), interpolation=cv2.INTER_NEAREST)
        mask_lat = mask_lat.astype(np.float32)

        # Condition dropout
        drop_depth = random.random() < self.dropout_prob
        drop_color = random.random() < self.dropout_prob

        def to_tensor(arr):
            return torch.from_numpy(arr).permute(2, 0, 1).float()

        input_t = to_tensor(np.clip(input_sdr, 0, 1) * 2.0 - 1.0)
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

    def _get_depth(self, item: dict, lat_size: int) -> np.ndarray:
        if self.depth_cache_dir:
            cache_key = f"{item['scene']}_{item.get('direction', 0)}"
            cache_path = self.depth_cache_dir / f"{cache_key}.npy"
            if cache_path.exists():
                depth = np.load(str(cache_path))
                import cv2
                return cv2.resize(depth, (lat_size, lat_size), interpolation=cv2.INTER_LINEAR)

        # Uniform depth fallback
        return np.ones((lat_size, lat_size), dtype=np.float32) * 0.5
