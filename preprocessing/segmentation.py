"""
Light Source Segmentation using SAM 2 (Ravi et al. 2024).

Per Section 3.4 and Supplementary B.2 of LightLab:
  "We use [Ravi et al. 2024] light source segmentation..."
  [Ravi et al. 2024] = SAM 2: Segment Anything in Images and Videos (arXiv:2408.00714)

The user provides a bounding box around the target light source, and SAM 2
produces a binary segmentation mask that is used as the spatial light condition.

The mask is used two ways:
  1. Intensity condition: mask × γ (target light intensity scalar)
  2. Color condition:     mask × ct_rgb (target light color in 3 channels)
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional, Tuple, Union


class LightSourceSegmenter:
    """
    Wraps SAM 2 for interactive light source segmentation.

    Accepts a user-defined bounding box (from the Gradio demo or CLI)
    and returns a binary mask at the VAE latent resolution.

    Args:
        checkpoint:  Path to SAM 2 model checkpoint (.pt file).
        model_cfg:   SAM 2 model config name (e.g. "sam2_hiera_l.yaml").
        device:      Computation device.
    """

    # Default SAM 2 Large checkpoint
    DEFAULT_CHECKPOINT = "checkpoints/sam2_hiera_large.pt"
    DEFAULT_CONFIG = "sam2_hiera_l.yaml"

    def __init__(
        self,
        checkpoint: str = DEFAULT_CHECKPOINT,
        model_cfg: str = DEFAULT_CONFIG,
        device: str = "cuda",
    ):
        self.device = device
        self._load_model(checkpoint, model_cfg, device)

    def _load_model(self, checkpoint: str, model_cfg: str, device: str):
        """Load SAM 2 model with graceful error handling."""
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            sam2_model = build_sam2(model_cfg, checkpoint, device=device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            print(f"SAM 2 loaded from {checkpoint}")
            self._sam2_available = True

        except (ImportError, FileNotFoundError) as e:
            print(
                f"Warning: SAM 2 not available ({e}). "
                "Falling back to bounding-box mask. "
                "Install SAM 2: pip install git+https://github.com/facebookresearch/sam2.git "
                "and download checkpoints from https://github.com/facebookresearch/sam2#download-checkpoints"
            )
            self.predictor = None
            self._sam2_available = False

    @torch.no_grad()
    def segment_from_bbox(
        self,
        image: np.ndarray,    # (H, W, 3) uint8 RGB
        bbox: List[int],      # [x_min, y_min, x_max, y_max] in image pixel coords
        target_h: int,
        target_w: int,
        select_mask_idx: int = 0,
    ) -> torch.Tensor:
        """
        Segment the light source within the given bounding box.

        Args:
            image:           Input image as uint8 numpy array (H, W, 3).
            bbox:            Bounding box [x_min, y_min, x_max, y_max].
            target_h:        Target mask height (VAE latent = image_h // 8).
            target_w:        Target mask width.
            select_mask_idx: Which SAM 2 candidate mask to use (0 = best confidence).

        Returns:
            Binary float mask, shape (1, 1, target_h, target_w), values ∈ {0, 1}.
        """
        if self._sam2_available and self.predictor is not None:
            return self._segment_with_sam2(image, bbox, target_h, target_w, select_mask_idx)
        else:
            return self._bbox_to_mask(image, bbox, target_h, target_w)

    def _segment_with_sam2(
        self,
        image: np.ndarray,
        bbox: List[int],
        target_h: int,
        target_w: int,
        select_mask_idx: int,
    ) -> torch.Tensor:
        """Use SAM 2 for precise light source segmentation."""
        self.predictor.set_image(image)

        bbox_arr = np.array(bbox, dtype=np.float32)[None, :]  # (1, 4)
        masks, scores, _ = self.predictor.predict(
            box=bbox_arr,
            multimask_output=True,  # Get multiple candidate masks
        )
        # masks: (num_masks, H, W) bool
        # scores: (num_masks,) — confidence scores

        # Select mask with highest confidence
        best_idx = int(np.argmax(scores))
        mask_bool = masks[best_idx]  # (H, W) bool

        mask = torch.from_numpy(mask_bool.astype(np.float32))  # (H, W)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        mask = F.interpolate(mask, size=(target_h, target_w), mode="nearest")
        return mask

    def _bbox_to_mask(
        self,
        image: np.ndarray,
        bbox: List[int],
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """
        Fallback: create a rectangular mask from the bounding box when SAM 2
        is not available. Less precise but functional.
        """
        H, W = image.shape[:2]
        x_min, y_min, x_max, y_max = bbox

        mask = np.zeros((H, W), dtype=np.float32)
        mask[y_min:y_max, x_min:x_max] = 1.0

        mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        mask_t = F.interpolate(mask_t, size=(target_h, target_w), mode="nearest")
        return mask_t

    def segment_from_points(
        self,
        image: np.ndarray,
        points: List[List[int]],    # List of [x, y] point coordinates
        point_labels: List[int],    # 1=foreground, 0=background
        target_h: int,
        target_w: int,
    ) -> torch.Tensor:
        """
        Alternative: segment using point prompts instead of bounding box.
        Useful when the user clicks on the light source in the UI.

        Args:
            image:        (H, W, 3) uint8 RGB.
            points:       List of [x, y] coordinates.
            point_labels: 1 for foreground (light source), 0 for background.
            target_h:     Target mask height.
            target_w:     Target mask width.

        Returns:
            Binary mask, shape (1, 1, target_h, target_w).
        """
        if not self._sam2_available:
            raise RuntimeError("SAM 2 is required for point-based segmentation.")

        self.predictor.set_image(image)

        points_arr = np.array(points, dtype=np.float32)        # (N, 2)
        labels_arr = np.array(point_labels, dtype=np.int32)    # (N,)

        masks, scores, _ = self.predictor.predict(
            point_coords=points_arr,
            point_labels=labels_arr,
            multimask_output=True,
        )

        best_idx = int(np.argmax(scores))
        mask_bool = masks[best_idx]

        mask = torch.from_numpy(mask_bool.astype(np.float32))
        mask = mask.unsqueeze(0).unsqueeze(0)
        mask = F.interpolate(mask, size=(target_h, target_w), mode="nearest")
        return mask

    def build_condition_masks(
        self,
        seg_mask: torch.Tensor,   # (1, 1, H, W)
        gamma: float,
        ct_rgb: Optional[List[float]],  # [R, G, B] ∈ [0, 1] or None
        device: str = "cuda",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build the two spatial condition masks from a segmentation mask:
          1. intensity_mask = seg_mask × γ       → (1, 1, H, W)
          2. color_mask     = seg_mask × ct_rgb  → (1, 3, H, W)

        Args:
            seg_mask: Binary float mask (1, 1, H, W).
            gamma:    Target light intensity ∈ [-1, 1] (negative = dim light).
            ct_rgb:   Target light color as [R, G, B] ∈ [0, 1]. None = white.
            device:   Target device.

        Returns:
            intensity_mask: (1, 1, H, W)
            color_mask:     (1, 3, H, W)
        """
        seg_mask = seg_mask.to(device)
        intensity_mask = seg_mask * gamma  # (1, 1, H, W)

        if ct_rgb is None:
            ct_rgb = [1.0, 1.0, 1.0]  # white = no color change

        ct = torch.tensor(ct_rgb, dtype=torch.float32, device=device).view(1, 3, 1, 1)
        color_mask = seg_mask.expand(-1, 3, -1, -1) * ct  # (1, 3, H, W)

        return intensity_mask, color_mask

    def batch_process_dataset(
        self,
        scene_dirs: List[str],
        output_dir: str,
        image_size: int = 1024,
        skip_existing: bool = True,
    ):
        """
        Batch-process a dataset to cache SAM 2 masks for training.

        Note: For training data, masks must be pre-computed from bounding boxes.
        This function expects each scene directory to contain a 'bbox.json' file
        with format {"bbox": [x_min, y_min, x_max, y_max]}.

        Args:
            scene_dirs:    List of scene directories.
            output_dir:    Directory to save mask .npy files.
            image_size:    Expected image spatial size.
            skip_existing: Skip if mask already exists.
        """
        import os
        import json
        from pathlib import Path
        from tqdm import tqdm

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        lat_size = image_size // 8

        for scene_dir in tqdm(scene_dirs):
            scene_path = Path(scene_dir)
            bbox_file = scene_path / "bbox.json"
            if not bbox_file.exists():
                continue

            scene_name = scene_path.name
            save_path = output_path / f"{scene_name}.npy"
            if skip_existing and save_path.exists():
                continue

            # Load bounding box annotation
            with open(bbox_file) as f:
                data = json.load(f)
            bbox = data["bbox"]  # [x_min, y_min, x_max, y_max]

            # Load the "on" image for segmentation
            on_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                p = scene_path / f"on{ext}"
                if p.exists():
                    on_path = p
                    break

            if on_path is None:
                continue

            image_pil = Image.open(on_path).convert("RGB")
            image_pil = image_pil.resize((image_size, image_size), Image.LANCZOS)
            image_np = np.array(image_pil)

            mask = self.segment_from_bbox(image_np, bbox, lat_size, lat_size)
            mask_np = mask.squeeze().cpu().numpy()  # (lat_H, lat_W)
            np.save(str(save_path), mask_np)

        print(f"Masks saved to {output_dir}")
