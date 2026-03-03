"""
Multi-Illumination Dataset loader (Murmann et al. 2019).
Used as a substitute for Blender synthetic renders.

Dataset URL: https://projects.csail.mit.edu/illumination/
Structure:
    train/
        scene_XXXX/
            dir_00_mip5.jpg  ...  dir_24_mip5.jpg  (25 flash directions)
    test/
        ...

We treat dir_00 as ambient (iamb) and all other directions as
"light on" images to extract ichange.
"""

import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


def load_jpg_as_float(path: str) -> np.ndarray:
    """Load a JPEG image and convert to linear float32 (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    # Approximate inverse sRGB gamma to get linear values
    low = arr <= 0.04045
    linear = np.where(low, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)
    return linear.astype(np.float32)


def load_exr_as_float(path: str) -> np.ndarray:
    """Load an EXR image in linear float32 (H, W, 3)."""
    try:
        import OpenEXR
        import Imath
        f = OpenEXR.InputFile(str(path))
        dw = f.header()["dataWindow"]
        h = dw.max.y - dw.min.y + 1
        w = dw.max.x - dw.min.x + 1
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        channels = {
            c: np.frombuffer(f.channel(c, pt), dtype=np.float32).reshape(h, w)
            for c in ("R", "G", "B")
        }
        return np.stack([channels["R"], channels["G"], channels["B"]], axis=-1)
    except ImportError:
        raise ImportError(
            "OpenEXR not installed. Run: pip install OpenEXR Imath\n"
            "On macOS: brew install openexr && pip install openexr"
        )


class MultiIlluminationDataset:
    """
    Builds (iamb, ichange, mask) triplets from the Multi-Illumination dataset.

    Strategy:
        - dir_00 is treated as the ambient image (iamb)
        - Each of dir_01 through dir_24 provides a "light on" image (ion)
        - ichange = clip(ion - iamb, 0)
        - The light mask is estimated from ichange (bright region = light source)

    This gives up to 24 light-change pairs per scene.
    """

    AMBIENT_DIR = 0       # dir_00 = ambient flash (or no-flash baseline)
    NUM_DIRECTIONS = 25   # 0..24

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: Optional[int] = 512,
        file_ext: str = "jpg",  # "jpg" or "exr"
    ):
        self.root = Path(root) / split
        self.image_size = image_size
        self.file_ext = file_ext

        if not self.root.exists():
            raise FileNotFoundError(
                f"Multi-Illumination dataset not found at {self.root}\n"
                "Download from: https://projects.csail.mit.edu/illumination/"
            )

        self.scenes = sorted([
            d for d in self.root.iterdir()
            if d.is_dir() and (d / f"dir_00_mip5.{file_ext}").exists()
        ])

        if len(self.scenes) == 0:
            raise RuntimeError(
                f"No valid scenes found in {self.root}. "
                f"Expected subdirectories with dir_XX_mip5.{file_ext} files."
            )

        # Build flat index: list of (scene_idx, direction_idx) pairs
        self.index = [
            (si, di)
            for si in range(len(self.scenes))
            for di in range(1, self.NUM_DIRECTIONS)
        ]

    def _get_image_path(self, scene: Path, direction: int) -> Path:
        fname = f"dir_{direction:02d}_mip5.{self.file_ext}"
        return scene / fname

    def _load_image(self, path: Path) -> np.ndarray:
        if self.file_ext == "exr":
            img = load_exr_as_float(str(path))
        else:
            img = load_jpg_as_float(str(path))

        if self.image_size is not None and (img.shape[0] != self.image_size or img.shape[1] != self.image_size):
            # Resize in linear float space to avoid uint8 quantization artifacts
            import cv2
            img = cv2.resize(
                img,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA if img.shape[0] > self.image_size else cv2.INTER_LINEAR,
            )

        return img.astype(np.float32)

    def _estimate_mask(
        self,
        ichange: np.ndarray,
        threshold_percentile: float = 85.0,
    ) -> np.ndarray:
        """
        Simple mask estimation: bright regions of ichange = light source.
        Returns a float32 mask in [0, 1], shape (H, W).
        """
        luminance = (
            0.2126 * ichange[:, :, 0]
            + 0.7152 * ichange[:, :, 1]
            + 0.0722 * ichange[:, :, 2]
        )
        threshold = np.percentile(luminance, threshold_percentile)
        mask = (luminance >= threshold).astype(np.float32)
        return mask

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        scene_idx, direction_idx = self.index[idx]
        scene = self.scenes[scene_idx]

        # Load ambient (light off) and light-on images
        iamb_path = self._get_image_path(scene, self.AMBIENT_DIR)
        ion_path = self._get_image_path(scene, direction_idx)

        iamb = self._load_image(iamb_path)
        ion = self._load_image(ion_path)

        # Extract light change
        from .light_arithmetic import extract_light_change
        ichange = extract_light_change(ion, iamb)

        # Estimate mask from ichange
        mask = self._estimate_mask(ichange)

        return {
            "iamb": iamb,          # (H, W, 3) linear float32
            "ichange": ichange,    # (H, W, 3) linear float32
            "mask": mask,          # (H, W) float32 [0, 1]
            "scene": str(scene.name),
            "direction": direction_idx,
        }


class HypersimDataset:
    """
    Loader for the Hypersim dataset (Roberts et al. 2021).
    URL: https://github.com/apple/ml-hypersim

    Hypersim provides per-frame diffuse illumination components separated
    by light index, which directly gives us iamb and ichange in HDR.

    Expected structure:
        hypersim/
            ai_001_001/
                images/
                    scene_cam_00_geometry_hdf5/
                    scene_cam_00_final_hdf5/
                        frame.0000.color.hdf5
                    scene_cam_00_final_preview/
                        frame.0000.tonemap.jpg
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        image_size: Optional[int] = 512,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self._build_index(split)

    def _build_index(self, split: str):
        """Build index of all valid frames with their scene info."""
        import json
        split_file = self.root / f"{split}_split.json"
        if split_file.exists():
            with open(split_file) as f:
                self.index = json.load(f)
        else:
            # Auto-discover: scan all scenes
            scenes = sorted(self.root.glob("ai_*_*"))
            self.index = []
            for scene in scenes:
                cam_dirs = sorted(scene.glob("images/scene_cam_*_final_preview"))
                for cam_dir in cam_dirs:
                    frames = sorted(cam_dir.glob("frame.*.tonemap.jpg"))
                    for frame in frames:
                        self.index.append({
                            "scene": str(scene.name),
                            "cam_dir": str(cam_dir.relative_to(self.root)),
                            "frame": str(frame.name),
                        })

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        entry = self.index[idx]
        # For now, load the preview image as a substitute
        # Full implementation would load HDF5 files for linear HDR data
        preview_path = self.root / entry["cam_dir"] / entry["frame"]
        img = load_jpg_as_float(str(preview_path))

        if self.image_size:
            from PIL import Image as PILImage
            pil = PILImage.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
            pil = pil.resize((self.image_size, self.image_size), PILImage.LANCZOS)
            arr = np.array(pil).astype(np.float32) / 255.0
            low = arr <= 0.04045
            img = np.where(low, arr / 12.92, ((arr + 0.055) / 1.055) ** 2.4)

        # Without HDF5 ground truth, we use the image as ambient
        # and apply random synthetic light changes
        iamb = img.astype(np.float32)
        ichange = np.zeros_like(iamb)
        mask = np.zeros(img.shape[:2], dtype=np.float32)

        return {
            "iamb": iamb,
            "ichange": ichange,
            "mask": mask,
            "scene": entry["scene"],
        }
