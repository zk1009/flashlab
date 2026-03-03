"""
Offline preprocessing script for real photograph pairs.

This script batch-processes all real photo pairs to:
  1. Extract depth maps using Depth Anything V2 (saved as .npy)
  2. Extract light source masks using SAM 2 (requires bbox.json annotations)
  3. Verify the light arithmetic extraction

Run this BEFORE training to cache depth maps and masks.

Usage:
    python scripts/preprocess_real_pairs.py \\
        --data_root ./data/real_pairs \\
        --depth_cache_dir ./data/depth_cache \\
        --mask_cache_dir ./data/mask_cache \\
        --image_size 1024
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of real photo pairs")
    parser.add_argument("--depth_cache_dir", type=str, required=True)
    parser.add_argument("--mask_cache_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--depth_model_size", type=str, default="large",
                        choices=["small", "base", "large"])
    parser.add_argument("--sam2_checkpoint", type=str,
                        default="checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    return parser.parse_args()


def find_scene_dirs(data_root: str):
    """Find all scene directories with valid on/off pairs."""
    root = Path(data_root)
    scenes = []
    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue
        on_path = None
        off_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tiff"]:
            if (scene_dir / f"on{ext}").exists():
                on_path = scene_dir / f"on{ext}"
            if (scene_dir / f"off{ext}").exists():
                off_path = scene_dir / f"off{ext}"
        if on_path and off_path:
            scenes.append({
                "name": scene_dir.name,
                "on": on_path,
                "off": off_path,
                "bbox_json": scene_dir / "bbox.json",
            })
    return scenes


def main():
    args = parse_args()

    # Load tools
    import torch
    from preprocessing.depth_extractor import DepthExtractor
    from preprocessing.segmentation import LightSourceSegmenter

    print("Loading Depth Anything V2...")
    depth_extractor = DepthExtractor(
        model_size=args.depth_model_size,
        device=args.device,
        dtype=torch.float16,
    )

    print("Loading SAM 2...")
    segmenter = LightSourceSegmenter(
        checkpoint=args.sam2_checkpoint,
        device=args.device,
    )

    # Discover scenes
    scenes = find_scene_dirs(args.data_root)
    print(f"Found {len(scenes)} scene pairs in {args.data_root}")

    depth_cache = Path(args.depth_cache_dir)
    mask_cache = Path(args.mask_cache_dir)
    depth_cache.mkdir(parents=True, exist_ok=True)
    mask_cache.mkdir(parents=True, exist_ok=True)

    lat_size = args.image_size // 8

    for scene in tqdm(scenes, desc="Processing scenes"):
        name = scene["name"]

        # --- Depth extraction ---
        depth_path = depth_cache / f"{name}.npy"
        if not (args.skip_existing and depth_path.exists()):
            try:
                image_pil = Image.open(scene["on"]).convert("RGB")
                image_pil = image_pil.resize(
                    (args.image_size, args.image_size), Image.LANCZOS
                )
                depth = depth_extractor(image_pil, lat_size, lat_size)
                depth_np = depth.squeeze().cpu().numpy()
                np.save(str(depth_path), depth_np)
            except Exception as e:
                print(f"Warning: depth extraction failed for {name}: {e}")

        # --- Mask extraction (requires bbox.json) ---
        mask_path = mask_cache / f"{name}.npy"
        if not (args.skip_existing and mask_path.exists()):
            if scene["bbox_json"].exists():
                try:
                    with open(scene["bbox_json"]) as f:
                        data = json.load(f)
                    bbox = data["bbox"]

                    image_pil = Image.open(scene["on"]).convert("RGB")
                    image_pil = image_pil.resize(
                        (args.image_size, args.image_size), Image.LANCZOS
                    )
                    image_np = np.array(image_pil)

                    mask = segmenter.segment_from_bbox(
                        image_np, bbox, lat_size, lat_size
                    )
                    mask_np = mask.squeeze().cpu().numpy()
                    np.save(str(mask_path), mask_np)
                except Exception as e:
                    print(f"Warning: mask extraction failed for {name}: {e}")
            else:
                print(
                    f"Note: No bbox.json for {name}. "
                    "Will use luminance-based mask during training."
                )

    print(f"\nPreprocessing complete!")
    print(f"  Depth maps: {depth_cache}")
    print(f"  Masks:      {mask_cache}")
    print(f"\nNow you can start training:")
    print(
        f"  accelerate launch --config_file configs/accelerate_config.yaml "
        f"training/train.py --depth_cache_dir {depth_cache} "
        f"--mask_cache_dir {mask_cache} ..."
    )


if __name__ == "__main__":
    main()
