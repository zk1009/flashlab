"""
Offline preprocessing for flash photograph pairs.

Only extracts depth maps (no SAM 2 masks needed since flash = full image).
Much simpler and faster than the original preprocess_real_pairs.py.

Usage:
    python scripts/preprocess_flash_pairs.py \\
        --data_root ./data/flash_pairs \\
        --depth_cache_dir ./data/depth_cache \\
        --image_size 1024

Expected structure:
    flash_pairs/
        scene_001/
            flash.jpg
            noflash.jpg
        scene_002/
            ...
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True,
                        help="Root directory of flash photo pairs")
    parser.add_argument("--depth_cache_dir", type=str, required=True,
                        help="Directory to save depth .npy files")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--depth_model_size", type=str, default="large",
                        choices=["small", "base", "large"])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    return parser.parse_args()


def find_scene_dirs(data_root: str):
    """Find all scene directories with valid flash/noflash pairs."""
    root = Path(data_root)
    scenes = []

    flash_stems = ["flash", "on", "lit"]
    noflash_stems = ["noflash", "off", "ambient", "amb", "no_flash"]
    exts = [".png", ".jpg", ".jpeg", ".tiff"]

    for scene_dir in sorted(root.iterdir()):
        if not scene_dir.is_dir():
            continue

        flash_path = None
        noflash_path = None
        for stem in flash_stems:
            for ext in exts:
                p = scene_dir / f"{stem}{ext}"
                if p.exists():
                    flash_path = p
                    break
            if flash_path:
                break

        for stem in noflash_stems:
            for ext in exts:
                p = scene_dir / f"{stem}{ext}"
                if p.exists():
                    noflash_path = p
                    break
            if noflash_path:
                break

        if flash_path and noflash_path:
            scenes.append({
                "name": scene_dir.name,
                "flash": flash_path,
                "noflash": noflash_path,
            })

    return scenes


def main():
    args = parse_args()

    import torch
    from preprocessing.depth_extractor import DepthExtractor

    print("Loading Depth Anything V2...")
    depth_extractor = DepthExtractor(
        model_size=args.depth_model_size,
        device=args.device,
        dtype=torch.float16,
    )

    scenes = find_scene_dirs(args.data_root)
    print(f"Found {len(scenes)} flash pairs in {args.data_root}")

    depth_cache = Path(args.depth_cache_dir)
    depth_cache.mkdir(parents=True, exist_ok=True)

    lat_size = args.image_size // 8

    for scene in tqdm(scenes, desc="Extracting depth"):
        name = scene["name"]
        depth_path = depth_cache / f"{name}.npy"

        if args.skip_existing and depth_path.exists():
            continue

        try:
            # Use the noflash image for depth (more natural ambient lighting)
            image_pil = Image.open(scene["noflash"]).convert("RGB")
            image_pil = image_pil.resize(
                (args.image_size, args.image_size), Image.LANCZOS
            )
            depth = depth_extractor(image_pil, lat_size, lat_size)
            depth_np = depth.squeeze().cpu().numpy()
            np.save(str(depth_path), depth_np)
        except Exception as e:
            print(f"Warning: depth extraction failed for {name}: {e}")

    print(f"\nDone! Depth maps saved to {depth_cache}")
    print(f"\nNext step — train:")
    print(
        f"  accelerate launch training/train_flash.py \\\n"
        f"    --flash_data_root {args.data_root} \\\n"
        f"    --depth_cache_dir {depth_cache} \\\n"
        f"    --output_dir ./checkpoints \\\n"
        f"    --max_train_steps 45000"
    )


if __name__ == "__main__":
    main()
