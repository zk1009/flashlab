"""
FlashLab Command-Line Inference.

Add, remove, or adjust camera flash lighting on any photograph.
No bounding box needed — flash affects the entire image.

Usage:
    # Add flash:
    python inference/infer_flash.py \\
        --checkpoint checkpoints/flashlab_step045000.pt \\
        --image photo.jpg \\
        --gamma 1.0 \\
        --output photo_flash.jpg

    # Remove flash (from a flash photo):
    python inference/infer_flash.py \\
        --image flash_photo.jpg \\
        --gamma -1.0 \\
        --checkpoint ...

    # Warm flash:
    python inference/infer_flash.py \\
        --image photo.jpg \\
        --gamma 0.8 \\
        --color 255 180 100 \\
        --checkpoint ...

    # Half-strength cool flash:
    python inference/infer_flash.py \\
        --image photo.jpg \\
        --gamma 0.5 \\
        --color 180 200 255 \\
        --checkpoint ...
"""

import argparse
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="FlashLab: Camera Flash Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to FlashLab/LightLab checkpoint (.pt)")
    parser.add_argument("--image", type=str, required=True,
                        help="Input image path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: input_flash.jpg)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Flash intensity [-1, 1]. 1.0=full flash, -1.0=remove flash (default: 1.0)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Ambient light change [-1, 1]. 0=no change (default: 0.0)")
    parser.add_argument("--color", type=int, nargs=3, default=None,
                        metavar=("R", "G", "B"),
                        help="Flash color [0-255]. Default: neutral white")
    parser.add_argument("--tonemap", type=str, default="together",
                        choices=["together", "separate"])
    parser.add_argument("--steps", type=int, default=15,
                        help="DDIM denoising steps (default: 15)")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--pretrained_model_id", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--depth_model_size", type=str, default="large",
                        choices=["small", "base", "large"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    from PIL import Image
    from models.pipeline_flash import FlashLabPipeline

    print(f"Loading FlashLab pipeline from {args.checkpoint}...")
    pipeline = FlashLabPipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        pretrained_model_id=args.pretrained_model_id,
        depth_model_size=args.depth_model_size,
        device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")

    # Parse color
    ct_rgb = None
    if args.color is not None:
        ct_rgb = [c / 255.0 for c in args.color]
        max_c = max(ct_rgb)
        if max_c > 1e-6:
            ct_rgb = [c / max_c for c in ct_rgb]

    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    print(f"Applying flash: gamma={args.gamma}, alpha={args.alpha}")
    result = pipeline(
        image=image,
        gamma=args.gamma,
        ct_rgb=ct_rgb,
        alpha=args.alpha,
        tonemap=args.tonemap,
        num_inference_steps=args.steps,
        image_size=args.image_size,
        generator=generator,
    )

    if args.output is None:
        stem = Path(args.image).stem
        output_path = f"{stem}_flash.jpg"
    else:
        output_path = args.output

    result.save(output_path)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
