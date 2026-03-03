"""
LightLab Command-Line Inference.

Usage:
    python inference/infer.py \\
        --checkpoint checkpoints/lightlab_step045000.pt \\
        --image examples/room.jpg \\
        --bbox 220 180 340 290 \\
        --gamma 1.0 \\
        --output room_lit.jpg

    # With color control:
    python inference/infer.py \\
        --image room.jpg \\
        --bbox 220 180 340 290 \\
        --gamma 1.0 \\
        --color 255 150 50 \\   # warm orange light
        --checkpoint ...

    # Sequential editing (multiple light sources):
    python inference/infer.py \\
        --image room.jpg \\
        --bbox 100 50 200 150 \\   # first light
        --gamma -1.0 \\            # turn off
        --checkpoint ...
    # Then run again on the output to edit the second light
"""

import argparse
import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(
        description="LightLab: Inference CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LightLab checkpoint (.pt)")
    parser.add_argument("--image", type=str, required=True,
                        help="Input image path")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: input_relit.jpg)")
    parser.add_argument("--bbox", type=int, nargs=4, required=True,
                        metavar=("X1", "Y1", "X2", "Y2"),
                        help="Bounding box of the target light source")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Target light intensity ∈ [-1, 1] (default: 1.0)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Ambient light change ∈ [-1, 1] (default: 0.0)")
    parser.add_argument("--color", type=int, nargs=3, default=None,
                        metavar=("R", "G", "B"),
                        help="Target light color in [0, 255] (default: no change)")
    parser.add_argument("--tonemap", type=str, default="together",
                        choices=["together", "separate"])
    parser.add_argument("--steps", type=int, default=15,
                        help="Number of DDIM denoising steps (default: 15)")
    parser.add_argument("--image_size", type=int, default=1024,
                        help="Processing resolution (default: 1024)")
    parser.add_argument("--pretrained_model_id", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--sam2_checkpoint", type=str,
                        default="checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--depth_model_size", type=str, default="large",
                        choices=["small", "base", "large"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    from PIL import Image
    from models.pipeline_lightlab import LightLabPipeline

    # Load pipeline
    print(f"Loading pipeline from {args.checkpoint}...")
    pipeline = LightLabPipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        pretrained_model_id=args.pretrained_model_id,
        depth_model_size=args.depth_model_size,
        sam2_checkpoint=args.sam2_checkpoint,
        device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )

    # Load input image
    print(f"Loading image from {args.image}...")
    image = Image.open(args.image).convert("RGB")

    # Parse color
    ct_rgb = None
    if args.color is not None:
        ct_rgb = [c / 255.0 for c in args.color]
        max_c = max(ct_rgb)
        if max_c > 1e-6:
            ct_rgb = [c / max_c for c in ct_rgb]

    # Set random seed for reproducibility
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    print(f"Editing lighting: γ={args.gamma}, α={args.alpha}, bbox={args.bbox}")
    result = pipeline(
        image=image,
        bbox=args.bbox,
        gamma=args.gamma,
        ct_rgb=ct_rgb,
        alpha=args.alpha,
        tonemap=args.tonemap,
        num_inference_steps=args.steps,
        image_size=args.image_size,
        generator=generator,
    )

    # Save output
    if args.output is None:
        stem = Path(args.image).stem
        output_path = f"{stem}_relit.jpg"
    else:
        output_path = args.output

    result.save(output_path)
    print(f"Saved result to {output_path}")


if __name__ == "__main__":
    main()
