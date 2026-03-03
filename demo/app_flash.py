"""
FlashLab Gradio Demo.

Simple web demo for camera flash control — no bounding box needed.
Upload a photo, adjust flash intensity and color, get the result.

Usage:
    python demo/app_flash.py --checkpoint checkpoints/flashlab_step045000.pt
    python demo/app_flash.py --checkpoint checkpoints/flashlab_step045000.pt --share
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


_pipeline = None
_image_size = 1024


def parse_args():
    parser = argparse.ArgumentParser(description="FlashLab Gradio Demo")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--pretrained_model_id", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--depth_model_size", type=str, default="large",
                        choices=["small", "base", "large"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--image_size", type=int, default=1024)
    return parser.parse_args()


def load_pipeline(args):
    global _pipeline, _image_size
    _image_size = args.image_size

    from models.pipeline_flash import FlashLabPipeline

    print(f"Loading FlashLab pipeline from {args.checkpoint}...")
    _pipeline = FlashLabPipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        pretrained_model_id=args.pretrained_model_id,
        depth_model_size=args.depth_model_size,
        device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    print("Pipeline loaded!")


def run_inference(
    input_image: np.ndarray,
    gamma: float,
    alpha: float,
    use_custom_color: bool,
    ct_r: float,
    ct_g: float,
    ct_b: float,
    tonemap: str,
    num_steps: int,
):
    if _pipeline is None:
        return None, "Pipeline not loaded."
    if input_image is None:
        return None, "Please upload an image."

    ct_rgb = None
    if use_custom_color:
        ct_rgb = [ct_r / 255.0, ct_g / 255.0, ct_b / 255.0]
        max_c = max(ct_rgb)
        if max_c > 1e-6:
            ct_rgb = [c / max_c for c in ct_rgb]

    try:
        pil_image = Image.fromarray(input_image)
        result = _pipeline(
            image=pil_image,
            gamma=gamma,
            ct_rgb=ct_rgb,
            alpha=alpha,
            tonemap=tonemap,
            num_inference_steps=num_steps,
            image_size=_image_size,
        )
        color_str = f", color=({ct_r:.0f},{ct_g:.0f},{ct_b:.0f})" if use_custom_color else ""
        return np.array(result), f"Done! gamma={gamma:.2f}, alpha={alpha:.2f}{color_str}"
    except Exception as e:
        import traceback
        return None, f"Error:\n{traceback.format_exc()}"


def build_demo():
    import gradio as gr

    with gr.Blocks(
        title="FlashLab: Camera Flash Control",
        theme=gr.themes.Soft(),
    ) as demo:

        gr.Markdown(
            """
            # FlashLab: Camera Flash Control
            Add, remove, or adjust camera flash on any photograph.
            Just upload an image and adjust the sliders — no bounding box needed.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(label="Input Image", type="numpy", height=400)

                with gr.Group():
                    gamma_slider = gr.Slider(
                        minimum=-1.0, maximum=1.5, value=1.0, step=0.05,
                        label="Flash Intensity (gamma)",
                        info="-1=remove flash, 0=no change, 1=full flash",
                    )
                    alpha_slider = gr.Slider(
                        minimum=-1.0, maximum=1.0, value=0.0, step=0.05,
                        label="Ambient Light Change (alpha)",
                        info="-1=darker ambient, 0=no change, 1=brighter ambient",
                    )

                with gr.Accordion("Flash Color", open=False):
                    use_color = gr.Checkbox(label="Custom Flash Color", value=False)
                    with gr.Row():
                        ct_r = gr.Slider(0, 255, value=255, step=1, label="R")
                        ct_g = gr.Slider(0, 255, value=255, step=1, label="G")
                        ct_b = gr.Slider(0, 255, value=255, step=1, label="B")
                    gr.Markdown("*Warm=255,180,100 | Neutral=255,255,255 | Cool=180,200,255*")

                with gr.Accordion("Quick Presets", open=False):
                    preset = gr.Radio(
                        choices=[
                            "Full Flash",
                            "Soft Flash",
                            "Remove Flash",
                            "Warm Flash",
                            "Cool Flash",
                        ],
                        label="Preset",
                        value=None,
                    )

                with gr.Accordion("Advanced", open=False):
                    tonemap = gr.Radio(
                        choices=["together", "separate"],
                        value="together",
                        label="Tone Mapping",
                    )
                    num_steps = gr.Slider(5, 50, value=15, step=1, label="Denoising Steps")

                run_btn = gr.Button("Apply Flash", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Result", type="numpy", height=400)
                status_text = gr.Textbox(label="Status", interactive=False, max_lines=3)

        # Presets
        presets = {
            "Full Flash":    (1.0, 0.0, False, 255, 255, 255, "together"),
            "Soft Flash":    (0.5, 0.0, False, 255, 255, 255, "together"),
            "Remove Flash":  (-1.0, 0.0, False, 255, 255, 255, "together"),
            "Warm Flash":    (1.0, 0.0, True, 255, 180, 100, "together"),
            "Cool Flash":    (1.0, 0.0, True, 180, 200, 255, "together"),
        }

        def apply_preset(name):
            if name is None:
                return 1.0, 0.0, False, 255, 255, 255, "together"
            return presets.get(name, (1.0, 0.0, False, 255, 255, 255, "together"))

        preset.change(
            fn=apply_preset,
            inputs=[preset],
            outputs=[gamma_slider, alpha_slider, use_color, ct_r, ct_g, ct_b, tonemap],
        )

        run_btn.click(
            fn=run_inference,
            inputs=[
                input_image,
                gamma_slider, alpha_slider,
                use_color, ct_r, ct_g, ct_b,
                tonemap, num_steps,
            ],
            outputs=[output_image, status_text],
        )

    return demo


def main():
    args = parse_args()
    load_pipeline(args)
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
