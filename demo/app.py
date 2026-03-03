"""
LightLab Gradio Demo.

Interactive web demo for controlling light sources in images.

Features:
  - Upload an image or use example images
  - Draw a bounding box around the target light source
  - Control light intensity (turn on/off, adjust brightness)
  - Control light color (color temperature or custom RGB)
  - Control ambient illumination
  - Sequential light editing

Usage:
    python demo/app.py --checkpoint checkpoints/lightlab_step045000.pt
    python demo/app.py --checkpoint checkpoints/lightlab_step045000.pt --share
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args():
    parser = argparse.ArgumentParser(description="LightLab Gradio Demo")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to LightLab checkpoint (.pt file)")
    parser.add_argument("--pretrained_model_id", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--sam2_checkpoint", type=str,
                        default="checkpoints/sam2_hiera_large.pt")
    parser.add_argument("--depth_model_size", type=str, default="large",
                        choices=["small", "base", "large"])
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--share", action="store_true",
                        help="Create a public shareable Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--image_size", type=int, default=1024)
    return parser.parse_args()


# Global pipeline instance (loaded once at startup)
_pipeline = None
_image_size = 1024


def load_pipeline(args):
    """Load the LightLab pipeline (called once at startup)."""
    global _pipeline, _image_size
    _image_size = args.image_size

    from models.pipeline_lightlab import LightLabPipeline

    print(f"Loading LightLab pipeline from {args.checkpoint}...")
    _pipeline = LightLabPipeline.from_checkpoint(
        checkpoint_path=args.checkpoint,
        pretrained_model_id=args.pretrained_model_id,
        depth_model_size=args.depth_model_size,
        sam2_checkpoint=args.sam2_checkpoint,
        device=args.device,
        dtype=torch.float16 if args.device == "cuda" else torch.float32,
    )
    print("Pipeline loaded successfully!")


def run_inference(
    input_image: np.ndarray,
    bbox_json: str,
    gamma: float,
    alpha: float,
    use_custom_color: bool,
    ct_r: float,
    ct_g: float,
    ct_b: float,
    tonemap: str,
    num_steps: int,
):
    """
    Main inference function called by Gradio.

    Args:
        input_image: Input image as numpy array (H, W, 3) uint8.
        bbox_json:   JSON string "[x1, y1, x2, y2]" defining bounding box.
        gamma:       Target light intensity ∈ [-1, 1].
        alpha:       Ambient light change ∈ [-1, 1].
        use_custom_color: Whether to apply custom color.
        ct_r/g/b:    Target color channels ∈ [0, 255].
        tonemap:     "separate" or "together".
        num_steps:   Number of DDIM steps.
    """
    if _pipeline is None:
        return None, "Error: Pipeline not loaded."

    if input_image is None:
        return None, "Please upload an image."

    if not bbox_json or bbox_json.strip() == "":
        return None, "Please specify a bounding box around the target light source."

    try:
        bbox = json.loads(bbox_json)
        if len(bbox) != 4:
            return None, "Bounding box must have 4 values: [x1, y1, x2, y2]."
    except json.JSONDecodeError:
        return None, f"Invalid bounding box format: {bbox_json}. Expected [x1, y1, x2, y2]."

    # Parse color
    ct_rgb = None
    if use_custom_color:
        ct_rgb = [ct_r / 255.0, ct_g / 255.0, ct_b / 255.0]
        # Normalize so max channel = 1
        max_c = max(ct_rgb)
        if max_c > 1e-6:
            ct_rgb = [c / max_c for c in ct_rgb]

    try:
        pil_image = Image.fromarray(input_image)
        result = _pipeline(
            image=pil_image,
            bbox=bbox,
            gamma=gamma,
            ct_rgb=ct_rgb,
            alpha=alpha,
            tonemap=tonemap,
            num_inference_steps=num_steps,
            image_size=_image_size,
        )
        return np.array(result), f"Done! gamma={gamma:.2f}, alpha={alpha:.2f}"
    except Exception as e:
        import traceback
        error_msg = f"Error during inference:\n{traceback.format_exc()}"
        print(error_msg)
        return None, error_msg


def apply_preset(preset_name: str):
    """Return preset parameters for quick demos."""
    presets = {
        "Turn On Light": (1.0, 0.0, False, 255, 200, 100, "together"),
        "Turn Off Light": (-1.0, 0.0, False, 255, 255, 255, "together"),
        "Warm Light": (1.0, 0.0, True, 255, 150, 50, "together"),
        "Cool Blue Light": (1.0, 0.0, True, 100, 150, 255, "together"),
        "Bright Ambient": (0.0, 0.8, False, 255, 255, 255, "separate"),
        "Dark Ambient": (0.0, -0.8, False, 255, 255, 255, "separate"),
    }
    return presets.get(preset_name, (1.0, 0.0, False, 255, 255, 255, "together"))


def build_demo():
    """Build and return the Gradio interface."""
    import gradio as gr

    with gr.Blocks(
        title="LightLab: Light Source Control in Images",
        theme=gr.themes.Soft(),
        css="""
        .title { text-align: center; font-size: 2em; margin-bottom: 0.5em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 1em; }
        """,
    ) as demo:

        gr.Markdown(
            """
            # LightLab: Controlling Light Sources in Images
            ### [arXiv:2505.09608](https://arxiv.org/abs/2505.09608) | Open-source reproduction
            Fine-grained, parametric control over light sources in photographs using diffusion models.
            **Instructions:**
            1. Upload an image with a visible light source
            2. Enter the bounding box of the target light source as `[x1, y1, x2, y2]`
            3. Adjust light intensity, color, and ambient parameters
            4. Click **Edit Lighting**
            """
        )

        with gr.Row():
            # --- Left Column: Inputs ---
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Input Image",
                    type="numpy",
                    height=400,
                )

                with gr.Group():
                    gr.Markdown("**Light Source Bounding Box** [x1, y1, x2, y2]")
                    bbox_input = gr.Textbox(
                        label="Bounding Box",
                        placeholder='e.g., [220, 180, 340, 290]',
                        info="Pixel coordinates in the original image",
                    )

                with gr.Accordion("Quick Presets", open=False):
                    preset = gr.Radio(
                        choices=[
                            "Turn On Light",
                            "Turn Off Light",
                            "Warm Light",
                            "Cool Blue Light",
                            "Bright Ambient",
                            "Dark Ambient",
                        ],
                        label="Preset",
                        value=None,
                    )

                with gr.Accordion("Light Intensity & Ambient", open=True):
                    gamma_slider = gr.Slider(
                        minimum=-1.0, maximum=1.5, value=1.0, step=0.05,
                        label="Target Light Intensity (γ)",
                        info="-1=fully off, 0=no change, +1=fully on",
                    )
                    alpha_slider = gr.Slider(
                        minimum=-1.0, maximum=1.0, value=0.0, step=0.05,
                        label="Ambient Light Change (α)",
                        info="-1=darker, 0=no change, +1=brighter",
                    )

                with gr.Accordion("Light Color (Optional)", open=False):
                    use_color = gr.Checkbox(label="Apply Custom Color", value=False)
                    with gr.Row():
                        ct_r = gr.Slider(0, 255, value=255, step=1, label="R")
                        ct_g = gr.Slider(0, 255, value=200, step=1, label="G")
                        ct_b = gr.Slider(0, 255, value=100, step=1, label="B")
                    gr.Markdown("*Common temperatures: Warm=255,150,50 | Neutral=255,255,200 | Cool=100,150,255*")

                with gr.Accordion("Advanced Settings", open=False):
                    tonemap = gr.Radio(
                        choices=["together", "separate"],
                        value="together",
                        label="Tone Mapping Strategy",
                        info='"together" = consistent brightness sequence; "separate" = auto-expose each image',
                    )
                    num_steps = gr.Slider(
                        5, 50, value=15, step=1,
                        label="Denoising Steps",
                        info="15 steps as per paper (faster) or up to 50 for better quality",
                    )

                run_btn = gr.Button("Edit Lighting", variant="primary", size="lg")

            # --- Right Column: Outputs ---
            with gr.Column(scale=1):
                output_image = gr.Image(
                    label="Result",
                    type="numpy",
                    height=400,
                )
                status_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    max_lines=3,
                )

                with gr.Accordion("Example Results", open=True):
                    gr.Examples(
                        examples=[
                            [
                                "examples/lamp_room.jpg",
                                "[220, 180, 340, 290]",
                                1.0, 0.0, False, 255, 200, 100, "together", 15,
                            ],
                            [
                                "examples/office.jpg",
                                "[100, 50, 250, 150]",
                                -1.0, 0.0, False, 255, 255, 255, "together", 15,
                            ],
                        ],
                        inputs=[
                            input_image, bbox_input,
                            gamma_slider, alpha_slider,
                            use_color, ct_r, ct_g, ct_b,
                            tonemap, num_steps,
                        ],
                        outputs=[output_image, status_text],
                        fn=run_inference,
                        cache_examples=False,
                    )

        # Preset handler
        def apply_preset_fn(preset_name):
            if preset_name is None:
                return 1.0, 0.0, False, 255, 200, 100, "together"
            vals = apply_preset(preset_name)
            return vals

        preset.change(
            fn=apply_preset_fn,
            inputs=[preset],
            outputs=[gamma_slider, alpha_slider, use_color, ct_r, ct_g, ct_b, tonemap],
        )

        # Main inference
        run_btn.click(
            fn=run_inference,
            inputs=[
                input_image, bbox_input,
                gamma_slider, alpha_slider,
                use_color, ct_r, ct_g, ct_b,
                tonemap, num_steps,
            ],
            outputs=[output_image, status_text],
            api_name="edit_lighting",
        )

        gr.Markdown(
            """
            ---
            **Paper:** [LightLab: Controlling Light Sources in Images with Diffusion Models](https://arxiv.org/abs/2505.09608)
            Magar et al., 2025 | Tel Aviv University & Google
            """
        )

    return demo


def main():
    args = parse_args()

    # Load pipeline
    load_pipeline(args)

    # Create example images directory
    os.makedirs("examples", exist_ok=True)

    # Build and launch demo
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
