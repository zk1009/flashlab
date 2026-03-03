# LightLab: Open-Source Reproduction

Open-source reproduction of **[LightLab: Controlling Light Sources in Images with Diffusion Models](https://arxiv.org/abs/2505.09608)** (Magar et al., 2025).

## Overview

LightLab fine-tunes an SDXL-based diffusion model to enable **parametric control** over visible light sources in photographs:
- **Intensity control**: Turn lights on/off or adjust brightness
- **Color control**: Change light color temperature or RGB
- **Ambient control**: Adjust overall scene illumination
- **Sequential editing**: Make multiple edits to different light sources

## Architecture

```
Input Image ─┐
              ├─ VAE Encode → image_latent (B, 4, H/8, W/8) ─┐
Depth Map ───┤                                                 ├─ SpatialConditionEncoder (1×1 conv, zero-init)
Light Mask ──┤                                                 │   9 channels → 4 channels
Color Mask ──┘                                                 │
                                                               ↓
Noise ─────────────────────────────────────── concat → (B, 8, H/8, W/8)
                                                               ↓
                                                      UNet (conv_in: 8→320)
                                                               ↓
Ambient α ──┐                                                  │
Tone-map ───┴─ Fourier Features → MLP → 2 tokens ──→ cross-attention
                                                               ↓
                                                      Denoised latent
                                                               ↓
                                                      VAE Decode → Output Image
```

## Project Structure

```
lightlab/
├── data/
│   ├── light_arithmetic.py     # Core: ichange = clip(ion - ioff, 0), irelit formula
│   ├── tone_mapping.py         # "separate" and "together" tone mapping
│   ├── dataset.py              # LightLabDataset (real + synthetic, with inflation)
│   └── multi_illumination.py   # Multi-Illumination Dataset loader (synthetic substitute)
├── models/
│   ├── spatial_encoder.py      # 1×1 Conv2d(9→4), zero-initialized
│   ├── global_conditioning.py  # FourierFeatures + MLP → 2 cross-attention tokens
│   ├── unet_lightlab.py        # Modified SDXL UNet: conv_in 4→8 channels
│   └── pipeline_lightlab.py    # Full inference pipeline
├── preprocessing/
│   ├── depth_extractor.py      # Depth Anything V2 wrapper
│   └── segmentation.py         # SAM 2 wrapper for light source segmentation
├── training/
│   └── train.py                # Multi-GPU training with HuggingFace Accelerate
├── inference/
│   └── infer.py                # CLI inference script
├── demo/
│   └── app.py                  # Gradio web demo
├── scripts/
│   └── preprocess_real_pairs.py # Offline depth/mask extraction
└── configs/
    └── accelerate_config.yaml   # Multi-GPU training config
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For SAM 2:
```bash
pip install git+https://github.com/facebookresearch/sam2.git
```

### 2. Download Model Weights

**Base SDXL model** (downloaded automatically by HuggingFace):
```bash
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0
```

**SAM 2 checkpoint**:
```bash
mkdir -p checkpoints
wget -P checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
mv checkpoints/sam2.1_hiera_large.pt checkpoints/sam2_hiera_large.pt
```

**Depth Anything V2** (downloaded automatically via HuggingFace Transformers).

### 3. Prepare Training Data

#### Option A: Use Multi-Illumination Dataset (recommended)

```bash
# Download from MIT CSAIL: https://projects.csail.mit.edu/illumination/
# Extract to ./data/multi_illumination/
```

#### Option B: Capture your own real pairs

Organize as:
```
data/real_pairs/
    scene_001/
        on.jpg      # image with light source ON
        off.jpg     # image with light source OFF (ambient)
        bbox.json   # {"bbox": [x1, y1, x2, y2]} — optional, for SAM 2 masks
    scene_002/
        ...
```

### 4. Preprocess (cache depth maps and masks)

```bash
python scripts/preprocess_real_pairs.py \
    --data_root ./data/real_pairs \
    --depth_cache_dir ./data/depth_cache \
    --mask_cache_dir ./data/mask_cache \
    --image_size 1024
```

## Training

### Multi-GPU (8 GPUs, effective batch = 128)

```bash
accelerate launch --config_file configs/accelerate_config.yaml \
    training/train.py \
    --output_dir ./checkpoints \
    --real_data_root ./data/real_pairs \
    --synthetic_data_root ./data/multi_illumination \
    --depth_cache_dir ./data/depth_cache \
    --mask_cache_dir ./data/mask_cache \
    --per_device_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --max_train_steps 45000 \
    --learning_rate 1e-5 \
    --mixed_precision bf16
```

### Single GPU (smaller scale)

```bash
accelerate launch --num_processes 1 \
    training/train.py \
    --output_dir ./checkpoints \
    --synthetic_data_root ./data/multi_illumination \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --max_train_steps 45000 \
    --learning_rate 1e-5 \
    --mixed_precision bf16 \
    --enable_gradient_checkpointing
```

> **Memory tip**: Use `--per_device_batch_size 1` and `--enable_gradient_checkpointing` on a 24GB GPU. Expect ~20GB VRAM usage.

## Inference

### Command Line

```bash
python inference/infer.py \
    --checkpoint checkpoints/lightlab_step045000.pt \
    --image examples/room.jpg \
    --bbox 220 180 340 290 \
    --gamma 1.0 \
    --output room_lit.jpg

# Turn off a light:
python inference/infer.py --image room.jpg --bbox 220 180 340 290 --gamma -1.0 ...

# Change color to warm light:
python inference/infer.py --image room.jpg --bbox 220 180 340 290 --gamma 1.0 --color 255 150 50 ...
```

### Gradio Web Demo

```bash
python demo/app.py --checkpoint checkpoints/lightlab_step045000.pt --port 7860
```

### Python API

```python
from PIL import Image
from models.pipeline_lightlab import LightLabPipeline

pipeline = LightLabPipeline.from_checkpoint("checkpoints/lightlab_step045000.pt")

result = pipeline(
    image=Image.open("room.jpg"),
    bbox=[220, 180, 340, 290],  # bounding box around the lamp
    gamma=1.0,                   # turn on fully
    alpha=0.0,                   # no ambient change
    ct_rgb=[1.0, 0.6, 0.2],     # warm orange color
    tonemap="together",
    num_inference_steps=15,
)
result.save("room_lit.jpg")
```

## Key Implementation Notes

### Light Arithmetic (Section 3.2)
```python
ichange = clip(ion - ioff, 0)                            # extract light contribution
c = ct ⊙ co^{-1}                                         # color change coefficient
irelit(α, γ, ct) = α · iamb + γ · ichange · c           # Eq. 1
```

### UNet Channel Expansion
- Original SDXL `conv_in`: `Conv2d(4, 320, 3×3)`
- Modified: `Conv2d(8, 320, 3×3)`
- Pretrained weights copied to channels 0–3 (noise), zeros at channels 4–7 (spatial conditions)
- Zero-init for spatial conditions ensures training stability (same principle as ControlNet)

### Condition Dropout (Section 3.4)
- Depth and color conditions dropped with 10% probability during training
- Allows the model to work without depth/color at inference time

## Deviations from Paper

| Paper | This Reproduction |
|-------|-------------------|
| Proprietary SDXL variant (trained on PaLI subset) | Public `stabilityai/stable-diffusion-xl-base-1.0` |
| 600 real photo pairs (proprietary) | User-provided pairs + Multi-Illumination dataset |
| 20 Blender scenes (~600K synthetic) | Multi-Illumination dataset (~24K pairs × 36 inflation) |
| 64 TPU v4s, 12 hours | 8 GPUs (estimated 2–5 days) |

## Citation

```bibtex
@article{magar2025lightlab,
  title={LightLab: Controlling Light Sources in Images with Diffusion Models},
  author={Magar, Nadav and Hertz, Amir and Tabellion, Eric and Pritch, Yael and
          Rav-Acha, Alex and Shamir, Ariel and Hoshen, Yedid},
  journal={arXiv preprint arXiv:2505.09608},
  year={2025}
}
```
