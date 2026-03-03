# FlashLab

给任意照片添加、去除或调整相机闪光灯效果。基于 SDXL 扩散模型微调，使用闪光灯开/关照片对训练。

受 [LightLab](https://arxiv.org/abs/2505.09608) (Magar et al., 2025) 启发，针对**相机闪光灯场景**做了简化——不需要框选灯具、不需要 SAM 2 分割，只需一张图即可控制闪光灯强度和颜色。

## 功能

- **添加闪光灯**：给无闪光的照片加上逼真的闪光灯效果
- **去除闪光灯**：从闪光灯照片中去除闪光灯痕迹
- **调节强度**：连续控制闪光灯亮度（gamma 参数）
- **调节颜色**：自定义闪光灯色温（暖光 / 冷光 / 自定义 RGB）

## 架构

```
Input Image ──── VAE Encode ──── image_latent (4ch) ─┐
                                                      ├─ SpatialEncoder (1×1 conv) → 4ch
Depth Map ───────────────────────────── depth  (1ch) ─┤
Flash Mask ──────────────────── mask × gamma   (1ch) ─┤    (flash = 全图 mask)
Flash Color ─────────────────── mask × ct_rgb  (3ch) ─┘
                                                      ↓
Noise (4ch) ──────────────────────── concat → UNet input (8ch)
                                                      ↓
                                                 SDXL UNet
                                                      ↓
Ambient α ──┐                                         │
Tone-map ───┴─ Fourier → MLP → 2 tokens ──→ cross-attention
                                                      ↓
                                              VAE Decode → Output
```

## 快速开始

### 1. 安装

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate xformers
pip install opencv-python Pillow numpy scipy tqdm tensorboard gradio
```

### 2. 准备数据

用手机 / 相机 + 三脚架拍摄同一场景的闪光灯开/关照片对：

```
data/flash_pairs/
    scene_001/
        flash.jpg       ← 开闪光灯
        noflash.jpg     ← 关闪光灯
    scene_002/
        flash.jpg
        noflash.jpg
    ...
```

### 3. 预处理（提取深度图）

```bash
python scripts/preprocess_flash_pairs.py \
    --data_root ./data/flash_pairs \
    --depth_cache_dir ./data/depth_cache
```

### 4. 训练

```bash
# 多卡
accelerate launch --config_file configs/accelerate_config.yaml \
    training/train_flash.py \
    --flash_data_root ./data/flash_pairs \
    --depth_cache_dir ./data/depth_cache \
    --output_dir ./checkpoints \
    --max_train_steps 30000

# 单卡 (24GB)
accelerate launch --num_processes 1 \
    training/train_flash.py \
    --flash_data_root ./data/flash_pairs \
    --depth_cache_dir ./data/depth_cache \
    --output_dir ./checkpoints \
    --per_device_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --enable_gradient_checkpointing
```

### 5. 推理

```bash
# 加闪光灯
python inference/infer_flash.py \
    --checkpoint checkpoints/checkpoint_030000.pt \
    --image photo.jpg \
    --gamma 1.0 \
    --output photo_flash.jpg

# 去闪光灯
python inference/infer_flash.py \
    --checkpoint checkpoints/checkpoint_030000.pt \
    --image flash_photo.jpg \
    --gamma -1.0 \
    --output photo_noflash.jpg

# 暖色闪光灯
python inference/infer_flash.py \
    --checkpoint checkpoints/checkpoint_030000.pt \
    --image photo.jpg \
    --gamma 0.8 \
    --color 255 180 100 \
    --output photo_warm_flash.jpg
```

### 6. Web Demo

```bash
python demo/app_flash.py \
    --checkpoint checkpoints/checkpoint_030000.pt \
    --port 7860
```

## Python API

```python
from PIL import Image
from models.pipeline_flash import FlashLabPipeline

pipeline = FlashLabPipeline.from_checkpoint("checkpoints/checkpoint_030000.pt")

result = pipeline(
    image=Image.open("photo.jpg"),
    gamma=0.8,                     # 闪光灯强度
    ct_rgb=[1.0, 0.85, 0.7],      # 暖色闪光
    alpha=0.0,                     # 环境光不变
)
result.save("photo_flash.jpg")
```

## 参数说明

| 参数 | 范围 | 说明 |
|------|------|------|
| `gamma` | [-1, 1] | 闪光灯强度。1.0 = 全开，0 = 不变，-1.0 = 去除闪光 |
| `alpha` | [-1, 1] | 环境光变化。0 = 不变，正值 = 增亮，负值 = 变暗 |
| `color` | [0-255] RGB | 闪光灯颜色。不填 = 白色中性闪光 |
| `tonemap` | together / separate | 色调映射策略。together = 保持亮度关系一致 |
| `steps` | 5-50 | DDIM 去噪步数，默认 15 |

## 项目结构

```
flashlab/
├── data/
│   ├── flash_dataset.py       # 闪光灯数据集（mask=全图）
│   ├── light_arithmetic.py    # 光线算术公式
│   └── tone_mapping.py        # 色调映射
├── models/
│   ├── pipeline_flash.py      # 推理 Pipeline（无需 SAM 2）
│   ├── spatial_encoder.py     # 空间条件编码（1×1 conv）
│   ├── global_conditioning.py # 全局条件（Fourier + MLP）
│   └── unet_lightlab.py       # 修改版 SDXL UNet（8通道输入）
├── training/
│   └── train_flash.py         # 训练脚本
├── inference/
│   └── infer_flash.py         # 命令行推理
├── demo/
│   └── app_flash.py           # Gradio Demo
├── preprocessing/
│   └── depth_extractor.py     # Depth Anything V2 深度估计
└── scripts/
    └── preprocess_flash_pairs.py  # 数据预处理
```

## 技术细节

### Light Arithmetic

从闪光灯开/关照片对中提取闪光灯贡献，再通过线性组合生成不同强度/颜色的训练样本：

```
flash_contribution = clip(flash_on - flash_off, 0)
output = alpha * ambient + gamma * flash_contribution * color_coeff
```

每对照片通过采样不同的 (gamma, alpha, color) 组合，膨胀为 ~30 个训练样本。

### 深度条件

使用 Depth Anything V2 提取单目深度图，帮助模型学习闪光灯的距离衰减特性（近亮远暗）。

### 训练策略

- 基座模型：SDXL (Stable Diffusion XL)
- UNet 输入通道：4 → 8（前 4 通道复制预训练权重，后 4 通道零初始化）
- 条件 Dropout：10% 概率丢弃深度 / 颜色条件，提升鲁棒性
- 损失：MSE + Min-SNR 加权
- 精度：bfloat16

## 硬件需求

| 用途 | 最低配置 | 推荐配置 |
|------|---------|---------|
| 推理 | 1× RTX 3090 (24GB) | 1× RTX 4090 |
| 训练 | 1× RTX 4090 (24GB) | 4-8× A100 (80GB) |

单卡训练预计时间：~1-2 周（30K 步）。多卡可线性加速。

## Acknowledgments

本项目基于 [LightLab](https://arxiv.org/abs/2505.09608) 的方法进行实现和改造。

```bibtex
@article{magar2025lightlab,
  title={LightLab: Controlling Light Sources in Images with Diffusion Models},
  author={Magar, Nadav and Hertz, Amir and Tabellion, Eric and Pritch, Yael and
          Rav-Acha, Alex and Shamir, Ariel and Hoshen, Yedid},
  journal={arXiv preprint arXiv:2505.09608},
  year={2025}
}
```
