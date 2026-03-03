# LightLab 复现 — 完整执行指南

## 环境要求

- Python 3.10+
- CUDA 12.1+（多张 GPU，推荐 8× A100/H100 或 4× RTX 4090）
- 磁盘空间：~200GB（模型权重 + 数据集）
- 内存：≥64GB RAM

---

## 第一步：克隆代码 & 安装依赖

```bash
# 把 lightlab/ 目录复制到新机器，然后进入
cd lightlab

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装主要依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install diffusers transformers accelerate xformers safetensors
pip install opencv-python Pillow numpy scipy tqdm tensorboard
pip install gradio omegaconf einops

# 安装 SAM 2
pip install git+https://github.com/facebookresearch/sam2.git

# （可选）安装 OpenEXR（读取 EXR 格式）
pip install openexr
# 如果失败，用 conda：conda install -c conda-forge openexr
```

---

## 第二步：下载预训练权重

### 2.1 SDXL 基础模型（自动下载，约 6.5GB）

```bash
# 运行时会自动从 HuggingFace 下载，无需手动操作
# 如果网络受限，手动下载：
huggingface-cli download stabilityai/stable-diffusion-xl-base-1.0 \
    --local-dir ./weights/sdxl-base
```

### 2.2 SAM 2 权重（约 900MB）

```bash
mkdir -p checkpoints
wget -O checkpoints/sam2_hiera_large.pt \
    https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

### 2.3 Depth Anything V2（自动下载，约 1.3GB）

```bash
# 运行时会自动从 HuggingFace 下载
# 手动下载（可选）：
huggingface-cli download depth-anything/Depth-Anything-V2-Large-hf \
    --local-dir ./weights/depth-anything-v2-large
```

---

## 第三步：准备训练数据

### 方案 A：Multi-Illumination 数据集（推荐，替代 Blender 合成数据）

```bash
# 1. 下载数据集（约 30GB，需申请）
# 申请页面：https://projects.csail.mit.edu/illumination/
# 下载后解压到 ./data/multi_illumination/

# 预期目录结构：
# data/multi_illumination/
#     train/
#         scene_0001/
#             dir_00_mip5.jpg  (ambient / no-flash)
#             dir_01_mip5.jpg
#             ...
#             dir_24_mip5.jpg
#         scene_0002/
#             ...
#     test/
#         ...
```

### 方案 B：自己拍摄真实照片对（可直接开始）

```bash
# 用手机 + 三脚架拍摄：同一场景，仅开/关一盏灯的区别
# 每对照片放入一个目录：

mkdir -p data/real_pairs/scene_001
# data/real_pairs/
#     scene_001/
#         on.jpg      ← 灯开着的照片
#         off.jpg     ← 灯关着的照片（仅环境光）
#         bbox.json   ← 灯具的边界框（可选，用于 SAM 2 分割）
#     scene_002/
#         ...

# bbox.json 格式：
# {"bbox": [x1, y1, x2, y2]}
# 坐标为图片像素坐标，框住灯具区域
```

---

## 第四步：离线预处理（提取深度图 & 分割 mask）

> **说明**：此步骤将所有深度图和 mask 提前缓存为 `.npy` 文件，避免训练时重复计算，节省 GPU 时间。

```bash
python3 scripts/preprocess_real_pairs.py \
    --data_root ./data/real_pairs \
    --depth_cache_dir ./data/depth_cache \
    --mask_cache_dir ./data/mask_cache \
    --image_size 1024 \
    --depth_model_size large \
    --sam2_checkpoint checkpoints/sam2_hiera_large.pt \
    --device cuda

# 预计用时：每张图约 1-2 秒，600 张约 20 分钟
```

---

## 第五步：配置多 GPU 训练

### 5.1 修改 accelerate 配置

编辑 `configs/accelerate_config.yaml`，根据你的 GPU 数量修改 `num_processes`：

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
num_processes: 8      # ← 改成你的 GPU 数量
mixed_precision: bf16
gradient_accumulation_steps: 8
main_process_port: 29500
machine_rank: 0
num_machines: 1
```

### 5.2 Batch Size 计算

论文 effective batch = 128，通过以下方式达到：

| GPU 数 | per_device_batch_size | gradient_accumulation_steps | Effective Batch |
|--------|----------------------|------------------------------|-----------------|
| 8 GPU  | 2                    | 8                            | 128 ✓           |
| 4 GPU  | 2                    | 16                           | 128 ✓           |
| 2 GPU  | 2                    | 32                           | 128 ✓           |
| 1 GPU  | 1                    | 128                          | 128 ✓（慢）      |

---

## 第六步：开始训练

### 多 GPU 训练（推荐）

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
    --mixed_precision bf16 \
    --checkpointing_steps 5000 \
    --real_weight 0.05 \
    --dropout_prob 0.10 \
    --enable_gradient_checkpointing \
    --logging_dir ./logs
```

### 单 GPU（调试用）

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

### 监控训练

```bash
# 在另一个终端运行
tensorboard --logdir ./logs --port 6006
```

### 训练时间参考

| 硬件 | 预计时间 |
|------|----------|
| 64× TPU v4（论文）| 12 小时 |
| 8× A100 80GB | 约 2 天 |
| 8× RTX 4090 24GB | 约 4 天 |
| 4× RTX 4090 24GB | 约 7 天 |

---

## 第七步：推理

### 命令行推理

```bash
# 打开灯（gamma=1.0）
python3 inference/infer.py \
    --checkpoint checkpoints/lightlab_step045000.pt \
    --image examples/room.jpg \
    --bbox 220 180 340 290 \
    --gamma 1.0 \
    --output room_lit.jpg

# 关灯（gamma=-1.0）
python3 inference/infer.py \
    --checkpoint checkpoints/lightlab_step045000.pt \
    --image examples/room.jpg \
    --bbox 220 180 340 290 \
    --gamma -1.0 \
    --output room_dark.jpg

# 改变颜色（暖黄光）
python3 inference/infer.py \
    --checkpoint checkpoints/lightlab_step045000.pt \
    --image examples/room.jpg \
    --bbox 220 180 340 290 \
    --gamma 1.0 \
    --color 255 150 50 \
    --output room_warm.jpg

# 改变环境光强度
python3 inference/infer.py \
    --checkpoint checkpoints/lightlab_step045000.pt \
    --image examples/room.jpg \
    --bbox 220 180 340 290 \
    --gamma 0.0 \
    --alpha 0.5 \
    --output room_bright_ambient.jpg
```

### Gradio Web Demo

```bash
python3 demo/app.py \
    --checkpoint checkpoints/lightlab_step045000.pt \
    --port 7860

# 公网访问（生成临时链接）：
python3 demo/app.py \
    --checkpoint checkpoints/lightlab_step045000.pt \
    --share
```

---

## 参数说明

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max_train_steps` | 45000 | 总训练步数（论文值） |
| `--learning_rate` | 1e-5 | 学习率（论文值） |
| `--real_weight` | 0.05 | 真实数据采样概率（5%） |
| `--dropout_prob` | 0.10 | 深度图/颜色条件 dropout 概率（论文 10%） |
| `--mixed_precision` | bf16 | 混合精度（bf16 最稳定） |
| `--image_size` | 1024 | 训练分辨率（论文值） |

### 推理参数

| 参数 | 说明 |
|------|------|
| `--gamma` | 目标光强度 ∈ [-1, 1]。1.0=完全开，-1.0=完全关 |
| `--alpha` | 环境光变化 ∈ [-1, 1]。0=不变，0.5=增亮 |
| `--color R G B` | 目标光颜色（0-255）。不填=保持原色 |
| `--tonemap` | `together`（推荐）或 `separate` |
| `--steps` | DDIM 步数，默认 15（论文值） |
| `--bbox X1 Y1 X2 Y2` | 灯具在原始图像中的像素坐标 |

---

## 常见问题

### OOM（显存不足）

```bash
# 1. 减小 per_device_batch_size
--per_device_batch_size 1

# 2. 开启 gradient checkpointing
--enable_gradient_checkpointing

# 3. 用 xformers（已在 requirements 中）
pip install xformers
```

### 数据集路径找不到

确保目录结构正确，数据集加载时会打印找到的样本数量。如果为 0，检查文件命名（`on.jpg`/`off.jpg` 或 `dir_00_mip5.jpg`）。

### SAM 2 安装失败

```bash
# 方案：跳过 SAM 2，使用矩形 mask（精度稍低但可用）
# segmentation.py 会自动 fallback 到 bounding box mask
```

### 恢复训练

```bash
accelerate launch ... training/train.py \
    --resume_from_checkpoint checkpoints/checkpoint_010000.pt \
    ...
```

---

## 文件说明

```
lightlab/
├── data/
│   ├── light_arithmetic.py   # 光线算术核心公式
│   ├── tone_mapping.py       # Tone mapping 策略
│   ├── dataset.py            # 训练数据集
│   └── multi_illumination.py # MIT 数据集加载
├── models/
│   ├── spatial_encoder.py    # 空间条件编码器（1×1 conv）
│   ├── global_conditioning.py# 全局条件（Fourier + MLP）
│   ├── unet_lightlab.py      # 修改版 SDXL UNet
│   └── pipeline_lightlab.py  # 推理 Pipeline
├── preprocessing/
│   ├── depth_extractor.py    # Depth Anything V2
│   └── segmentation.py       # SAM 2 分割
├── training/
│   └── train.py              # 训练脚本
├── inference/
│   └── infer.py              # 命令行推理
├── demo/
│   └── app.py                # Gradio Demo
├── scripts/
│   └── preprocess_real_pairs.py  # 离线预处理
├── configs/
│   └── accelerate_config.yaml    # 多 GPU 配置
├── requirements.txt
├── README.md                 # 架构说明
└── SETUP.md                  # 本文件（执行指南）
```
