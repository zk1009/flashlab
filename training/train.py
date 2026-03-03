"""
LightLab Training Script.

Fine-tunes an SDXL-based latent diffusion model for parametric light source
control, following LightLab (arXiv:2505.09608) Section 3.4 and Supplementary C.

Training setup per paper:
  - Base model:  SDXL architecture
  - Steps:       45,000
  - LR:          1e-5
  - Batch:       128 (achieved via gradient accumulation across GPUs)
  - Resolution:  1024 × 1024
  - Precision:   bfloat16

Usage:
    accelerate launch --config_file configs/accelerate_config.yaml \\
        training/train.py \\
        --output_dir ./checkpoints \\
        --real_data_root ./data/real_pairs \\
        --synthetic_data_root ./data/multi_illumination \\
        --per_device_batch_size 2 \\
        --gradient_accumulation_steps 8 \\
        --max_train_steps 45000 \\
        --learning_rate 1e-5 \\
        --mixed_precision bf16
"""

import argparse
import logging
import math
import os
import sys
import random
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm.auto import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.dataset import LightLabDataset
from models.spatial_encoder import SpatialConditionEncoder
from models.global_conditioning import GlobalConditionEmbedder, append_global_conditions
from models.unet_lightlab import build_lightlab_unet, save_lightlab_checkpoint

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="LightLab Training Script")

    # Data
    parser.add_argument("--real_data_root", type=str, default=None,
                        help="Directory with real photograph pairs (on/off)")
    parser.add_argument("--synthetic_data_root", type=str, default=None,
                        help="Directory with Multi-Illumination dataset")
    parser.add_argument("--depth_cache_dir", type=str, default=None,
                        help="Directory with pre-computed depth maps (.npy)")
    parser.add_argument("--mask_cache_dir", type=str, default=None,
                        help="Directory with pre-computed SAM 2 masks (.npy)")
    parser.add_argument("--synthetic_type", type=str, default="multi_illumination",
                        choices=["multi_illumination", "hypersim"])

    # Model
    parser.add_argument("--pretrained_model_id", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0",
                        help="HuggingFace model ID for base SDXL model")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a LightLab checkpoint to resume from")

    # Training
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save checkpoints and logs")
    parser.add_argument("--max_train_steps", type=int, default=45000)
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--real_weight", type=float, default=0.05,
                        help="Probability of sampling from real data (default 5%)")
    parser.add_argument("--dropout_prob", type=float, default=0.10,
                        help="Probability of dropping depth/color conditions")

    # Optimization
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--use_snr_weighting", action="store_true", default=True,
                        help="Use min-SNR loss weighting (Hang et al. 2023)")

    # Logging and checkpointing
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--validation_steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # VRAM optimization
    parser.add_argument("--enable_gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)

    return parser.parse_args()


def encode_null_prompt(
    tokenizer_1,
    tokenizer_2,
    text_encoder_1,
    text_encoder_2,
    batch_size: int,
    device: str,
    dtype: torch.dtype,
):
    """
    Encode empty/null prompts for SDXL's dual CLIP text encoders.

    LightLab uses no text prompts — all conditioning is spatial/global.
    Returns hidden states and pooled embeddings from empty string input.

    Returns:
        text_embeddings:  (B, 77, 2048) — SDXL concatenated hidden states
        pooled_embeddings: (B, 1280) — SDXL pooled text embedding
    """
    empty = [""] * batch_size

    # Tokenize for both text encoders
    tokens_1 = tokenizer_1(
        empty, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt"
    )
    tokens_2 = tokenizer_2(
        empty, padding="max_length", max_length=77,
        truncation=True, return_tensors="pt"
    )

    with torch.no_grad():
        # CLIP-L (text_encoder_1): outputs last_hidden_state (B, 77, 768)
        enc_out_1 = text_encoder_1(
            tokens_1.input_ids.to(device),
            output_hidden_states=True,
        )
        hidden_1 = enc_out_1.hidden_states[-2].to(dtype)  # (B, 77, 768)

        # OpenCLIP-ViT/G (text_encoder_2): outputs last_hidden_state (B, 77, 1280)
        enc_out_2 = text_encoder_2(
            tokens_2.input_ids.to(device),
            output_hidden_states=True,
        )
        hidden_2 = enc_out_2.hidden_states[-2].to(dtype)  # (B, 77, 1280)
        pooled = enc_out_2[0].to(dtype)                   # (B, 1280)

    # Concatenate to match SDXL cross_attention_dim = 2048
    text_embeddings = torch.cat([hidden_1, hidden_2], dim=-1)  # (B, 77, 2048)

    return text_embeddings, pooled


def compute_time_ids(
    original_size: tuple,
    crop_coords: tuple,
    target_size: tuple,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Compute SDXL's additional text/time conditioning IDs.
    Required by SDXL's time_text_embed layer (added conditioning).
    """
    add_time_ids = list(original_size) + list(crop_coords) + list(target_size)
    add_time_ids = torch.tensor([add_time_ids], dtype=dtype, device=device)
    return add_time_ids  # (1, 6)


def main():
    args = parse_args()

    # Initialize Accelerator
    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed
    from accelerate.logging import get_logger

    os.makedirs(args.output_dir, exist_ok=True)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir,
        logging_dir=args.logging_dir,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=accelerator_project_config,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    if args.seed is not None:
        set_seed(args.seed)

    # -------------------------------------------------------------------------
    # Load frozen components (VAE, text encoders)
    # -------------------------------------------------------------------------
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else \
                   torch.float16 if args.mixed_precision == "fp16" else torch.float32

    model_id = args.pretrained_model_id

    logger.info("Loading frozen model components...")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

    tokenizer_1 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")

    text_encoder_1 = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        model_id, subfolder="text_encoder_2"
    )

    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

    # Freeze everything except the three trainable components
    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)

    # Move frozen models to GPU
    vae.to(accelerator.device, dtype=torch.float32)  # VAE needs float32
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # -------------------------------------------------------------------------
    # Load trainable components
    # -------------------------------------------------------------------------
    logger.info("Building LightLab trainable components...")

    if args.resume_from_checkpoint:
        from models.unet_lightlab import load_lightlab_checkpoint
        components = load_lightlab_checkpoint(
            args.resume_from_checkpoint,
            pretrained_model_id=model_id,
            device="cpu",
            torch_dtype=torch.float32,
        )
        unet = components["unet"]
        spatial_encoder = components["spatial_encoder"]
        global_embedder = components["global_embedder"]
        start_step = components["step"]
        logger.info(f"Resumed from checkpoint at step {start_step}")
    else:
        unet = build_lightlab_unet(
            pretrained_model_id=model_id,
            enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        )
        spatial_encoder = SpatialConditionEncoder(in_channels=9, out_channels=4)
        global_embedder = GlobalConditionEmbedder(output_dim=2048)
        start_step = 0

    # Enable xformers memory-efficient attention if available
    try:
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        logger.info("xformers memory-efficient attention enabled.")
    except ImportError:
        logger.info("xformers not available; using default attention.")

    # Collect trainable parameters
    params_to_optimize = (
        list(unet.parameters()) +
        list(spatial_encoder.parameters()) +
        list(global_embedder.parameters())
    )
    logger.info(
        f"Trainable parameters: "
        f"UNet={sum(p.numel() for p in unet.parameters()) / 1e6:.1f}M, "
        f"SpatialEncoder={sum(p.numel() for p in spatial_encoder.parameters()) / 1e3:.1f}K, "
        f"GlobalEmbedder={sum(p.numel() for p in global_embedder.parameters()) / 1e3:.1f}K"
    )

    # -------------------------------------------------------------------------
    # Dataset and DataLoader
    # -------------------------------------------------------------------------
    logger.info("Building dataset...")
    train_dataset = LightLabDataset(
        real_root=args.real_data_root,
        synthetic_root=args.synthetic_data_root,
        real_weight=args.real_weight,
        image_size=args.image_size,
        depth_cache_dir=args.depth_cache_dir,
        mask_cache_dir=args.mask_cache_dir,
        dropout_prob=args.dropout_prob,
        synthetic_type=args.synthetic_type,
        total_samples=args.max_train_steps * args.per_device_batch_size * accelerator.num_processes * args.gradient_accumulation_steps,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # -------------------------------------------------------------------------
    # Optimizer and LR scheduler
    # -------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    from diffusers.optimization import get_scheduler
    lr_scheduler = get_scheduler(
        "constant_with_warmup",
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # -------------------------------------------------------------------------
    # Prepare with Accelerator (handles DDP, mixed precision, etc.)
    # -------------------------------------------------------------------------
    unet, spatial_encoder, global_embedder, optimizer, train_dataloader, lr_scheduler = \
        accelerator.prepare(
            unet, spatial_encoder, global_embedder, optimizer,
            train_dataloader, lr_scheduler
        )

    # Effective batch size
    total_batch_size = (
        args.per_device_batch_size *
        accelerator.num_processes *
        args.gradient_accumulation_steps
    )

    logger.info("***** Starting Training *****")
    logger.info(f"  Max train steps = {args.max_train_steps}")
    logger.info(f"  Effective batch size = {total_batch_size}")
    logger.info(f"  Per-device batch size = {args.per_device_batch_size}")
    logger.info(f"  Gradient accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Num GPUs = {accelerator.num_processes}")
    logger.info(f"  Mixed precision = {args.mixed_precision}")

    # -------------------------------------------------------------------------
    # Pre-compute null prompt embeddings (same for every batch)
    # -------------------------------------------------------------------------
    null_text_emb, null_pooled = encode_null_prompt(
        tokenizer_1, tokenizer_2,
        text_encoder_1, text_encoder_2,
        batch_size=args.per_device_batch_size,
        device=accelerator.device,
        dtype=weight_dtype,
    )
    # time_ids for SDXL: [orig_h, orig_w, crop_y, crop_x, target_h, target_w]
    base_time_ids = compute_time_ids(
        original_size=(args.image_size, args.image_size),
        crop_coords=(0, 0),
        target_size=(args.image_size, args.image_size),
        device=accelerator.device,
        dtype=weight_dtype,
    ).expand(args.per_device_batch_size, -1)  # (B, 6)

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    global_step = start_step
    progress_bar = tqdm(
        range(start_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    if args.use_snr_weighting:
        from diffusers.training_utils import compute_snr

    unet.train()
    spatial_encoder.train()
    global_embedder.train()

    for step, batch in enumerate(train_dataloader):
        if global_step >= args.max_train_steps:
            break

        with accelerator.accumulate(unet, spatial_encoder, global_embedder):

            B = batch["input_image"].shape[0]
            device = accelerator.device

            # -----------------------------------------------------------------
            # 1. Encode input and target images with frozen VAE
            # -----------------------------------------------------------------
            with torch.no_grad():
                input_images = batch["input_image"].to(device, dtype=torch.float32)
                target_images = batch["target_image"].to(device, dtype=torch.float32)

                # VAE encoding
                image_latent = vae.encode(input_images).latent_dist.sample()
                image_latent = image_latent * vae.config.scaling_factor  # (B, 4, H/8, W/8)
                image_latent = image_latent.to(dtype=weight_dtype)

                target_latent = vae.encode(target_images).latent_dist.sample()
                target_latent = target_latent * vae.config.scaling_factor
                target_latent = target_latent.to(dtype=weight_dtype)

            # -----------------------------------------------------------------
            # 2. Build spatial condition (B, 4, lH, lW) via SpatialConditionEncoder
            # -----------------------------------------------------------------
            depth_map = batch["depth_map"].to(device, dtype=weight_dtype)      # (B, 1, lH, lW)
            intensity_mask = batch["intensity_mask"].to(device, dtype=weight_dtype)  # (B, 1, lH, lW)
            color_mask = batch["color_mask"].to(device, dtype=weight_dtype)    # (B, 3, lH, lW)

            spatial_cond = spatial_encoder(
                image_latent, depth_map, intensity_mask, color_mask
            )  # (B, 4, lH, lW)

            # -----------------------------------------------------------------
            # 3. Add noise to target latents (forward diffusion process)
            # -----------------------------------------------------------------
            noise = torch.randn_like(target_latent)
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (B,),
                device=device,
            ).long()
            noisy_latents = noise_scheduler.add_noise(target_latent, noise, timesteps)

            # -----------------------------------------------------------------
            # 4. Concatenate noisy target latents with spatial conditions
            # -----------------------------------------------------------------
            model_input = torch.cat([noisy_latents, spatial_cond], dim=1)  # (B, 8, lH, lW)

            # -----------------------------------------------------------------
            # 5. Build text + global condition embeddings for cross-attention
            # -----------------------------------------------------------------
            # Repeat null embeddings for batch size B
            if null_text_emb.shape[0] != B:
                text_emb = null_text_emb[:1].expand(B, -1, -1)  # (B, 77, 2048)
                pooled = null_pooled[:1].expand(B, -1)            # (B, 1280)
            else:
                text_emb = null_text_emb
                pooled = null_pooled

            # Append global condition tokens (B, 79, 2048)
            ambient_alpha = batch["ambient_alpha"].to(device)   # (B,)
            tonemap_flag = batch["tonemap_flag"].to(device)     # (B,)
            encoder_hidden_states = append_global_conditions(
                text_emb, global_embedder, ambient_alpha, tonemap_flag
            )

            # SDXL added conditioning
            if base_time_ids.shape[0] != B:
                time_ids = base_time_ids[:1].expand(B, -1)
            else:
                time_ids = base_time_ids

            added_cond_kwargs = {
                "text_embeds": pooled,
                "time_ids": time_ids,
            }

            # -----------------------------------------------------------------
            # 6. UNet forward pass
            # -----------------------------------------------------------------
            model_pred = unet(
                model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]  # (B, 4, lH, lW)

            # -----------------------------------------------------------------
            # 7. Compute loss
            # -----------------------------------------------------------------
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(target_latent, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type: {noise_scheduler.config.prediction_type}"
                )

            if args.use_snr_weighting:
                # Min-SNR loss weighting (Hang et al. 2023) for stable training
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack(
                    [snr, 5.0 * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0] / snr
                loss = F.mse_loss(
                    model_pred.float(), target.float(), reduction="none"
                )
                loss = loss.mean(dim=list(range(1, len(loss.shape))))  # per-sample
                loss = (loss * mse_loss_weights).mean()
            else:
                loss = F.mse_loss(model_pred.float(), target.float())

            # -----------------------------------------------------------------
            # 8. Backward pass and optimizer step
            # -----------------------------------------------------------------
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # ---------------------------------------------------------------------
        # Logging and checkpointing (only when gradients synced)
        # ---------------------------------------------------------------------
        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            # Log metrics
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # Save checkpoint
            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                save_lightlab_checkpoint(
                    unet, spatial_encoder, global_embedder,
                    step=global_step,
                    output_dir=args.output_dir,
                    accelerator=accelerator,
                )

    # Final save
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lightlab_checkpoint(
            unet, spatial_encoder, global_embedder,
            step=global_step,
            output_dir=args.output_dir,
            accelerator=accelerator,
        )
        logger.info(f"Training complete! Final checkpoint saved at step {global_step}.")

    accelerator.end_training()


if __name__ == "__main__":
    main()
