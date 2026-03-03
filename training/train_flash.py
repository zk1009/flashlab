"""
FlashLab Training Script.

Fine-tunes SDXL for camera flash control using flash on/off photograph pairs.
Simplified from train.py — no SAM 2, no synthetic data, no bounding boxes.

Training setup (same hyperparams as LightLab paper):
  - Base model:  SDXL
  - Steps:       45,000 (can reduce to ~20,000 for flash-only task)
  - LR:          1e-5
  - Batch:       128 (via gradient accumulation)
  - Resolution:  1024 x 1024
  - Precision:   bfloat16

Usage:
    accelerate launch --config_file configs/accelerate_config.yaml \\
        training/train_flash.py \\
        --flash_data_root ./data/flash_pairs \\
        --depth_cache_dir ./data/depth_cache \\
        --output_dir ./checkpoints \\
        --max_train_steps 30000 \\
        --per_device_batch_size 2 \\
        --gradient_accumulation_steps 8

    # Single GPU:
    accelerate launch --num_processes 1 \\
        training/train_flash.py \\
        --flash_data_root ./data/flash_pairs \\
        --output_dir ./checkpoints \\
        --per_device_batch_size 1 \\
        --gradient_accumulation_steps 128 \\
        --enable_gradient_checkpointing
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from data.flash_dataset import FlashLabDataset
from models.spatial_encoder import SpatialConditionEncoder
from models.global_conditioning import GlobalConditionEmbedder, append_global_conditions
from models.unet_lightlab import build_lightlab_unet, save_lightlab_checkpoint

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="FlashLab Training")

    # Data
    parser.add_argument("--flash_data_root", type=str, required=True,
                        help="Directory with flash on/off pairs")
    parser.add_argument("--depth_cache_dir", type=str, default=None,
                        help="Directory with pre-computed depth maps (.npy)")

    # Model
    parser.add_argument("--pretrained_model_id", type=str,
                        default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # Training
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_train_steps", type=int, default=30000,
                        help="Total steps (flash task may converge faster than 45K)")
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--dropout_prob", type=float, default=0.10)

    # Optimization
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--use_snr_weighting", action="store_true", default=True)

    # Logging
    parser.add_argument("--logging_dir", type=str, default="./logs")
    parser.add_argument("--checkpointing_steps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataloader_num_workers", type=int, default=4)

    # VRAM
    parser.add_argument("--enable_gradient_checkpointing", action="store_true", default=True)

    return parser.parse_args()


def encode_null_prompt(tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2,
                       batch_size, device, dtype):
    """Encode empty prompt for SDXL dual text encoders."""
    empty = [""] * batch_size
    tokens_1 = tokenizer_1(empty, padding="max_length", max_length=77,
                           truncation=True, return_tensors="pt")
    tokens_2 = tokenizer_2(empty, padding="max_length", max_length=77,
                           truncation=True, return_tensors="pt")

    with torch.no_grad():
        enc_out_1 = text_encoder_1(tokens_1.input_ids.to(device), output_hidden_states=True)
        hidden_1 = enc_out_1.hidden_states[-2].to(dtype)

        enc_out_2 = text_encoder_2(tokens_2.input_ids.to(device), output_hidden_states=True)
        hidden_2 = enc_out_2.hidden_states[-2].to(dtype)
        pooled = enc_out_2[0].to(dtype)

    text_embeddings = torch.cat([hidden_1, hidden_2], dim=-1)
    return text_embeddings, pooled


def main():
    args = parse_args()

    from accelerate import Accelerator
    from accelerate.utils import ProjectConfiguration, set_seed

    os.makedirs(args.output_dir, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        project_config=ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=args.logging_dir,
        ),
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    # Load frozen components
    from diffusers import AutoencoderKL, DDPMScheduler
    from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

    weight_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else \
                   torch.float16 if args.mixed_precision == "fp16" else torch.float32

    model_id = args.pretrained_model_id

    logger.info("Loading frozen SDXL components...")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    tokenizer_1 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
    text_encoder_1 = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")

    vae.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    vae.to(accelerator.device, dtype=torch.float32)
    text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)

    # Load trainable components
    logger.info("Building trainable components...")
    if args.resume_from_checkpoint:
        from models.unet_lightlab import load_lightlab_checkpoint
        components = load_lightlab_checkpoint(
            args.resume_from_checkpoint, pretrained_model_id=model_id,
            device="cpu", torch_dtype=torch.float32,
        )
        unet = components["unet"]
        spatial_encoder = components["spatial_encoder"]
        global_embedder = components["global_embedder"]
        start_step = components["step"]
    else:
        unet = build_lightlab_unet(
            pretrained_model_id=model_id,
            enable_gradient_checkpointing=args.enable_gradient_checkpointing,
        )
        spatial_encoder = SpatialConditionEncoder(in_channels=9, out_channels=4)
        global_embedder = GlobalConditionEmbedder(output_dim=2048)
        start_step = 0

    try:
        import xformers
        unet.enable_xformers_memory_efficient_attention()
        logger.info("xformers enabled.")
    except ImportError:
        pass

    params_to_optimize = (
        list(unet.parameters()) +
        list(spatial_encoder.parameters()) +
        list(global_embedder.parameters())
    )

    # Dataset
    logger.info("Building flash dataset...")
    train_dataset = FlashLabDataset(
        flash_root=args.flash_data_root,
        image_size=args.image_size,
        depth_cache_dir=args.depth_cache_dir,
        dropout_prob=args.dropout_prob,
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

    # Optimizer
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

    # Prepare
    unet, spatial_encoder, global_embedder, optimizer, train_dataloader, lr_scheduler = \
        accelerator.prepare(
            unet, spatial_encoder, global_embedder, optimizer,
            train_dataloader, lr_scheduler
        )

    total_batch_size = (
        args.per_device_batch_size *
        accelerator.num_processes *
        args.gradient_accumulation_steps
    )
    logger.info("***** FlashLab Training *****")
    logger.info(f"  Flash pairs: {len(train_dataset.flash_dataset.scenes)}")
    logger.info(f"  Inflated samples: {len(train_dataset.flash_dataset)}")
    logger.info(f"  Max steps: {args.max_train_steps}")
    logger.info(f"  Effective batch size: {total_batch_size}")
    logger.info(f"  Num GPUs: {accelerator.num_processes}")

    # Pre-compute null embeddings
    null_text_emb, null_pooled = encode_null_prompt(
        tokenizer_1, tokenizer_2, text_encoder_1, text_encoder_2,
        batch_size=args.per_device_batch_size,
        device=accelerator.device, dtype=weight_dtype,
    )

    add_time_ids = torch.tensor(
        [[args.image_size, args.image_size, 0, 0, args.image_size, args.image_size]],
        dtype=weight_dtype, device=accelerator.device
    ).expand(args.per_device_batch_size, -1)

    if args.use_snr_weighting:
        from diffusers.training_utils import compute_snr

    # Training loop
    global_step = start_step
    progress_bar = tqdm(
        range(start_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )

    unet.train()
    spatial_encoder.train()
    global_embedder.train()

    for step, batch in enumerate(train_dataloader):
        if global_step >= args.max_train_steps:
            break

        with accelerator.accumulate(unet, spatial_encoder, global_embedder):
            B = batch["input_image"].shape[0]
            device = accelerator.device

            with torch.no_grad():
                input_images = batch["input_image"].to(device, dtype=torch.float32)
                target_images = batch["target_image"].to(device, dtype=torch.float32)

                image_latent = vae.encode(input_images).latent_dist.sample()
                image_latent = image_latent * vae.config.scaling_factor
                image_latent = image_latent.to(dtype=weight_dtype)

                target_latent = vae.encode(target_images).latent_dist.sample()
                target_latent = target_latent * vae.config.scaling_factor
                target_latent = target_latent.to(dtype=weight_dtype)

            depth_map = batch["depth_map"].to(device, dtype=weight_dtype)
            intensity_mask = batch["intensity_mask"].to(device, dtype=weight_dtype)
            color_mask = batch["color_mask"].to(device, dtype=weight_dtype)

            spatial_cond = spatial_encoder(image_latent, depth_map, intensity_mask, color_mask)

            noise = torch.randn_like(target_latent)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (B,), device=device
            ).long()
            noisy_latents = noise_scheduler.add_noise(target_latent, noise, timesteps)

            model_input = torch.cat([noisy_latents, spatial_cond], dim=1)

            if null_text_emb.shape[0] != B:
                text_emb = null_text_emb[:1].expand(B, -1, -1)
                pooled = null_pooled[:1].expand(B, -1)
            else:
                text_emb = null_text_emb
                pooled = null_pooled

            ambient_alpha = batch["ambient_alpha"].to(device)
            tonemap_flag = batch["tonemap_flag"].to(device)
            encoder_hidden_states = append_global_conditions(
                text_emb, global_embedder, ambient_alpha, tonemap_flag
            )

            if add_time_ids.shape[0] != B:
                time_ids = add_time_ids[:1].expand(B, -1)
            else:
                time_ids = add_time_ids

            added_cond_kwargs = {"text_embeds": pooled, "time_ids": time_ids}

            model_pred = unet(
                model_input, timesteps,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(target_latent, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

            if args.use_snr_weighting:
                snr = compute_snr(noise_scheduler, timesteps)
                mse_loss_weights = torch.stack(
                    [snr, 5.0 * torch.ones_like(timesteps)], dim=1
                ).min(dim=1)[0] / snr
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape))))
                loss = (loss * mse_loss_weights).mean()
            else:
                loss = F.mse_loss(model_pred.float(), target.float())

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(params_to_optimize, args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.sync_gradients:
            progress_bar.update(1)
            global_step += 1

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step % args.checkpointing_steps == 0 and accelerator.is_main_process:
                save_lightlab_checkpoint(
                    unet, spatial_encoder, global_embedder,
                    step=global_step, output_dir=args.output_dir,
                    accelerator=accelerator,
                )

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_lightlab_checkpoint(
            unet, spatial_encoder, global_embedder,
            step=global_step, output_dir=args.output_dir,
            accelerator=accelerator,
        )
        logger.info(f"Training complete at step {global_step}.")
    accelerator.end_training()


if __name__ == "__main__":
    main()
