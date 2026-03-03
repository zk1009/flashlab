"""
FlashLab Inference Pipeline.

Simplified pipeline for camera flash control — no SAM 2 or bounding box needed.
The flash mask covers the entire image since flash illuminates globally.

Usage:
    from models.pipeline_flash import FlashLabPipeline

    pipeline = FlashLabPipeline.from_checkpoint(
        checkpoint_path="checkpoints/flashlab_step045000.pt",
        device="cuda",
    )
    result = pipeline(
        image=Image.open("photo.jpg"),
        gamma=0.8,                   # flash intensity
        ct_rgb=[1.0, 0.85, 0.7],    # warm flash color
    )
    result.save("photo_flash.jpg")
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional, Tuple


class FlashLabPipeline:
    """
    Inference pipeline for flash photography control.

    Simplified from LightLabPipeline:
      - No SAM 2 dependency (mask = full image)
      - No bounding box input
      - Only needs: Depth Anything V2 + trained LightLab components

    Args:
        unet:              Modified LightLab UNet (8-channel conv_in).
        spatial_encoder:   SpatialConditionEncoder.
        global_embedder:   GlobalConditionEmbedder.
        vae:               SDXL VAE (frozen).
        text_encoder_1:    CLIP-L text encoder (frozen).
        text_encoder_2:    OpenCLIP-ViT/G text encoder (frozen).
        tokenizer_1:       CLIP-L tokenizer.
        tokenizer_2:       OpenCLIP-ViT/G tokenizer.
        depth_extractor:   DepthExtractor (Depth Anything V2).
        device:            Computation device.
        dtype:             Model dtype for inference.
    """

    def __init__(
        self,
        unet,
        spatial_encoder,
        global_embedder,
        vae,
        text_encoder_1,
        text_encoder_2,
        tokenizer_1,
        tokenizer_2,
        depth_extractor,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        self.unet = unet.eval().to(device, dtype=dtype)
        self.spatial_encoder = spatial_encoder.eval().to(device, dtype=dtype)
        self.global_embedder = global_embedder.eval().to(device, dtype=dtype)
        self.vae = vae.eval().to(device, dtype=torch.float32)  # VAE needs float32
        self.text_encoder_1 = text_encoder_1.eval().to(device, dtype=dtype)
        self.text_encoder_2 = text_encoder_2.eval().to(device, dtype=dtype)
        self.tokenizer_1 = tokenizer_1
        self.tokenizer_2 = tokenizer_2
        self.depth_extractor = depth_extractor
        self.device = device
        self.dtype = dtype

        from diffusers import DDIMScheduler
        self.scheduler = DDIMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        pretrained_model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
        depth_model_size: str = "large",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "FlashLabPipeline":
        """
        Construct a FlashLabPipeline from a saved checkpoint.

        No SAM 2 needed — significantly faster to load than LightLabPipeline.
        """
        from diffusers import AutoencoderKL
        from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                                   CLIPTokenizer)
        from models.unet_lightlab import load_lightlab_checkpoint
        from preprocessing.depth_extractor import DepthExtractor

        print(f"Loading FlashLab checkpoint from {checkpoint_path}...")
        components = load_lightlab_checkpoint(
            checkpoint_path,
            pretrained_model_id=pretrained_model_id,
            device=device,
            torch_dtype=dtype,
        )

        print(f"Loading frozen SDXL components from {pretrained_model_id}...")
        vae = AutoencoderKL.from_pretrained(pretrained_model_id, subfolder="vae")
        text_encoder_1 = CLIPTextModel.from_pretrained(
            pretrained_model_id, subfolder="text_encoder"
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            pretrained_model_id, subfolder="text_encoder_2"
        )
        tokenizer_1 = CLIPTokenizer.from_pretrained(
            pretrained_model_id, subfolder="tokenizer"
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            pretrained_model_id, subfolder="tokenizer_2"
        )

        print(f"Loading Depth Anything V2 ({depth_model_size})...")
        depth_extractor = DepthExtractor(model_size=depth_model_size, device=device, dtype=dtype)

        return cls(
            unet=components["unet"],
            spatial_encoder=components["spatial_encoder"],
            global_embedder=components["global_embedder"],
            vae=vae,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            depth_extractor=depth_extractor,
            device=device,
            dtype=dtype,
        )

    def _pil_to_tensor(self, image: Image.Image, target_size: int) -> torch.Tensor:
        """Convert PIL image to normalized tensor [-1, 1]."""
        image = image.convert("RGB").resize((target_size, target_size), Image.LANCZOS)
        arr = np.array(image).astype(np.float32) / 127.5 - 1.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device, dtype=torch.float32)

    def _tensor_to_pil(self, latent_decoded: torch.Tensor) -> Image.Image:
        """Convert decoded output tensor to PIL image."""
        img = latent_decoded.squeeze(0).permute(1, 2, 0)
        img = ((img.float() + 1.0) / 2.0).clamp(0, 1)
        img_np = (img.cpu().numpy() * 255).astype(np.uint8)
        return Image.fromarray(img_np)

    def _encode_null_prompt(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode empty prompt for SDXL's dual text encoders."""
        empty = [""] * batch_size
        t1 = self.tokenizer_1(
            empty, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        t2 = self.tokenizer_2(
            empty, padding="max_length", max_length=77,
            truncation=True, return_tensors="pt"
        )
        with torch.no_grad():
            out1 = self.text_encoder_1(
                t1.input_ids.to(self.device), output_hidden_states=True
            )
            h1 = out1.hidden_states[-2].to(self.dtype)

            out2 = self.text_encoder_2(
                t2.input_ids.to(self.device), output_hidden_states=True
            )
            h2 = out2.hidden_states[-2].to(self.dtype)
            pooled = out2[0].to(self.dtype)

        text_emb = torch.cat([h1, h2], dim=-1)
        return text_emb, pooled

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        gamma: float = 1.0,        # Flash intensity ∈ [-1, 1]
        ct_rgb: Optional[List[float]] = None,  # Flash color [R,G,B] ∈ [0,1]
        alpha: float = 0.0,        # Ambient light change ∈ [-1, 1]
        tonemap: str = "together",
        num_inference_steps: int = 15,
        guidance_scale: float = 1.0,
        image_size: int = 1024,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        """
        Add/remove/adjust flash lighting on an image.

        Args:
            image:               Input PIL image.
            gamma:               Flash intensity. 1.0=full flash, 0=no flash, -1=remove flash.
            ct_rgb:              Flash color [R, G, B] ∈ [0, 1]. None = neutral white.
            alpha:               Ambient light change ∈ [-1, 1]. 0 = no change.
            tonemap:             Tone mapping strategy: "separate" or "together".
            num_inference_steps: DDIM steps (default 15).
            guidance_scale:      CFG scale (1.0 = no CFG, paper default).
            image_size:          Processing resolution.
            generator:           Optional RNG for reproducibility.

        Returns:
            Output PIL image with flash effect applied.
        """
        image_resized = image.convert("RGB").resize(
            (image_size, image_size), Image.LANCZOS
        )

        lat_size = image_size // 8
        B = 1

        # Step 1: Extract depth map
        depth = self.depth_extractor(image_resized, lat_size, lat_size)
        depth = depth.to(self.device, dtype=self.dtype)

        # Step 2: Full-image mask (flash illuminates everything)
        seg_mask = torch.ones(1, 1, lat_size, lat_size, dtype=self.dtype, device=self.device)

        # Step 3: Build condition masks
        intensity_mask = seg_mask * gamma

        if ct_rgb is None:
            ct_rgb = [1.0, 1.0, 1.0]
        ct = torch.tensor(ct_rgb, dtype=self.dtype, device=self.device).view(1, 3, 1, 1)
        color_mask = seg_mask.expand(-1, 3, -1, -1) * ct

        # Step 4: Encode input image with VAE
        img_tensor = self._pil_to_tensor(image_resized, image_size)
        image_latent = self.vae.encode(img_tensor).latent_dist.mode()
        image_latent = image_latent * self.vae.config.scaling_factor
        image_latent = image_latent.to(dtype=self.dtype)

        # Step 5: Build spatial condition
        spatial_cond = self.spatial_encoder(
            image_latent, depth, intensity_mask, color_mask
        )

        # Step 6: Build text + global embeddings
        text_emb, pooled = self._encode_null_prompt(B)

        ambient_t = torch.tensor([alpha], dtype=self.dtype, device=self.device)
        tonemap_t = torch.tensor(
            [1.0 if tonemap == "together" else 0.0],
            dtype=self.dtype, device=self.device
        )

        from models.global_conditioning import append_global_conditions
        encoder_hidden_states = append_global_conditions(
            text_emb, self.global_embedder, ambient_t, tonemap_t
        )

        add_time_ids = torch.tensor(
            [[image_size, image_size, 0, 0, image_size, image_size]],
            dtype=self.dtype, device=self.device
        )
        added_cond_kwargs = {
            "text_embeds": pooled,
            "time_ids": add_time_ids,
        }

        # Step 7: DDIM denoising loop
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        latents = torch.randn(
            (B, 4, lat_size, lat_size),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        latents = latents * self.scheduler.init_noise_sigma

        for t in timesteps:
            model_input = torch.cat([latents, spatial_cond], dim=1)
            model_input = self.scheduler.scale_model_input(model_input, t)

            noise_pred = self.unet(
                model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            if guidance_scale > 1.0:
                uncond_spatial = torch.zeros_like(spatial_cond)
                model_input_uncond = torch.cat([latents, uncond_spatial], dim=1)
                model_input_uncond = self.scheduler.scale_model_input(model_input_uncond, t)

                uncond_ambient = torch.zeros(B, dtype=self.dtype, device=self.device)
                uncond_tonemap = torch.zeros(B, dtype=self.dtype, device=self.device)
                uncond_hidden_states = append_global_conditions(
                    text_emb, self.global_embedder, uncond_ambient, uncond_tonemap
                )

                noise_pred_uncond = self.unet(
                    model_input_uncond,
                    t,
                    encoder_hidden_states=uncond_hidden_states,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Step 8: Decode
        latents = latents.to(torch.float32) / self.vae.config.scaling_factor
        decoded = self.vae.decode(latents).sample

        return self._tensor_to_pil(decoded)
