"""
LightLab Inference Pipeline.

End-to-end pipeline for light source control in images:
  1. Segment target light source with SAM 2 (from user bounding box)
  2. Extract depth map with Depth Anything V2
  3. Build spatial and global condition tensors
  4. Run DDIM denoising (15 steps per paper)
  5. Decode latents to SDR output image

Usage:
    from models.pipeline_lightlab import LightLabPipeline

    pipeline = LightLabPipeline.from_checkpoint(
        checkpoint_path="checkpoints/lightlab_step045000.pt",
        device="cuda",
    )
    result = pipeline(
        image=Image.open("room.jpg"),
        bbox=[220, 180, 340, 290],  # light source bounding box
        gamma=1.0,   # turn light fully on
        alpha=0.0,   # no ambient change
    )
    result.save("room_lit.jpg")
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from typing import List, Optional, Tuple, Union


class LightLabPipeline:
    """
    Full inference pipeline for LightLab light source control.

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
        segmenter:         LightSourceSegmenter (SAM 2).
        device:            Computation device.
        dtype:             Model dtype for inference (float16 recommended).
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
        segmenter,
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
        self.segmenter = segmenter
        self.device = device
        self.dtype = dtype

        # Load DDIM scheduler for 15-step inference
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
        sam2_checkpoint: str = "checkpoints/sam2_hiera_large.pt",
        sam2_config: str = "sam2_hiera_l.yaml",
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ) -> "LightLabPipeline":
        """
        Construct a LightLabPipeline from a saved checkpoint.

        Args:
            checkpoint_path:     Path to a .pt checkpoint file.
            pretrained_model_id: Base SDXL model for frozen components.
            depth_model_size:    Depth Anything V2 model size.
            sam2_checkpoint:     Path to SAM 2 checkpoint.
            sam2_config:         SAM 2 config name.
            device:              Target device.
            dtype:               Model dtype.
        """
        from diffusers import AutoencoderKL
        from transformers import (CLIPTextModel, CLIPTextModelWithProjection,
                                   CLIPTokenizer)
        from models.unet_lightlab import load_lightlab_checkpoint
        from preprocessing.depth_extractor import DepthExtractor
        from preprocessing.segmentation import LightSourceSegmenter

        print(f"Loading LightLab checkpoint from {checkpoint_path}...")
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

        depth_extractor = DepthExtractor(model_size=depth_model_size, device=device, dtype=dtype)
        segmenter = LightSourceSegmenter(
            checkpoint=sam2_checkpoint,
            model_cfg=sam2_config,
            device=device,
        )

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
            segmenter=segmenter,
            device=device,
            dtype=dtype,
        )

    def _pil_to_tensor(self, image: Image.Image, target_size: int) -> torch.Tensor:
        """Convert PIL image to normalized tensor [-1, 1], shape (1, 3, H, W)."""
        image = image.convert("RGB").resize((target_size, target_size), Image.LANCZOS)
        arr = np.array(image).astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        return tensor.to(self.device, dtype=torch.float32)

    def _tensor_to_pil(self, latent_decoded: torch.Tensor) -> Image.Image:
        """Convert decoded output tensor to PIL image."""
        img = latent_decoded.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
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
            h1 = out1.hidden_states[-2].to(self.dtype)  # (B, 77, 768)

            out2 = self.text_encoder_2(
                t2.input_ids.to(self.device), output_hidden_states=True
            )
            h2 = out2.hidden_states[-2].to(self.dtype)  # (B, 77, 1280)
            pooled = out2[0].to(self.dtype)              # (B, 1280)

        text_emb = torch.cat([h1, h2], dim=-1)  # (B, 77, 2048)
        return text_emb, pooled

    @torch.no_grad()
    def __call__(
        self,
        image: Image.Image,
        bbox: List[int],           # [x_min, y_min, x_max, y_max]
        gamma: float = 1.0,        # Target light intensity ∈ [-1, 1]
                                   #  1.0 = fully on, -1.0 = fully off
        ct_rgb: Optional[List[float]] = None,  # Target color [R,G,B] ∈ [0,1]
        alpha: float = 0.0,        # Ambient light change ∈ [-1, 1]
        tonemap: str = "together", # "separate" or "together"
        num_inference_steps: int = 15,
        guidance_scale: float = 1.0,  # ≥1.0; paper uses no CFG (=1.0)
        image_size: int = 1024,
        generator: Optional[torch.Generator] = None,
    ) -> Image.Image:
        """
        Run LightLab inference to edit the lighting of an image.

        Args:
            image:               Input PIL image.
            bbox:                Bounding box around target light source [x1,y1,x2,y2].
            gamma:               Target light intensity change. ∈ [-1, 1].
                                 Positive = brighter, negative = dimmer.
            ct_rgb:              Target light color [R, G, B] ∈ [0, 1]. None = keep original.
            alpha:               Ambient light change ∈ [-1, 1]. 0 = no change.
            tonemap:             Tone mapping strategy: "separate" or "together".
            num_inference_steps: Number of DDIM denoising steps (paper uses 15).
            guidance_scale:      Classifier-free guidance scale (paper uses 1.0 = no CFG).
            image_size:          Target image size (should match training resolution).
            generator:           Optional random number generator for reproducibility.

        Returns:
            Relit PIL image.
        """
        image_resized = image.convert("RGB").resize(
            (image_size, image_size), Image.LANCZOS
        )
        image_np = np.array(image_resized)  # (H, W, 3) uint8

        lat_size = image_size // 8
        B = 1

        # -----------------------------------------------------------------
        # Step 1: Extract depth map
        # -----------------------------------------------------------------
        depth = self.depth_extractor(image_resized, lat_size, lat_size)
        depth = depth.to(self.device, dtype=self.dtype)  # (1, 1, lat_H, lat_W)

        # -----------------------------------------------------------------
        # Step 2: Segment light source with SAM 2
        # -----------------------------------------------------------------
        # Scale bbox from original image coordinates to image_size
        orig_w, orig_h = image.size
        scale_x = image_size / orig_w
        scale_y = image_size / orig_h
        bbox_scaled = [
            int(bbox[0] * scale_x),
            int(bbox[1] * scale_y),
            int(bbox[2] * scale_x),
            int(bbox[3] * scale_y),
        ]

        seg_mask = self.segmenter.segment_from_bbox(
            image_np, bbox_scaled, lat_size, lat_size
        ).to(self.device, dtype=self.dtype)  # (1, 1, lat_H, lat_W)

        # -----------------------------------------------------------------
        # Step 3: Build condition masks
        # -----------------------------------------------------------------
        intensity_mask = seg_mask * gamma  # (1, 1, lat_H, lat_W)

        if ct_rgb is None:
            ct_rgb = [1.0, 1.0, 1.0]  # white = neutral
        ct = torch.tensor(ct_rgb, dtype=self.dtype, device=self.device).view(1, 3, 1, 1)
        color_mask = seg_mask.expand(-1, 3, -1, -1) * ct  # (1, 3, lat_H, lat_W)

        # -----------------------------------------------------------------
        # Step 4: Encode input image with VAE
        # -----------------------------------------------------------------
        img_tensor = self._pil_to_tensor(image_resized, image_size)  # (1, 3, H, W)
        image_latent = self.vae.encode(img_tensor).latent_dist.mode()
        image_latent = image_latent * self.vae.config.scaling_factor   # (1, 4, lat_H, lat_W)
        image_latent = image_latent.to(dtype=self.dtype)

        # -----------------------------------------------------------------
        # Step 5: Build spatial condition
        # -----------------------------------------------------------------
        spatial_cond = self.spatial_encoder(
            image_latent, depth, intensity_mask, color_mask
        )  # (1, 4, lat_H, lat_W)

        # -----------------------------------------------------------------
        # Step 6: Build text + global embeddings
        # -----------------------------------------------------------------
        text_emb, pooled = self._encode_null_prompt(B)  # (1, 77, 2048), (1, 1280)

        ambient_t = torch.tensor([alpha], dtype=self.dtype, device=self.device)
        tonemap_t = torch.tensor(
            [1.0 if tonemap == "together" else 0.0],
            dtype=self.dtype, device=self.device
        )

        from models.global_conditioning import append_global_conditions
        encoder_hidden_states = append_global_conditions(
            text_emb, self.global_embedder, ambient_t, tonemap_t
        )  # (1, 79, 2048)

        # SDXL added conditioning
        add_time_ids = torch.tensor(
            [[image_size, image_size, 0, 0, image_size, image_size]],
            dtype=self.dtype, device=self.device
        )
        added_cond_kwargs = {
            "text_embeds": pooled,
            "time_ids": add_time_ids,
        }

        # -----------------------------------------------------------------
        # Step 7: DDIM denoising loop (15 steps)
        # -----------------------------------------------------------------
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.scheduler.timesteps

        # Start from random noise in the latent space
        latents = torch.randn(
            (B, 4, lat_size, lat_size),
            dtype=self.dtype,
            device=self.device,
            generator=generator,
        )
        latents = latents * self.scheduler.init_noise_sigma

        for t in timesteps:
            # Concatenate spatial condition to noisy latents
            model_input = torch.cat([latents, spatial_cond], dim=1)  # (1, 8, lH, lW)
            model_input = self.scheduler.scale_model_input(model_input, t)

            # UNet prediction
            noise_pred = self.unet(
                model_input,
                t,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # Classifier-free guidance (if scale > 1)
            if guidance_scale > 1.0:
                # Run unconditional pass with zeroed spatial and global conditions
                uncond_spatial = torch.zeros_like(spatial_cond)
                model_input_uncond = torch.cat([latents, uncond_spatial], dim=1)
                model_input_uncond = self.scheduler.scale_model_input(model_input_uncond, t)

                # Zero out global condition tokens as well
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

            # DDIM step
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # -----------------------------------------------------------------
        # Step 8: Decode latents to image
        # -----------------------------------------------------------------
        latents = latents.to(torch.float32) / self.vae.config.scaling_factor
        decoded = self.vae.decode(latents).sample  # (1, 3, H, W)

        return self._tensor_to_pil(decoded)

    def batch_edit(
        self,
        images: List[Image.Image],
        bboxes: List[List[int]],
        gammas: Optional[List[float]] = None,
        ct_rgbs: Optional[List[Optional[List[float]]]] = None,
        alphas: Optional[List[float]] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Edit multiple images sequentially (for animation / sequential editing).

        Args:
            images:   List of input PIL images.
            bboxes:   List of bounding boxes, one per image.
            gammas:   List of gamma values. If None, uses 1.0 for all.
            ct_rgbs:  List of target colors. If None, uses None for all.
            alphas:   List of ambient values. If None, uses 0.0 for all.

        Returns:
            List of relit PIL images.
        """
        N = len(images)
        gammas = gammas or [1.0] * N
        ct_rgbs = ct_rgbs or [None] * N
        alphas = alphas or [0.0] * N

        results = []
        for i, (img, bbox, gamma, ct, alpha) in enumerate(
            zip(images, bboxes, gammas, ct_rgbs, alphas)
        ):
            print(f"Processing image {i+1}/{N}...")
            result = self(img, bbox, gamma=gamma, ct_rgb=ct, alpha=alpha, **kwargs)
            results.append(result)

        return results
