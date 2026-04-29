from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import library.sdxl_model_util as sdxl_model_util


def decode_pipeline_output(*, vae, is_sdxl, latents, vae_batch_size, batch_size, output_type):
    latents = 1 / (sdxl_model_util.VAE_SCALE_FACTOR if is_sdxl else 0.18215) * latents
    if vae_batch_size >= batch_size:
        image = vae.decode(latents.to(vae.dtype)).sample
    else:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        images = []
        for i in tqdm(range(0, batch_size, vae_batch_size)):
            images.append(
                vae.decode(
                    (latents[i : i + vae_batch_size] if vae_batch_size > 1 else latents[i].unsqueeze(0)).to(vae.dtype)
                ).sample
            )
        image = torch.cat(images)

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if output_type == "pil":
        image = (image * 255).round().astype("uint8")
        image = [Image.fromarray(im) for im in image]

    return image


__all__ = ["decode_pipeline_output"]
