from __future__ import annotations

import PIL
import torch
from tqdm import tqdm

import library.sdxl_model_util as sdxl_model_util
from library.gen_img_preprocess_util import preprocess_image, preprocess_mask


def prepare_pipeline_latents(
    *,
    scheduler,
    device,
    unet,
    vae,
    is_sdxl,
    batch_size,
    num_images_per_prompt,
    height,
    width,
    latents,
    latents_dtype,
    generator,
    init_image,
    mask_image,
    vae_batch_size,
    num_inference_steps,
    strength,
    force_scheduler_zero_steps_offset,
    img2img_noise,
):
    init_latents_orig = None
    mask = None

    if init_image is None:
        latents_shape = (
            batch_size * num_images_per_prompt,
            unet.in_channels,
            height // 8,
            width // 8,
        )

        if latents is None:
            if device.type == "mps":
                latents = torch.randn(
                    latents_shape,
                    generator=generator,
                    device="cpu",
                    dtype=latents_dtype,
                ).to(device)
            else:
                latents = torch.randn(
                    latents_shape,
                    generator=generator,
                    device=device,
                    dtype=latents_dtype,
                )
        else:
            if latents.shape != latents_shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
            latents = latents.to(device)

        timesteps = scheduler.timesteps.to(device)
        latents = latents * scheduler.init_noise_sigma
        return latents, timesteps, init_latents_orig, mask, init_image

    if isinstance(init_image, PIL.Image.Image):
        init_image = [init_image]
    if isinstance(init_image[0], PIL.Image.Image):
        init_image = [preprocess_image(im) for im in init_image]
        init_image = torch.cat(init_image)
    if isinstance(init_image, list):
        init_image = torch.stack(init_image)

    if mask_image is not None:
        if isinstance(mask_image, PIL.Image.Image):
            mask_image = [mask_image]
        if isinstance(mask_image[0], PIL.Image.Image):
            mask_image = torch.cat([preprocess_mask(im) for im in mask_image])

    init_image = init_image.to(device=device, dtype=latents_dtype)
    if init_image.size()[-2:] == (height // 8, width // 8):
        init_latents = init_image
    else:
        if vae_batch_size >= batch_size:
            init_latent_dist = vae.encode(init_image.to(vae.dtype)).latent_dist
            init_latents = init_latent_dist.sample(generator=generator)
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            init_latents = []
            for i in tqdm(range(0, min(batch_size, len(init_image)), vae_batch_size)):
                init_latent_dist = vae.encode(
                    (init_image[i : i + vae_batch_size] if vae_batch_size > 1 else init_image[i].unsqueeze(0)).to(vae.dtype)
                ).latent_dist
                init_latents.append(init_latent_dist.sample(generator=generator))
            init_latents = torch.cat(init_latents)

        init_latents = (sdxl_model_util.VAE_SCALE_FACTOR if is_sdxl else 0.18215) * init_latents

    if len(init_latents) == 1:
        init_latents = init_latents.repeat((batch_size, 1, 1, 1))
    init_latents_orig = init_latents

    if mask_image is not None:
        mask = mask_image.to(device=device, dtype=latents_dtype)
        if len(mask) == 1:
            mask = mask.repeat((batch_size, 1, 1, 1))
        if not mask.shape == init_latents.shape:
            raise ValueError("The mask and init_image should be the same size!")

    offset = 0 if force_scheduler_zero_steps_offset else scheduler.config.get("steps_offset", 0)
    init_timestep = int(num_inference_steps * strength) + offset
    init_timestep = min(init_timestep, num_inference_steps)

    timesteps = scheduler.timesteps[-init_timestep]
    timesteps = torch.tensor([timesteps] * batch_size * num_images_per_prompt, device=device)
    latents = scheduler.add_noise(init_latents, img2img_noise, timesteps)

    t_start = max(num_inference_steps - init_timestep + offset, 0)
    timesteps = scheduler.timesteps[t_start:].to(device)

    return latents, timesteps, init_latents_orig, mask, init_image


__all__ = ["prepare_pipeline_latents"]
