from __future__ import annotations

import os
import time

import numpy as np
import PIL
import torch
from PIL.PngImagePlugin import PngInfo


def process_diffusers_generation_batch(
    batch,
    *,
    highres_fix,
    highres_1st,
    args,
    batch_size,
    latent_channels,
    downsampling_factor,
    device,
    dtype,
    scheduler_num_noises_per_step,
    noise_manager,
    control_nets,
    networks,
    network_default_muls,
    regional_network,
    network_pre_calc,
    pipe,
    max_embeddings_multiples,
    init_images_ref,
    logger,
):
    (
        return_latents,
        (step_first, _, _, _, init_image, mask_image, _, guide_image, _),
        (width, height, steps, scale, negative_scale, strength, network_muls, num_sub_prompts),
    ) = batch[0]
    noise_shape = (latent_channels, height // downsampling_factor, width // downsampling_factor)

    prompts = []
    negative_prompts = []
    raw_prompts = []
    start_code = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
    noises = [
        torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
        for _ in range(steps * scheduler_num_noises_per_step)
    ]
    seeds = []
    clip_prompts = []

    if init_image is not None:
        i2i_noises = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
        init_images = []

        if mask_image is not None:
            mask_images = []
        else:
            mask_images = None
    else:
        i2i_noises = None
        init_images = None
        mask_images = None

    if guide_image is not None:
        guide_images = []
    else:
        guide_images = None

    all_images_are_same = True
    all_masks_are_same = True
    all_guide_images_are_same = True
    for i, (
        _,
        (_, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image, raw_prompt),
        _,
    ) in enumerate(batch):
        prompts.append(prompt)
        negative_prompts.append(negative_prompt)
        seeds.append(seed)
        clip_prompts.append(clip_prompt)
        raw_prompts.append(raw_prompt)

        if init_image is not None:
            init_images.append(init_image)
            if i > 0 and all_images_are_same:
                all_images_are_same = init_images[-2] is init_image

        if mask_image is not None:
            mask_images.append(mask_image)
            if i > 0 and all_masks_are_same:
                all_masks_are_same = mask_images[-2] is mask_image

        if guide_image is not None:
            if type(guide_image) is list:
                guide_images.extend(guide_image)
                all_guide_images_are_same = False
            else:
                guide_images.append(guide_image)
                if i > 0 and all_guide_images_are_same:
                    all_guide_images_are_same = guide_images[-2] is guide_image

        torch.manual_seed(seed)
        start_code[i] = torch.randn(noise_shape, device=device, dtype=dtype)

        for j in range(steps * scheduler_num_noises_per_step):
            noises[j][i] = torch.randn(noise_shape, device=device, dtype=dtype)

        if i2i_noises is not None:
            i2i_noises[i] = torch.randn(noise_shape, device=device, dtype=dtype)

    noise_manager.reset_sampler_noises(noises)

    if init_images is not None and all_images_are_same:
        init_images = init_images[0]
    if mask_images is not None and all_masks_are_same:
        mask_images = mask_images[0]
    if guide_images is not None and all_guide_images_are_same:
        guide_images = guide_images[0]

    if control_nets:
        guide_images = guide_images if type(guide_images) == list else [guide_images]
        guide_images = [image.resize((width, height), resample=PIL.Image.LANCZOS) for image in guide_images]
        if len(guide_images) == 1:
            guide_images = guide_images[0]

    if networks:
        shared = {}
        for network, network_mul in zip(networks, network_muls if network_muls else network_default_muls):
            network.set_multiplier(network_mul)
            if regional_network:
                network.set_current_generation(batch_size, num_sub_prompts, width, height, shared)

        if not regional_network and network_pre_calc:
            for network in networks:
                network.restore_weights()
            for network in networks:
                network.pre_calculation()
            logger.info("pre-calculation... done")

    images = pipe(
        prompts,
        negative_prompts,
        init_images,
        mask_images,
        height,
        width,
        steps,
        scale,
        negative_scale,
        strength,
        latents=start_code,
        output_type="pil",
        max_embeddings_multiples=max_embeddings_multiples,
        img2img_noise=i2i_noises,
        vae_batch_size=args.vae_batch_size,
        return_latents=return_latents,
        clip_prompts=clip_prompts,
        clip_guide_images=guide_images,
    )[0]
    if highres_1st and not args.highres_fix_save_1st:
        return images

    highres_prefix = ("0" if highres_1st else "1") if highres_fix else ""
    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    for i, (image, prompt, negative_prompt, seed, clip_prompt, raw_prompt) in enumerate(
        zip(images, prompts, negative_prompts, seeds, clip_prompts, raw_prompts)
    ):
        if highres_fix:
            seed -= 1
        metadata = PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("seed", str(seed))
        metadata.add_text("sampler", args.sampler)
        metadata.add_text("steps", str(steps))
        metadata.add_text("scale", str(scale))
        if negative_prompt is not None:
            metadata.add_text("negative-prompt", negative_prompt)
        if negative_scale is not None:
            metadata.add_text("negative-scale", str(negative_scale))
        if clip_prompt is not None:
            metadata.add_text("clip-prompt", clip_prompt)
        if raw_prompt is not None:
            metadata.add_text("raw-prompt", raw_prompt)

        if args.use_original_file_name and init_images_ref is not None:
            if type(init_images_ref) is list:
                filename = os.path.splitext(os.path.basename(init_images_ref[i % len(init_images_ref)].filename))[0] + ".png"
            else:
                filename = os.path.splitext(os.path.basename(init_images_ref.filename))[0] + ".png"
        elif args.sequential_file_name:
            filename = f"im_{highres_prefix}{step_first + i + 1:06d}.png"
        else:
            filename = f"im_{ts_str}_{highres_prefix}{i:03d}_{seed}.png"

        image.save(os.path.join(args.outdir, filename), pnginfo=metadata)

    if not args.no_preview and not highres_1st and args.interactive:
        try:
            import cv2

            for prompt, image in zip(prompts, images):
                cv2.imshow(prompt[:128], np.array(image)[:, :, ::-1])
                cv2.waitKey()
                cv2.destroyAllWindows()
        except ImportError:
            logger.info(
                "opencv-python is not installed, cannot preview / opencv-pythonがインストールされていないためプレビューできません"
            )

    return images


__all__ = ["process_diffusers_generation_batch"]
