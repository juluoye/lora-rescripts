from __future__ import annotations

import random
from typing import Any, NamedTuple

import PIL
import torch


class PreparedProcessBatchInputs(NamedTuple):
    return_latents: bool
    step_first: int
    width: int
    height: int
    original_width: int
    original_height: int
    original_width_negative: int
    original_height_negative: int
    crop_left: int
    crop_top: int
    steps: int
    scale: float
    negative_scale: float
    strength: float
    network_muls: Any
    num_sub_prompts: int
    prompts: list[str]
    negative_prompts: list[str]
    raw_prompts: list[str]
    filenames: list[str]
    start_code: Any
    seeds: list[int]
    clip_prompts: list[Any]
    i2i_noises: Any
    init_images: Any
    mask_images: Any
    guide_images: Any


def prepare_process_batch_inputs(
    batch,
    *,
    latent_channels: int,
    downsampling_factor: int,
    scheduler_num_noises_per_step: int,
    device,
    dtype,
    args,
    noise_manager,
    control_nets,
    control_net_lllites,
    pyramid_noise_like_fn,
    logger,
):
    (
        return_latents,
        (step_first, _, _, _, init_image, mask_image, _, guide_image, _, _),
        (
            width,
            height,
            original_width,
            original_height,
            original_width_negative,
            original_height_negative,
            crop_left,
            crop_top,
            steps,
            scale,
            negative_scale,
            strength,
            network_muls,
            num_sub_prompts,
        ),
    ) = batch[0]
    noise_shape = (latent_channels, height // downsampling_factor, width // downsampling_factor)
    batch_size = len(batch)

    prompts = []
    negative_prompts = []
    raw_prompts = []
    filenames = []
    start_code = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
    noises = [torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype) for _ in range(steps * scheduler_num_noises_per_step)]
    seeds = []
    clip_prompts = []

    if init_image is not None:
        i2i_noises = torch.zeros((batch_size, *noise_shape), device=device, dtype=dtype)
        init_images = []
        mask_images = [] if mask_image is not None else None
    else:
        i2i_noises = None
        init_images = None
        mask_images = None

    guide_images = [] if guide_image is not None else None

    all_images_are_same = True
    all_masks_are_same = True
    all_guide_images_are_same = True
    for i, (_, (_, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image, raw_prompt, filename), _) in enumerate(batch):
        prompts.append(prompt)
        negative_prompts.append(negative_prompt)
        seeds.append(seed)
        clip_prompts.append(clip_prompt)
        raw_prompts.append(raw_prompt)
        filenames.append(filename)

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

        if args.pyramid_noise_prob is not None and random.random() < args.pyramid_noise_prob:
            min_discount, max_discount = args.pyramid_noise_discount_range
            discount = torch.rand(1, device=device, dtype=dtype) * (max_discount - min_discount) + min_discount
            logger.info(f"apply pyramid noise to start code: {start_code[i].shape}, discount: {discount.item()}")
            start_code[i] = pyramid_noise_like_fn(start_code[i].unsqueeze(0), device=device, discount=discount).squeeze(0)

        if args.noise_offset_prob is not None and random.random() < args.noise_offset_prob:
            min_offset, max_offset = args.noise_offset_range
            noise_offset = torch.randn(1, device=device, dtype=dtype) * (max_offset - min_offset) + min_offset
            logger.info(f"apply noise offset to start code: {start_code[i].shape}, offset: {noise_offset.item()}")
            start_code[i] += noise_offset

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

    if control_nets or control_net_lllites:
        guide_images = guide_images if type(guide_images) == list else [guide_images]
        guide_images = [image.resize((width, height), resample=PIL.Image.LANCZOS) for image in guide_images]
        if len(guide_images) == 1:
            guide_images = guide_images[0]

    return PreparedProcessBatchInputs(
        return_latents=return_latents,
        step_first=step_first,
        width=width,
        height=height,
        original_width=original_width,
        original_height=original_height,
        original_width_negative=original_width_negative,
        original_height_negative=original_height_negative,
        crop_left=crop_left,
        crop_top=crop_top,
        steps=steps,
        scale=scale,
        negative_scale=negative_scale,
        strength=strength,
        network_muls=network_muls,
        num_sub_prompts=num_sub_prompts,
        prompts=prompts,
        negative_prompts=negative_prompts,
        raw_prompts=raw_prompts,
        filenames=filenames,
        start_code=start_code,
        seeds=seeds,
        clip_prompts=clip_prompts,
        i2i_noises=i2i_noises,
        init_images=init_images,
        mask_images=mask_images,
        guide_images=guide_images,
    )


def prepare_networks_for_generation(
    networks,
    *,
    network_muls,
    network_default_muls,
    regional_network: bool,
    batch_size: int,
    num_sub_prompts: int,
    width: int,
    height: int,
    unet,
    network_pre_calc: bool,
    logger,
):
    if not networks:
        return

    shared = {}
    for network, multiplier in zip(networks, network_muls if network_muls else network_default_muls):
        network.set_multiplier(multiplier)
        if regional_network:
            network.set_current_generation(batch_size, num_sub_prompts, width, height, shared, unet.ds_ratio)

    if not regional_network and network_pre_calc:
        for network in networks:
            network.restore_weights()
        for network in networks:
            network.pre_calculation()
        logger.info("pre-calculation... done")


__all__ = [
    "PreparedProcessBatchInputs",
    "prepare_networks_for_generation",
    "prepare_process_batch_inputs",
]
