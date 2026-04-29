from __future__ import annotations

import os
import time

import numpy as np
import PIL
import torch
from PIL.PngImagePlugin import PngInfo


def process_sdxl_generation_batch(
    batch,
    *,
    highres_fix,
    highres_1st=False,
    args,
    upscaler,
    batch_data_cls,
    batch_data_base_cls,
    batch_data_ext_cls,
    pipe,
    vae,
    dtype,
    device,
    scheduler_num_noises_per_step,
    noise_manager,
    control_nets,
    networks,
    network_default_muls,
    regional_network,
    network_pre_calc,
    max_embeddings_multiples,
    latent_channels,
    downsampling_factor,
    logger,
):
    batch_size = len(batch)

    if highres_fix and not highres_1st:
        is_1st_latent = upscaler.support_latents() if upscaler else args.highres_fix_latents_upscaling

        logger.info("process 1st stage")
        batch_1st = []
        for _, base, ext in batch:
            def scale_and_round(value):
                if value is None:
                    return None
                return int(value * args.highres_fix_scale + 0.5)

            width_1st = scale_and_round(ext.width)
            height_1st = scale_and_round(ext.height)
            width_1st = width_1st - width_1st % 32
            height_1st = height_1st - height_1st % 32

            ext_1st = batch_data_ext_cls(
                width_1st,
                height_1st,
                scale_and_round(ext.original_width),
                scale_and_round(ext.original_height),
                scale_and_round(ext.original_width_negative),
                scale_and_round(ext.original_height_negative),
                scale_and_round(ext.crop_left),
                scale_and_round(ext.crop_top),
                args.highres_fix_steps,
                ext.scale,
                ext.negative_scale,
                ext.strength if args.highres_fix_strength is None else args.highres_fix_strength,
                ext.network_muls,
                ext.num_sub_prompts,
            )
            batch_1st.append(batch_data_cls(is_1st_latent, base, ext_1st))

        pipe.set_enable_control_net(True)
        images_1st = process_sdxl_generation_batch(
            batch_1st,
            highres_fix=True,
            highres_1st=True,
            args=args,
            upscaler=upscaler,
            batch_data_cls=batch_data_cls,
            batch_data_base_cls=batch_data_base_cls,
            batch_data_ext_cls=batch_data_ext_cls,
            pipe=pipe,
            vae=vae,
            dtype=dtype,
            device=device,
            scheduler_num_noises_per_step=scheduler_num_noises_per_step,
            noise_manager=noise_manager,
            control_nets=control_nets,
            networks=networks,
            network_default_muls=network_default_muls,
            regional_network=regional_network,
            network_pre_calc=network_pre_calc,
            max_embeddings_multiples=max_embeddings_multiples,
            latent_channels=latent_channels,
            downsampling_factor=downsampling_factor,
            logger=logger,
        )

        logger.info("process 2nd stage")
        width_2nd, height_2nd = batch[0].ext.width, batch[0].ext.height

        if upscaler:
            lowreso_imgs = None if is_1st_latent else images_1st
            lowreso_latents = None if not is_1st_latent else images_1st

            vae_batch_size = (
                batch_size
                if args.vae_batch_size is None
                else (max(1, int(batch_size * args.vae_batch_size)) if args.vae_batch_size < 1 else args.vae_batch_size)
            )
            vae_batch_size = int(vae_batch_size)
            images_1st = upscaler.upscale(vae, lowreso_imgs, lowreso_latents, dtype, width_2nd, height_2nd, batch_size, vae_batch_size)
        elif args.highres_fix_latents_upscaling:
            original_dtype = images_1st.dtype
            if images_1st.dtype == torch.bfloat16:
                images_1st = images_1st.to(torch.float)
            images_1st = torch.nn.functional.interpolate(
                images_1st,
                (batch[0].ext.height // 8, batch[0].ext.width // 8),
                mode="bilinear",
            )
            images_1st = images_1st.to(original_dtype)
        else:
            images_1st = [image.resize((width_2nd, height_2nd), resample=PIL.Image.LANCZOS) for image in images_1st]

        batch = [
            batch_data_cls(
                False,
                batch_data_base_cls(*bd.base[0:3], bd.base.seed + 1, image, None, *bd.base[6:]),
                bd.ext,
            )
            for bd, image in zip(batch, images_1st)
        ]

        if args.highres_fix_disable_control_net:
            pipe.set_enable_control_net(False)

    (
        return_latents,
        (step_first, _, _, _, init_image, mask_image, _, guide_image, _),
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

    prompts = []
    negative_prompts = []
    raw_prompts = []
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

    for index, (_, (_, prompt, negative_prompt, seed, init_image, mask_image, clip_prompt, guide_image, raw_prompt), _) in enumerate(batch):
        prompts.append(prompt)
        negative_prompts.append(negative_prompt)
        seeds.append(seed)
        clip_prompts.append(clip_prompt)
        raw_prompts.append(raw_prompt)

        if init_image is not None:
            init_images.append(init_image)
            if index > 0 and all_images_are_same:
                all_images_are_same = init_images[-2] is init_image

        if mask_image is not None:
            mask_images.append(mask_image)
            if index > 0 and all_masks_are_same:
                all_masks_are_same = mask_images[-2] is mask_image

        if guide_image is not None:
            if type(guide_image) is list:
                guide_images.extend(guide_image)
                all_guide_images_are_same = False
            else:
                guide_images.append(guide_image)
                if index > 0 and all_guide_images_are_same:
                    all_guide_images_are_same = guide_images[-2] is guide_image

        torch.manual_seed(seed)
        start_code[index] = torch.randn(noise_shape, device=device, dtype=dtype)

        for noise_index in range(steps * scheduler_num_noises_per_step):
            noises[noise_index][index] = torch.randn(noise_shape, device=device, dtype=dtype)

        if i2i_noises is not None:
            i2i_noises[index] = torch.randn(noise_shape, device=device, dtype=dtype)

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
        original_height,
        original_width,
        original_height_negative,
        original_width_negative,
        crop_top,
        crop_left,
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
    )
    if highres_1st and not args.highres_fix_save_1st:
        return images

    highres_prefix = ("0" if highres_1st else "1") if highres_fix else ""
    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    for index, (image, prompt, negative_prompt, seed, clip_prompt, raw_prompt) in enumerate(
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
        metadata.add_text("original-height", str(original_height))
        metadata.add_text("original-width", str(original_width))
        metadata.add_text("original-height-negative", str(original_height_negative))
        metadata.add_text("original-width-negative", str(original_width_negative))
        metadata.add_text("crop-top", str(crop_top))
        metadata.add_text("crop-left", str(crop_left))

        if args.use_original_file_name and init_images is not None:
            if type(init_images) is list:
                filename = os.path.splitext(os.path.basename(init_images[index % len(init_images)].filename))[0] + ".png"
            else:
                filename = os.path.splitext(os.path.basename(init_images.filename))[0] + ".png"
        elif args.sequential_file_name:
            filename = f"im_{highres_prefix}{step_first + index + 1:06d}.png"
        else:
            filename = f"im_{ts_str}_{highres_prefix}{index:03d}_{seed}.png"

        image.save(os.path.join(args.outdir, filename), pnginfo=metadata)

    if not args.no_preview and not highres_1st and args.interactive:
        try:
            import cv2

            for prompt, image in zip(prompts, images):
                cv2.imshow(prompt[:128], np.array(image)[:, :, ::-1])
                cv2.waitKey()
                cv2.destroyAllWindows()
        except ImportError:
            logger.error(
                "opencv-python is not installed, cannot preview / opencv-pythonがインストールされていないためプレビューできません"
            )

    return images


__all__ = ["process_sdxl_generation_batch"]
