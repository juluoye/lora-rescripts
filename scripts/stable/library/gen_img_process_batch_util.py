from __future__ import annotations

import library.gen_img_batch_prepare_util as gen_img_batch_prepare_util
import library.gen_img_highres_util as gen_img_highres_util
import library.gen_img_output_util as gen_img_output_util


def process_generation_batch(
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
    control_net_lllites,
    networks,
    network_default_muls,
    regional_network,
    unet,
    network_pre_calc,
    max_embeddings_multiples,
    latent_channels,
    downsampling_factor,
    is_sdxl,
    logger,
    pyramid_noise_like_fn,
):
    batch_size = len(batch)

    if highres_fix and not highres_1st:
        logger.info("process 1st stage")
        _, batch_1st = gen_img_highres_util.build_highres_first_stage_batch(
            batch,
            args=args,
            upscaler=upscaler,
            batch_data_cls=batch_data_cls,
            batch_data_ext_cls=batch_data_ext_cls,
        )

        pipe.set_enable_control_net(True)
        images_1st = process_generation_batch(
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
            control_net_lllites=control_net_lllites,
            networks=networks,
            network_default_muls=network_default_muls,
            regional_network=regional_network,
            unet=unet,
            network_pre_calc=network_pre_calc,
            max_embeddings_multiples=max_embeddings_multiples,
            latent_channels=latent_channels,
            downsampling_factor=downsampling_factor,
            is_sdxl=is_sdxl,
            logger=logger,
            pyramid_noise_like_fn=pyramid_noise_like_fn,
        )

        logger.info("process 2nd stage")
        images_1st = gen_img_highres_util.upscale_first_stage_outputs(
            images_1st,
            batch=batch_1st,
            args=args,
            upscaler=upscaler,
            vae=vae,
            dtype=dtype,
        )
        batch = gen_img_highres_util.build_highres_second_stage_batch(
            batch,
            images_1st,
            batch_data_cls=batch_data_cls,
            batch_data_base_cls=batch_data_base_cls,
        )

        if args.highres_fix_disable_control_net:
            pipe.set_enable_control_net(False)

    prepared_batch_inputs = gen_img_batch_prepare_util.prepare_process_batch_inputs(
        batch,
        latent_channels=latent_channels,
        downsampling_factor=downsampling_factor,
        scheduler_num_noises_per_step=scheduler_num_noises_per_step,
        device=device,
        dtype=dtype,
        args=args,
        noise_manager=noise_manager,
        control_nets=control_nets,
        control_net_lllites=control_net_lllites,
        pyramid_noise_like_fn=pyramid_noise_like_fn,
        logger=logger,
    )
    return_latents = prepared_batch_inputs.return_latents
    step_first = prepared_batch_inputs.step_first
    width = prepared_batch_inputs.width
    height = prepared_batch_inputs.height
    original_width = prepared_batch_inputs.original_width
    original_height = prepared_batch_inputs.original_height
    original_width_negative = prepared_batch_inputs.original_width_negative
    original_height_negative = prepared_batch_inputs.original_height_negative
    crop_left = prepared_batch_inputs.crop_left
    crop_top = prepared_batch_inputs.crop_top
    steps = prepared_batch_inputs.steps
    scale = prepared_batch_inputs.scale
    negative_scale = prepared_batch_inputs.negative_scale
    strength = prepared_batch_inputs.strength
    network_muls = prepared_batch_inputs.network_muls
    num_sub_prompts = prepared_batch_inputs.num_sub_prompts
    prompts = prepared_batch_inputs.prompts
    negative_prompts = prepared_batch_inputs.negative_prompts
    raw_prompts = prepared_batch_inputs.raw_prompts
    filenames = prepared_batch_inputs.filenames
    start_code = prepared_batch_inputs.start_code
    seeds = prepared_batch_inputs.seeds
    clip_prompts = prepared_batch_inputs.clip_prompts
    i2i_noises = prepared_batch_inputs.i2i_noises
    init_images = prepared_batch_inputs.init_images
    mask_images = prepared_batch_inputs.mask_images
    guide_images = prepared_batch_inputs.guide_images

    gen_img_batch_prepare_util.prepare_networks_for_generation(
        networks,
        network_muls=network_muls,
        network_default_muls=network_default_muls,
        regional_network=regional_network,
        batch_size=batch_size,
        num_sub_prompts=num_sub_prompts,
        width=width,
        height=height,
        unet=unet,
        network_pre_calc=network_pre_calc,
        logger=logger,
    )

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
        emb_normalize_mode=args.emb_normalize_mode,
        force_scheduler_zero_steps_offset=args.force_scheduler_zero_steps_offset,
    )
    if highres_1st and not args.highres_fix_save_1st:
        return images

    gen_img_output_util.save_generated_images(
        args,
        images=images,
        prompts=prompts,
        negative_prompts=negative_prompts,
        seeds=seeds,
        clip_prompts=clip_prompts,
        raw_prompts=raw_prompts,
        filenames=filenames,
        highres_fix=highres_fix,
        highres_1st=highres_1st,
        steps=steps,
        scale=scale,
        negative_scale=negative_scale,
        is_sdxl=is_sdxl,
        original_height=original_height,
        original_width=original_width,
        original_height_negative=original_height_negative,
        original_width_negative=original_width_negative,
        crop_top=crop_top,
        crop_left=crop_left,
        init_images=init_images,
        step_first=step_first,
        logger=logger,
    )
    gen_img_output_util.preview_generated_images(
        args,
        highres_1st=highres_1st,
        prompts=prompts,
        images=images,
        logger=logger,
    )

    return images


__all__ = ["process_generation_batch"]
