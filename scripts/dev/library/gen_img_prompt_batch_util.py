from __future__ import annotations


def build_prompt_batch_entry(
    *,
    global_step,
    prompt,
    negative_prompt,
    seed,
    raw_prompt,
    filename,
    clip_prompt,
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
    highres_fix,
    init_images,
    mask_images,
    guide_images,
    control_nets,
    control_net_lllites,
    regional_network,
    networks,
    batch_data_cls,
    batch_data_base_cls,
    batch_data_ext_cls,
    logger,
):
    init_image = mask_image = guide_image = None

    if init_images is not None:
        init_image = init_images[global_step % len(init_images)]

        # In img2img mode we follow the source image size, rounded down to a multiple of 32.
        if not highres_fix:
            width, height = init_image.size
            width = width - width % 32
            height = height - height % 32
            if width != init_image.size[0] or height != init_image.size[1]:
                logger.warning(
                    "img2img image size is not divisible by 32 so aspect ratio is changed / "
                    "img2imgの画像サイズが32で割り切れないためリサイズされます。画像が歪みます"
                )

    if mask_images is not None:
        mask_image = mask_images[global_step % len(mask_images)]

    if guide_images is not None:
        if control_nets or control_net_lllites:
            control_count = max(len(control_nets), len(control_net_lllites))
            guide_index = global_step % (len(guide_images) // control_count)
            guide_image = guide_images[guide_index * control_count : guide_index * control_count + control_count]
        else:
            guide_image = guide_images[global_step % len(guide_images)]

    if regional_network:
        num_sub_prompts = len(prompt.split(" AND "))
        assert len(networks) <= num_sub_prompts, "Number of networks must be less than or equal to number of sub prompts."
    else:
        num_sub_prompts = None

    return batch_data_cls(
        False,
        batch_data_base_cls(
            global_step,
            prompt,
            negative_prompt,
            seed,
            init_image,
            mask_image,
            clip_prompt,
            guide_image,
            raw_prompt,
            filename,
        ),
        batch_data_ext_cls(
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
            tuple(network_muls) if network_muls else None,
            num_sub_prompts,
        ),
    )


__all__ = ["build_prompt_batch_entry"]
