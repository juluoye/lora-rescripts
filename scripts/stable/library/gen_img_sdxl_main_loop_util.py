from __future__ import annotations

import random
import re

from library.utils import GradualLatent


def run_sdxl_generation_iterations(
    *,
    args,
    prompt_list,
    predefined_seeds,
    init_images,
    mask_images,
    guide_images,
    highres_fix,
    handle_dynamic_prompt_variants_fn,
    process_batch_fn,
    batch_data_cls,
    batch_data_base_cls,
    batch_data_ext_cls,
    networks,
    control_nets,
    regional_network,
    unet,
    pipe,
    logger,
):
    for gen_iter in range(args.n_iter):
        logger.info(f"iteration {gen_iter+1}/{args.n_iter}")
        iter_seed = random.randint(0, 0x7FFFFFFF)

        prompt_index = 0
        global_step = 0
        batch_data = []
        while args.interactive or prompt_index < len(prompt_list):
            if len(prompt_list) == 0:
                valid = False
                while not valid:
                    logger.info("")
                    logger.info("Type prompt:")
                    try:
                        raw_prompt = input()
                    except EOFError:
                        break

                    valid = len(raw_prompt.strip().split(" --")[0].strip()) > 0
                if not valid:
                    break
            else:
                raw_prompt = prompt_list[prompt_index]

            raw_prompts = handle_dynamic_prompt_variants_fn(raw_prompt, args.images_per_prompt)

            for pi in range(args.images_per_prompt if len(raw_prompts) == 1 else len(raw_prompts)):
                raw_prompt = raw_prompts[pi] if len(raw_prompts) > 1 else raw_prompts[0]

                if pi == 0 or len(raw_prompts) > 1:
                    width = args.W
                    height = args.H
                    original_width = args.original_width
                    original_height = args.original_height
                    original_width_negative = args.original_width_negative
                    original_height_negative = args.original_height_negative
                    crop_top = args.crop_top
                    crop_left = args.crop_left
                    scale = args.scale
                    negative_scale = args.negative_scale
                    steps = args.steps
                    seed = None
                    seeds = None
                    strength = 0.8 if args.strength is None else args.strength
                    negative_prompt = ""
                    clip_prompt = None
                    network_muls = None

                    ds_depth_1 = None
                    ds_timesteps_1 = args.ds_timesteps_1
                    ds_depth_2 = args.ds_depth_2
                    ds_timesteps_2 = args.ds_timesteps_2
                    ds_ratio = args.ds_ratio

                    gl_timesteps = None
                    gl_ratio = args.gradual_latent_ratio
                    gl_every_n_steps = args.gradual_latent_every_n_steps
                    gl_ratio_step = args.gradual_latent_ratio_step
                    gl_s_noise = args.gradual_latent_s_noise
                    gl_unsharp_params = args.gradual_latent_unsharp_params

                    prompt_args = raw_prompt.strip().split(" --")
                    prompt = prompt_args[0]
                    logger.info(f"prompt {prompt_index+1}/{len(prompt_list)}: {prompt}")

                    for parg in prompt_args[1:]:
                        try:
                            m = re.match(r"w (\d+)", parg, re.IGNORECASE)
                            if m:
                                width = int(m.group(1))
                                logger.info(f"width: {width}")
                                continue

                            m = re.match(r"h (\d+)", parg, re.IGNORECASE)
                            if m:
                                height = int(m.group(1))
                                logger.info(f"height: {height}")
                                continue

                            m = re.match(r"ow (\d+)", parg, re.IGNORECASE)
                            if m:
                                original_width = int(m.group(1))
                                logger.info(f"original width: {original_width}")
                                continue

                            m = re.match(r"oh (\d+)", parg, re.IGNORECASE)
                            if m:
                                original_height = int(m.group(1))
                                logger.info(f"original height: {original_height}")
                                continue

                            m = re.match(r"nw (\d+)", parg, re.IGNORECASE)
                            if m:
                                original_width_negative = int(m.group(1))
                                logger.info(f"original width negative: {original_width_negative}")
                                continue

                            m = re.match(r"nh (\d+)", parg, re.IGNORECASE)
                            if m:
                                original_height_negative = int(m.group(1))
                                logger.info(f"original height negative: {original_height_negative}")
                                continue

                            m = re.match(r"ct (\d+)", parg, re.IGNORECASE)
                            if m:
                                crop_top = int(m.group(1))
                                logger.info(f"crop top: {crop_top}")
                                continue

                            m = re.match(r"cl (\d+)", parg, re.IGNORECASE)
                            if m:
                                crop_left = int(m.group(1))
                                logger.info(f"crop left: {crop_left}")
                                continue

                            m = re.match(r"s (\d+)", parg, re.IGNORECASE)
                            if m:
                                steps = max(1, min(1000, int(m.group(1))))
                                logger.info(f"steps: {steps}")
                                continue

                            m = re.match(r"d ([\d,]+)", parg, re.IGNORECASE)
                            if m:
                                seeds = [int(d) for d in m.group(1).split(",")]
                                logger.info(f"seeds: {seeds}")
                                continue

                            m = re.match(r"l ([\d\.]+)", parg, re.IGNORECASE)
                            if m:
                                scale = float(m.group(1))
                                logger.info(f"scale: {scale}")
                                continue

                            m = re.match(r"nl ([\d\.]+|none|None)", parg, re.IGNORECASE)
                            if m:
                                negative_scale = None if m.group(1).lower() == "none" else float(m.group(1))
                                logger.info(f"negative scale: {negative_scale}")
                                continue

                            m = re.match(r"t ([\d\.]+)", parg, re.IGNORECASE)
                            if m:
                                strength = float(m.group(1))
                                logger.info(f"strength: {strength}")
                                continue

                            m = re.match(r"n (.+)", parg, re.IGNORECASE)
                            if m:
                                negative_prompt = m.group(1)
                                logger.info(f"negative prompt: {negative_prompt}")
                                continue

                            m = re.match(r"c (.+)", parg, re.IGNORECASE)
                            if m:
                                clip_prompt = m.group(1)
                                logger.info(f"clip prompt: {clip_prompt}")
                                continue

                            m = re.match(r"am ([\d\.\-,]+)", parg, re.IGNORECASE)
                            if m:
                                network_muls = [float(v) for v in m.group(1).split(",")]
                                while len(network_muls) < len(networks):
                                    network_muls.append(network_muls[-1])
                                logger.info(f"network mul: {network_muls}")
                                continue

                            m = re.match(r"dsd1 ([\d\.]+)", parg, re.IGNORECASE)
                            if m:
                                ds_depth_1 = int(m.group(1))
                                logger.info(f"deep shrink depth 1: {ds_depth_1}")
                                continue

                            m = re.match(r"dst1 ([\d\.]+)", parg, re.IGNORECASE)
                            if m:
                                ds_timesteps_1 = int(m.group(1))
                                ds_depth_1 = ds_depth_1 if ds_depth_1 is not None else -1
                                logger.info(f"deep shrink timesteps 1: {ds_timesteps_1}")
                                continue

                            m = re.match(r"dsd2 ([\d\.]+)", parg, re.IGNORECASE)
                            if m:
                                ds_depth_2 = int(m.group(1))
                                ds_depth_1 = ds_depth_1 if ds_depth_1 is not None else -1
                                logger.info(f"deep shrink depth 2: {ds_depth_2}")
                                continue

                            m = re.match(r"dst2 ([\d\.]+)", parg, re.IGNORECASE)
                            if m:
                                ds_timesteps_2 = int(m.group(1))
                                ds_depth_1 = ds_depth_1 if ds_depth_1 is not None else -1
                                logger.info(f"deep shrink timesteps 2: {ds_timesteps_2}")
                                continue

                            m = re.match(r"dsr ([\d\.]+)", parg, re.IGNORECASE)
                            if m:
                                ds_ratio = float(m.group(1))
                                ds_depth_1 = ds_depth_1 if ds_depth_1 is not None else -1
                                logger.info(f"deep shrink ratio: {ds_ratio}")
                                continue

                            m = re.match(r"glt ([\d\.]+)", parg, re.IGNORECASE)
                            if m:
                                gl_timesteps = int(m.group(1))
                                logger.info(f"gradual latent timesteps: {gl_timesteps}")
                                continue

                            m = re.match(r"glr ([\d\.]+)", parg, re.IGNORECASE)
                            if m:
                                gl_ratio = float(m.group(1))
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1
                                logger.info(f"gradual latent ratio: {ds_ratio}")
                                continue

                            m = re.match(r"gle ([\d\.]+)", parg, re.IGNORECASE)
                            if m:
                                gl_every_n_steps = int(m.group(1))
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1
                                logger.info(f"gradual latent every n steps: {gl_every_n_steps}")
                                continue

                            m = re.match(r"gls ([\d\.]+)", parg, re.IGNORECASE)
                            if m:
                                gl_ratio_step = float(m.group(1))
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1
                                logger.info(f"gradual latent ratio step: {gl_ratio_step}")
                                continue

                            m = re.match(r"glsn ([\d\.]+)", parg, re.IGNORECASE)
                            if m:
                                gl_s_noise = float(m.group(1))
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1
                                logger.info(f"gradual latent s noise: {gl_s_noise}")
                                continue

                            m = re.match(r"glus ([\d\.\-,]+)", parg, re.IGNORECASE)
                            if m:
                                gl_unsharp_params = m.group(1)
                                gl_timesteps = gl_timesteps if gl_timesteps is not None else -1
                                logger.info(f"gradual latent unsharp params: {gl_unsharp_params}")
                                continue

                        except ValueError as ex:
                            logger.error(f"Exception in parsing / 解析エラー: {parg}")
                            logger.error(f"{ex}")

                    if ds_depth_1 is not None:
                        if ds_depth_1 < 0:
                            ds_depth_1 = args.ds_depth_1 or 3
                        unet.set_deep_shrink(ds_depth_1, ds_timesteps_1, ds_depth_2, ds_timesteps_2, ds_ratio)

                    if gl_timesteps is not None:
                        if gl_timesteps < 0:
                            gl_timesteps = args.gradual_latent_timesteps or 650
                        if gl_unsharp_params is not None:
                            unsharp_params = gl_unsharp_params.split(",")
                            us_ksize, us_sigma, us_strength = [float(v) for v in unsharp_params[:3]]
                            us_target_x = True if len(unsharp_params) < 4 else bool(int(unsharp_params[3]))
                            us_ksize = int(us_ksize)
                        else:
                            us_ksize, us_sigma, us_strength, us_target_x = None, None, None, None
                        pipe.set_gradual_latent(
                            GradualLatent(
                                gl_ratio,
                                gl_timesteps,
                                gl_every_n_steps,
                                gl_ratio_step,
                                gl_s_noise,
                                us_ksize,
                                us_sigma,
                                us_strength,
                                us_target_x,
                            )
                        )

                    if seeds is not None:
                        if len(seeds) > 0:
                            seed = seeds.pop(0)
                    else:
                        if predefined_seeds is not None:
                            if len(predefined_seeds) > 0:
                                seed = predefined_seeds.pop(0)
                            else:
                                logger.error("predefined seeds are exhausted")
                                seed = None
                        elif args.iter_same_seed:
                            seeds = iter_seed
                        else:
                            seed = None

                    if seed is None:
                        seed = random.randint(0, 0x7FFFFFFF)
                    if args.interactive:
                        logger.info(f"seed: {seed}")

                    init_image = mask_image = guide_image = None
                    if init_images is not None:
                        init_image = init_images[global_step % len(init_images)]
                        if not highres_fix:
                            width, height = init_image.size
                            width = width - width % 32
                            height = height - height % 32
                            if width != init_image.size[0] or height != init_image.size[1]:
                                logger.warning(
                                    "img2img image size is not divisible by 32 so aspect ratio is changed / img2imgの画像サイズが32で割り切れないためリサイズされます。画像が歪みます"
                                )

                    if mask_images is not None:
                        mask_image = mask_images[global_step % len(mask_images)]

                    if guide_images is not None:
                        if control_nets:
                            control_net_count = len(control_nets)
                            guide_index = global_step % (len(guide_images) // control_net_count)
                            guide_image = guide_images[guide_index * control_net_count : guide_index * control_net_count + control_net_count]
                        else:
                            guide_image = guide_images[global_step % len(guide_images)]

                    if regional_network:
                        num_sub_prompts = len(prompt.split(" AND "))
                        assert len(networks) <= num_sub_prompts, "Number of networks must be less than or equal to number of sub prompts."
                    else:
                        num_sub_prompts = None

                    current_batch = batch_data_cls(
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

                if len(batch_data) > 0 and batch_data[-1].ext != current_batch.ext:
                    process_batch_fn(batch_data, highres_fix)
                    batch_data.clear()

                batch_data.append(current_batch)
                if len(batch_data) == args.batch_size:
                    process_batch_fn(batch_data, highres_fix)
                    batch_data.clear()

                global_step += 1

            prompt_index += 1

        if len(batch_data) > 0:
            process_batch_fn(batch_data, highres_fix)
            batch_data.clear()


__all__ = ["run_sdxl_generation_iterations"]
