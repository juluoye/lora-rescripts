from __future__ import annotations

import re
from typing import Any, NamedTuple

from library.utils import GradualLatent


class ParsedPromptOverrides(NamedTuple):
    prompt: str
    width: int
    height: int
    original_width: int
    original_height: int
    original_width_negative: int
    original_height_negative: int
    crop_top: int
    crop_left: int
    scale: float
    negative_scale: Any
    steps: int
    strength: float
    negative_prompt: str
    clip_prompt: Any
    network_muls: Any
    filename: Any
    ds_depth_1: Any
    ds_timesteps_1: Any
    ds_depth_2: Any
    ds_timesteps_2: Any
    ds_ratio: Any
    gl_timesteps: Any
    gl_ratio: Any
    gl_every_n_steps: Any
    gl_ratio_step: Any
    gl_s_noise: Any
    gl_unsharp_params: Any


def parse_prompt_overrides(raw_prompt, *, args, prompter, prompt_index: int, networks, logger):
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
    strength = 0.8 if args.strength is None else args.strength
    negative_prompt = ""
    clip_prompt = None
    network_muls = None
    filename = None

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
    length = len(prompter) if hasattr(prompter, "__len__") else 0
    logger.info(f"prompt {prompt_index+1}/{length}: {prompt}")

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

            m = re.match(r"f (.+)", parg, re.IGNORECASE)
            if m:
                filename = m.group(1)
                logger.info(f"filename: {filename}")
                continue
        except ValueError as ex:
            logger.error(f"Exception in parsing / 解析エラー: {parg}")
            logger.error(f"{ex}")

    return ParsedPromptOverrides(
        prompt=prompt,
        width=width,
        height=height,
        original_width=original_width,
        original_height=original_height,
        original_width_negative=original_width_negative,
        original_height_negative=original_height_negative,
        crop_top=crop_top,
        crop_left=crop_left,
        scale=scale,
        negative_scale=negative_scale,
        steps=steps,
        strength=strength,
        negative_prompt=negative_prompt,
        clip_prompt=clip_prompt,
        network_muls=network_muls,
        filename=filename,
        ds_depth_1=ds_depth_1,
        ds_timesteps_1=ds_timesteps_1,
        ds_depth_2=ds_depth_2,
        ds_timesteps_2=ds_timesteps_2,
        ds_ratio=ds_ratio,
        gl_timesteps=gl_timesteps,
        gl_ratio=gl_ratio,
        gl_every_n_steps=gl_every_n_steps,
        gl_ratio_step=gl_ratio_step,
        gl_s_noise=gl_s_noise,
        gl_unsharp_params=gl_unsharp_params,
    )


def apply_prompt_runtime_overrides(args, *, parsed_prompt: ParsedPromptOverrides, unet, pipe):
    if parsed_prompt.ds_depth_1 is not None:
        ds_depth_1 = parsed_prompt.ds_depth_1
        if ds_depth_1 < 0:
            ds_depth_1 = args.ds_depth_1 or 3
        unet.set_deep_shrink(
            ds_depth_1,
            parsed_prompt.ds_timesteps_1,
            parsed_prompt.ds_depth_2,
            parsed_prompt.ds_timesteps_2,
            parsed_prompt.ds_ratio,
        )

    if parsed_prompt.gl_timesteps is not None:
        gl_timesteps = parsed_prompt.gl_timesteps
        if gl_timesteps < 0:
            gl_timesteps = args.gradual_latent_timesteps or 650
        if parsed_prompt.gl_unsharp_params is not None:
            unsharp_params = parsed_prompt.gl_unsharp_params.split(",")
            us_ksize, us_sigma, us_strength = [float(v) for v in unsharp_params[:3]]
            us_target_x = True if len(unsharp_params) < 4 else bool(int(unsharp_params[3]))
            us_ksize = int(us_ksize)
        else:
            us_ksize, us_sigma, us_strength, us_target_x = None, None, None, None
        gradual_latent = GradualLatent(
            parsed_prompt.gl_ratio,
            gl_timesteps,
            parsed_prompt.gl_every_n_steps,
            parsed_prompt.gl_ratio_step,
            parsed_prompt.gl_s_noise,
            us_ksize,
            us_sigma,
            us_strength,
            us_target_x,
        )
        pipe.set_gradual_latent(gradual_latent)


def prepare_generation_seed(args, *, seeds, iter_seed, seed_random, logger, previous_seed=None):
    if seeds is not None:
        if len(seeds) > 0:
            seed = seeds.pop(0)
        else:
            seed = previous_seed
    else:
        seed = iter_seed if args.iter_same_seed else None

    if seed is None:
        seed = seed_random.randint(0, 2**32 - 1)
    if args.interactive:
        logger.info(f"seed: {seed}")
    return seed


__all__ = [
    "ParsedPromptOverrides",
    "apply_prompt_runtime_overrides",
    "parse_prompt_overrides",
    "prepare_generation_seed",
]
