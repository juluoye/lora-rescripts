from __future__ import annotations

from typing import Any, NamedTuple

import library.gen_img_prompt_batch_util as gen_img_prompt_batch_util
import library.gen_img_prompt_parse_util as gen_img_prompt_parse_util


class PromptRenderState(NamedTuple):
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


class PreparedPromptIteration(NamedTuple):
    state: PromptRenderState
    seed: int
    batch_entry: Any


def prepare_prompt_iteration(
    raw_prompt,
    *,
    should_reparse: bool,
    prompt_state: PromptRenderState | None,
    args,
    prompter,
    prompt_index: int,
    networks,
    logger,
    unet,
    pipe,
    seeds,
    iter_seed,
    seed_random,
    previous_seed,
    global_step: int,
    highres_fix,
    init_images,
    mask_images,
    guide_images,
    control_nets,
    control_net_lllites,
    regional_network,
    batch_data_cls,
    batch_data_base_cls,
    batch_data_ext_cls,
):
    filename = None

    if should_reparse:
        parsed_prompt = gen_img_prompt_parse_util.parse_prompt_overrides(
            raw_prompt,
            args=args,
            prompter=prompter,
            prompt_index=prompt_index,
            networks=networks,
            logger=logger,
        )
        prompt_state = PromptRenderState(
            prompt=parsed_prompt.prompt,
            width=parsed_prompt.width,
            height=parsed_prompt.height,
            original_width=parsed_prompt.original_width,
            original_height=parsed_prompt.original_height,
            original_width_negative=parsed_prompt.original_width_negative,
            original_height_negative=parsed_prompt.original_height_negative,
            crop_top=parsed_prompt.crop_top,
            crop_left=parsed_prompt.crop_left,
            scale=parsed_prompt.scale,
            negative_scale=parsed_prompt.negative_scale,
            steps=parsed_prompt.steps,
            strength=parsed_prompt.strength,
            negative_prompt=parsed_prompt.negative_prompt,
            clip_prompt=parsed_prompt.clip_prompt,
            network_muls=parsed_prompt.network_muls,
        )
        filename = parsed_prompt.filename

        gen_img_prompt_parse_util.apply_prompt_runtime_overrides(
            args,
            parsed_prompt=parsed_prompt,
            unet=unet,
            pipe=pipe,
        )
    elif prompt_state is None:
        raise ValueError("prompt_state must be provided when prompt parsing is skipped")

    seed = gen_img_prompt_parse_util.prepare_generation_seed(
        args,
        seeds=seeds,
        iter_seed=iter_seed,
        seed_random=seed_random,
        logger=logger,
        previous_seed=previous_seed,
    )

    batch_entry = gen_img_prompt_batch_util.build_prompt_batch_entry(
        global_step=global_step,
        prompt=prompt_state.prompt,
        negative_prompt=prompt_state.negative_prompt,
        seed=seed,
        raw_prompt=raw_prompt,
        filename=filename,
        clip_prompt=prompt_state.clip_prompt,
        width=prompt_state.width,
        height=prompt_state.height,
        original_width=prompt_state.original_width,
        original_height=prompt_state.original_height,
        original_width_negative=prompt_state.original_width_negative,
        original_height_negative=prompt_state.original_height_negative,
        crop_left=prompt_state.crop_left,
        crop_top=prompt_state.crop_top,
        steps=prompt_state.steps,
        scale=prompt_state.scale,
        negative_scale=prompt_state.negative_scale,
        strength=prompt_state.strength,
        network_muls=prompt_state.network_muls,
        highres_fix=highres_fix,
        init_images=init_images,
        mask_images=mask_images,
        guide_images=guide_images,
        control_nets=control_nets,
        control_net_lllites=control_net_lllites,
        regional_network=regional_network,
        networks=networks,
        batch_data_cls=batch_data_cls,
        batch_data_base_cls=batch_data_base_cls,
        batch_data_ext_cls=batch_data_ext_cls,
        logger=logger,
    )

    return PreparedPromptIteration(prompt_state, seed, batch_entry)


__all__ = ["PreparedPromptIteration", "PromptRenderState", "prepare_prompt_iteration"]
