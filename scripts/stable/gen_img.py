import itertools
import json
from typing import Any, List, NamedTuple, Optional, Tuple, Union, Callable
import glob
import importlib
import inspect
import time
import zipfile
from diffusers.utils import deprecate
from diffusers.configuration_utils import FrozenDict
import argparse
import math
import os
import random
import re

import diffusers
import numpy as np
import torch

from library.device_utils import init_ipex
from library.strategy_sd import SdTokenizeStrategy

init_ipex()

import torchvision
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    # UNet2DConditionModel,
    StableDiffusionPipeline,
)
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
from accelerate import init_empty_weights
import PIL
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import library.model_util as model_util
import library.train_util as train_util
import library.sdxl_model_util as sdxl_model_util
import library.sdxl_train_util as sdxl_train_util
from library.gen_img_attention_patch_util import replace_unet_modules, replace_vae_modules
import library.gen_img_input_util as gen_img_input_util
import library.gen_img_dynamic_prompt_util as gen_img_dynamic_prompt_util
import library.gen_img_model_setup_util as gen_img_model_setup_util
from library.gen_img_main_loop_util import run_generation_iterations
from library.gen_img_pipeline_conditioning_util import prepare_clip_guidance_images, prepare_sdxl_vector_embeddings
from library.gen_img_pipeline_control_util import prepare_control_conditions, prepare_scheduler_step_kwargs
from library.gen_img_pipeline_decode_util import decode_pipeline_output
from library.gen_img_pipeline_gradual_latent_util import prepare_gradual_latent_state
from library.gen_img_pipeline_latent_util import prepare_pipeline_latents
from library.gen_img_pipeline_loop_util import run_pipeline_denoising_loop
from library.gen_img_pipeline_text_util import prepare_pipeline_text_embeddings
import library.gen_img_pipeline_setup_util as gen_img_pipeline_setup_util
from library.gen_img_preprocess_util import preprocess_image, preprocess_mask
from library.gen_img_parser_util import setup_parser
import library.gen_img_prompt_iteration_util as gen_img_prompt_iteration_util
import library.gen_img_prompt_runtime_util as gen_img_prompt_runtime_util
import library.gen_img_process_batch_util as gen_img_process_batch_util
import library.gen_img_scheduler_util as gen_img_scheduler_util
import library.gen_img_textual_inversion_util as gen_img_textual_inversion_util
from library.gen_img_types_util import BatchData, BatchDataBase, BatchDataExt, ListPrompter
from networks.lora import LoRANetwork
import tools.original_control_net as original_control_net
from tools.original_control_net import ControlNetInfo
from library.original_unet import UNet2DConditionModel, InferUNet2DConditionModel
from library.sdxl_original_unet import InferSdxlUNet2DConditionModel
from library.sdxl_original_control_net import SdxlControlNet
from library.original_unet import FlashAttentionFunction
from library.custom_train_functions import pyramid_noise_like
from networks.control_net_lllite import ControlNetLLLite
from library.utils import GradualLatent, EulerAncestralDiscreteSchedulerGL
from library.utils import setup_logging, add_logging_arguments

setup_logging()
import logging

logger = logging.getLogger(__name__)

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

# その他の設定
LATENT_CHANNELS = 4
DOWNSAMPLING_FACTOR = 8

CLIP_VISION_MODEL = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"

# region モジュール入れ替え部
"""
高速化のためのモジュール入れ替え
"""


# endregion

# region 画像生成の本体：lpw_stable_diffusion.py （ASL）からコピーして修正
# https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion.py
# Pipelineだけ独立して使えないのと機能追加するのとでコピーして修正


class PipelineLike:
    def __init__(
        self,
        is_sdxl,
        device,
        vae: AutoencoderKL,
        text_encoders: List[CLIPTextModel],
        tokenizers: List[CLIPTokenizer],
        unet: InferSdxlUNet2DConditionModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        clip_skip: int,
    ):
        super().__init__()
        self.is_sdxl = is_sdxl
        self.device = device
        self.clip_skip = clip_skip

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.vae = vae
        self.text_encoders = text_encoders
        self.tokenizers = tokenizers
        self.unet: Union[InferUNet2DConditionModel, InferSdxlUNet2DConditionModel] = unet
        self.scheduler = scheduler
        self.safety_checker = None

        self.clip_vision_model: CLIPVisionModelWithProjection = None
        self.clip_vision_processor: CLIPImageProcessor = None
        self.clip_vision_strength = 0.0

        # Textual Inversion
        self.token_replacements_list = []
        for _ in range(len(self.text_encoders)):
            self.token_replacements_list.append({})

        # ControlNet
        self.control_nets: List[Union[ControlNetInfo, Tuple[SdxlControlNet, float]]] = []
        self.control_net_lllites: List[Tuple[ControlNetLLLite, float]] = []
        self.control_net_enabled = True  # control_netsが空ならTrueでもFalseでもControlNetは動作しない

        self.gradual_latent: GradualLatent = None

    # Textual Inversion
    def add_token_replacement(self, text_encoder_index, target_token_id, rep_token_ids):
        self.token_replacements_list[text_encoder_index][target_token_id] = rep_token_ids

    def set_enable_control_net(self, en: bool):
        self.control_net_enabled = en

    def get_token_replacer(self, tokenizer):
        tokenizer_index = self.tokenizers.index(tokenizer)
        token_replacements = self.token_replacements_list[tokenizer_index]

        def replace_tokens(tokens):
            # print("replace_tokens", tokens, "=>", token_replacements)
            if isinstance(tokens, torch.Tensor):
                tokens = tokens.tolist()

            new_tokens = []
            for token in tokens:
                if token in token_replacements:
                    replacement = token_replacements[token]
                    new_tokens.extend(replacement)
                else:
                    new_tokens.append(token)
            return new_tokens

        return replace_tokens

    def set_control_nets(self, ctrl_nets):
        self.control_nets = ctrl_nets

    def set_control_net_lllites(self, ctrl_net_lllites):
        self.control_net_lllites = ctrl_net_lllites

    def set_gradual_latent(self, gradual_latent):
        if gradual_latent is None:
            logger.info("gradual_latent is disabled")
            self.gradual_latent = None
        else:
            logger.info(f"gradual_latent is enabled: {gradual_latent}")
            self.gradual_latent = gradual_latent  # (ds_ratio, start_timesteps, every_n_steps, ratio_step)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        init_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        height: int = 1024,
        width: int = 1024,
        original_height: int = None,
        original_width: int = None,
        original_height_negative: int = None,
        original_width_negative: int = None,
        crop_top: int = 0,
        crop_left: int = 0,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_scale: float = None,
        strength: float = 0.8,
        # num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        vae_batch_size: float = None,
        return_latents: bool = False,
        # return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: Optional[int] = 1,
        img2img_noise=None,
        clip_guide_images=None,
        emb_normalize_mode: str = "original",
        force_scheduler_zero_steps_offset: bool = False,
        **kwargs,
    ):
        # TODO support secondary prompt
        num_images_per_prompt = 1  # fixed because already prompt is repeated

        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        regional_network = " AND " in prompt[0]

        vae_batch_size = (
            batch_size
            if vae_batch_size is None
            else (int(vae_batch_size) if vae_batch_size >= 1 else max(1, int(batch_size * vae_batch_size)))
        )

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type" f" {type(callback_steps)}."
            )

        # get prompt text embeddings

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if not do_classifier_free_guidance and negative_scale is not None:
            logger.warning(f"negative_scale is ignored if guidance scalle <= 1.0")
            negative_scale = None

        text_embeddings, text_pool, uncond_pool = prepare_pipeline_text_embeddings(
            is_sdxl=self.is_sdxl,
            tokenizers=self.tokenizers,
            text_encoders=self.text_encoders,
            get_token_replacer_fn=self.get_token_replacer,
            prompt=prompt,
            negative_prompt=negative_prompt,
            batch_size=batch_size,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_scale=negative_scale,
            max_embeddings_multiples=max_embeddings_multiples,
            clip_skip=self.clip_skip,
            device=self.device,
            emb_normalize_mode=emb_normalize_mode,
            extra_kwargs=kwargs,
        )

        clip_guide_images = prepare_clip_guidance_images(
            clip_guide_images,
            has_control_net_lllite=bool(self.control_net_lllites),
            has_sdxl_control_net=bool(self.control_nets and self.is_sdxl),
            device=self.device,
            dtype=text_embeddings.dtype,
        )

        if self.is_sdxl:
            vector_embeddings, text_pool, uncond_pool = prepare_sdxl_vector_embeddings(
                device=self.device,
                dtype=text_embeddings.dtype,
                batch_size=batch_size,
                height=height,
                width=width,
                original_height=original_height,
                original_width=original_width,
                original_height_negative=original_height_negative,
                original_width_negative=original_width_negative,
                crop_top=crop_top,
                crop_left=crop_left,
                regional_network=regional_network,
                text_pool=text_pool,
                uncond_pool=uncond_pool,
                init_image=init_image,
                clip_vision_model=self.clip_vision_model,
                clip_vision_processor=self.clip_vision_processor,
                clip_vision_strength=self.clip_vision_strength,
                do_classifier_free_guidance=do_classifier_free_guidance,
            )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, self.device)

        latents_dtype = text_embeddings.dtype
        latents, timesteps, init_latents_orig, mask, init_image = prepare_pipeline_latents(
            scheduler=self.scheduler,
            device=self.device,
            unet=self.unet,
            vae=self.vae,
            is_sdxl=self.is_sdxl,
            batch_size=batch_size,
            num_images_per_prompt=num_images_per_prompt,
            height=height,
            width=width,
            latents=latents,
            latents_dtype=latents_dtype,
            generator=generator,
            init_image=init_image,
            mask_image=mask_image,
            vae_batch_size=vae_batch_size,
            num_inference_steps=num_inference_steps,
            strength=strength,
            force_scheduler_zero_steps_offset=force_scheduler_zero_steps_offset,
            img2img_noise=img2img_noise,
        )

        extra_step_kwargs = prepare_scheduler_step_kwargs(self.scheduler, eta)

        num_latent_input = (3 if negative_scale is not None else 2) if do_classifier_free_guidance else 1

        control_state = prepare_control_conditions(
            control_nets=self.control_nets,
            control_net_lllites=self.control_net_lllites,
            control_net_enabled=self.control_net_enabled,
            is_sdxl=self.is_sdxl,
            clip_guide_images=clip_guide_images,
            num_latent_input=num_latent_input,
            batch_size=batch_size,
        )
        guided_hints = control_state.guided_hints
        each_control_net_enabled = control_state.each_control_net_enabled
        clip_guide_images = control_state.clip_guide_images

        gradual_latent_state = prepare_gradual_latent_state(
            scheduler=self.scheduler,
            gradual_latent=self.gradual_latent,
            latents=latents,
        )
        latents = gradual_latent_state.latents

        latents, each_control_net_enabled, gradual_latent_state = run_pipeline_denoising_loop(
            scheduler=self.scheduler,
            gradual_latent=self.gradual_latent,
            gradual_latent_state=gradual_latent_state,
            timesteps=timesteps,
            latents=latents,
            callback_steps=callback_steps,
            callback=callback,
            is_cancelled_callback=is_cancelled_callback,
            num_latent_input=num_latent_input,
            control_net_lllites=self.control_net_lllites,
            control_nets=self.control_nets,
            each_control_net_enabled=each_control_net_enabled,
            text_embeddings=text_embeddings,
            batch_size=batch_size,
            regional_network=regional_network,
            unet=self.unet,
            is_sdxl=self.is_sdxl,
            vector_embeddings=vector_embeddings if self.is_sdxl else None,
            clip_guide_images=clip_guide_images,
            guided_hints=guided_hints,
            control_net_enabled=self.control_net_enabled,
            do_classifier_free_guidance=do_classifier_free_guidance,
            negative_scale=negative_scale,
            guidance_scale=guidance_scale,
            extra_step_kwargs=extra_step_kwargs,
            mask=mask,
            init_latents_orig=init_latents_orig,
            img2img_noise=img2img_noise,
        )
        if latents is None:
            return None

        if return_latents:
            return latents

        return decode_pipeline_output(
            vae=self.vae,
            is_sdxl=self.is_sdxl,
            latents=latents,
            vae_batch_size=vae_batch_size,
            batch_size=batch_size,
            output_type=output_type,
        )

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)


# endregion

# def load_clip_l14_336(dtype):
#   print(f"loading CLIP: {CLIP_ID_L14_336}")
#   text_encoder = CLIPTextModel.from_pretrained(CLIP_ID_L14_336, torch_dtype=dtype)
#   return text_encoder


def main(args):
    if args.fp16:
        dtype = torch.float16
    elif args.bf16:
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    highres_fix = args.highres_fix_scale is not None
    # assert not highres_fix or args.image_path is None, f"highres_fix doesn't work with img2img / highres_fixはimg2imgと同時に使えません"

    if args.v2 and args.clip_skip is not None:
        logger.warning("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません")

    prepared_scheduler = gen_img_scheduler_util.prepare_scheduler_runtime(
        args,
        scheduler_linear_start=SCHEDULER_LINEAR_START,
        scheduler_linear_end=SCHEDULER_LINEAR_END,
        scheduler_timesteps=SCHEDULER_TIMESTEPS,
        scheduler_schedule=SCHEDLER_SCHEDULE,
    )
    scheduler = prepared_scheduler.scheduler
    scheduler_num_noises_per_step = prepared_scheduler.scheduler_num_noises_per_step
    noise_manager = prepared_scheduler.noise_manager

    prepared_model_setup = gen_img_model_setup_util.prepare_gen_img_model_setup(
        args,
        dtype=dtype,
        replace_unet_modules_fn=replace_unet_modules,
        replace_vae_modules_fn=replace_vae_modules,
        logger=logger,
    )
    is_sdxl = prepared_model_setup.is_sdxl
    device = prepared_model_setup.device
    vae_dtype = prepared_model_setup.vae_dtype
    text_encoders = prepared_model_setup.text_encoders
    tokenizers = prepared_model_setup.tokenizers
    vae = prepared_model_setup.vae
    unet = prepared_model_setup.unet

    prepared_pipeline_setup = gen_img_pipeline_setup_util.prepare_gen_img_pipeline_setup(
        args,
        dtype=dtype,
        device=device,
        vae=vae,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        unet=unet,
        scheduler=scheduler,
        is_sdxl=is_sdxl,
        pipeline_like_cls=PipelineLike,
        logger=logger,
    )
    networks = prepared_pipeline_setup.networks
    network_default_muls = prepared_pipeline_setup.network_default_muls
    network_pre_calc = prepared_pipeline_setup.network_pre_calc
    upscaler = prepared_pipeline_setup.upscaler
    control_nets = prepared_pipeline_setup.control_nets
    control_net_lllites = prepared_pipeline_setup.control_net_lllites
    pipe = prepared_pipeline_setup.pipe

    gen_img_textual_inversion_util.load_textual_inversion_embeddings(
        args,
        is_sdxl=is_sdxl,
        tokenizers=tokenizers,
        text_encoders=text_encoders,
        pipe=pipe,
        logger=logger,
    )

    prepared_inputs = gen_img_input_util.prepare_gen_img_inputs(
        args,
        pipe=pipe,
        networks=networks,
        highres_fix=highres_fix,
        is_sdxl=is_sdxl,
        device=device,
        dtype=dtype,
        clip_vision_model_name=CLIP_VISION_MODEL,
        list_prompter_cls=ListPrompter,
        logger=logger,
    )
    prompter = prepared_inputs.prompter
    init_images = prepared_inputs.init_images
    mask_images = prepared_inputs.mask_images
    regional_network = prepared_inputs.regional_network
    guide_images = prepared_inputs.guide_images
    seed_random = prepared_inputs.seed_random

    # 画像生成のループ
    os.makedirs(args.outdir, exist_ok=True)
    max_embeddings_multiples = 1 if args.max_embeddings_multiples is None else args.max_embeddings_multiples

    run_generation_iterations(
        args=args,
        prompter=prompter,
        seed_random=seed_random,
        highres_fix=highres_fix,
        upscaler=upscaler,
        batch_data_cls=BatchData,
        batch_data_base_cls=BatchDataBase,
        batch_data_ext_cls=BatchDataExt,
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
        latent_channels=LATENT_CHANNELS,
        downsampling_factor=DOWNSAMPLING_FACTOR,
        is_sdxl=is_sdxl,
        logger=logger,
        pyramid_noise_like_fn=pyramid_noise_like,
        init_images=init_images,
        mask_images=mask_images,
        guide_images=guide_images,
    )

    logger.info("done!")


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    main(args)
