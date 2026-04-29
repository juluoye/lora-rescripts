# common functions for training

import argparse
import asyncio
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import datetime
import importlib
import json
import logging
import pathlib
import re
import shutil
import time
import typing
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union
from accelerate import Accelerator, InitProcessGroupKwargs, DistributedDataParallelKwargs, PartialState
import glob
import math
import os
import random
import hashlib
import subprocess
from io import BytesIO
import toml

# from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from packaging.version import Version

import torch
from library.device_utils import init_ipex, clean_memory_on_device
from library.strategy_base import (
    LatentsCachingStrategy,
    TokenizeStrategy,
    TextEncoderOutputsCachingStrategy,
    TextEncodingStrategy,
    configure_latents_cache_runtime,
)
from library.latents_disk_cache import normalize_latents_disk_cache_format
from mikazuki.utils.runtime_sageattention import load_runtime_sageattention_symbols
from mikazuki.utils.runtime_mode import infer_attention_runtime_mode, is_amd_rocm_runtime, is_intel_xpu_runtime
from mikazuki.utils.runtime_safe_preview import (
    clamp_safe_preview_request,
    temporary_diffusion_safe_preview_backend,
)

init_ipex()

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torchvision import transforms
from transformers import CLIPTokenizer, CLIPTextModel, CLIPTextModelWithProjection
import transformers
from diffusers import (
    StableDiffusionPipeline,
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
    AutoencoderKL,
)
from library import anima_caption_util, custom_train_functions, optimizer_scheduler_util, optimizer_util, sd3_utils
from library import attention as unified_attention
from library.original_unet import UNet2DConditionModel
from huggingface_hub import hf_hub_download
import numpy as np
from PIL import Image
import imagesize
import cv2
import safetensors.torch
from library.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline
from library.sdxl_lpw_stable_diffusion import SdxlStableDiffusionLongPromptWeightingPipeline
import library.model_util as model_util
import library.dataset_argument_groups_util as dataset_argument_groups_util
import library.train_argument_groups_util as train_argument_groups_util
import library.huggingface_util as huggingface_util
import library.sai_model_spec as sai_model_spec
import library.deepspeed_utils as deepspeed_utils
from library.utils import setup_logging, resize_image, validate_interpolation_fn
try:
    from mikazuki.utils.nvidia_smi import query_gpu_metrics, resolve_visible_gpu_targets_from_env
except Exception:
    query_gpu_metrics = None
    resolve_visible_gpu_targets_from_env = None

setup_logging()
import logging

logger = logging.getLogger(__name__)

EPOCH_STATE_NAME = '{}-{:06d}-state'
EPOCH_FILE_NAME = '{}-{:06d}'
LAST_STATE_NAME = '{}-state'
DEFAULT_EPOCH_NAME = 'epoch'
DEFAULT_LAST_OUTPUT_NAME = 'last'
DEFAULT_STEP_NAME = 'at'
STEP_STATE_NAME = '{}-step{:08d}-state'

def default_if_none(value, default):
    return default if value is None else value


def get_epoch_ckpt_name(args: argparse.Namespace, ext: str, epoch_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ext


def get_step_ckpt_name(args: argparse.Namespace, ext: str, step_no: int):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
    return STEP_FILE_NAME.format(model_name, step_no) + ext


def get_last_ckpt_name(args: argparse.Namespace, ext: str):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)
    return model_name + ext


def get_remove_epoch_no(args: argparse.Namespace, epoch_no: int):
    if args.save_last_n_epochs is None:
        return None

    remove_epoch_no = epoch_no - args.save_every_n_epochs * args.save_last_n_epochs
    if remove_epoch_no < 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: argparse.Namespace, step_no: int):
    if args.save_last_n_steps is None:
        return None

    # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
    # save_every_n_steps=10, save_last_n_steps=30の場合、50step目には30step分残し、10step目を削除する
    remove_step_no = step_no - args.save_last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


# epochとstepの保存、メタデータにepoch/stepが含まれ引数が同じになるため、統合している
# on_epoch_end: Trueならepoch終了時、Falseならstep経過時
def save_sd_model_on_epoch_end_or_stepwise(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    src_path: str,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    text_encoder,
    unet,
    vae,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = get_sai_model_spec(None, args, False, False, False, is_stable_diffusion_ckpt=True)
        model_util.save_stable_diffusion_checkpoint(
            args.v2, ckpt_file, text_encoder, unet, src_path, epoch_no, global_step, sai_metadata, save_dtype, vae
        )

    def diffusers_saver(out_dir):
        model_util.save_diffusers_checkpoint(
            args.v2, out_dir, text_encoder, unet, src_path, vae=vae, use_safetensors=use_safetensors
        )

    save_sd_model_on_epoch_end_or_stepwise_common(
        args,
        on_epoch_end,
        accelerator,
        save_stable_diffusion_format,
        use_safetensors,
        epoch,
        num_train_epochs,
        global_step,
        sd_saver,
        diffusers_saver,
    )


def save_sd_model_on_epoch_end_or_stepwise_common(
    args: argparse.Namespace,
    on_epoch_end: bool,
    accelerator,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    num_train_epochs: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    if on_epoch_end:
        epoch_no = epoch + 1
        saving = epoch_no % args.save_every_n_epochs == 0 and epoch_no < num_train_epochs
        if not saving:
            return

        model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)
        remove_no = get_remove_epoch_no(args, epoch_no)
    else:
        # 保存するか否かは呼び出し側で判断済み

        model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)
        epoch_no = epoch  # 例: 最初のepochの途中で保存したら0になる、SDモデルに保存される
        remove_no = get_remove_step_no(args, global_step)

    os.makedirs(args.output_dir, exist_ok=True)
    if save_stable_diffusion_format:
        ext = ".safetensors" if use_safetensors else ".ckpt"

        if on_epoch_end:
            ckpt_name = get_epoch_ckpt_name(args, ext, epoch_no)
        else:
            ckpt_name = get_step_ckpt_name(args, ext, global_step)

        ckpt_file = os.path.join(args.output_dir, ckpt_name)
        logger.info("")
        logger.info(f"saving checkpoint: {ckpt_file}")
        sd_saver(ckpt_file, epoch_no, global_step)

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_ckpt_name = get_epoch_ckpt_name(args, ext, remove_no)
            else:
                remove_ckpt_name = get_step_ckpt_name(args, ext, remove_no)

            remove_ckpt_file = os.path.join(args.output_dir, remove_ckpt_name)
            if os.path.exists(remove_ckpt_file):
                logger.info(f"removing old checkpoint: {remove_ckpt_file}")
                os.remove(remove_ckpt_file)

    else:
        if on_epoch_end:
            out_dir = os.path.join(args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, epoch_no))
        else:
            out_dir = os.path.join(args.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, global_step))

        logger.info("")
        logger.info(f"saving model: {out_dir}")
        diffusers_saver(out_dir)

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, out_dir, "/" + model_name)

        # remove older checkpoints
        if remove_no is not None:
            if on_epoch_end:
                remove_out_dir = os.path.join(args.output_dir, EPOCH_DIFFUSERS_DIR_NAME.format(model_name, remove_no))
            else:
                remove_out_dir = os.path.join(args.output_dir, STEP_DIFFUSERS_DIR_NAME.format(model_name, remove_no))

            if os.path.exists(remove_out_dir):
                logger.info(f"removing old model: {remove_out_dir}")
                shutil.rmtree(remove_out_dir)

    if args.save_state:
        if on_epoch_end:
            save_and_remove_state_on_epoch_end(args, accelerator, epoch_no)
        else:
            save_and_remove_state_stepwise(args, accelerator, global_step)


def save_and_remove_state_on_epoch_end(args: argparse.Namespace, accelerator, epoch_no):
    model_name = default_if_none(args.output_name, DEFAULT_EPOCH_NAME)

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)
    _ensure_scheduler_state_compatibility(accelerator, args, state_dir)
    if args.save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + EPOCH_STATE_NAME.format(model_name, epoch_no))

    last_n_epochs = args.save_last_n_epochs_state if args.save_last_n_epochs_state else args.save_last_n_epochs
    if last_n_epochs is not None:
        remove_epoch_no = epoch_no - args.save_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
        if os.path.exists(state_dir_old):
            logger.info(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_and_remove_state_stepwise(args: argparse.Namespace, accelerator, step_no):
    model_name = default_if_none(args.output_name, DEFAULT_STEP_NAME)

    logger.info("")
    logger.info(f"saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, step_no))
    accelerator.save_state(state_dir)
    _ensure_scheduler_state_compatibility(accelerator, args, state_dir)
    if args.save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + STEP_STATE_NAME.format(model_name, step_no))

    last_n_steps = args.save_last_n_steps_state if args.save_last_n_steps_state else args.save_last_n_steps
    if last_n_steps is not None:
        # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)

        if remove_step_no > 0:
            state_dir_old = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no))
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def save_state_on_train_end(args: argparse.Namespace, accelerator):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)
    _ensure_scheduler_state_compatibility(accelerator, args, state_dir)

    if args.save_state_to_huggingface:
        logger.info("uploading last state to huggingface.")
        huggingface_util.upload(args, state_dir, "/" + LAST_STATE_NAME.format(model_name))


def save_sd_model_on_train_end(
    args: argparse.Namespace,
    src_path: str,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    save_dtype: torch.dtype,
    epoch: int,
    global_step: int,
    text_encoder,
    unet,
    vae,
):
    def sd_saver(ckpt_file, epoch_no, global_step):
        sai_metadata = get_sai_model_spec(None, args, False, False, False, is_stable_diffusion_ckpt=True)
        model_util.save_stable_diffusion_checkpoint(
            args.v2, ckpt_file, text_encoder, unet, src_path, epoch_no, global_step, sai_metadata, save_dtype, vae
        )

    def diffusers_saver(out_dir):
        model_util.save_diffusers_checkpoint(
            args.v2, out_dir, text_encoder, unet, src_path, vae=vae, use_safetensors=use_safetensors
        )

    save_sd_model_on_train_end_common(
        args, save_stable_diffusion_format, use_safetensors, epoch, global_step, sd_saver, diffusers_saver
    )


def save_sd_model_on_train_end_common(
    args: argparse.Namespace,
    save_stable_diffusion_format: bool,
    use_safetensors: bool,
    epoch: int,
    global_step: int,
    sd_saver,
    diffusers_saver,
):
    model_name = default_if_none(args.output_name, DEFAULT_LAST_OUTPUT_NAME)

    if save_stable_diffusion_format:
        os.makedirs(args.output_dir, exist_ok=True)

        ckpt_name = model_name + (".safetensors" if use_safetensors else ".ckpt")
        ckpt_file = os.path.join(args.output_dir, ckpt_name)

        logger.info(f"save trained model as StableDiffusion checkpoint to {ckpt_file}")
        sd_saver(ckpt_file, epoch, global_step)

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, ckpt_file, "/" + ckpt_name, force_sync_upload=True)
    else:
        out_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(out_dir, exist_ok=True)

        logger.info(f"save trained model as Diffusers to {out_dir}")
        diffusers_saver(out_dir)

        if args.huggingface_repo_id is not None:
            huggingface_util.upload(args, out_dir, "/" + model_name, force_sync_upload=True)

__all__ = [
    'default_if_none',
    'get_epoch_ckpt_name',
    'get_step_ckpt_name',
    'get_last_ckpt_name',
    'get_remove_epoch_no',
    'get_remove_step_no',
    'save_sd_model_on_epoch_end_or_stepwise',
    'save_sd_model_on_epoch_end_or_stepwise_common',
    'save_and_remove_state_on_epoch_end',
    'save_and_remove_state_stepwise',
    'save_state_on_train_end',
    'save_sd_model_on_train_end',
    'save_sd_model_on_train_end_common',
]
