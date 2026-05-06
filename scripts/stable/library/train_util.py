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


def resolve_cache_latents_runtime_kwargs(args: argparse.Namespace) -> dict:
    from library.strategy_base import normalize_latent_cache_disk_dtype

    raw_workers = getattr(args, "cache_latents_cpu_workers", None)
    try:
        resolved_workers = None if raw_workers in (None, "") else max(0, int(raw_workers))
    except (TypeError, ValueError):
        resolved_workers = None

    raw_prefetch = getattr(args, "cache_latents_prefetch_batches", None)
    try:
        resolved_prefetch = None if raw_prefetch in (None, "") else max(1, int(raw_prefetch))
    except (TypeError, ValueError):
        resolved_prefetch = None

    resolved_format = normalize_latents_disk_cache_format(getattr(args, "latent_cache_disk_format", None))
    resolved_dtype = normalize_latent_cache_disk_dtype(getattr(args, "latent_cache_disk_dtype", None))
    if resolved_dtype == "bf16" and resolved_format == "npz":
        resolved_dtype = "fp32"

    return {
        "preprocess_workers": resolved_workers,
        "prefetch_batches": resolved_prefetch,
        "disk_cache_format": resolved_format,
        "disk_cache_dtype": resolved_dtype,
    }


def resolve_text_encoder_outputs_cache_runtime_kwargs(args: argparse.Namespace) -> dict:
    from library.strategy_base import normalize_text_encoder_outputs_cache_dtype

    resolved_dtype = normalize_text_encoder_outputs_cache_dtype(
        getattr(args, "text_encoder_outputs_cache_dtype", None)
    )
    resolved_format = normalize_latents_disk_cache_format(getattr(args, "text_encoder_outputs_cache_disk_format", None))
    if resolved_dtype != "auto" and resolved_format == "npz":
        resolved_format = "safetensors"

    return {
        "disk_cache_format": resolved_format,
        "disk_cache_dtype": resolved_dtype,
    }


import library.train_dataset_util as _train_dataset_state
import library.train_dataset_util as _train_dataset_util
import library.train_patch_util as _train_patch_util
import library.train_config_util as _train_config_util
import library.train_prepare_util as _train_prepare_util
import library.train_checkpoint_util as _train_checkpoint_util
import library.train_loss_util as _train_loss_util
import library.train_sampling_util as _train_sampling_util
import library.train_support_util as _train_support_util


def _reexport_public_api(*modules):
    exported_names = []
    for module in modules:
        for name in getattr(module, "__all__", ()):
            globals()[name] = getattr(module, name)
            exported_names.append(name)
    return tuple(exported_names)

def __getattr__(name: str):
    if name == "HIGH_VRAM":
        return _train_dataset_state.HIGH_VRAM
    if name == "IMAGE_EXTENSIONS":
        return _train_dataset_state.IMAGE_EXTENSIONS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# from library.attention_processors import FlashAttnProcessor
# from library.hypernetwork import replace_attentions_for_hypernetwork
from library.original_unet import UNet2DConditionModel

# HIGH_VRAM state moved to library.train_dataset_util

_ARGPARSE_ZH_HELP_FALLBACK = " / 中文说明：请参考前面的英文描述。"
_EPOCH_COOLDOWN_WARNING_KEYS = set()
def configure_bucket_runtime_policy(*, mode: Optional[str] = None, target_edge: Optional[int] = None) -> None:
    _train_dataset_util.configure_bucket_runtime_policy(mode=mode, target_edge=target_edge)


def get_bucket_runtime_policy() -> dict[str, Optional[int | str]]:
    return _train_dataset_util.get_bucket_runtime_policy()


def resolve_dataloader_runtime_kwargs(args: argparse.Namespace, n_workers: int) -> dict:
    resolved_workers = max(0, int(n_workers or 0))
    persistent_requested = bool(getattr(args, "persistent_data_loader_workers", False))
    persistent_workers = bool(persistent_requested and resolved_workers > 0)
    if persistent_requested and resolved_workers <= 0:
        logger.warning(
            "persistent_data_loader_workers is enabled, but max_data_loader_n_workers resolved to 0. "
            "The current run will disable persistent workers for compatibility."
        )

    prefetch_factor = 2
    try:
        raw_prefetch = int(getattr(args, "data_loader_prefetch_factor", 2) or 2)
        prefetch_factor = max(1, raw_prefetch)
    except (TypeError, ValueError):
        prefetch_factor = 2

    kwargs = {
        "num_workers": resolved_workers,
        "persistent_workers": persistent_workers,
        "pin_memory": bool(torch.cuda.is_available()),
    }
    if resolved_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def _build_trilingual_status_pattern(
    english_and_japanese: str,
    chinese: str,
    *,
    separator: str = ": ",
):
    pattern = re.compile(rf"^(?P<indent>\s*){re.escape(english_and_japanese)}{re.escape(separator)}(?P<value>.+)$")

    def repl(match: re.Match[str]) -> str:
        return f"{match.group('indent')}{english_and_japanese} / {chinese}{separator}{match.group('value')}"

    return pattern, repl


_TRAINING_STATUS_TRANSLATORS: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str]]] = [
    (
        re.compile(r"^(?P<indent>\s*)running training / 学習開始$"),
        lambda match: f"{match.group('indent')}running training / 学習開始 / 开始训练",
    ),
    (
        re.compile(
            r"^(?P<indent>\s*)override steps\. (?P<english>steps for .+? epochs is) / 指定エポックまでのステップ数: (?P<value>.+)$"
        ),
        lambda match: (
            f"{match.group('indent')}override steps. {match.group('english')} / 指定エポックまでのステップ数 / "
            f"指定轮次对应的总步数: {match.group('value')}"
        ),
    ),
    _build_trilingual_status_pattern(
        "num train images * repeats / 学習画像の数×繰り返し回数",
        "训练图像数×重复次数",
    ),
    _build_trilingual_status_pattern(
        "num validation images * repeats / 学習画像の数×繰り返し回数",
        "验证图像数×重复次数",
    ),
    _build_trilingual_status_pattern(
        "num reg images / 正則化画像の数",
        "正则化图像数",
    ),
    _build_trilingual_status_pattern(
        "num examples / サンプル数",
        "样本数",
    ),
    _build_trilingual_status_pattern(
        "num batches per epoch / 1epochのバッチ数",
        "每个 epoch 的批次数",
    ),
    _build_trilingual_status_pattern(
        "num epochs / epoch数",
        "训练轮次",
    ),
    _build_trilingual_status_pattern(
        "batch size per device / バッチサイズ",
        "每张设备的批大小",
    ),
    _build_trilingual_status_pattern(
        "gradient accumulation steps / 勾配を合計するステップ数",
        "梯度累积步数",
        separator=" = ",
    ),
    _build_trilingual_status_pattern(
        "total optimization steps / 学習ステップ数",
        "总优化步数",
    ),
]


def translate_training_status_line(message: Any) -> Any:
    if not isinstance(message, str):
        return message
    if message.count(" / ") >= 2:
        return message

    for pattern, replacer in _TRAINING_STATUS_TRANSLATORS:
        match = pattern.match(message)
        if match:
            return replacer(match)
    return message


def _patch_accelerator_print_with_trilingual_training_status() -> None:
    if getattr(Accelerator, "_mikazuki_trilingual_training_status_patched", False):
        return

    original_print = Accelerator.print

    def print_with_trilingual_status(self, *args, **kwargs):
        translated_args = tuple(translate_training_status_line(arg) for arg in args)
        return original_print(self, *translated_args, **kwargs)

    Accelerator.print = print_with_trilingual_status
    Accelerator._mikazuki_trilingual_training_status_patched = True


def _patch_logger_info_with_trilingual_training_status() -> None:
    if getattr(logging.Logger, "_mikazuki_trilingual_training_status_patched", False):
        return

    original_info = logging.Logger.info

    def info_with_trilingual_status(self, msg, *args, **kwargs):
        return original_info(self, translate_training_status_line(msg), *args, **kwargs)

    logging.Logger.info = info_with_trilingual_status
    logging.Logger._mikazuki_trilingual_training_status_patched = True


def _normalize_argparse_help_text(help_text: Optional[str]) -> Optional[str]:
    if not isinstance(help_text, str):
        return help_text
    if " / " not in help_text:
        return help_text
    if re.search(r"[\u4e00-\u9fff]", help_text):
        return help_text
    return f"{help_text}{_ARGPARSE_ZH_HELP_FALLBACK}"


def _patch_parser_add_argument_with_zh_help(parser: argparse.ArgumentParser) -> None:
    if getattr(parser, "_mikazuki_zh_help_patched", False):
        return

    original_add_argument = parser.add_argument

    def add_argument_with_zh_help(*args, **kwargs):
        if "help" in kwargs:
            kwargs["help"] = _normalize_argparse_help_text(kwargs.get("help"))
        return original_add_argument(*args, **kwargs)

    parser.add_argument = add_argument_with_zh_help
    parser._mikazuki_zh_help_patched = True


_patch_accelerator_print_with_trilingual_training_status()
_patch_logger_info_with_trilingual_training_status()

# checkpointファイル名
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
DEFAULT_EPOCH_NAME = "epoch"
DEFAULT_LAST_OUTPUT_NAME = "last"

DEFAULT_STEP_NAME = "at"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"

_TRAIN_UTIL_COMPAT_EXPORTS = _reexport_public_api(
    _train_dataset_util,
    _train_patch_util,
    _train_config_util,
    _train_prepare_util,
    _train_checkpoint_util,
    _train_loss_util,
)


def load_tokenizer(args):
    from library.strategy_sd import SdTokenizeStrategy

    tokenize_strategy = SdTokenizeStrategy(
        bool(getattr(args, "v2", False)),
        getattr(args, "max_token_length", None),
        tokenizer_cache_dir=getattr(args, "tokenizer_cache_dir", None),
    )
    return tokenize_strategy.tokenizer


def append_lr_to_logs(logs, lr_scheduler, optimizer_type, including_unet=True):
    optimizer_scheduler_util.append_lr_to_logs(logs, lr_scheduler, optimizer_type, including_unet=including_unet)


def append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names):
    optimizer_scheduler_util.append_lr_to_logs_with_names(logs, lr_scheduler, optimizer_type, names)


# scheduler:
_TRAIN_UTIL_COMPAT_EXPORTS += _reexport_public_api(
    _train_sampling_util,
    _train_support_util,
)

__all__ = list(
    dict.fromkeys(
        [
            "load_tokenizer",
            "append_lr_to_logs",
            "append_lr_to_logs_with_names",
            *_TRAIN_UTIL_COMPAT_EXPORTS,
        ]
    )
)
