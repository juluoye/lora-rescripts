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
_EPOCH_COOLDOWN_WARNING_KEYS = set()

from library.argument_help_util import normalize_help_text
import library.train_dataset_util as dataset_util


def _patch_parser_add_argument_with_zh_help(parser: argparse.ArgumentParser) -> None:
    if getattr(parser, '_mikazuki_zh_help_patched', False):
        return

    original_add_argument = parser.add_argument

    def add_argument_with_zh_help(*args, **kwargs):
        if 'help' in kwargs:
            kwargs['help'] = normalize_help_text(kwargs.get('help'))
        return original_add_argument(*args, **kwargs)

    parser.add_argument = add_argument_with_zh_help
    parser._mikazuki_zh_help_patched = True


def load_metadata_from_safetensors(safetensors_file: str) -> dict:
    """r
    This method locks the file. see https://github.com/huggingface/safetensors/issues/164
    If the file isn't .safetensors or doesn't have metadata, return empty dict.
    """
    if os.path.splitext(safetensors_file)[1] != ".safetensors":
        return {}

    with safetensors.safe_open(safetensors_file, framework="pt", device="cpu") as f:
        metadata = f.metadata()
    if metadata is None:
        metadata = {}
    return metadata


def resolve_attention_backend(args, default: str = "default") -> str:
    attn_mode = str(getattr(args, "attn_mode", "") or "").strip().lower()
    if attn_mode in {"flash", "flashattn"}:
        return "flashattn"
    if attn_mode in {"sageattn", "sage"}:
        return "sageattn"
    if attn_mode == "xformers":
        return "xformers"
    if attn_mode == "sdpa":
        return "sdpa"
    if attn_mode == "torch":
        return "torch"

    if bool(getattr(args, "flashattn", False)):
        return "flashattn"
    if bool(getattr(args, "sageattn", False)) or bool(getattr(args, "use_sage_attn", False)):
        return "sageattn"
    if bool(getattr(args, "xformers", False)):
        return "xformers"
    if bool(getattr(args, "sdpa", False)):
        return "sdpa"
    if bool(getattr(args, "mem_eff_attn", False)):
        return "mem_eff_attn"

    return default


# this metadata is referred from train_network and various scripts, so we wrote here
SS_METADATA_KEY_V2 = "ss_v2"
SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
]


def build_minimum_network_metadata(
    v2: Optional[str],
    base_model: Optional[str],
    network_module: str,
    network_dim: str,
    network_alpha: str,
    network_args: Optional[dict],
):
    # old LoRA doesn't have base_model
    metadata = {
        SS_METADATA_KEY_NETWORK_MODULE: network_module,
        SS_METADATA_KEY_NETWORK_DIM: network_dim,
        SS_METADATA_KEY_NETWORK_ALPHA: network_alpha,
    }
    if v2 is not None:
        metadata[SS_METADATA_KEY_V2] = v2
    if base_model is not None:
        metadata[SS_METADATA_KEY_BASE_MODEL_VERSION] = base_model
    if network_args is not None:
        metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(network_args)
    return metadata


def get_sai_model_spec(
    state_dict: dict,
    args: argparse.Namespace,
    sdxl: bool,
    lora: bool,
    textual_inversion: bool,
    is_stable_diffusion_ckpt: Optional[bool] = None,  # None for TI and LoRA
    sd3: str = None,
    flux: str = None,  # "dev", "schnell" or "chroma"
    lumina: str = None,
    optional_metadata: dict[str, str] | None = None,
):
    timestamp = time.time()

    v2 = args.v2
    v_parameterization = args.v_parameterization
    reso = args.resolution

    title = args.metadata_title if args.metadata_title is not None else args.output_name

    if args.min_timestep is not None or args.max_timestep is not None:
        min_time_step = args.min_timestep if args.min_timestep is not None else 0
        max_time_step = args.max_timestep if args.max_timestep is not None else 1000
        timesteps = (min_time_step, max_time_step)
    else:
        timesteps = None

    # Convert individual model parameters to model_config dict
    # TODO: Update calls to this function to pass in the model config
    model_config = {}
    if sd3 is not None:
        model_config["sd3"] = sd3
    if flux is not None:
        model_config["flux"] = flux
    if lumina is not None:
        model_config["lumina"] = lumina

    # Extract metadata_* fields from args and merge with optional_metadata
    extracted_metadata = {}

    # Extract all metadata_* attributes from args
    for attr_name in dir(args):
        if attr_name.startswith("metadata_") and not attr_name.startswith("metadata___"):
            value = getattr(args, attr_name, None)
            if value is not None:
                # Remove metadata_ prefix and exclude already handled fields
                field_name = attr_name[9:]  # len("metadata_") = 9
                if field_name not in ["title", "author", "description", "license", "tags"]:
                    extracted_metadata[field_name] = value

    # Merge extracted metadata with provided optional_metadata
    all_optional_metadata = {**extracted_metadata}
    if optional_metadata:
        all_optional_metadata.update(optional_metadata)

    metadata = sai_model_spec.build_metadata(
        state_dict,
        v2,
        v_parameterization,
        sdxl,
        lora,
        textual_inversion,
        timestamp,
        title=title,
        reso=reso,
        is_stable_diffusion_ckpt=is_stable_diffusion_ckpt,
        author=args.metadata_author,
        description=args.metadata_description,
        license=args.metadata_license,
        tags=args.metadata_tags,
        timesteps=timesteps,
        clip_skip=args.clip_skip,  # None or int
        model_config=model_config,
        optional_metadata=all_optional_metadata if all_optional_metadata else None,
    )
    return metadata


def get_sai_model_spec_dataclass(
    state_dict: dict,
    args: argparse.Namespace,
    sdxl: bool,
    lora: bool,
    textual_inversion: bool,
    is_stable_diffusion_ckpt: Optional[bool] = None,
    sd3: str = None,
    flux: str = None,
    lumina: str = None,
    hunyuan_image: str = None,
    anima: str = None,
    optional_metadata: dict[str, str] | None = None,
) -> sai_model_spec.ModelSpecMetadata:
    """
    Get ModelSpec metadata as a dataclass - preferred for new code.
    Automatically extracts metadata_* fields from args.
    """
    timestamp = time.time()

    v2 = args.v2
    v_parameterization = args.v_parameterization
    reso = args.resolution

    title = args.metadata_title if args.metadata_title is not None else args.output_name

    if args.min_timestep is not None or args.max_timestep is not None:
        min_time_step = args.min_timestep if args.min_timestep is not None else 0
        max_time_step = args.max_timestep if args.max_timestep is not None else 1000
        timesteps = (min_time_step, max_time_step)
    else:
        timesteps = None

    # Convert individual model parameters to model_config dict
    model_config = {}
    if sd3 is not None:
        model_config["sd3"] = sd3
    if flux is not None:
        model_config["flux"] = flux
    if lumina is not None:
        model_config["lumina"] = lumina
    if hunyuan_image is not None:
        model_config["hunyuan_image"] = hunyuan_image
    if anima is not None:
        model_config["anima"] = anima
    # Use the dataclass function directly
    return sai_model_spec.build_metadata_dataclass(
        state_dict,
        v2,
        v_parameterization,
        sdxl,
        lora,
        textual_inversion,
        timestamp,
        title=title,
        reso=reso,
        is_stable_diffusion_ckpt=is_stable_diffusion_ckpt,
        author=args.metadata_author,
        description=args.metadata_description,
        license=args.metadata_license,
        tags=args.metadata_tags,
        timesteps=timesteps,
        clip_skip=args.clip_skip,
        model_config=model_config,
        optional_metadata=optional_metadata,
    )


def add_sd_models_arguments(parser: argparse.ArgumentParser):
    _patch_parser_add_argument_with_zh_help(parser)

    # for pretrained models
    parser.add_argument(
        "--v2", action="store_true", help="load Stable Diffusion v2.0 model / Stable Diffusion 2.0のモデルを読み込む"
    )
    parser.add_argument(
        "--v_parameterization", action="store_true", help="enable v-parameterization training / v-parameterization学習を有効にする"
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model to train, directory to Diffusers model or StableDiffusion checkpoint / 学習元モデル、Diffusers形式モデルのディレクトリまたはStableDiffusionのckptファイル",
    )
    parser.add_argument(
        "--tokenizer_cache_dir",
        type=str,
        default=None,
        help="directory for caching Tokenizer (for offline training) / Tokenizerをキャッシュするディレクトリ（ネット接続なしでの学習のため）",
    )


def add_optimizer_arguments(parser: argparse.ArgumentParser):
    _patch_parser_add_argument_with_zh_help(parser)

    def int_or_float(value):
        if value.endswith("%"):
            try:
                return float(value[:-1]) / 100.0
            except ValueError:
                raise argparse.ArgumentTypeError(f"Value '{value}' is not a valid percentage")
        try:
            float_value = float(value)
            if float_value >= 1:
                return int(value)
            return float(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"'{value}' is not an int or float")

    parser.add_argument(
        "--optimizer_type",
        type=str,
        default="",
        help="Optimizer to use / オプティマイザの種類: AdamW (default), AdamW8bit, PagedAdamW, PagedAdamW8bit, PagedAdamW32bit, "
        "Lion8bit, PagedLion8bit, Lion, SGDNesterov, SGDNesterov8bit, "
        "DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, "
        "AdaFactor. "
        "Also, you can use any optimizer by specifying the full path to the class, like 'bitsandbytes.optim.AdEMAMix8bit' or 'bitsandbytes.optim.PagedAdEMAMix8bit'.",
    )

    # backward compatibility
    parser.add_argument(
        "--use_8bit_adam",
        action="store_true",
        help="use 8bit AdamW optimizer (requires bitsandbytes) / 8bit Adamオプティマイザを使う（bitsandbytesのインストールが必要）",
    )
    parser.add_argument(
        "--use_lion_optimizer",
        action="store_true",
        help="use Lion optimizer (requires lion-pytorch) / Lionオプティマイザを使う（ lion-pytorch のインストールが必要）",
    )

    parser.add_argument("--learning_rate", type=float, default=2.0e-6, help="learning rate / 学習率")
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=None,
        help="weight decay for optimizer (equivalent to setting weight_decay=... in optimizer_args) / オプティマイザのweight decay（optimizer_argsの weight_decay=... と同等）",
    )
    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm, 0 for no clipping / 勾配正規化の最大norm、0でclippingを行わない",
    )

    parser.add_argument(
        "--optimizer_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for optimizer (like "weight_decay=0.01 betas=0.9,0.999 ...") / オプティマイザの追加引数（例： "weight_decay=0.01 betas=0.9,0.999 ..."）',
    )
    parser.add_argument("--prodigy_d0", type=float, default=None, help="Prodigy d0 / Prodigy の d0")
    parser.add_argument("--prodigy_d_coef", type=float, default=None, help="Prodigy d coefficient / Prodigy の d 係数")

    # parser.add_argument(
    #     "--optimizer_schedulefree_wrapper",
    #     action="store_true",
    #     help="use schedulefree_wrapper any optimizer / 任意のオプティマイザにschedulefree_wrapperを使用",
    # )

    # parser.add_argument(
    #     "--schedulefree_wrapper_args",
    #     type=str,
    #     default=None,
    #     nargs="*",
    #     help='additional arguments for schedulefree_wrapper (like "momentum=0.9 weight_decay_at_y=0.1 ...") / オプティマイザの追加引数（例： "momentum=0.9 weight_decay_at_y=0.1 ..."）',
    # )

    parser.add_argument("--lr_scheduler_type", type=str, default="", help="custom scheduler module / 使用するスケジューラ")
    parser.add_argument(
        "--lr_scheduler_args",
        type=str,
        default=None,
        nargs="*",
        help='additional arguments for scheduler (like "T_max=100") / スケジューラの追加引数（例： "T_max100"）',
    )

    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help="scheduler to use for learning rate / 学習率のスケジューラ: linear, cosine, cosine_with_restarts, polynomial, constant (default), constant_with_warmup, adafactor",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the warmup in the lr scheduler (default is 0) or float with ratio of train steps"
        " / 学習率のスケジューラをウォームアップするステップ数（デフォルト0）、または学習ステップの比率（1未満のfloat値の場合）",
    )
    parser.add_argument(
        "--lr_decay_steps",
        type=int_or_float,
        default=0,
        help="Int number of steps for the decay in the lr scheduler (default is 0) or float (<1) with ratio of train steps"
        " / 学習率のスケジューラを減衰させるステップ数（デフォルト0）、または学習ステップの比率（1未満のfloat値の場合）",
    )
    parser.add_argument(
        "--lr_scheduler_num_cycles",
        type=int,
        default=1,
        help="Number of restarts for cosine scheduler with restarts / cosine with restartsスケジューラでのリスタート回数",
    )
    parser.add_argument(
        "--lr_scheduler_power",
        type=float,
        default=1,
        help="Polynomial power for polynomial scheduler / polynomialスケジューラでのpolynomial power",
    )
    parser.add_argument(
        "--fused_backward_pass",
        action="store_true",
        help="Combines backward pass and optimizer step to reduce VRAM usage. Only available in SDXL, SD3 and FLUX"
        " / バックワードパスとオプティマイザステップを組み合わせてVRAMの使用量を削減します。SDXL、SD3、FLUXでのみ利用可能",
    )
    parser.add_argument(
        "--lr_scheduler_timescale",
        type=int,
        default=None,
        help="Inverse sqrt timescale for inverse sqrt scheduler,defaults to `num_warmup_steps`"
        + " / 逆平方根スケジューラのタイムスケール、デフォルトは`num_warmup_steps`",
    )
    parser.add_argument(
        "--lr_scheduler_min_lr_ratio",
        type=float,
        default=None,
        help="The minimum learning rate as a ratio of the initial learning rate for cosine with min lr scheduler and warmup decay scheduler"
        + " / 初期学習率の比率としての最小学習率を指定する、cosine with min lr と warmup decay スケジューラ で有効",
    )


def add_training_arguments(parser: argparse.ArgumentParser, support_dreambooth: bool):
    _patch_parser_add_argument_with_zh_help(parser)
    train_argument_groups_util.add_training_arguments(parser, support_dreambooth)


def add_masked_loss_arguments(parser: argparse.ArgumentParser):
    _patch_parser_add_argument_with_zh_help(parser)

    parser.add_argument(
        "--conditioning_data_dir",
        type=str,
        default=None,
        help="conditioning data directory / 条件付けデータのディレクトリ",
    )
    parser.add_argument(
        "--masked_loss",
        action="store_true",
        help="apply mask for calculating loss. conditioning_data_dir is required for dataset. / 損失計算時にマスクを適用する。datasetにはconditioning_data_dirが必要",
    )


def add_dit_training_arguments(parser: argparse.ArgumentParser):
    _patch_parser_add_argument_with_zh_help(parser)

    # Text encoder related arguments
    parser.add_argument(
        "--cache_text_encoder_outputs", action="store_true", help="cache text encoder outputs / text encoderの出力をキャッシュする"
    )
    parser.add_argument(
        "--cache_text_encoder_outputs_to_disk",
        action="store_true",
        help="cache text encoder outputs to disk / text encoderの出力をディスクにキャッシュする",
    )
    parser.add_argument(
        "--text_encoder_batch_size",
        type=int,
        default=None,
        help="text encoder batch size (default: None, use dataset's batch size)"
        + " / text encoderのバッチサイズ（デフォルト: None, データセットのバッチサイズを使用）",
    )

    # Model loading optimization
    parser.add_argument(
        "--disable_mmap_load_safetensors",
        action="store_true",
        help="disable mmap load for safetensors. Speed up model loading in WSL environment / safetensorsのmmapロードを無効にする。WSL環境等でモデル読み込みを高速化できる",
    )

    # Training arguments. partial copy from Diffusers
    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="uniform",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none", "uniform"],
        help="weighting scheme for timestep distribution. Default is uniform, uniform and none are the same behavior"
        " / タイムステップ分布の重み付けスキーム、デフォルトはuniform、uniform と none は同じ挙動",
    )
    parser.add_argument(
        "--logit_mean",
        type=float,
        default=0.0,
        help="mean to use when using the `'logit_normal'` weighting scheme / `'logit_normal'`重み付けスキームを使用する場合の平均",
    )
    parser.add_argument(
        "--logit_std",
        type=float,
        default=1.0,
        help="std to use when using the `'logit_normal'` weighting scheme / `'logit_normal'`重み付けスキームを使用する場合のstd",
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme` / モード重み付けスキームのスケール",
    )

    # offloading
    parser.add_argument(
        "--blocks_to_swap",
        type=int,
        default=None,
        help="[EXPERIMENTAL] "
        "Sets the number of blocks to swap during the forward and backward passes."
        "Increasing this number lowers the overall VRAM used during training at the expense of training speed (s/it)."
        " / 順伝播および逆伝播中にスワップするブロックの数を設定します。"
        "この数を増やすと、トレーニング中のVRAM使用量が減りますが、トレーニング速度（s/it）も低下します。",
    )


def get_sanitized_config_or_none(args: argparse.Namespace):
    # if `--log_config` is enabled, return args for logging. if not, return None.
    # when `--log_config is enabled, filter out sensitive values from args
    # if wandb is not enabled, the log is not exposed to the public, but it is fine to filter out sensitive values to be safe

    if not args.log_config:
        return None

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "pretrained_model_name_or_path",
        "vae",
        "tokenizer_cache_dir",
        "train_data_dir",
        "conditioning_data_dir",
        "reg_data_dir",
        "output_dir",
        "logging_dir",
        "logging_run_dir",
    ]
    filtered_args = {}
    for k, v in vars(args).items():
        # filter out sensitive values and convert to string if necessary
        if k not in sensitive_args + sensitive_path_args:
            # Accelerate values need to have type `bool`,`str`, `float`, `int`, or `None`.
            if v is None or isinstance(v, bool) or isinstance(v, str) or isinstance(v, float) or isinstance(v, int):
                filtered_args[k] = v
            # accelerate does not support lists
            elif isinstance(v, list):
                filtered_args[k] = f"{v}"
            # accelerate does not support objects
            elif isinstance(v, object):
                filtered_args[k] = f"{v}"

    return filtered_args


# verify command line args for training
def verify_command_line_training_args(args: argparse.Namespace):
    # if wandb is enabled, the command line is exposed to the public
    # check whether sensitive options are included in the command line arguments
    # if so, warn or inform the user to move them to the configuration file
    # wandbが有効な場合、コマンドラインが公開される
    # 学習用のコマンドライン引数に敏感なオプションが含まれているかどうかを確認し、
    # 含まれている場合は設定ファイルに移動するようにユーザーに警告または通知する

    wandb_enabled = args.log_with is not None and args.log_with != "tensorboard"  # "all" or "wandb"
    if not wandb_enabled:
        return

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "pretrained_model_name_or_path",
        "vae",
        "tokenizer_cache_dir",
        "train_data_dir",
        "conditioning_data_dir",
        "reg_data_dir",
        "output_dir",
        "logging_dir",
        "logging_run_dir",
    ]

    for arg in sensitive_args:
        if getattr(args, arg, None) is not None:
            logger.warning(
                f"wandb is enabled, but option `{arg}` is included in the command line. Because the command line is exposed to the public, it is recommended to move it to the `.toml` file."
                + f" / wandbが有効で、かつオプション `{arg}` がコマンドラインに含まれています。コマンドラインは公開されるため、`.toml`ファイルに移動することをお勧めします。"
            )

    # if path is absolute, it may include sensitive information
    for arg in sensitive_path_args:
        if getattr(args, arg, None) is not None and os.path.isabs(getattr(args, arg)):
            logger.info(
                f"wandb is enabled, but option `{arg}` is included in the command line and it is an absolute path. Because the command line is exposed to the public, it is recommended to move it to the `.toml` file or use relative path."
                + f" / wandbが有効で、かつオプション `{arg}` がコマンドラインに含まれており、絶対パスです。コマンドラインは公開されるため、`.toml`ファイルに移動するか、相対パスを使用することをお勧めします。"
            )

    if getattr(args, "config_file", None) is not None:
        logger.info(
            f"wandb is enabled, but option `config_file` is included in the command line. Because the command line is exposed to the public, please be careful about the information included in the path."
            + f" / wandbが有効で、かつオプション `config_file` がコマンドラインに含まれています。コマンドラインは公開されるため、パスに含まれる情報にご注意ください。"
        )

    # other sensitive options
    if args.huggingface_repo_id is not None and args.huggingface_repo_visibility != "public":
        logger.info(
            f"wandb is enabled, but option huggingface_repo_id is included in the command line and huggingface_repo_visibility is not 'public'. Because the command line is exposed to the public, it is recommended to move it to the `.toml` file."
            + f" / wandbが有効で、かつオプション huggingface_repo_id がコマンドラインに含まれており、huggingface_repo_visibility が 'public' ではありません。コマンドラインは公開されるため、`.toml`ファイルに移動することをお勧めします。"
        )


def enable_high_vram(args: argparse.Namespace):
    if args.highvram:
        logger.info("highvram is enabled / highvramが有効です / highvram 已启用")
        dataset_util.set_high_vram(True)


def verify_training_args(args: argparse.Namespace):
    r"""
    Verify training arguments. Also reflect highvram option to global variable
    学習用引数を検証する。あわせて highvram オプションの指定をグローバル変数に反映する
    """
    enable_high_vram(args)

    if args.v2 and args.clip_skip is not None:
        logger.warning("v2 with clip_skip will be unexpected / v2でclip_skipを使用することは想定されていません / v2 与 clip_skip 同时使用结果可能异常")

    if args.cache_latents_to_disk and not args.cache_latents:
        args.cache_latents = True
        logger.warning(
            "cache_latents_to_disk is enabled, so cache_latents is also enabled / cache_latents_to_diskが有効なため、cache_latentsを有効にします / 已启用 cache_latents_to_disk，因此同时启用 cache_latents"
        )

    # noise_offset, perlin_noise, multires_noise_iterations cannot be enabled at the same time
    # # Listを使って数えてもいいけど並べてしまえ
    # if args.noise_offset is not None and args.multires_noise_iterations is not None:
    #     raise ValueError(
    #         "noise_offset and multires_noise_iterations cannot be enabled at the same time / noise_offsetとmultires_noise_iterationsを同時に有効にできません"
    #     )
    # if args.noise_offset is not None and args.perlin_noise is not None:
    #     raise ValueError("noise_offset and perlin_noise cannot be enabled at the same time / noise_offsetとperlin_noiseは同時に有効にできません")
    # if args.perlin_noise is not None and args.multires_noise_iterations is not None:
    #     raise ValueError(
    #         "perlin_noise and multires_noise_iterations cannot be enabled at the same time / perlin_noiseとmultires_noise_iterationsを同時に有効にできません"
    #     )

    if args.adaptive_noise_scale is not None and args.noise_offset is None:
        raise ValueError("adaptive_noise_scale requires noise_offset / adaptive_noise_scaleを使用するにはnoise_offsetが必要です / 使用 adaptive_noise_scale 需要同时设置 noise_offset")

    if args.scale_v_pred_loss_like_noise_pred and not args.v_parameterization:
        raise ValueError(
            "scale_v_pred_loss_like_noise_pred can be enabled only with v_parameterization / scale_v_pred_loss_like_noise_predはv_parameterizationが有効なときのみ有効にできます / 仅在启用 v_parameterization 时可开启 scale_v_pred_loss_like_noise_pred"
        )

    if args.v_pred_like_loss and args.v_parameterization:
        raise ValueError(
            "v_pred_like_loss cannot be enabled with v_parameterization / v_pred_like_lossはv_parameterizationが有効なときには有効にできません / 启用 v_parameterization 时不能启用 v_pred_like_loss"
        )

    if args.zero_terminal_snr and not args.v_parameterization:
        logger.warning(
            f"zero_terminal_snr is enabled, but v_parameterization is not enabled. training will be unexpected"
            + " / zero_terminal_snrが有効ですが、v_parameterizationが有効ではありません。学習結果は想定外になる可能性があります / zero_terminal_snr 已启用但 v_parameterization 未启用，训练结果可能不符合预期"
        )

    if args.sample_every_n_epochs is not None and args.sample_every_n_epochs <= 0:
        logger.warning(
            "sample_every_n_epochs is less than or equal to 0, so it will be disabled / sample_every_n_epochsに0以下の値が指定されたため無効になります / sample_every_n_epochs 小于等于 0，已自动禁用"
        )
        args.sample_every_n_epochs = None

    if args.sample_every_n_steps is not None and args.sample_every_n_steps <= 0:
        logger.warning(
            "sample_every_n_steps is less than or equal to 0, so it will be disabled / sample_every_n_stepsに0以下の値が指定されたため無効になります / sample_every_n_steps 小于等于 0，已自动禁用"
        )
        args.sample_every_n_steps = None

    if getattr(args, "cooldown_every_n_epochs", None) is not None and args.cooldown_every_n_epochs <= 0:
        logger.warning(
            "cooldown_every_n_epochs is less than or equal to 0, so it will be disabled / cooldown_every_n_epochsに0以下の値が指定されたため無効になります / cooldown_every_n_epochs 小于等于 0，已自动禁用"
        )
        args.cooldown_every_n_epochs = None

    if getattr(args, "cooldown_minutes", None) is not None and args.cooldown_minutes <= 0:
        logger.warning(
            "cooldown_minutes is less than or equal to 0, so fixed cooldown time will be disabled / cooldown_minutesに0以下の値が指定されたため固定待機を無効にします / cooldown_minutes 小于等于 0，已自动禁用固定等待"
        )
        args.cooldown_minutes = None

    if getattr(args, "cooldown_until_temp_c", None) is not None and args.cooldown_until_temp_c <= 0:
        logger.warning(
            "cooldown_until_temp_c is less than or equal to 0, so temperature cooldown will be disabled / cooldown_until_temp_cに0以下の値が指定されたため温度待機を無効にします / cooldown_until_temp_c 小于等于 0，已自动禁用温度等待"
        )
        args.cooldown_until_temp_c = None

    if getattr(args, "cooldown_poll_seconds", None) is None or args.cooldown_poll_seconds <= 0:
        if getattr(args, "cooldown_poll_seconds", None) is not None and args.cooldown_poll_seconds <= 0:
            logger.warning(
                "cooldown_poll_seconds is less than or equal to 0, so it will be reset to 15 seconds / cooldown_poll_secondsに0以下の値が指定されたため15秒に戻します / cooldown_poll_seconds 小于等于 0，已自动重置为 15 秒"
            )
        args.cooldown_poll_seconds = 15

    if getattr(args, "cooldown_every_n_epochs", None) is not None:
        if getattr(args, "cooldown_minutes", None) is None and getattr(args, "cooldown_until_temp_c", None) is None:
            logger.warning(
                "cooldown_every_n_epochs is set, but neither cooldown_minutes nor cooldown_until_temp_c is configured, so cooldown will be disabled / cooldown_every_n_epochsは設定されていますが、cooldown_minutesとcooldown_until_temp_cのどちらも未設定のため無効にします / 已设置 cooldown_every_n_epochs，但未配置 cooldown_minutes 或 cooldown_until_temp_c，已自动禁用冷却"
            )
            args.cooldown_every_n_epochs = None
    elif getattr(args, "cooldown_minutes", None) is not None or getattr(args, "cooldown_until_temp_c", None) is not None:
        logger.warning(
            "cooldown_minutes or cooldown_until_temp_c is set without cooldown_every_n_epochs, so cooldown will not run / cooldown_every_n_epochsが未設定のため温度待機・固定待機は実行されません / 未设置 cooldown_every_n_epochs，冷却暂停不会执行"
        )

    if getattr(args, "gpu_power_limit_w", None) is not None and args.gpu_power_limit_w <= 0:
        logger.warning(
            "gpu_power_limit_w is less than or equal to 0, so it will be disabled / gpu_power_limit_wに0以下の値が指定されたため無効になります / gpu_power_limit_w 小于等于 0，已自动禁用"
        )
        args.gpu_power_limit_w = None


def _warn_epoch_cooldown_once(key: str, message: str):
    if key in _EPOCH_COOLDOWN_WARNING_KEYS:
        return
    _EPOCH_COOLDOWN_WARNING_KEYS.add(key)
    logger.warning(message)


def _summarize_gpu_temperatures(gpus: Sequence[dict]) -> str:
    summaries = []
    for gpu in gpus or []:
        gpu_id = str(gpu.get("index", "?"))
        temperature_c = gpu.get("temperature_c")
        if temperature_c is None:
            summaries.append(f"{gpu_id}:N/A")
        else:
            summaries.append(f"{gpu_id}:{int(round(float(temperature_c)))}C")
    return ", ".join(summaries) if summaries else "N/A"


def maybe_run_epoch_cooldown(
    args: argparse.Namespace,
    accelerator: Accelerator,
    epoch_no: int,
    total_epochs: Optional[int] = None,
    context_label: str = "training",
) -> bool:
    cooldown_every_n_epochs = getattr(args, "cooldown_every_n_epochs", None)
    cooldown_minutes = getattr(args, "cooldown_minutes", None)
    cooldown_until_temp_c = getattr(args, "cooldown_until_temp_c", None)
    cooldown_poll_seconds = int(getattr(args, "cooldown_poll_seconds", 15) or 15)

    if cooldown_every_n_epochs is None or cooldown_every_n_epochs <= 0:
        return False
    if epoch_no is None or epoch_no <= 0:
        return False
    if epoch_no % cooldown_every_n_epochs != 0:
        return False
    if total_epochs is not None and epoch_no >= total_epochs:
        return False
    if cooldown_minutes is None and cooldown_until_temp_c is None:
        return False

    accelerator.wait_for_everyone()
    try:
        if not accelerator.is_local_main_process:
            return True

        logger.info(
            f"[cooldown] {context_label}: epoch {epoch_no}"
            + (f"/{total_epochs}" if total_epochs is not None else "")
            + " reached cooldown point."
        )

        if cooldown_minutes is not None:
            cooldown_seconds = max(0.0, float(cooldown_minutes) * 60.0)
            if cooldown_seconds > 0:
                logger.info(
                    f"[cooldown] {context_label}: sleeping for at least {float(cooldown_minutes):.2f} minute(s)."
                )
                time.sleep(cooldown_seconds)

        if cooldown_until_temp_c is None:
            return True

        if query_gpu_metrics is None or resolve_visible_gpu_targets_from_env is None:
            _warn_epoch_cooldown_once(
                "cooldown-import-missing",
                "[cooldown] GPU temperature cooldown is unavailable because the nvidia-smi helper could not be imported. / GPU温度クールダウン用ヘルパーを読み込めないため温度待機をスキップします / 无法导入 nvidia-smi 辅助模块，已跳过温度冷却等待",
            )
            return True

        target_ids = resolve_visible_gpu_targets_from_env() or None
        metrics = query_gpu_metrics(target_ids)
        if not metrics.get("ok"):
            _warn_epoch_cooldown_once(
                f"cooldown-query-failed:{metrics.get('error', '')}",
                f"[cooldown] Unable to query GPU temperature, skip temperature wait: {metrics.get('error', 'unknown error')} / GPU温度を取得できないため温度待機をスキップします / 无法读取 GPU 温度，已跳过温度等待",
            )
            return True

        gpus = metrics.get("gpus") or []
        temperatures = [gpu.get("temperature_c") for gpu in gpus if gpu.get("temperature_c") is not None]
        if not temperatures:
            _warn_epoch_cooldown_once(
                "cooldown-temperature-unavailable",
                "[cooldown] GPU temperature telemetry is unavailable, skip temperature wait. / GPU温度テレメトリが利用できないため温度待機をスキップします / GPU 温度遥测不可用，已跳过温度等待",
            )
            return True

        logger.info(
            f"[cooldown] {context_label}: waiting until max GPU temperature is <= {int(cooldown_until_temp_c)}C "
            + f"(current: {_summarize_gpu_temperatures(gpus)})."
        )

        while True:
            max_temperature = max(temp for temp in temperatures if temp is not None)
            if max_temperature <= cooldown_until_temp_c:
                logger.info(
                    f"[cooldown] {context_label}: cooldown finished at {_summarize_gpu_temperatures(gpus)}."
                )
                break

            time.sleep(max(1, cooldown_poll_seconds))
            metrics = query_gpu_metrics(target_ids)
            if not metrics.get("ok"):
                _warn_epoch_cooldown_once(
                    f"cooldown-query-loop-failed:{metrics.get('error', '')}",
                    f"[cooldown] GPU temperature polling failed, stop temperature wait: {metrics.get('error', 'unknown error')} / GPU温度ポーリングに失敗したため温度待機を終了します / GPU 温度轮询失败，已结束温度等待",
                )
                break

            gpus = metrics.get("gpus") or []
            temperatures = [gpu.get("temperature_c") for gpu in gpus if gpu.get("temperature_c") is not None]
            if not temperatures:
                _warn_epoch_cooldown_once(
                    "cooldown-temperature-loop-unavailable",
                    "[cooldown] GPU temperature telemetry disappeared during cooldown, stop temperature wait. / クールダウン中にGPU温度テレメトリが利用できなくなったため温度待機を終了します / 冷却过程中 GPU 温度遥测不可用，已结束温度等待",
                )
                break
    finally:
        accelerator.wait_for_everyone()

    return True


def add_dataset_arguments(
    parser: argparse.ArgumentParser, support_dreambooth: bool, support_caption: bool, support_caption_dropout: bool
):
    _patch_parser_add_argument_with_zh_help(parser)
    dataset_argument_groups_util.add_dataset_arguments(
        parser,
        support_dreambooth=support_dreambooth,
        support_caption=support_caption,
        support_caption_dropout=support_caption_dropout,
    )


def add_sd_saving_arguments(parser: argparse.ArgumentParser):
    _patch_parser_add_argument_with_zh_help(parser)

    parser.add_argument(
        "--save_model_as",
        type=str,
        default=None,
        choices=[None, "ckpt", "safetensors", "diffusers", "diffusers_safetensors"],
        help="format to save the model (default is same to original) / モデル保存時の形式（未指定時は元モデルと同じ）",
    )
    parser.add_argument(
        "--use_safetensors",
        action="store_true",
        help="use safetensors format to save (if save_model_as is not specified) / checkpoint、モデルをsafetensors形式で保存する（save_model_as未指定時）",
    )


def read_config_from_file(args: argparse.Namespace, parser: argparse.ArgumentParser):
    if not args.config_file:
        return args

    config_path = args.config_file + ".toml" if not args.config_file.endswith(".toml") else args.config_file

    if args.output_config:
        # check if config file exists
        if os.path.exists(config_path):
            logger.error(
                f"Config file already exists. Aborting... / 出力先の設定ファイルが既に存在します: {config_path} / 配置文件已存在，操作已中止: {config_path}"
            )
            exit(1)

        # convert args to dictionary
        args_dict = vars(args)

        # remove unnecessary keys
        for key in ["config_file", "output_config", "wandb_api_key"]:
            if key in args_dict:
                del args_dict[key]

        # get default args from parser
        default_args = vars(parser.parse_args([]))

        # remove default values: cannot use args_dict.items directly because it will be changed during iteration
        for key, value in list(args_dict.items()):
            if key in default_args and value == default_args[key]:
                del args_dict[key]

        # convert Path to str in dictionary
        for key, value in args_dict.items():
            if isinstance(value, pathlib.Path):
                args_dict[key] = str(value)

        # convert to toml and output to file
        with open(config_path, "w") as f:
            toml.dump(args_dict, f)

        logger.info(f"Saved config file / 設定ファイルを保存しました / 已保存配置文件: {config_path}")
        exit(0)

    if not os.path.exists(config_path):
        logger.info(f"{config_path} not found.")
        exit(1)

    logger.info(f"Loading settings from {config_path}...")
    with open(config_path, "r", encoding="utf-8") as f:
        config_dict = toml.load(f)

    # combine all sections into one
    ignore_nesting_dict = {}
    for section_name, section_dict in config_dict.items():
        # if value is not dict, save key and value as is
        if not isinstance(section_dict, dict):
            ignore_nesting_dict[section_name] = section_dict
            continue

        if section_name == "custom_attributes":
            ignore_nesting_dict[section_name] = section_dict
            continue

        # if value is dict, save all key and value into one dict
        for key, value in section_dict.items():
            ignore_nesting_dict[key] = value

    config_args = argparse.Namespace(**ignore_nesting_dict)
    args = parser.parse_args(namespace=config_args)
    args.config_file = os.path.splitext(args.config_file)[0]

    return args


# endregion

__all__ = [
    'SS_METADATA_KEY_V2',
    'SS_METADATA_KEY_BASE_MODEL_VERSION',
    'SS_METADATA_KEY_NETWORK_MODULE',
    'SS_METADATA_KEY_NETWORK_DIM',
    'SS_METADATA_KEY_NETWORK_ALPHA',
    'SS_METADATA_KEY_NETWORK_ARGS',
    'SS_METADATA_MINIMUM_KEYS',
    'load_metadata_from_safetensors',
    'resolve_attention_backend',
    'build_minimum_network_metadata',
    'get_sai_model_spec',
    'get_sai_model_spec_dataclass',
    'add_sd_models_arguments',
    'add_optimizer_arguments',
    'add_training_arguments',
    'add_masked_loss_arguments',
    'add_dit_training_arguments',
    'get_sanitized_config_or_none',
    'verify_command_line_training_args',
    'enable_high_vram',
    'verify_training_args',
    'maybe_run_epoch_cooldown',
    'add_dataset_arguments',
    'add_sd_saving_arguments',
    'read_config_from_file',
]
