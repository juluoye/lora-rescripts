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

from library.train_dataset_util import *
from library.train_patch_util import *
from library.train_config_util import *

def _resume_state_allows_missing_scheduler(args: argparse.Namespace) -> bool:
    if optimizer_scheduler_util.is_schedulefree_mode(args, include_wrapper=True):
        return True

    lr_scheduler = str(getattr(args, "lr_scheduler", "") or "").strip().lower()
    lr_scheduler_type = str(getattr(args, "lr_scheduler_type", "") or "").strip()
    return lr_scheduler == "constant" and not lr_scheduler_type


def _ensure_scheduler_state_compatibility(accelerator, args: argparse.Namespace, state_dir: str) -> None:
    if not state_dir or not os.path.isdir(state_dir):
        return

    scheduler_file = os.path.join(state_dir, "scheduler.bin")
    if os.path.isfile(scheduler_file) or not _resume_state_allows_missing_scheduler(args):
        return

    schedulers = list(getattr(accelerator, "_schedulers", []) or [])
    if len(schedulers) == 0:
        logger.info(
            "resume/save state dir has no scheduler.bin, but this optimizer path does not register a resumable scheduler. continuing."
        )
        return

    written_files = []
    for i, scheduler in enumerate(schedulers):
        state_dict_fn = getattr(scheduler, "state_dict", None)
        if not callable(state_dict_fn):
            continue

        state_path = os.path.join(state_dir, "scheduler.bin" if i == 0 else f"scheduler_{i}.bin")
        try:
            torch.save(state_dict_fn(), state_path)
            written_files.append(state_path)
        except Exception as exc:
            logger.warning(f"failed to synthesize scheduler state for compatibility: {state_path} ({exc})")

    if written_files:
        logger.info(f"wrote scheduler compatibility state file(s): {', '.join(written_files)}")


def resume_from_local_or_hf_if_specified(accelerator, args):
    if not args.resume:
        return

    if not args.resume_from_huggingface:
        logger.info(f"resume training from local state: {args.resume}")
        _ensure_scheduler_state_compatibility(accelerator, args, args.resume)
        accelerator.load_state(args.resume)
        return

    logger.info(f"resume training from huggingface state: {args.resume}")
    repo_id = args.resume.split("/")[0] + "/" + args.resume.split("/")[1]
    path_in_repo = "/".join(args.resume.split("/")[2:])
    revision = None
    repo_type = None
    if ":" in path_in_repo:
        divided = path_in_repo.split(":")
        if len(divided) == 2:
            path_in_repo, revision = divided
            repo_type = "model"
        else:
            path_in_repo, revision, repo_type = divided
    logger.info(f"Downloading state from huggingface: {repo_id}/{path_in_repo}@{revision}")

    list_files = huggingface_util.list_dir(
        repo_id=repo_id,
        subfolder=path_in_repo,
        revision=revision,
        token=args.huggingface_token,
        repo_type=repo_type,
    )

    async def download(filename) -> str:
        def task():
            return hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type=repo_type,
                token=args.huggingface_token,
            )

        return await asyncio.get_event_loop().run_in_executor(None, task)

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(asyncio.gather(*[download(filename=filename.rfilename) for filename in list_files]))
    if len(results) == 0:
        raise ValueError(
            "No files found in the specified repo id/path/revision / 指定されたリポジトリID/パス/リビジョンにファイルが見つかりませんでした / 未在指定的 repo id/path/revision 中找到文件"
        )
    dirname = os.path.dirname(results[0])
    _ensure_scheduler_state_compatibility(accelerator, args, dirname)
    accelerator.load_state(dirname)


def _materialize_optimizer_params(trainable_params):
    """
    Normalize optimizer params into concrete lists so we can validate them before
    handing them to torch / bitsandbytes optimizers.
    """

    if isinstance(trainable_params, dict):
        trainable_params = [trainable_params]

    items = list(trainable_params)
    if len(items) == 0:
        return [], 0, 0

    if all(isinstance(item, dict) for item in items):
        normalized_groups = []
        tensor_count = 0
        element_count = 0

        for group in items:
            normalized_group = dict(group)
            params = list(normalized_group.get("params", []))
            if len(params) == 0:
                continue
            normalized_group["params"] = params
            normalized_groups.append(normalized_group)
            tensor_count += len(params)
            element_count += sum(p.numel() for p in params if getattr(p, "requires_grad", True))

        return normalized_groups, tensor_count, element_count

    params = list(items)
    tensor_count = len(params)
    element_count = sum(p.numel() for p in params if getattr(p, "requires_grad", True))
    return params, tensor_count, element_count


def get_optimizer(args, trainable_params) -> tuple[str, str, object]:
    # "Optimizer to use: AdamW, AdamW8bit, Lion, SGDNesterov, SGDNesterov8bit, PagedAdamW, PagedAdamW8bit, PagedAdamW32bit, Lion8bit, PagedLion8bit, AdEMAMix8bit, PagedAdEMAMix8bit, DAdaptation(DAdaptAdamPreprint), DAdaptAdaGrad, DAdaptAdam, DAdaptAdan, DAdaptAdanIP, DAdaptLion, DAdaptSGD, Adafactor"

    trainable_params, optimizer_tensor_count, optimizer_element_count = _materialize_optimizer_params(trainable_params)
    if optimizer_tensor_count == 0 or optimizer_element_count == 0:
        raise ValueError(
            "No trainable parameters were collected for the optimizer. "
            "This usually means the selected training targets cancel each other out "
            "(for example both network_train_unet_only and network_train_text_encoder_only are enabled), "
            "or all candidate modules were filtered/frozen by the current network settings, "
            "or every active learning rate resolved to 0."
        )

    optimizer_type = optimizer_util.resolve_optimizer_type(args, logger)
    optimizer_util.validate_optimizer_choice(args, optimizer_type)
    optimizer_kwargs = optimizer_util.parse_optimizer_kwargs(args, logger)
    lr = args.learning_rate
    optimizer_class, optimizer = optimizer_util.build_optimizer(
        args=args,
        trainable_params=trainable_params,
        optimizer_type=optimizer_type,
        optimizer_kwargs=optimizer_kwargs,
        lr=lr,
        logger=logger,
    )

    """
    # wrap any of above optimizer with schedulefree, if optimizer is not schedulefree
    if args.optimizer_schedulefree_wrapper and not optimizer_type.endswith("schedulefree".lower()):
        try:
            import schedulefree as sf
        except ImportError:
            raise ImportError("No schedulefree / schedulefreeがインストールされていないようです / 未安装 schedulefree")

        schedulefree_wrapper_kwargs = {}
        if args.schedulefree_wrapper_args is not None and len(args.schedulefree_wrapper_args) > 0:
            for arg in args.schedulefree_wrapper_args:
                key, value = arg.split("=")
                value = ast.literal_eval(value)
                schedulefree_wrapper_kwargs[key] = value

        sf_wrapper = sf.ScheduleFreeWrapper(optimizer, **schedulefree_wrapper_kwargs)
        sf_wrapper.train()  # make optimizer as train mode

        # we need to make optimizer as a subclass of torch.optim.Optimizer, we make another Proxy class over SFWrapper
        class OptimizerProxy(torch.optim.Optimizer):
            def __init__(self, sf_wrapper):
                self._sf_wrapper = sf_wrapper

            def __getattr__(self, name):
                return getattr(self._sf_wrapper, name)

            # override properties
            @property
            def state(self):
                return self._sf_wrapper.state

            @state.setter
            def state(self, state):
                self._sf_wrapper.state = state

            @property
            def param_groups(self):
                return self._sf_wrapper.param_groups

            @param_groups.setter
            def param_groups(self, param_groups):
                self._sf_wrapper.param_groups = param_groups

            @property
            def defaults(self):
                return self._sf_wrapper.defaults

            @defaults.setter
            def defaults(self, defaults):
                self._sf_wrapper.defaults = defaults

            def add_param_group(self, param_group):
                self._sf_wrapper.add_param_group(param_group)

            def load_state_dict(self, state_dict):
                self._sf_wrapper.load_state_dict(state_dict)

            def state_dict(self):
                return self._sf_wrapper.state_dict()

            def zero_grad(self):
                self._sf_wrapper.zero_grad()

            def step(self, closure=None):
                self._sf_wrapper.step(closure)

            def train(self):
                self._sf_wrapper.train()

            def eval(self):
                self._sf_wrapper.eval()

            # isinstance チェックをパスするためのメソッド
            def __instancecheck__(self, instance):
                return isinstance(instance, (type(self), Optimizer))

        optimizer = OptimizerProxy(sf_wrapper)

        logger.info(f"wrap optimizer with ScheduleFreeWrapper | {schedulefree_wrapper_kwargs}")
    """

    # for logging
    optimizer_name = optimizer_class.__module__ + "." + optimizer_class.__name__
    optimizer_args = ",".join([f"{k}={v}" for k, v in optimizer_kwargs.items()])

    if hasattr(optimizer, "train") and callable(optimizer.train):
        # make optimizer as train mode before training for schedulefree optimizer. the optimizer will be in eval mode in sampling and saving.
        optimizer.train()

    return optimizer_name, optimizer_args, optimizer


def get_optimizer_train_eval_fn(optimizer: Optimizer, args: argparse.Namespace) -> Tuple[Callable, Callable]:
    return optimizer_scheduler_util.get_optimizer_train_eval_fn(optimizer, args)


def is_schedulefree_optimizer(optimizer: Optimizer, args: argparse.Namespace) -> bool:
    return optimizer_scheduler_util.is_schedulefree_optimizer(optimizer, args)


def get_dummy_scheduler(optimizer: Optimizer) -> Any:
    return optimizer_scheduler_util.get_dummy_scheduler(optimizer)


# Modified version of get_scheduler() function from diffusers.optimizer.get_scheduler
# Add some checking and features to the original function.


def get_scheduler_fix(args, optimizer: Optimizer, num_processes: int):
    return optimizer_scheduler_util.get_scheduler_fix(args, optimizer, num_processes, logger)


def prepare_dataset_args(args: argparse.Namespace, support_metadata: bool):
    # backward compatibility
    if args.caption_extention is not None:
        args.caption_extension = args.caption_extention
        args.caption_extention = None

    cache_latents_runtime = resolve_cache_latents_runtime_kwargs(args)
    if hasattr(args, "cache_latents_cpu_workers"):
        args.cache_latents_cpu_workers = cache_latents_runtime["preprocess_workers"]
    if hasattr(args, "cache_latents_prefetch_batches"):
        args.cache_latents_prefetch_batches = cache_latents_runtime["prefetch_batches"]
    if hasattr(args, "latent_cache_disk_format"):
        args.latent_cache_disk_format = cache_latents_runtime["disk_cache_format"]
    configure_latents_cache_runtime(**cache_latents_runtime)

    if hasattr(args, "bucket_selection_mode") and args.bucket_selection_mode is not None:
        args.bucket_selection_mode = str(args.bucket_selection_mode).strip().lower()
    if hasattr(args, "caption_tag_dropout_target_mode") and args.caption_tag_dropout_target_mode is not None:
        args.caption_tag_dropout_target_mode = str(args.caption_tag_dropout_target_mode).strip().lower()
    if hasattr(args, "caption_tag_dropout_target_count"):
        args.caption_tag_dropout_target_count = max(1, int(args.caption_tag_dropout_target_count or 1))

    def _parse_scalar_or_twodim_resolution_arg(raw_value, arg_name: str):
        if raw_value is None:
            return None

        if isinstance(raw_value, (tuple, list)):
            parsed = tuple(int(r) for r in raw_value)
        else:
            parsed = tuple(int(r) for r in str(raw_value).split(","))

        if len(parsed) == 1:
            parsed = (parsed[0], parsed[0])
        assert (
            len(parsed) == 2
        ), f"{arg_name} must be 'size' or 'width,height' / {arg_name}（解像度）は'サイズ'または'幅','高さ'で指定してください / {arg_name} 需填写为“size”或“width,height”: {parsed}"
        return parsed

    # assert args.resolution is not None, f"resolution is required / resolution（解像度）を指定してください"
    args.resolution = _parse_scalar_or_twodim_resolution_arg(args.resolution, "resolution")
    if hasattr(args, "skip_image_resolution"):
        args.skip_image_resolution = _parse_scalar_or_twodim_resolution_arg(
            args.skip_image_resolution, "skip_image_resolution"
        )

    if args.face_crop_aug_range is not None:
        args.face_crop_aug_range = tuple([float(r) for r in args.face_crop_aug_range.split(",")])
        assert (
            len(args.face_crop_aug_range) == 2 and args.face_crop_aug_range[0] <= args.face_crop_aug_range[1]
        ), f"face_crop_aug_range must be two floats / face_crop_aug_rangeは'下限,上限'で指定してください / face_crop_aug_range 需要按“下限,上限”指定两个浮点数: {args.face_crop_aug_range}"
    else:
        args.face_crop_aug_range = None

    if support_metadata:
        if args.in_json is not None and (args.color_aug or args.random_crop):
            logger.warning(
                "disk-cached latents are ignored when color_aug or random_crop is True / "
                "color_augまたはrandom_cropを有効にした場合、ディスクキャッシュされた latent は無視されます / "
                "启用 color_aug 或 random_crop 时，磁盘缓存的 latents 会被忽略"
            )


def _runtime_flag_enabled(value, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def resolve_torch_compile_runtime(args: argparse.Namespace) -> tuple[bool, list[str]]:
    if not _runtime_flag_enabled(getattr(args, "torch_compile", False), default=False):
        return False, []

    reasons: list[str] = []
    if _runtime_flag_enabled(getattr(args, "deepspeed", False), default=False):
        reasons.append("deepspeed")
    if _runtime_flag_enabled(getattr(args, "sdxl_fixed_block_swap", False), default=False):
        reasons.append("sdxl_fixed_block_swap")
    if _runtime_flag_enabled(getattr(args, "sdxl_component_cpu_residency", False), default=False):
        reasons.append("sdxl_component_cpu_residency")

    if reasons:
        return False, reasons
    return True, []


def prepare_accelerator(args: argparse.Namespace):
    """
    this function also prepares deepspeed plugin
    """

    fixed_logging_dir = getattr(args, "logging_run_dir", None)
    if fixed_logging_dir is not None and str(fixed_logging_dir).strip() != "":
        logging_dir = str(fixed_logging_dir).strip()
    elif args.logging_dir is None:
        logging_dir = None
    else:
        log_prefix = "" if args.log_prefix is None else args.log_prefix
        logging_dir = args.logging_dir + "/" + log_prefix + time.strftime("%Y%m%d%H%M%S", time.localtime())

    if args.log_with is None:
        if logging_dir is not None:
            log_with = "tensorboard"
        else:
            log_with = None
    else:
        log_with = args.log_with
        if log_with in ["tensorboard", "all"]:
            if logging_dir is None:
                raise ValueError(
                    "logging_dir is required when log_with is tensorboard / Tensorboardを使う場合、logging_dirを指定してください / 当 log_with 使用 tensorboard 时必须指定 logging_dir"
                )
        if log_with in ["wandb", "all"]:
            try:
                import wandb
            except ImportError:
                raise ImportError("No wandb / wandb がインストールされていないようです / 未安装 wandb")
            if logging_dir is not None:
                os.makedirs(logging_dir, exist_ok=True)
                os.environ["WANDB_DIR"] = logging_dir
            if args.wandb_api_key is not None:
                wandb.login(key=args.wandb_api_key)

    # torch.compile のオプション。 NO の場合は torch.compile は使わない
    torch_compile_enabled, compile_guard_reasons = resolve_torch_compile_runtime(args)
    if not torch_compile_enabled and _runtime_flag_enabled(getattr(args, "torch_compile", False), default=False):
        logger.warning(
            "torch.compile was requested but will be disabled for runtime stability because: "
            + ", ".join(compile_guard_reasons)
        )
        logger.warning(
            "已请求 torch.compile，但为保证运行时稳定性将自动关闭，原因："
            + "、".join(compile_guard_reasons)
        )
        args.torch_compile = False

    dynamo_backend = "NO"
    if torch_compile_enabled:
        dynamo_backend = args.dynamo_backend
        logger.info(f"Enable torch.compile for training (dynamo backend: {dynamo_backend})")
        logger.info(f"当前已启用 torch.compile（dynamo 后端：{dynamo_backend}）")

    kwargs_handlers = [
        (
            InitProcessGroupKwargs(
                backend="gloo" if os.name == "nt" or not torch.cuda.is_available() else "nccl",
                init_method=(
                    "env://?use_libuv=False" if os.name == "nt" and Version(torch.__version__) >= Version("2.4.0") else None
                ),
                timeout=datetime.timedelta(minutes=args.ddp_timeout) if args.ddp_timeout else None,
            )
            if torch.cuda.device_count() > 1
            else None
        ),
        (
            DistributedDataParallelKwargs(
                gradient_as_bucket_view=args.ddp_gradient_as_bucket_view, static_graph=args.ddp_static_graph
            )
            if args.ddp_gradient_as_bucket_view or args.ddp_static_graph
            else None
        ),
    ]
    kwargs_handlers = [i for i in kwargs_handlers if i is not None]
    deepspeed_plugin = deepspeed_utils.prepare_deepspeed_plugin(args)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=log_with,
        project_dir=logging_dir,
        kwargs_handlers=kwargs_handlers,
        dynamo_backend=dynamo_backend,
        deepspeed_plugin=deepspeed_plugin,
    )
    print("accelerator device:", accelerator.device)
    return accelerator


def prepare_dtype(args: argparse.Namespace):
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    save_dtype = None
    if args.save_precision == "fp16":
        save_dtype = torch.float16
    elif args.save_precision == "bf16":
        save_dtype = torch.bfloat16
    elif args.save_precision == "float":
        save_dtype = torch.float32

    return weight_dtype, save_dtype


def _load_target_model(args: argparse.Namespace, weight_dtype, device="cpu", unet_use_linear_projection_in_v2=False):
    name_or_path = args.pretrained_model_name_or_path
    name_or_path = os.path.realpath(name_or_path) if os.path.islink(name_or_path) else name_or_path
    load_stable_diffusion_format = os.path.isfile(name_or_path)  # determine SD or Diffusers
    if load_stable_diffusion_format:
        logger.info(f"load StableDiffusion checkpoint: {name_or_path}")
        text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(
            args.v2, name_or_path, device, unet_use_linear_projection_in_v2=unet_use_linear_projection_in_v2
        )
    else:
        # Diffusers model is loaded to CPU
        logger.info(f"load Diffusers pretrained models: {name_or_path}")
        try:
            pipe = StableDiffusionPipeline.from_pretrained(name_or_path, tokenizer=None, safety_checker=None)
        except EnvironmentError as ex:
            logger.error(
                f"model is not found as a file or in Hugging Face, perhaps file name is wrong? / 指定したモデル名のファイル、またはHugging Faceのモデルが見つかりません。ファイル名が誤っているかもしれません: {name_or_path} / 未在本地文件或 Hugging Face 中找到该模型，文件名可能有误: {name_or_path}"
            )
            raise ex
        text_encoder = pipe.text_encoder
        vae = pipe.vae
        unet = pipe.unet
        del pipe

        # Diffusers U-Net to original U-Net
        # TODO *.ckpt/*.safetensorsのv2と同じ形式にここで変換すると良さそう
        # logger.info(f"unet config: {unet.config}")
        original_unet = UNet2DConditionModel(
            unet.config.sample_size,
            unet.config.attention_head_dim,
            unet.config.cross_attention_dim,
            unet.config.use_linear_projection,
            unet.config.upcast_attention,
        )
        original_unet.load_state_dict(unet.state_dict())
        unet = original_unet
        logger.info("U-Net converted to original U-Net")

    # VAEを読み込む
    if args.vae is not None:
        vae = model_util.load_vae(args.vae, weight_dtype)
        logger.info("additional VAE loaded")

    return text_encoder, vae, unet, load_stable_diffusion_format


def load_target_model(args, weight_dtype, accelerator, unet_use_linear_projection_in_v2=False):
    for pi in range(accelerator.state.num_processes):
        if pi == accelerator.state.local_process_index:
            logger.info(f"loading model for process {accelerator.state.local_process_index}/{accelerator.state.num_processes}")

            text_encoder, vae, unet, load_stable_diffusion_format = _load_target_model(
                args,
                weight_dtype,
                accelerator.device if args.lowram else "cpu",
                unet_use_linear_projection_in_v2=unet_use_linear_projection_in_v2,
            )
            # work on low-ram device
            if args.lowram:
                text_encoder.to(accelerator.device)
                unet.to(accelerator.device)
                vae.to(accelerator.device)

            clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()
    return text_encoder, vae, unet, load_stable_diffusion_format


def patch_accelerator_for_fp16_training(accelerator):

    from accelerate import DistributedType

    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        return

    org_unscale_grads = accelerator.scaler._unscale_grads_

    def _unscale_grads_replacer(optimizer, inv_scale, found_inf, allow_fp16):
        return org_unscale_grads(optimizer, inv_scale, found_inf, True)

    accelerator.scaler._unscale_grads_ = _unscale_grads_replacer


def get_hidden_states(args: argparse.Namespace, input_ids, tokenizer, text_encoder, weight_dtype=None):
    # with no_token_padding, the length is not max length, return result immediately
    if input_ids.size()[-1] != tokenizer.model_max_length:
        return text_encoder(input_ids)[0]

    # input_ids: b,n,77
    b_size = input_ids.size()[0]
    input_ids = input_ids.reshape((-1, tokenizer.model_max_length))  # batch_size*3, 77

    if args.clip_skip is None:
        encoder_hidden_states = text_encoder(input_ids)[0]
    else:
        enc_out = text_encoder(input_ids, output_hidden_states=True, return_dict=True)
        encoder_hidden_states = enc_out["hidden_states"][-args.clip_skip]
        encoder_hidden_states = text_encoder.text_model.final_layer_norm(encoder_hidden_states)

    # bs*3, 77, 768 or 1024
    encoder_hidden_states = encoder_hidden_states.reshape((b_size, -1, encoder_hidden_states.shape[-1]))

    if args.max_token_length is not None:
        if args.v2:
            # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, args.max_token_length, tokenizer.model_max_length):
                chunk = encoder_hidden_states[:, i : i + tokenizer.model_max_length - 2]  # <BOS> の後から 最後の前まで
                if i > 0:
                    for j in range(len(chunk)):
                        if input_ids[j, 1] == tokenizer.eos_token:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
                            chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
                states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
            encoder_hidden_states = torch.cat(states_list, dim=1)
        else:
            # v1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
            states_list = [encoder_hidden_states[:, 0].unsqueeze(1)]  # <BOS>
            for i in range(1, args.max_token_length, tokenizer.model_max_length):
                states_list.append(
                    encoder_hidden_states[:, i : i + tokenizer.model_max_length - 2]
                )  # <BOS> の後から <EOS> の前まで
            states_list.append(encoder_hidden_states[:, -1].unsqueeze(1))  # <EOS>
            encoder_hidden_states = torch.cat(states_list, dim=1)

    if weight_dtype is not None:
        # this is required for additional network training
        encoder_hidden_states = encoder_hidden_states.to(weight_dtype)

    return encoder_hidden_states


def pool_workaround(
    text_encoder: CLIPTextModelWithProjection, last_hidden_state: torch.Tensor, input_ids: torch.Tensor, eos_token_id: int
):
    r"""
    workaround for CLIP's pooling bug: it returns the hidden states for the max token id as the pooled output
    instead of the hidden states for the EOS token
    If we use Textual Inversion, we need to use the hidden states for the EOS token as the pooled output

    Original code from CLIP's pooling function:

    \# text_embeds.shape = [batch_size, sequence_length, transformer.width]
    \# take features from the eot embedding (eot_token is the highest number in each sequence)
    \# casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
    pooled_output = last_hidden_state[
        torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
        input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
    ]
    """

    # input_ids: b*n,77
    # find index for EOS token

    # Following code is not working if one of the input_ids has multiple EOS tokens (very odd case)
    # eos_token_index = torch.where(input_ids == eos_token_id)[1]
    # eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # Create a mask where the EOS tokens are
    eos_token_mask = (input_ids == eos_token_id).int()

    # Use argmax to find the last index of the EOS token for each element in the batch
    eos_token_index = torch.argmax(eos_token_mask, dim=1)  # this will be 0 if there is no EOS token, it's fine
    eos_token_index = eos_token_index.to(device=last_hidden_state.device)

    # get hidden states for EOS token
    pooled_output = last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), eos_token_index]

    # apply projection: projection may be of different dtype than last_hidden_state
    pooled_output = text_encoder.text_projection(pooled_output.to(text_encoder.text_projection.weight.dtype))
    pooled_output = pooled_output.to(last_hidden_state.dtype)

    return pooled_output


def get_hidden_states_sdxl(
    max_token_length: int,
    input_ids1: torch.Tensor,
    input_ids2: torch.Tensor,
    tokenizer1: CLIPTokenizer,
    tokenizer2: CLIPTokenizer,
    text_encoder1: CLIPTextModel,
    text_encoder2: CLIPTextModelWithProjection,
    weight_dtype: Optional[str] = None,
    accelerator: Optional[Accelerator] = None,
):
    from library import strategy_sdxl

    # input_ids: b,n,77 -> b*n, 77
    input_ids1 = strategy_sdxl.normalize_sdxl_token_tensor(input_ids1, tokenizer1.model_max_length)
    input_ids2 = strategy_sdxl.normalize_sdxl_token_tensor(input_ids2, tokenizer2.model_max_length)
    b_size = input_ids1.size()[0]
    if input_ids1.size()[1] == 1:
        max_token_length = None
    else:
        max_token_length = input_ids1.size()[1] * input_ids1.size()[2]
    input_ids1 = input_ids1.reshape((-1, tokenizer1.model_max_length))  # batch_size*n, 77
    input_ids2 = input_ids2.reshape((-1, tokenizer2.model_max_length))  # batch_size*n, 77

    # text_encoder1
    enc_out = text_encoder1(input_ids1, output_hidden_states=True, return_dict=True)
    hidden_states1 = enc_out["hidden_states"][11]

    # text_encoder2
    enc_out = text_encoder2(input_ids2, output_hidden_states=True, return_dict=True)
    hidden_states2 = enc_out["hidden_states"][-2]  # penuultimate layer

    # pool2 = enc_out["text_embeds"]
    unwrapped_text_encoder2 = text_encoder2 if accelerator is None else accelerator.unwrap_model(text_encoder2)
    pool2 = pool_workaround(unwrapped_text_encoder2, enc_out["last_hidden_state"], input_ids2, tokenizer2.eos_token_id)

    # b*n, 77, 768 or 1280 -> b, n*77, 768 or 1280
    n_size = 1 if max_token_length is None else max_token_length // 75
    hidden_states1 = hidden_states1.reshape((b_size, -1, hidden_states1.shape[-1]))
    hidden_states2 = hidden_states2.reshape((b_size, -1, hidden_states2.shape[-1]))

    if max_token_length is not None:
        # bs*3, 77, 768 or 1024
        # encoder1: <BOS>...<EOS> の三連を <BOS>...<EOS> へ戻す
        states_list = [hidden_states1[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer1.model_max_length):
            states_list.append(hidden_states1[:, i : i + tokenizer1.model_max_length - 2])  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states1[:, -1].unsqueeze(1))  # <EOS>
        hidden_states1 = torch.cat(states_list, dim=1)

        # v2: <BOS>...<EOS> <PAD> ... の三連を <BOS>...<EOS> <PAD> ... へ戻す　正直この実装でいいのかわからん
        states_list = [hidden_states2[:, 0].unsqueeze(1)]  # <BOS>
        for i in range(1, max_token_length, tokenizer2.model_max_length):
            chunk = hidden_states2[:, i : i + tokenizer2.model_max_length - 2]  # <BOS> の後から 最後の前まで
            # this causes an error:
            # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation
            # if i > 1:
            #     for j in range(len(chunk)):  # batch_size
            #         if input_ids2[n_index + j * n_size, 1] == tokenizer2.eos_token_id:  # 空、つまり <BOS> <EOS> <PAD> ...のパターン
            #             chunk[j, 0] = chunk[j, 1]  # 次の <PAD> の値をコピーする
            states_list.append(chunk)  # <BOS> の後から <EOS> の前まで
        states_list.append(hidden_states2[:, -1].unsqueeze(1))  # <EOS> か <PAD> のどちらか
        hidden_states2 = torch.cat(states_list, dim=1)

        # pool はnの最初のものを使う
        pool2 = pool2[::n_size]

    if weight_dtype is not None:
        # this is required for additional network training
        hidden_states1 = hidden_states1.to(weight_dtype)
        hidden_states2 = hidden_states2.to(weight_dtype)

    return hidden_states1, hidden_states2, pool2

__all__ = [
    'resume_from_local_or_hf_if_specified',
    'get_optimizer',
    'get_optimizer_train_eval_fn',
    'is_schedulefree_optimizer',
    'get_dummy_scheduler',
    'get_scheduler_fix',
    'prepare_dataset_args',
    'resolve_torch_compile_runtime',
    'prepare_accelerator',
    'prepare_dtype',
    'load_target_model',
    'patch_accelerator_for_fp16_training',
    'get_hidden_states',
    'pool_workaround',
    'get_hidden_states_sdxl',
]
