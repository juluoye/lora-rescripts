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

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def model_hash(filename):
    """Old model hash used by stable-diffusion-webui"""
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def calculate_sha256(filename):
    """New model hash used by stable-diffusion-webui"""
    try:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def get_git_revision_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(__file__),
                stderr=subprocess.DEVNULL,
            )
            .decode("ascii")
            .strip()
        )
    except Exception:
        return "(unknown)"


# def replace_unet_modules(unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, mem_eff_attn, xformers):
#     replace_attentions_for_hypernetwork()
#     # unet is not used currently, but it is here for future use
#     unet.enable_xformers_memory_efficient_attention()
#     return
#     if mem_eff_attn:
#         unet.set_attn_processor(FlashAttnProcessor())
#     elif xformers:
#         unet.enable_xformers_memory_efficient_attention()


# def replace_unet_cross_attn_to_xformers():
#     logger.info("CrossAttention.forward has been replaced to enable xformers.")
#     try:
#         import xformers.ops
#     except ImportError:
#         raise ImportError("No xformers / xformersがインストールされていないようです")

#     def forward_xformers(self, x, context=None, mask=None):
#         h = self.heads
#         q_in = self.to_q(x)

#         context = default(context, x)
#         context = context.to(x.dtype)

#         if hasattr(self, "hypernetwork") and self.hypernetwork is not None:
#             context_k, context_v = self.hypernetwork.forward(x, context)
#             context_k = context_k.to(x.dtype)
#             context_v = context_v.to(x.dtype)
#         else:
#             context_k = context
#             context_v = context

#         k_in = self.to_k(context_k)
#         v_in = self.to_v(context_v)

#         q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
#         del q_in, k_in, v_in

#         q = q.contiguous()
#         k = k.contiguous()
#         v = v.contiguous()
#         out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)  # 最適なのを選んでくれる

#         out = rearrange(out, "b n h d -> b n (h d)", h=h)

#         # diffusers 0.7.0~
#         out = self.to_out[0](out)
#         out = self.to_out[1](out)
#         return out


#     diffusers.models.attention.CrossAttention.forward = forward_xformers
def enable_sageattention(*named_models):
    logger.info("Enable SageAttention")
    logger.info("Training attention backend: sageattn")
    logger.info("当前训练使用的注意力后端：sageattn")
    try:
        _sageattn, _, _sageattention_source = load_runtime_sageattention_symbols()  # noqa: F841
    except Exception:
        raise ImportError("No SageAttention / SageAttentionがインストールされていないようです / 未检测到 SageAttention")

    for model_name, model in named_models:
        if model is None:
            raise ValueError(f"{model_name} is missing / {model_name} が存在しません / {model_name} 不存在")
        if not hasattr(model, "set_use_sageattn"):
            raise ValueError(
                f"SageAttention is not supported by {model_name} / "
                f"{model_name} は SageAttention をサポートしていません / {model_name} 当前不支持 SageAttention"
            )
        model.set_use_sageattn(True)


def _flashattention_available() -> tuple[bool, str]:
    if not torch.cuda.is_available():
        return False, "CUDA is not available"
    if bool(getattr(torch.version, "hip", None)):
        return False, "FlashAttention 2 is not supported on ROCm in this runtime"
    if unified_attention.flash_attn_func is None:
        return False, "flash-attn is not installed"

    try:
        capability = torch.cuda.get_device_capability(torch.cuda.current_device())
    except Exception:
        capability = None

    if capability is not None and capability < (8, 0):
        return False, f"GPU capability {capability} is below SM80"

    return True, "ok"


def enable_flashattention(*named_models):
    logger.info("Enable FlashAttention for training models")
    logger.info("Training attention backend: flashattn")
    logger.info("当前训练使用的注意力后端：flashattn")

    flashattention_ok, flashattention_reason = _flashattention_available()
    if not flashattention_ok:
        raise ImportError(
            "FlashAttention 2 is unavailable in the current runtime"
            f" / 当前运行时不可用 FlashAttention 2: {flashattention_reason}"
        )

    for model_name, model in named_models:
        if model is None:
            raise ValueError(f"{model_name} is missing / {model_name} が存在しません / {model_name} 不存在")
        if not hasattr(model, "set_use_flashattn"):
            raise ValueError(
                f"FlashAttention is not supported by {model_name} / "
                f"{model_name} は FlashAttention をサポートしていません / {model_name} 当前不支持 FlashAttention"
            )
        model.set_use_flashattn(True)


def replace_unet_modules(
    unet: UNet2DConditionModel,
    mem_eff_attn,
    xformers,
    sdpa,
    sageattn=False,
    flashattn=False,
    cross_attn_fused_kv=False,
):
    if mem_eff_attn:
        logger.info("Enable memory efficient attention for U-Net")
        logger.info("Training attention backend: mem_eff_attn")
        logger.info("当前训练使用的注意力后端：mem_eff_attn")
        unet.set_use_memory_efficient_attention(False, True)
    elif flashattn:
        enable_flashattention(("U-Net", unet))
    elif sageattn:
        enable_sageattention(("U-Net", unet))
    elif xformers:
        logger.info("Enable xformers for U-Net")
        logger.info("Training attention backend: xformers")
        logger.info("当前训练使用的注意力后端：xformers")
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("No xformers / xformersがインストールされていないようです / 未检测到 xformers")

        unet.set_use_memory_efficient_attention(True, False)
    elif sdpa:
        logger.info("Enable SDPA for U-Net")
        logger.info("Training attention backend: sdpa")
        logger.info("当前训练使用的注意力后端：sdpa")
        unet.set_use_sdpa(True)

    if cross_attn_fused_kv:
        if hasattr(unet, "set_use_cross_attn_fused_kv"):
            logger.info("Enable experimental fused K/V projection for SDXL cross attention")
            logger.info("当前已启用 SDXL cross-attn 的 fused K/V projection 实验开关")
            unet.set_use_cross_attn_fused_kv(True)
        else:
            logger.warning("cross_attn_fused_kv was requested, but the current U-Net does not expose this experimental hook.")


def _module_has_channels_last_candidate(module: torch.nn.Module) -> bool:
    for tensor in list(module.parameters(recurse=True)) + list(module.buffers(recurse=True)):
        if tensor is not None and tensor.is_floating_point() and tensor.ndim == 4:
            return True
    return False


def apply_opt_channels_last(args, *named_models):
    if not getattr(args, "opt_channels_last", False):
        return []

    applied = []
    skipped = []

    for item in named_models:
        if isinstance(item, tuple):
            display_name, model = item
        else:
            display_name, model = type(item).__name__, item

        if model is None:
            continue

        if not _module_has_channels_last_candidate(model):
            skipped.append(display_name)
            continue

        try:
            model.to(memory_format=torch.channels_last)
            applied.append(display_name)
        except Exception as exc:
            logger.warning(f"Failed to enable channels_last for {display_name}: {exc}")

    if applied:
        logger.info("Enable channels_last memory format")
        logger.info(f"channels_last applied to: {', '.join(applied)}")
        logger.info(f"当前已启用 channels_last 内存格式：{', '.join(applied)}")
    elif skipped:
        logger.info(
            "channels_last was requested, but no 4D convolution-style weights were found in the selected training models."
        )
        logger.info("当前已请求 channels_last，但当前训练主模型中未检测到适合切换的 4D 卷积权重。")

    return applied


def maybe_apply_channels_last_to_tensor(args, tensor: Optional[torch.Tensor]):
    if not getattr(args, "opt_channels_last", False):
        return tensor
    if tensor is None or not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
        return tensor
    return tensor.contiguous(memory_format=torch.channels_last)


"""
def replace_vae_modules(vae: diffusers.models.AutoencoderKL, mem_eff_attn, xformers):
    # vae is not used currently, but it is here for future use
    if mem_eff_attn:
        replace_vae_attn_to_memory_efficient()
    elif xformers:
        # とりあえずDiffusersのxformersを使う。AttentionがあるのはMidBlockのみ
        logger.info("Use Diffusers xformers for VAE")
        vae.encoder.mid_block.attentions[0].set_use_memory_efficient_attention_xformers(True)
        vae.decoder.mid_block.attentions[0].set_use_memory_efficient_attention_xformers(True)


def replace_vae_attn_to_memory_efficient():
    logger.info("AttentionBlock.forward has been replaced to FlashAttention (not xformers)")
    flash_func = FlashAttentionFunction

    def forward_flash_attn(self, hidden_states):
        logger.info("forward_flash_attn")
        q_bucket_size = 512
        k_bucket_size = 1024

        residual = hidden_states
        batch, channel, height, width = hidden_states.shape

        # norm
        hidden_states = self.group_norm(hidden_states)

        hidden_states = hidden_states.view(batch, channel, height * width).transpose(1, 2)

        # proj to q, k, v
        query_proj = self.query(hidden_states)
        key_proj = self.key(hidden_states)
        value_proj = self.value(hidden_states)

        query_proj, key_proj, value_proj = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads), (query_proj, key_proj, value_proj)
        )

        out = flash_func.apply(query_proj, key_proj, value_proj, None, False, q_bucket_size, k_bucket_size)

        out = rearrange(out, "b h n d -> b n (h d)")

        # compute next hidden_states
        hidden_states = self.proj_attn(hidden_states)
        hidden_states = hidden_states.transpose(-1, -2).reshape(batch, channel, height, width)

        # res connect and rescale
        hidden_states = (hidden_states + residual) / self.rescale_output_factor
        return hidden_states

    diffusers.models.attention.AttentionBlock.forward = forward_flash_attn
"""


# endregion

__all__ = [
    'exists',
    'default',
    'model_hash',
    'calculate_sha256',
    'precalculate_safetensors_hashes',
    'addnet_hash_legacy',
    'addnet_hash_safetensors',
    'get_git_revision_hash',
    'enable_sageattention',
    'enable_flashattention',
    'replace_unet_modules',
    'apply_opt_channels_last',
    'maybe_apply_channels_last_to_tensor',
]
