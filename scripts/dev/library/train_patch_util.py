# common functions for training

import argparse
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
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
from library.strategy_base import LatentsCachingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy, TextEncodingStrategy

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
from library import custom_train_functions, optimizer_scheduler_util, optimizer_util, sd3_utils
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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)).decode("ascii").strip()
    except:
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
def replace_unet_modules(unet: UNet2DConditionModel, mem_eff_attn, xformers, sdpa):
    if mem_eff_attn:
        logger.info("Enable memory efficient attention for U-Net")
        unet.set_use_memory_efficient_attention(False, True)
    elif xformers:
        logger.info("Enable xformers for U-Net")
        try:
            import xformers.ops
        except ImportError:
            raise ImportError("No xformers / xformersがインストールされていないようです")

        unet.set_use_memory_efficient_attention(True, False)
    elif sdpa:
        logger.info("Enable SDPA for U-Net")
        unet.set_use_sdpa(True)


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
    'replace_unet_modules',
    'maybe_apply_channels_last_to_tensor',
]
