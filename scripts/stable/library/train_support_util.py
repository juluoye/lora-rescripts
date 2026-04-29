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

class ImageLoadingDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths):
        self.images = image_paths

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            # convert to tensor temporarily so dataloader will accept it
            tensor_pil = transforms.functional.pil_to_tensor(image)
        except Exception as e:
            logger.error(f"Could not load image path / 画像を読み込めません / 无法读取图像: {img_path}, error: {e}")
            return None

        return (tensor_pil, img_path)


# endregion


# collate_fn用 epoch,stepはmultiprocessing.Value
class collator_class:
    def __init__(self, epoch, step, dataset):
        self.current_epoch = epoch
        self.current_step = step
        self.dataset = dataset  # not used if worker_info is not None, in case of multiprocessing

    def __call__(self, examples):
        worker_info = torch.utils.data.get_worker_info()
        # worker_info is None in the main process
        if worker_info is not None:
            dataset = worker_info.dataset
        else:
            dataset = self.dataset

        # set epoch and step
        dataset.set_current_epoch(self.current_epoch.value)
        dataset.set_current_step(self.current_step.value)
        return examples[0]


class LossRecorder:
    def __init__(self):
        self.loss_list: List[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            while len(self.loss_list) <= step:
                self.loss_list.append(0.0)
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        losses = len(self.loss_list)
        if losses == 0:
            return 0
        return self.loss_total / losses


class SafeGuardDecision(NamedTuple):
    skip_step: bool
    stop_training: bool
    reason: Optional[str] = None


class TrainingSafeGuard:
    def __init__(
        self,
        *,
        nan_check_interval: int = 1,
        max_nan_count: int = 3,
        loss_spike_threshold: float = 5.0,
        loss_window_size: int = 20,
        auto_reduce_lr: bool = False,
        lr_reduction_factor: float = 0.5,
    ):
        self.nan_check_interval = max(1, int(nan_check_interval))
        self.max_nan_count = max(1, int(max_nan_count))
        self.loss_spike_threshold = max(1.0, float(loss_spike_threshold))
        self.loss_window_size = max(2, int(loss_window_size))
        self.auto_reduce_lr = bool(auto_reduce_lr)
        self.lr_reduction_factor = float(lr_reduction_factor)
        self.loss_window: deque[float] = deque(maxlen=self.loss_window_size)
        self.nan_count = 0
        self.skipped_steps = 0
        self.lr_reduction_count = 0

    def inspect_loss(self, loss_value: float, global_step: int, optimizer: Optional[Optimizer] = None) -> SafeGuardDecision:
        check_step = max(1, int(global_step))
        if check_step % self.nan_check_interval != 0:
            return SafeGuardDecision(False, False, None)

        if not math.isfinite(loss_value):
            self.nan_count += 1
            self.skipped_steps += 1
            message = (
                f"SafeGuard detected non-finite loss at step {global_step}. "
                f"Current consecutive bad count: {self.nan_count}/{self.max_nan_count}."
            )
            self._maybe_reduce_lr(optimizer, reason="non-finite loss")
            if self.nan_count >= self.max_nan_count:
                return SafeGuardDecision(True, True, message)
            return SafeGuardDecision(True, False, message)

        baseline = self.get_loss_baseline()
        if baseline is not None and baseline > 1e-12:
            threshold_value = baseline * self.loss_spike_threshold
            if loss_value > threshold_value:
                self.skipped_steps += 1
                message = (
                    f"SafeGuard skipped step {global_step} because loss spiked to {loss_value:.6f} "
                    f"(rolling avg {baseline:.6f}, threshold x{self.loss_spike_threshold:.2f})."
                )
                self._maybe_reduce_lr(optimizer, reason="loss spike")
                return SafeGuardDecision(True, False, message)

        return SafeGuardDecision(False, False, None)

    def record_loss(self, loss_value: float) -> None:
        if not math.isfinite(loss_value):
            return
        self.nan_count = 0
        self.loss_window.append(float(loss_value))

    def get_loss_baseline(self) -> Optional[float]:
        if len(self.loss_window) < self.loss_window_size:
            return None
        return float(sum(self.loss_window) / len(self.loss_window))

    def _maybe_reduce_lr(self, optimizer: Optional[Optimizer], *, reason: str) -> None:
        if optimizer is None or not self.auto_reduce_lr:
            return

        reduced = False
        for param_group in optimizer.param_groups:
            current_lr = param_group.get("lr")
            if current_lr is None:
                continue
            new_lr = max(float(current_lr) * self.lr_reduction_factor, 0.0)
            if new_lr != current_lr:
                param_group["lr"] = new_lr
                reduced = True

        if reduced:
            self.lr_reduction_count += 1
            logger.warning(
                f"SafeGuard reduced learning rates by factor {self.lr_reduction_factor:.4f} due to {reason}. "
                f"Total LR reductions: {self.lr_reduction_count}"
            )


class ModelEMA:
    def __init__(
        self,
        named_models: Sequence[Tuple[str, torch.nn.Module]],
        *,
        decay: float = 0.999,
        update_every: int = 1,
        update_after_step: int = 0,
        use_warmup: bool = False,
        inv_gamma: float = 1.0,
        power: float = 0.75,
    ):
        self.decay = min(max(float(decay), 0.0), 0.99999)
        self.update_every = max(1, int(update_every))
        self.update_after_step = max(0, int(update_after_step))
        self.use_warmup = bool(use_warmup)
        self.inv_gamma = max(float(inv_gamma), 1e-6)
        self.power = max(float(power), 1e-6)
        self.num_updates = 0
        self.entries: list[dict[str, Any]] = []

        for display_name, module in named_models:
            if module is None:
                continue
            shadow_params = {}
            for name, param in module.named_parameters():
                if not param.requires_grad or not param.dtype.is_floating_point:
                    continue
                shadow_params[name] = param.detach().clone()

            if shadow_params:
                self.entries.append(
                    {
                        "name": display_name,
                        "module": module,
                        "shadow_params": shadow_params,
                    }
                )

    @property
    def enabled(self) -> bool:
        return len(self.entries) > 0

    @property
    def tracked_model_names(self) -> list[str]:
        return [entry["name"] for entry in self.entries]

    def _current_decay(self) -> float:
        if not self.use_warmup:
            return self.decay

        warmup_step = max(0, self.num_updates - self.update_after_step)
        if warmup_step <= 0:
            return 0.0

        warmup_decay = 1.0 - (1.0 + warmup_step / self.inv_gamma) ** (-self.power)
        return min(self.decay, max(0.0, warmup_decay))

    def should_update(self, global_step: int) -> bool:
        if not self.enabled:
            return False
        if global_step <= self.update_after_step:
            return False
        return global_step % self.update_every == 0

    def update(self, global_step: int) -> bool:
        if not self.should_update(global_step):
            return False

        decay = self._current_decay()
        for entry in self.entries:
            named_params = dict(entry["module"].named_parameters())
            shadow_params = entry["shadow_params"]
            for name, shadow in shadow_params.items():
                param = named_params.get(name)
                if param is None:
                    continue
                target = param.detach()
                if shadow.device != target.device:
                    shadow = shadow.to(device=target.device, dtype=target.dtype)
                    shadow_params[name] = shadow
                elif shadow.dtype != target.dtype:
                    shadow = shadow.to(dtype=target.dtype)
                    shadow_params[name] = shadow

                shadow.mul_(decay).add_(target, alpha=1.0 - decay)

        self.num_updates += 1
        return True

    @contextmanager
    def apply_to_models(self):
        if not self.enabled:
            yield
            return

        backups: list[Tuple[torch.nn.Parameter, torch.Tensor]] = []
        try:
            for entry in self.entries:
                named_params = dict(entry["module"].named_parameters())
                shadow_params = entry["shadow_params"]
                for name, shadow in shadow_params.items():
                    param = named_params.get(name)
                    if param is None:
                        continue
                    backups.append((param, param.detach().clone()))
                    source = shadow
                    if source.device != param.device or source.dtype != param.dtype:
                        source = source.to(device=param.device, dtype=param.dtype)
                    param.data.copy_(source)
            yield
        finally:
            for param, backup in backups:
                param.data.copy_(backup)


def create_training_safeguard(args: argparse.Namespace) -> Optional[TrainingSafeGuard]:
    if not getattr(args, "safeguard_enabled", False):
        return None

    safeguard = TrainingSafeGuard(
        nan_check_interval=getattr(args, "safeguard_nan_check_interval", 1),
        max_nan_count=getattr(args, "safeguard_max_nan_count", 3),
        loss_spike_threshold=getattr(args, "safeguard_loss_spike_threshold", 5.0),
        loss_window_size=getattr(args, "safeguard_loss_window_size", 20),
        auto_reduce_lr=getattr(args, "safeguard_auto_reduce_lr", False),
        lr_reduction_factor=getattr(args, "safeguard_lr_reduction_factor", 0.5),
    )
    logger.info(
        "SafeGuard enabled: "
        f"window={safeguard.loss_window_size}, spike_threshold={safeguard.loss_spike_threshold:.2f}, "
        f"nan_check_interval={safeguard.nan_check_interval}, max_nan_count={safeguard.max_nan_count}, "
        f"auto_reduce_lr={safeguard.auto_reduce_lr}"
    )
    return safeguard


def create_model_ema(args: argparse.Namespace, named_models: Sequence[Tuple[str, torch.nn.Module]]) -> Optional[ModelEMA]:
    if not getattr(args, "ema_enabled", False):
        return None

    ema = ModelEMA(
        named_models,
        decay=getattr(args, "ema_decay", 0.999),
        update_every=getattr(args, "ema_update_every", 1),
        update_after_step=getattr(args, "ema_update_after_step", 0),
        use_warmup=getattr(args, "ema_use_warmup", False),
        inv_gamma=getattr(args, "ema_inv_gamma", 1.0),
        power=getattr(args, "ema_power", 0.75),
    )

    if not ema.enabled:
        logger.warning("EMA was requested but no floating-point trainable parameters were found. EMA is disabled for this run.")
        return None

    logger.info(
        "EMA enabled: "
        f"models={', '.join(ema.tracked_model_names)}, decay={ema.decay:.5f}, update_every={ema.update_every}, "
        f"update_after_step={ema.update_after_step}, warmup={ema.use_warmup}"
    )
    return ema


def call_with_ema(ema: Optional[ModelEMA], callback: Callable, *args, **kwargs):
    if ema is None:
        return callback(*args, **kwargs)
    with ema.apply_to_models():
        return callback(*args, **kwargs)

__all__ = [
    'ImageLoadingDataset',
    'collator_class',
    'LossRecorder',
    'SafeGuardDecision',
    'TrainingSafeGuard',
    'ModelEMA',
    'create_training_safeguard',
    'create_model_ema',
    'call_with_ema',
]
