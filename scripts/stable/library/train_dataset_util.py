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

HIGH_VRAM = False
_RUNTIME_BUCKET_POLICY = {
    "mode": None,
    "target_edge": None,
}


def set_high_vram(enabled: bool) -> None:
    global HIGH_VRAM
    HIGH_VRAM = bool(enabled)


def configure_bucket_runtime_policy(*, mode: Optional[str] = None, target_edge: Optional[int] = None) -> None:
    normalized_mode = str(mode or "").strip().lower() or None
    if normalized_mode not in {"long_edge", "short_edge"}:
        normalized_mode = None

    normalized_target_edge = None
    if target_edge is not None:
        try:
            normalized_target_edge = max(64, int(target_edge))
        except (TypeError, ValueError):
            normalized_target_edge = None

    _RUNTIME_BUCKET_POLICY["mode"] = normalized_mode
    _RUNTIME_BUCKET_POLICY["target_edge"] = normalized_target_edge


def get_bucket_runtime_policy() -> dict[str, Optional[int | str]]:
    return dict(_RUNTIME_BUCKET_POLICY)


# region dataset

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".PNG", ".JPG", ".JPEG", ".WEBP", ".BMP"]

try:
    import pillow_avif

    IMAGE_EXTENSIONS.extend([".avif", ".AVIF"])
except:
    pass

# JPEG-XL on Linux
try:
    from jxlpy import JXLImagePlugin
    from library.jpeg_xl_util import get_jxl_size

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

# JPEG-XL on Linux and Windows
try:
    import pillow_jxl
    from library.jpeg_xl_util import get_jxl_size

    IMAGE_EXTENSIONS.extend([".jxl", ".JXL"])
except:
    pass

IMAGE_TRANSFORMS = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]
)

TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX = "_te_outputs.npz"
TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3 = "_sd3_te.npz"


def split_train_val(
    paths: List[str],
    sizes: List[Optional[Tuple[int, int]]],
    is_training_dataset: bool,
    validation_split: float,
    validation_seed: int | None,
) -> Tuple[List[str], List[Optional[Tuple[int, int]]]]:
    """
    Split the dataset into train and validation

    Shuffle the dataset based on the validation_seed or the current random seed.
    For example if the split of 0.2 of 100 images.
    [0:80] = 80 training images
    [80:] = 20 validation images
    """
    dataset = list(zip(paths, sizes))
    if validation_seed is not None:
        logging.info(f"Using validation seed: {validation_seed}")
        prevstate = random.getstate()
        random.seed(validation_seed)
        random.shuffle(dataset)
        random.setstate(prevstate)
    else:
        random.shuffle(dataset)

    paths, sizes = zip(*dataset)
    paths = list(paths)
    sizes = list(sizes)
    # Split the dataset between training and validation
    if is_training_dataset:
        # Training dataset we split to the first part
        split = math.ceil(len(paths) * (1 - validation_split))
        return paths[0:split], sizes[0:split]
    else:
        # Validation dataset we split to the second part
        split = len(paths) - round(len(paths) * validation_split)
        return paths[split:], sizes[split:]


class ImageInfo:
    def __init__(
        self, image_key: str, num_repeats: int, caption: str, is_reg: bool, absolute_path: str, caption_dropout_rate: float = 0.0
    ) -> None:
        self.image_key: str = image_key
        self.num_repeats: int = num_repeats
        self.caption: str = caption
        self.is_reg: bool = is_reg
        self.absolute_path: str = absolute_path
        self.caption_dropout_rate: float = caption_dropout_rate
        self.image_size: Tuple[int, int] = None
        self.resized_size: Tuple[int, int] = None
        self.bucket_reso: Tuple[int, int] = None
        self.latents: Optional[torch.Tensor] = None
        self.latents_flipped: Optional[torch.Tensor] = None
        self.latents_npz: Optional[str] = None  # set in cache_latents
        self.latents_disk_cache_ref: Optional[Any] = None
        self.latents_cache_root: Optional[str] = None
        self.latents_original_size: Optional[Tuple[int, int]] = None  # original image size, not latents size
        self.latents_crop_ltrb: Optional[Tuple[int, int]] = (
            None  # crop left top right bottom in resized / target pixel space, not latent space
        )
        self.cond_img_path: Optional[str] = None
        self.image: Optional[Image.Image] = None  # optional, original PIL Image
        self.text_encoder_outputs_npz: Optional[str] = None  # filename. set in cache_text_encoder_outputs

        # new
        self.text_encoder_outputs: Optional[List[torch.Tensor]] = None
        # old
        self.text_encoder_outputs1: Optional[torch.Tensor] = None
        self.text_encoder_outputs2: Optional[torch.Tensor] = None
        self.text_encoder_pool2: Optional[torch.Tensor] = None

        self.alpha_mask: Optional[torch.Tensor] = None  # alpha mask can be flipped in runtime
        self.resize_interpolation: Optional[str] = None


def parse_tag_text_list(raw_text: Optional[str]) -> List[str]:
    if raw_text is None:
        return []
    text = str(raw_text).strip()
    if not text:
        return []
    parts = re.split(r"[\r\n,;]+", text)
    return [part.strip() for part in parts if part and part.strip()]


def normalize_tag_token(token: str) -> str:
    return re.sub(r"\s+", " ", str(token).strip()).lower()


def parse_bucket_resolution_list(raw_text: Optional[str], reso_steps: int) -> List[Tuple[int, int]]:
    resos: List[Tuple[int, int]] = []
    seen = set()
    if raw_text is None:
        return resos
    for match in re.finditer(r"(\d+)\s*(?:x|X|,)\s*(\d+)", str(raw_text)):
        width = int(match.group(1))
        height = int(match.group(2))
        if width < reso_steps or height < reso_steps:
            continue
        width = width - width % reso_steps
        height = height - height % reso_steps
        if width < reso_steps or height < reso_steps:
            continue
        reso = (width, height)
        if reso in seen:
            continue
        seen.add(reso)
        resos.append(reso)
    resos.sort()
    return resos


class BucketManager:
    def __init__(
        self,
        no_upscale,
        max_reso,
        min_size,
        max_size,
        reso_steps,
        bucket_selection_mode: str = "legacy",
        bucket_custom_resos: Optional[str] = None,
    ) -> None:
        if max_size is not None:
            if max_reso is not None:
                assert max_size >= max_reso[0], "the max_size should be larger than the width of max_reso"
                assert max_size >= max_reso[1], "the max_size should be larger than the height of max_reso"
            if min_size is not None:
                assert max_size >= min_size, "the max_size should be larger than the min_size"

        self.no_upscale = no_upscale
        if max_reso is None:
            self.max_reso = None
            self.max_area = None
        else:
            self.max_reso = max_reso
            self.max_area = max_reso[0] * max_reso[1]
        self.min_size = min_size
        self.max_size = max_size
        self.reso_steps = reso_steps
        self.bucket_selection_mode = str(bucket_selection_mode or "legacy").strip().lower()
        if self.bucket_selection_mode not in {"legacy", "nearest_only", "custom_only"}:
            logger.warning(
                f"unknown bucket_selection_mode: {bucket_selection_mode}, fallback to legacy / 未知 bucket_selection_mode，已回退到 legacy: {bucket_selection_mode}"
            )
            self.bucket_selection_mode = "legacy"
        self.bucket_custom_resos_raw = bucket_custom_resos
        runtime_bucket_policy = get_bucket_runtime_policy()
        self.runtime_resolution_mode = runtime_bucket_policy["mode"]
        self.runtime_target_edge = runtime_bucket_policy["target_edge"]

        self.resos = []
        self.reso_to_id = {}
        self.buckets = []  # 前処理時は (image_key, image, original size, crop left/top)、学習時は image_key

    def add_image(self, reso, image_or_info):
        bucket_id = self.reso_to_id[reso]
        self.buckets[bucket_id].append(image_or_info)

    def shuffle(self):
        for bucket in self.buckets:
            random.shuffle(bucket)

    def sort(self):
        # 解像度順にソートする（表示時、メタデータ格納時の見栄えをよくするためだけ）。bucketsも入れ替えてreso_to_idも振り直す
        sorted_resos = self.resos.copy()
        sorted_resos.sort()

        sorted_buckets = []
        sorted_reso_to_id = {}
        for i, reso in enumerate(sorted_resos):
            bucket_id = self.reso_to_id[reso]
            sorted_buckets.append(self.buckets[bucket_id])
            sorted_reso_to_id[reso] = i

        self.resos = sorted_resos
        self.buckets = sorted_buckets
        self.reso_to_id = sorted_reso_to_id

    def make_buckets(self):
        if self.bucket_selection_mode == "custom_only":
            resos = parse_bucket_resolution_list(self.bucket_custom_resos_raw, self.reso_steps)
            if not resos:
                raise ValueError(
                    "custom_only bucket mode requires valid bucket_custom_resos / custom_only 需要提供合法的 bucket_custom_resos"
                )
        else:
            resos = model_util.make_bucket_resolutions(self.max_reso, self.min_size, self.max_size, self.reso_steps)
        self.set_predefined_resos(resos)

    def make_buckets_by_nearest_image_aspect(self, image_sizes: Sequence[Tuple[int, int]]):
        assert self.max_area is not None, "max_area is required for nearest_only bucket mode"
        min_edge = self.reso_steps if self.min_size is None else max(self.reso_steps, self.min_size)
        resos = set()
        for image_width, image_height in image_sizes:
            if image_width is None or image_height is None or image_width <= 0 or image_height <= 0:
                continue
            aspect_ratio = image_width / image_height
            target_width = math.sqrt(self.max_area * aspect_ratio)
            target_height = self.max_area / target_width

            b_width_rounded = max(min_edge, self.round_to_steps(target_width))
            b_height_in_wr = max(min_edge, self.round_to_steps(b_width_rounded / aspect_ratio))
            ar_width_rounded = b_width_rounded / b_height_in_wr

            b_height_rounded = max(min_edge, self.round_to_steps(target_height))
            b_width_in_hr = max(min_edge, self.round_to_steps(b_height_rounded * aspect_ratio))
            ar_height_rounded = b_width_in_hr / b_height_rounded

            if abs(ar_width_rounded - aspect_ratio) <= abs(ar_height_rounded - aspect_ratio):
                resos.add((b_width_rounded, b_height_in_wr))
            else:
                resos.add((b_width_in_hr, b_height_rounded))

        if not resos and self.max_reso is not None:
            resos.add(self.max_reso)
        self.set_predefined_resos(sorted(resos))

    def set_predefined_resos(self, resos):
        # 規定サイズから選ぶ場合の解像度、aspect ratioの情報を格納しておく
        self.predefined_resos = resos.copy()
        self.predefined_resos_set = set(resos)
        self.predefined_aspect_ratios = np.array([w / h for w, h in resos])

    def add_if_new_reso(self, reso):
        if reso not in self.reso_to_id:
            bucket_id = len(self.resos)
            self.reso_to_id[reso] = bucket_id
            self.resos.append(reso)
            self.buckets.append([])
            # logger.info(reso, bucket_id, len(self.buckets))

    def round_to_steps(self, x):
        x = int(x + 0.5)
        return x - x % self.reso_steps

    def select_bucket(self, image_width, image_height):
        aspect_ratio = image_width / image_height
        use_predefined_buckets = not self.no_upscale or self.bucket_selection_mode != "legacy"
        if use_predefined_buckets:
            # 拡大および縮小を行う
            # 同じaspect ratioがあるかもしれないので（fine tuningで、no_upscale=Trueで前処理した場合）、解像度が同じものを優先する
            reso = (image_width, image_height)
            if reso in self.predefined_resos_set:
                pass
            else:
                ar_errors = self.predefined_aspect_ratios - aspect_ratio
                predefined_bucket_id = np.abs(ar_errors).argmin()  # 当該解像度以外でaspect ratio errorが最も少ないもの
                reso = self.predefined_resos[predefined_bucket_id]

            ar_reso = reso[0] / reso[1]
            if aspect_ratio > ar_reso:  # 横が長い→縦を合わせる
                scale = reso[1] / image_height
            else:
                scale = reso[0] / image_width

            resized_size = (int(image_width * scale + 0.5), int(image_height * scale + 0.5))
            # logger.info(f"use predef, {image_width}, {image_height}, {reso}, {resized_size}")
        else:
            # 縮小のみを行う
            if self.runtime_resolution_mode in {"long_edge", "short_edge"} and self.runtime_target_edge is not None:
                target_edge = max(self.reso_steps, int(self.runtime_target_edge))
                if self.runtime_resolution_mode == "short_edge":
                    current_edge = min(image_width, image_height)
                else:
                    current_edge = max(image_width, image_height)

                scale = 1.0 if current_edge <= 0 else min(1.0, target_edge / current_edge)
                if self.max_size is not None:
                    scale = min(scale, self.max_size / max(image_width, image_height))

                resized_width = max(self.reso_steps, int(image_width * scale + 0.5))
                resized_height = max(self.reso_steps, int(image_height * scale + 0.5))
                resized_size = (resized_width, resized_height)
                bucket_width = max(self.reso_steps, self.round_to_steps(resized_width))
                bucket_height = max(self.reso_steps, self.round_to_steps(resized_height))
                reso = (bucket_width, bucket_height)
            elif image_width * image_height > self.max_area:
                # 画像が大きすぎるのでアスペクト比を保ったまま縮小することを前提にbucketを決める
                resized_width = math.sqrt(self.max_area * aspect_ratio)
                resized_height = self.max_area / resized_width
                assert abs(resized_width / resized_height - aspect_ratio) < 1e-2, "aspect is illegal"

                # リサイズ後の短辺または長辺をreso_steps単位にする：aspect ratioの差が少ないほうを選ぶ
                # 元のbucketingと同じロジック
                b_width_rounded = self.round_to_steps(resized_width)
                b_height_in_wr = self.round_to_steps(b_width_rounded / aspect_ratio)
                ar_width_rounded = b_width_rounded / b_height_in_wr

                b_height_rounded = self.round_to_steps(resized_height)
                b_width_in_hr = self.round_to_steps(b_height_rounded * aspect_ratio)
                ar_height_rounded = b_width_in_hr / b_height_rounded

                # logger.info(b_width_rounded, b_height_in_wr, ar_width_rounded)
                # logger.info(b_width_in_hr, b_height_rounded, ar_height_rounded)

                if abs(ar_width_rounded - aspect_ratio) < abs(ar_height_rounded - aspect_ratio):
                    resized_size = (b_width_rounded, int(b_width_rounded / aspect_ratio + 0.5))
                else:
                    resized_size = (int(b_height_rounded * aspect_ratio + 0.5), b_height_rounded)
                # logger.info(resized_size)
            else:
                resized_size = (image_width, image_height)  # リサイズは不要

            if self.runtime_resolution_mode not in {"long_edge", "short_edge"} or self.runtime_target_edge is None:
                # 画像のサイズ未満をbucketのサイズとする（paddingせずにcroppingする）
                bucket_width = resized_size[0] - resized_size[0] % self.reso_steps
                bucket_height = resized_size[1] - resized_size[1] % self.reso_steps
                # logger.info(f"use arbitrary {image_width}, {image_height}, {resized_size}, {bucket_width}, {bucket_height}")

                reso = (bucket_width, bucket_height)

        self.add_if_new_reso(reso)

        ar_error = (reso[0] / reso[1]) - aspect_ratio
        return reso, resized_size, ar_error

    @staticmethod
    def get_crop_ltrb(bucket_reso: Tuple[int, int], image_size: Tuple[int, int]):
        # Stability AIの前処理に合わせてcrop left/topを計算する。crop rightはflipのaugmentationのために求める
        # Calculate crop left/top according to the preprocessing of Stability AI. Crop right is calculated for flip augmentation.

        bucket_ar = bucket_reso[0] / bucket_reso[1]
        image_ar = image_size[0] / image_size[1]
        if bucket_ar > image_ar:
            # bucketのほうが横長→縦を合わせる
            resized_width = bucket_reso[1] * image_ar
            resized_height = bucket_reso[1]
        else:
            resized_width = bucket_reso[0]
            resized_height = bucket_reso[0] / image_ar
        crop_left = int((bucket_reso[0] - resized_width) / 2 + 0.5)
        crop_top = int((bucket_reso[1] - resized_height) / 2 + 0.5)
        crop_right = int(crop_left + resized_width + 0.5)
        crop_bottom = int(crop_top + resized_height + 0.5)
        crop_left = max(0, min(crop_left, bucket_reso[0]))
        crop_top = max(0, min(crop_top, bucket_reso[1]))
        crop_right = max(crop_left, min(crop_right, bucket_reso[0]))
        crop_bottom = max(crop_top, min(crop_bottom, bucket_reso[1]))
        return crop_left, crop_top, crop_right, crop_bottom


class BucketBatchIndex(NamedTuple):
    bucket_index: int
    bucket_batch_size: int
    batch_index: int


class AugHelper:
    # albumentationsへの依存をなくしたがとりあえず同じinterfaceを持たせる

    def __init__(self):
        pass

    def color_aug(self, image: np.ndarray):
        # self.color_aug_method = albu.OneOf(
        #     [
        #         albu.HueSaturationValue(8, 0, 0, p=0.5),
        #         albu.RandomGamma((95, 105), p=0.5),
        #     ],
        #     p=0.33,
        # )
        hue_shift_limit = 8

        # remove dependency to albumentations
        if random.random() <= 0.33:
            if random.random() > 0.5:
                # hue shift
                hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                hue_shift = random.uniform(-hue_shift_limit, hue_shift_limit)
                if hue_shift < 0:
                    hue_shift = 180 + hue_shift
                hsv_img[:, :, 0] = (hsv_img[:, :, 0] + hue_shift) % 180
                image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
            else:
                # random gamma
                gamma = random.uniform(0.95, 1.05)
                image = np.clip(image**gamma, 0, 255).astype(np.uint8)

        return {"image": image}

    def get_augmentor(self, use_color_aug: bool):  # -> Optional[Callable[[np.ndarray], Dict[str, np.ndarray]]]:
        return self.color_aug if use_color_aug else None


class BaseSubset:
    def __init__(
        self,
        image_dir: Optional[str],
        alpha_mask: Optional[bool],
        num_repeats: int,
        shuffle_caption: bool,
        caption_separator: str,
        keep_tokens: int,
        keep_tokens_separator: str,
        secondary_separator: Optional[str],
        enable_wildcard: bool,
        color_aug: bool,
        flip_aug: bool,
        face_crop_aug_range: Optional[Tuple[float, float]],
        random_crop: bool,
        caption_dropout_rate: float,
        caption_dropout_every_n_epochs: int,
        caption_tag_dropout_rate: float,
        caption_tag_dropout_targets: Optional[str],
        caption_tag_dropout_target_mode: str,
        caption_tag_dropout_target_count: int,
        caption_prefix: Optional[str],
        caption_suffix: Optional[str],
        token_warmup_min: int,
        token_warmup_step: Union[float, int],
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        self.image_dir = image_dir
        self.alpha_mask = alpha_mask if alpha_mask is not None else False
        self.num_repeats = num_repeats
        self.shuffle_caption = shuffle_caption
        self.caption_separator = caption_separator
        self.keep_tokens = keep_tokens
        self.keep_tokens_separator = keep_tokens_separator
        self.secondary_separator = secondary_separator
        self.enable_wildcard = enable_wildcard
        self.color_aug = color_aug
        self.flip_aug = flip_aug
        self.face_crop_aug_range = face_crop_aug_range
        self.random_crop = random_crop
        self.caption_dropout_rate = caption_dropout_rate
        self.caption_dropout_every_n_epochs = caption_dropout_every_n_epochs
        self.caption_tag_dropout_rate = caption_tag_dropout_rate
        self.caption_tag_dropout_targets = caption_tag_dropout_targets
        self.caption_tag_dropout_target_mode = str(caption_tag_dropout_target_mode or "drop_all").strip().lower()
        if self.caption_tag_dropout_target_mode not in {"drop_all", "random_n"}:
            self.caption_tag_dropout_target_mode = "drop_all"
        self.caption_tag_dropout_target_count = max(1, int(caption_tag_dropout_target_count or 1))
        self.caption_prefix = caption_prefix
        self.caption_suffix = caption_suffix

        self.token_warmup_min = token_warmup_min  # step=0におけるタグの数
        self.token_warmup_step = token_warmup_step  # N（N<1ならN*max_train_steps）ステップ目でタグの数が最大になる

        self.custom_attributes = custom_attributes if custom_attributes is not None else {}

        self.img_count = 0

        self.validation_seed = validation_seed
        self.validation_split = validation_split

        self.resize_interpolation = resize_interpolation


class DreamBoothSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        is_reg: bool,
        class_tokens: Optional[str],
        caption_extension: str,
        cache_info: bool,
        alpha_mask: bool,
        num_repeats,
        shuffle_caption,
        caption_separator: str,
        keep_tokens,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_tag_dropout_targets,
        caption_tag_dropout_target_mode,
        caption_tag_dropout_target_count,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dirは指定が必須です / 必须指定 image_dir"

        super().__init__(
            image_dir,
            alpha_mask,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_tag_dropout_targets,
            caption_tag_dropout_target_mode,
            caption_tag_dropout_target_count,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes=custom_attributes,
            validation_seed=validation_seed,
            validation_split=validation_split,
            resize_interpolation=resize_interpolation,
        )

        self.is_reg = is_reg
        self.class_tokens = class_tokens
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension
        self.cache_info = cache_info

    def __eq__(self, other) -> bool:
        if not isinstance(other, DreamBoothSubset):
            return NotImplemented
        return self.image_dir == other.image_dir


class FineTuningSubset(BaseSubset):
    def __init__(
        self,
        image_dir,
        metadata_file: str,
        alpha_mask: bool,
        num_repeats,
        shuffle_caption,
        caption_separator,
        keep_tokens,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_tag_dropout_targets,
        caption_tag_dropout_target_mode,
        caption_tag_dropout_target_count,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        assert metadata_file is not None, "metadata_file must be specified / metadata_fileは指定が必須です / 必须指定 metadata_file"

        super().__init__(
            image_dir,
            alpha_mask,
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_tag_dropout_targets,
            caption_tag_dropout_target_mode,
            caption_tag_dropout_target_count,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes=custom_attributes,
            validation_seed=validation_seed,
            validation_split=validation_split,
            resize_interpolation=resize_interpolation,
        )

        self.metadata_file = metadata_file

    def __eq__(self, other) -> bool:
        if not isinstance(other, FineTuningSubset):
            return NotImplemented
        return self.metadata_file == other.metadata_file


class ControlNetSubset(BaseSubset):
    def __init__(
        self,
        image_dir: str,
        conditioning_data_dir: str,
        caption_extension: str,
        cache_info: bool,
        num_repeats,
        shuffle_caption,
        caption_separator,
        keep_tokens,
        keep_tokens_separator,
        secondary_separator,
        enable_wildcard,
        color_aug,
        flip_aug,
        face_crop_aug_range,
        random_crop,
        caption_dropout_rate,
        caption_dropout_every_n_epochs,
        caption_tag_dropout_rate,
        caption_tag_dropout_targets,
        caption_tag_dropout_target_mode,
        caption_tag_dropout_target_count,
        caption_prefix,
        caption_suffix,
        token_warmup_min,
        token_warmup_step,
        custom_attributes: Optional[Dict[str, Any]] = None,
        validation_seed: Optional[int] = None,
        validation_split: Optional[float] = 0.0,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        assert image_dir is not None, "image_dir must be specified / image_dirは指定が必須です / 必须指定 image_dir"

        super().__init__(
            image_dir,
            False,  # alpha_mask
            num_repeats,
            shuffle_caption,
            caption_separator,
            keep_tokens,
            keep_tokens_separator,
            secondary_separator,
            enable_wildcard,
            color_aug,
            flip_aug,
            face_crop_aug_range,
            random_crop,
            caption_dropout_rate,
            caption_dropout_every_n_epochs,
            caption_tag_dropout_rate,
            caption_tag_dropout_targets,
            caption_tag_dropout_target_mode,
            caption_tag_dropout_target_count,
            caption_prefix,
            caption_suffix,
            token_warmup_min,
            token_warmup_step,
            custom_attributes=custom_attributes,
            validation_seed=validation_seed,
            validation_split=validation_split,
            resize_interpolation=resize_interpolation,
        )

        self.conditioning_data_dir = conditioning_data_dir
        self.caption_extension = caption_extension
        if self.caption_extension and not self.caption_extension.startswith("."):
            self.caption_extension = "." + self.caption_extension
        self.cache_info = cache_info

    def __eq__(self, other) -> bool:
        if not isinstance(other, ControlNetSubset):
            return NotImplemented
        return self.image_dir == other.image_dir and self.conditioning_data_dir == other.conditioning_data_dir


class BaseDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        resolution: Optional[Tuple[int, int]],
        skip_image_resolution: Optional[Tuple[int, int]],
        network_multiplier: float,
        debug_dataset: bool,
        resize_interpolation: Optional[str] = None,
    ) -> None:
        super().__init__()

        # width/height is used when enable_bucket==False
        self.width, self.height = (None, None) if resolution is None else resolution
        self.skip_image_resolution = None if skip_image_resolution is None else tuple(skip_image_resolution)
        self.network_multiplier = network_multiplier
        self.debug_dataset = debug_dataset

        self.subsets: List[Union[DreamBoothSubset, FineTuningSubset]] = []

        self.token_padding_disabled = False
        self.tag_frequency = {}
        self.XTI_layers = None
        self.token_strings = None

        self.enable_bucket = False
        self.bucket_manager: BucketManager = None  # not initialized
        self.min_bucket_reso = None
        self.max_bucket_reso = None
        self.bucket_reso_steps = None
        self.bucket_no_upscale = None
        self.bucket_selection_mode = "legacy"
        self.bucket_custom_resos = None
        self.bucket_info = None  # for metadata

        self.current_epoch: int = 0  # インスタンスがepochごとに新しく作られるようなので外側から渡さないとダメ

        self.current_step: int = 0
        self.max_train_steps: int = 0
        self.seed: int = 0

        # augmentation
        self.aug_helper = AugHelper()

        self.image_transforms = IMAGE_TRANSFORMS

        if resize_interpolation is not None:
            assert validate_interpolation_fn(
                resize_interpolation
            ), f'Resize interpolation "{resize_interpolation}" is not a valid interpolation'
        self.resize_interpolation = resize_interpolation

        self.image_data: Dict[str, ImageInfo] = {}
        self.image_to_subset: Dict[str, Union[DreamBoothSubset, FineTuningSubset]] = {}

        self.replacements = {}

        # caching
        self.caching_mode = None  # None, 'latents', 'text'

        self.tokenize_strategy = None
        self.text_encoder_output_caching_strategy = None
        self.latents_caching_strategy = None
        self._skip_image_resolution_applied = False

    def set_current_strategies(self):
        self.tokenize_strategy = TokenizeStrategy.get_strategy()
        self.text_encoder_output_caching_strategy = TextEncoderOutputsCachingStrategy.get_strategy()
        self.latents_caching_strategy = LatentsCachingStrategy.get_strategy()

    def _refresh_registered_image_counts(self):
        for subset in self.subsets:
            subset.img_count = 0

        self.num_train_images = 0
        self.num_reg_images = 0

        for image_key, image_info in self.image_data.items():
            subset = self.image_to_subset.get(image_key)
            if subset is None:
                continue

            subset.img_count += 1
            if getattr(subset, "is_reg", False):
                self.num_reg_images += image_info.num_repeats
            else:
                self.num_train_images += image_info.num_repeats

    def apply_skip_image_resolution_filter(self):
        if self.skip_image_resolution is None or self._skip_image_resolution_applied:
            return

        min_width, min_height = self.skip_image_resolution
        min_area = min_width * min_height
        skipped_images = 0
        skipped_train_repeats = 0
        skipped_reg_repeats = 0

        for image_key, image_info in list(self.image_data.items()):
            image_size = image_info.image_size
            if image_size is None:
                continue

            image_width, image_height = image_size
            if image_width is None or image_height is None or image_width <= 0 or image_height <= 0:
                continue

            if image_width * image_height > min_area:
                continue

            subset = self.image_to_subset.pop(image_key, None)
            self.image_data.pop(image_key, None)
            skipped_images += 1
            if getattr(subset, "is_reg", False):
                skipped_reg_repeats += image_info.num_repeats
            else:
                skipped_train_repeats += image_info.num_repeats

        self._skip_image_resolution_applied = True

        if skipped_images == 0:
            logger.info(
                f"skip_image_resolution is enabled: images with original area <= {min_width}x{min_height} ({min_area}) will be skipped if found"
            )
            return

        self._refresh_registered_image_counts()
        logger.warning(
            f"skip_image_resolution filtered {skipped_images} images with original area <= {min_width}x{min_height} ({min_area})"
            f" / 跳过了 {skipped_images} 张原始面积小于等于阈值的图像"
            f" / skipped repeats: train={skipped_train_repeats}, reg={skipped_reg_repeats}"
        )

    def adjust_min_max_bucket_reso_by_steps(
        self, resolution: Tuple[int, int], min_bucket_reso: int, max_bucket_reso: int, bucket_reso_steps: int
    ) -> Tuple[int, int]:
        # make min/max bucket reso to be multiple of bucket_reso_steps
        if min_bucket_reso % bucket_reso_steps != 0:
            adjusted_min_bucket_reso = min_bucket_reso - min_bucket_reso % bucket_reso_steps
            logger.warning(
                f"min_bucket_reso is adjusted to be multiple of bucket_reso_steps"
                f" / min_bucket_resoがbucket_reso_stepsの倍数になるように調整されました: {min_bucket_reso} -> {adjusted_min_bucket_reso}"
                f" / min_bucket_reso 已调整为 bucket_reso_steps 的整数倍: {min_bucket_reso} -> {adjusted_min_bucket_reso}"
            )
            min_bucket_reso = adjusted_min_bucket_reso
        if max_bucket_reso % bucket_reso_steps != 0:
            adjusted_max_bucket_reso = max_bucket_reso + bucket_reso_steps - max_bucket_reso % bucket_reso_steps
            logger.warning(
                f"max_bucket_reso is adjusted to be multiple of bucket_reso_steps"
                f" / max_bucket_resoがbucket_reso_stepsの倍数になるように調整されました: {max_bucket_reso} -> {adjusted_max_bucket_reso}"
                f" / max_bucket_reso 已调整为 bucket_reso_steps 的整数倍: {max_bucket_reso} -> {adjusted_max_bucket_reso}"
            )
            max_bucket_reso = adjusted_max_bucket_reso

        assert (
            min(resolution) >= min_bucket_reso
        ), f"min_bucket_reso must be equal or less than resolution / min_bucket_resoは最小解像度より大きくできません。解像度を大きくするかmin_bucket_resoを小さくしてください / min_bucket_reso 不能大于最小分辨率，请提高分辨率或降低 min_bucket_reso"
        assert (
            max(resolution) <= max_bucket_reso
        ), f"max_bucket_reso must be equal or greater than resolution / max_bucket_resoは最大解像度より小さくできません。解像度を小さくするかmin_bucket_resoを大きくしてください / max_bucket_reso 不能小于最大分辨率，请降低分辨率或提高 max_bucket_reso"

        return min_bucket_reso, max_bucket_reso

    def set_seed(self, seed):
        self.seed = seed

    def set_caching_mode(self, mode):
        self.caching_mode = mode

    def set_current_epoch(self, epoch):
        if not self.current_epoch == epoch:  # epochが切り替わったらバケツをシャッフルする
            if epoch > self.current_epoch:
                num_epochs = epoch - self.current_epoch
                for _ in range(num_epochs):
                    self.current_epoch += 1
                    self.shuffle_buckets()
                # self.current_epoch seem to be set to 0 again in the next epoch. it may be caused by skipped_dataloader?
            else:
                # DataLoader workers can observe stale local epoch state briefly.
                # Updating silently avoids noisy duplicated console output.
                self.current_epoch = epoch

    def set_current_step(self, step):
        self.current_step = step

    def set_max_train_steps(self, max_train_steps):
        self.max_train_steps = max_train_steps

    def set_tag_frequency(self, dir_name, captions):
        frequency_for_dir = self.tag_frequency.get(dir_name, {})
        self.tag_frequency[dir_name] = frequency_for_dir
        for caption in captions:
            payload = anima_caption_util.decode_special_caption_payload(caption)
            if payload is not None:
                caption = anima_caption_util.build_caption_from_payload(
                    payload,
                    shuffle_appearance=False,
                    shuffle_tags=False,
                    shuffle_environment=False,
                    tag_dropout=0.0,
                )
            for tag in caption.split(","):
                tag = tag.strip()
                if tag:
                    tag = tag.lower()
                    frequency = frequency_for_dir.get(tag, 0)
                    frequency_for_dir[tag] = frequency + 1

    def disable_token_padding(self):
        self.token_padding_disabled = True

    def enable_XTI(self, layers=None, token_strings=None):
        self.XTI_layers = layers
        self.token_strings = token_strings

    def add_replacement(self, str_from, str_to):
        self.replacements[str_from] = str_to

    def apply_targeted_tag_dropout(self, subset: BaseSubset, caption: str) -> str:
        target_tokens = parse_tag_text_list(getattr(subset, "caption_tag_dropout_targets", None))
        if not target_tokens:
            return caption

        caption_separator = getattr(subset, "caption_separator", ",") or ","
        tokens = [token.strip() for token in caption.split(caption_separator)]
        normalized_targets = {normalize_tag_token(token) for token in target_tokens if normalize_tag_token(token)}
        if not normalized_targets:
            return caption

        matched_indices = [
            index for index, token in enumerate(tokens) if token and normalize_tag_token(token) in normalized_targets
        ]
        if not matched_indices:
            return caption

        mode = getattr(subset, "caption_tag_dropout_target_mode", "drop_all")
        if mode == "random_n":
            drop_count = min(len(matched_indices), max(1, int(getattr(subset, "caption_tag_dropout_target_count", 1) or 1)))
            drop_indices = set(random.sample(matched_indices, drop_count))
        else:
            drop_indices = set(matched_indices)

        kept_tokens = [token for index, token in enumerate(tokens) if token and index not in drop_indices]
        return f"{caption_separator} ".join(kept_tokens)

    def process_caption(self, subset: BaseSubset, caption):
        structured_caption_payload = anima_caption_util.decode_special_caption_payload(caption)
        structured_caption_rendered = structured_caption_payload is not None
        if structured_caption_payload is not None:
            caption = anima_caption_util.build_caption_from_payload(
                structured_caption_payload,
                shuffle_appearance=subset.shuffle_caption,
                shuffle_tags=subset.shuffle_caption,
                shuffle_environment=subset.shuffle_caption,
                tag_dropout=subset.caption_tag_dropout_rate,
            )

        # dropoutの決定：tag dropがこのメソッド内にあるのでここで行うのが良い
        is_drop_out = subset.caption_dropout_rate > 0 and random.random() < subset.caption_dropout_rate
        is_drop_out = (
            is_drop_out
            or subset.caption_dropout_every_n_epochs > 0
            and self.current_epoch % subset.caption_dropout_every_n_epochs == 0
        )

        if is_drop_out:
            caption = ""
        else:
            # process wildcards
            if subset.enable_wildcard:
                # if caption is multiline, random choice one line
                if "\n" in caption:
                    caption = random.choice(caption.split("\n"))

                # wildcard is like '{aaa|bbb|ccc...}'
                # escape the curly braces like {{ or }}
                replacer1 = "⦅"
                replacer2 = "⦆"
                while replacer1 in caption or replacer2 in caption:
                    replacer1 += "⦅"
                    replacer2 += "⦆"

                caption = caption.replace("{{", replacer1).replace("}}", replacer2)

                # replace the wildcard
                def replace_wildcard(match):
                    return random.choice(match.group(1).split("|"))

                caption = re.sub(r"\{([^}]+)\}", replace_wildcard, caption)

                # unescape the curly braces
                caption = caption.replace(replacer1, "{").replace(replacer2, "}")
            else:
                # if caption is multiline, use the first line
                caption = caption.split("\n")[0]

            caption = self.apply_targeted_tag_dropout(subset, caption)

            if not structured_caption_rendered and (
                subset.shuffle_caption or subset.token_warmup_step > 0 or subset.caption_tag_dropout_rate > 0
            ):
                fixed_tokens = []
                flex_tokens = []
                fixed_suffix_tokens = []
                if (
                    hasattr(subset, "keep_tokens_separator")
                    and subset.keep_tokens_separator
                    and subset.keep_tokens_separator in caption
                ):
                    fixed_part, flex_part = caption.split(subset.keep_tokens_separator, 1)
                    if subset.keep_tokens_separator in flex_part:
                        flex_part, fixed_suffix_part = flex_part.split(subset.keep_tokens_separator, 1)
                        fixed_suffix_tokens = [t.strip() for t in fixed_suffix_part.split(subset.caption_separator) if t.strip()]

                    fixed_tokens = [t.strip() for t in fixed_part.split(subset.caption_separator) if t.strip()]
                    flex_tokens = [t.strip() for t in flex_part.split(subset.caption_separator) if t.strip()]
                else:
                    tokens = [t.strip() for t in caption.strip().split(subset.caption_separator)]
                    flex_tokens = tokens[:]
                    if subset.keep_tokens > 0:
                        fixed_tokens = flex_tokens[: subset.keep_tokens]
                        flex_tokens = tokens[subset.keep_tokens :]

                if subset.token_warmup_step < 1:  # 初回に上書きする
                    subset.token_warmup_step = math.floor(subset.token_warmup_step * self.max_train_steps)
                if subset.token_warmup_step and self.current_step < subset.token_warmup_step:
                    tokens_len = (
                        math.floor(
                            (self.current_step) * ((len(flex_tokens) - subset.token_warmup_min) / (subset.token_warmup_step))
                        )
                        + subset.token_warmup_min
                    )
                    flex_tokens = flex_tokens[:tokens_len]

                def dropout_tags(tokens):
                    if subset.caption_tag_dropout_rate <= 0:
                        return tokens
                    l = []
                    for token in tokens:
                        if random.random() >= subset.caption_tag_dropout_rate:
                            l.append(token)
                    return l

                if subset.shuffle_caption:
                    random.shuffle(flex_tokens)

                flex_tokens = dropout_tags(flex_tokens)

                caption = ", ".join(fixed_tokens + flex_tokens + fixed_suffix_tokens)

            if subset.caption_prefix:
                caption = subset.caption_prefix + " " + caption
            if subset.caption_suffix:
                caption = caption + " " + subset.caption_suffix

            # process secondary separator
            if subset.secondary_separator:
                caption = caption.replace(subset.secondary_separator, subset.caption_separator)

            # textual inversion対応
            for str_from, str_to in self.replacements.items():
                if str_from == "":
                    # replace all
                    if type(str_to) == list:
                        caption = random.choice(str_to)
                    else:
                        caption = str_to
                else:
                    caption = caption.replace(str_from, str_to)

        return caption

    def get_input_ids(self, caption, tokenizer=None):
        if tokenizer is None:
            tokenizer = self.tokenizers[0]

        input_ids = tokenizer(
            caption, padding="max_length", truncation=True, max_length=self.tokenizer_max_length, return_tensors="pt"
        ).input_ids

        if self.tokenizer_max_length > tokenizer.model_max_length:
            input_ids = input_ids.squeeze(0)
            iids_list = []
            if tokenizer.pad_token_id == tokenizer.eos_token_id:
                # v1
                # 77以上の時は "<BOS> .... <EOS> <EOS> <EOS>" でトータル227とかになっているので、"<BOS>...<EOS>"の三連に変換する
                # 1111氏のやつは , で区切る、とかしているようだが　とりあえず単純に
                for i in range(
                    1, self.tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2
                ):  # (1, 152, 75)
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )
                    ids_chunk = torch.cat(ids_chunk)
                    iids_list.append(ids_chunk)
            else:
                # v2 or SDXL
                # 77以上の時は "<BOS> .... <EOS> <PAD> <PAD>..." でトータル227とかになっているので、"<BOS>...<EOS> <PAD> <PAD> ..."の三連に変換する
                for i in range(1, self.tokenizer_max_length - tokenizer.model_max_length + 2, tokenizer.model_max_length - 2):
                    ids_chunk = (
                        input_ids[0].unsqueeze(0),  # BOS
                        input_ids[i : i + tokenizer.model_max_length - 2],
                        input_ids[-1].unsqueeze(0),
                    )  # PAD or EOS
                    ids_chunk = torch.cat(ids_chunk)

                    # 末尾が <EOS> <PAD> または <PAD> <PAD> の場合は、何もしなくてよい
                    # 末尾が x <PAD/EOS> の場合は末尾を <EOS> に変える（x <EOS> なら結果的に変化なし）
                    if ids_chunk[-2] != tokenizer.eos_token_id and ids_chunk[-2] != tokenizer.pad_token_id:
                        ids_chunk[-1] = tokenizer.eos_token_id
                    # 先頭が <BOS> <PAD> ... の場合は <BOS> <EOS> <PAD> ... に変える
                    if ids_chunk[1] == tokenizer.pad_token_id:
                        ids_chunk[1] = tokenizer.eos_token_id

                    iids_list.append(ids_chunk)

            input_ids = torch.stack(iids_list)  # 3,77
        return input_ids

    def _build_registered_image_key(
        self, base_image_key: str, image_size: Optional[Tuple[int, int]], duplicate_index: int = 0
    ) -> str:
        resolved_key = str(base_image_key)
        if image_size is not None and len(image_size) >= 2:
            resolved_key = f"{resolved_key}#orig={int(image_size[0])}x{int(image_size[1])}"
        if duplicate_index > 0:
            resolved_key = f"{resolved_key}#dup={duplicate_index}"
        return resolved_key

    def register_image(self, info: ImageInfo, subset: BaseSubset):
        base_image_key = str(info.image_key)
        resolved_image_key = self._build_registered_image_key(base_image_key, info.image_size)
        duplicate_index = 0
        while resolved_image_key in self.image_data:
            duplicate_index += 1
            resolved_image_key = self._build_registered_image_key(base_image_key, info.image_size, duplicate_index)

        if resolved_image_key != info.image_key:
            logger.info(
                f"resolved duplicated image key: {info.image_key} -> {resolved_image_key}"
                f" / 已为重复图像生成稳定内部键"
            )
            info.image_key = resolved_image_key

        self.image_data[info.image_key] = info
        self.image_to_subset[info.image_key] = subset

    def make_buckets(self):
        """
        bucketingを行わない場合も呼び出し必須（ひとつだけbucketを作る）
        min_size and max_size are ignored when enable_bucket is False
        """
        logger.info("loading image sizes.")
        for info in tqdm(self.image_data.values()):
            if info.image_size is None:
                info.image_size = self.get_image_size(info.absolute_path)

        self.apply_skip_image_resolution_filter()

        # # run in parallel
        # max_workers = min(os.cpu_count(), len(self.image_data))  # TODO consider multi-gpu (processes)
        # with ThreadPoolExecutor(max_workers) as executor:
        #     futures = []
        #     for info in tqdm(self.image_data.values(), desc="loading image sizes"):
        #         if info.image_size is None:
        #             def get_and_set_image_size(info):
        #                 info.image_size = self.get_image_size(info.absolute_path)
        #             futures.append(executor.submit(get_and_set_image_size, info))
        #             # consume futures to reduce memory usage and prevent Ctrl-C hang
        #             if len(futures) >= max_workers:
        #                 for future in futures:
        #                     future.result()
        #                 futures = []
        #     for future in futures:
        #         future.result()

        if self.enable_bucket:
            logger.info("make buckets")
        else:
            logger.info("prepare dataset")

        # bucketを作成し、画像をbucketに振り分ける
        if self.enable_bucket:
            if self.bucket_manager is None:  # fine tuningの場合でmetadataに定義がある場合は、すでに初期化済み
                self.bucket_manager = BucketManager(
                    self.bucket_no_upscale,
                    (self.width, self.height),
                    self.min_bucket_reso,
                    self.max_bucket_reso,
                    self.bucket_reso_steps,
                    self.bucket_selection_mode,
                    self.bucket_custom_resos,
                )
                if self.bucket_selection_mode == "nearest_only":
                    self.bucket_manager.make_buckets_by_nearest_image_aspect(
                        [info.image_size for info in self.image_data.values()]
                    )
                elif not self.bucket_no_upscale or self.bucket_selection_mode == "custom_only":
                    self.bucket_manager.make_buckets()
                else:
                    logger.warning(
                        "min_bucket_reso and max_bucket_reso are ignored if bucket_no_upscale is set, because bucket reso is defined by image size automatically / bucket_no_upscaleが指定された場合は、bucketの解像度は画像サイズから自動計算されるため、min_bucket_resoとmax_bucket_resoは無視されます / 当启用 bucket_no_upscale 时，bucket 分辨率会根据图像尺寸自动计算，因此 min_bucket_reso 和 max_bucket_reso 会被忽略"
                    )
                if self.bucket_selection_mode != "legacy" and self.bucket_no_upscale:
                    logger.warning(
                        f"bucket_no_upscale is ignored when bucket_selection_mode={self.bucket_selection_mode} / 当 bucket_selection_mode={self.bucket_selection_mode} 时，bucket_no_upscale 将被忽略"
                    )

            img_ar_errors = []
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, ar_error = self.bucket_manager.select_bucket(
                    image_width, image_height
                )

                # logger.info(image_info.image_key, image_info.bucket_reso)
                img_ar_errors.append(abs(ar_error))

            self.bucket_manager.sort()
        else:
            self.bucket_manager = BucketManager(False, (self.width, self.height), None, None, None)
            self.bucket_manager.set_predefined_resos([(self.width, self.height)])  # ひとつの固定サイズbucketのみ
            for image_info in self.image_data.values():
                image_width, image_height = image_info.image_size
                image_info.bucket_reso, image_info.resized_size, _ = self.bucket_manager.select_bucket(image_width, image_height)

        for image_info in self.image_data.values():
            for _ in range(image_info.num_repeats):
                self.bucket_manager.add_image(image_info.bucket_reso, image_info.image_key)

        # bucket情報を表示、格納する
        if self.enable_bucket:
            self.bucket_info = {"buckets": {}}
            logger.info("number of images (including repeats) / 各bucketの画像枚数（繰り返し回数を含む） / 图像数量（包含重复次数）")
            for i, (reso, bucket) in enumerate(zip(self.bucket_manager.resos, self.bucket_manager.buckets)):
                count = len(bucket)
                if count > 0:
                    self.bucket_info["buckets"][i] = {"resolution": reso, "count": len(bucket)}
                    logger.info(f"bucket {i}: resolution {reso}, count: {len(bucket)}")

            if len(img_ar_errors) == 0:
                mean_img_ar_error = 0  # avoid NaN
            else:
                img_ar_errors = np.array(img_ar_errors)
                mean_img_ar_error = np.mean(np.abs(img_ar_errors))
            self.bucket_info["mean_img_ar_error"] = mean_img_ar_error
            logger.info(f"mean ar error (without repeats): {mean_img_ar_error}")

        # データ参照用indexを作る。このindexはdatasetのshuffleに用いられる
        self.buckets_indices: List[BucketBatchIndex] = []
        for bucket_index, bucket in enumerate(self.bucket_manager.buckets):
            batch_count = int(math.ceil(len(bucket) / self.batch_size))
            for batch_index in range(batch_count):
                self.buckets_indices.append(BucketBatchIndex(bucket_index, self.batch_size, batch_index))

        self.shuffle_buckets()
        self._length = len(self.buckets_indices)

    def shuffle_buckets(self):
        # set random seed for this epoch
        random.seed(self.seed + self.current_epoch)

        random.shuffle(self.buckets_indices)
        self.bucket_manager.shuffle()

    def verify_bucket_reso_steps(self, min_steps: int):
        assert self.bucket_reso_steps is None or self.bucket_reso_steps % min_steps == 0, (
            f"bucket_reso_steps is {self.bucket_reso_steps}. it must be divisible by {min_steps}.\n"
            + f"bucket_reso_stepsが{self.bucket_reso_steps}です。{min_steps}で割り切れる必要があります"
        )

    def is_latent_cacheable(self):
        return all([not subset.color_aug and not subset.random_crop for subset in self.subsets])

    def is_text_encoder_output_cacheable(self, cache_supports_dropout: bool = False):
        return all(
            [
                not (
                    subset.caption_dropout_rate > 0 and not cache_supports_dropout
                    or subset.shuffle_caption
                    or subset.token_warmup_step > 0
                    or subset.caption_tag_dropout_rate > 0
                    or bool(parse_tag_text_list(getattr(subset, "caption_tag_dropout_targets", None)))
                )
                for subset in self.subsets
            ]
        )

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        r"""
        a brand new method to cache latents. This method caches latents with caching strategy.
        normal cache_latents method is used by default, but this method is used when caching strategy is specified.
        """
        logger.info("caching latents with caching strategy.")
        caching_strategy = LatentsCachingStrategy.get_strategy()
        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution and some conditions
        class Condition:
            def __init__(self, reso, flip_aug, alpha_mask, random_crop, cache_root):
                self.reso = reso
                self.flip_aug = flip_aug
                self.alpha_mask = alpha_mask
                self.random_crop = random_crop
                self.cache_root = cache_root

            def __eq__(self, other):
                return (
                    other is not None
                    and self.reso == other.reso
                    and self.flip_aug == other.flip_aug
                    and self.alpha_mask == other.alpha_mask
                    and self.random_crop == other.random_crop
                    and self.cache_root == other.cache_root
                )

        batch: List[ImageInfo] = []
        current_condition = None
        current_cache_batch_size = None

        # support multiple-gpus
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        def finalize_caching_strategy() -> None:
            finalize_caching = getattr(caching_strategy, "finalize_caching", None)
            if callable(finalize_caching):
                finalize_caching()

        def flush_pending_disk_cache() -> None:
            flush_cache = getattr(caching_strategy, "flush_pending_disk_cache", None)
            if callable(flush_cache):
                flush_cache()

        use_pipelined_preprocessing = bool(
            getattr(caching_strategy, "preprocess_workers", 0) > 0 and hasattr(caching_strategy, "cache_batch_latents_prepared")
        )

        if use_pipelined_preprocessing:
            preprocess_workers = max(1, int(caching_strategy.preprocess_workers))
            prefetch_batches = max(1, int(getattr(caching_strategy, "prefetch_batches", 1) or 1))
            preprocess_executor = ThreadPoolExecutor(max_workers=preprocess_workers)
            prepared_batches: deque[tuple[List[ImageInfo], Condition, Future]] = deque()

            def submit_prepared_batch(batch_items, cond):
                batch_copy = list(batch_items)
                future = preprocess_executor.submit(
                    caching_strategy.prepare_batch_latents,
                    batch_copy,
                    cond.alpha_mask,
                    cond.random_crop,
                )
                prepared_batches.append((batch_copy, cond, future))

            def consume_prepared_batches(force_wait: bool) -> None:
                while prepared_batches and (force_wait or len(prepared_batches) >= prefetch_batches):
                    queued_batch, queued_condition, future = prepared_batches.popleft()
                    prepared_batch = future.result()
                    caching_strategy.cache_batch_latents_prepared(
                        model,
                        queued_batch,
                        prepared_batch,
                        queued_condition.flip_aug,
                        queued_condition.alpha_mask,
                        queued_condition.random_crop,
                    )

            try:
                logger.info(
                    "caching latents with pipelined CPU preprocessing: "
                    f"workers={preprocess_workers}, prefetch_batches={prefetch_batches}"
                )
                for i, info in enumerate(tqdm(image_infos)):
                    subset = self.image_to_subset[info.image_key]

                    if info.latents_npz is not None:  # fine tuning dataset
                        continue

                    info.latents_cache_root = caching_strategy.resolve_disk_cache_root(
                        info.absolute_path, getattr(subset, "image_dir", None)
                    )

                    if caching_strategy.cache_to_disk:
                        if i % num_processes != process_index:
                            continue

                        cache_ref = caching_strategy.find_existing_latents_disk_cache_ref(
                            info.absolute_path,
                            info.image_size,
                            cache_root=info.latents_cache_root,
                            bucket_reso=info.bucket_reso,
                            flip_aug=subset.flip_aug,
                            alpha_mask=subset.alpha_mask,
                        )
                        if cache_ref is not None:
                            if cache_ref.format == "npz":
                                info.latents_npz = cache_ref.path
                            else:
                                info.latents_disk_cache_ref = cache_ref
                            continue
                        info.latents_disk_cache_ref = None
                        info.latents_npz = (
                            caching_strategy.get_latents_npz_path(info.absolute_path, info.image_size)
                            if caching_strategy.disk_cache_format == "npz"
                            else None
                        )

                    condition = Condition(
                        info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop, info.latents_cache_root
                    )
                    if len(batch) > 0 and current_condition != condition:
                        submit_prepared_batch(batch, current_condition)
                        batch = []
                        if caching_strategy.uses_safetensors_disk_cache:
                            consume_prepared_batches(force_wait=True)
                            flush_pending_disk_cache()
                        else:
                            consume_prepared_batches(force_wait=False)
                    if condition != current_condition and HIGH_VRAM:
                        clean_memory_on_device(accelerator.device)

                    if condition != current_condition or current_cache_batch_size is None:
                        current_cache_batch_size = caching_strategy.resolve_cache_batch_size(condition.reso)

                    batch.append(info)
                    current_condition = condition

                    if len(batch) >= current_cache_batch_size:
                        submit_prepared_batch(batch, current_condition)
                        batch = []
                        consume_prepared_batches(force_wait=False)

                if len(batch) > 0:
                    submit_prepared_batch(batch, current_condition)

                consume_prepared_batches(force_wait=True)
            finally:
                preprocess_executor.shutdown()
                finalize_caching_strategy()
            return

        # define a function to submit a batch to cache
        def submit_batch(batch, cond):
            for info in batch:
                if info.image is not None and isinstance(info.image, Future):
                    info.image = info.image.result()  # future to image
            caching_strategy.cache_batch_latents(model, batch, cond.flip_aug, cond.alpha_mask, cond.random_crop)

            # remove image from memory
            for info in batch:
                info.image = None

        # define ThreadPoolExecutor to load images in parallel
        max_workers = min(os.cpu_count(), len(image_infos))
        max_workers = max(1, max_workers // num_processes)  # consider multi-gpu
        max_workers = min(max_workers, getattr(caching_strategy, "max_batch_size", caching_strategy.batch_size))
        executor = ThreadPoolExecutor(max_workers)

        try:
            # iterate images
            logger.info("caching latents...")
            for i, info in enumerate(tqdm(image_infos)):
                subset = self.image_to_subset[info.image_key]

                if info.latents_npz is not None:  # fine tuning dataset
                    continue

                info.latents_cache_root = caching_strategy.resolve_disk_cache_root(
                    info.absolute_path, getattr(subset, "image_dir", None)
                )

                # check disk cache exists and size of latents
                if caching_strategy.cache_to_disk:
                    # if the modulo of num_processes is not equal to process_index, skip caching
                    # this makes each process cache different latents
                    if i % num_processes != process_index:
                        continue

                    cache_ref = caching_strategy.find_existing_latents_disk_cache_ref(
                        info.absolute_path,
                        info.image_size,
                        cache_root=info.latents_cache_root,
                        bucket_reso=info.bucket_reso,
                        flip_aug=subset.flip_aug,
                        alpha_mask=subset.alpha_mask,
                    )
                    if cache_ref is not None:  # do not add to batch
                        if cache_ref.format == "npz":
                            info.latents_npz = cache_ref.path
                        else:
                            info.latents_disk_cache_ref = cache_ref
                        continue
                    info.latents_disk_cache_ref = None
                    info.latents_npz = (
                        caching_strategy.get_latents_npz_path(info.absolute_path, info.image_size)
                        if caching_strategy.disk_cache_format == "npz"
                        else None
                    )

                condition = Condition(
                    info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop, info.latents_cache_root
                )
                if len(batch) > 0 and current_condition != condition:
                    submit_batch(batch, current_condition)
                    flush_pending_disk_cache()
                    batch = []
                if condition != current_condition and HIGH_VRAM:  # even with high VRAM, if shape is changed
                    clean_memory_on_device(accelerator.device)

                if condition != current_condition or current_cache_batch_size is None:
                    current_cache_batch_size = caching_strategy.resolve_cache_batch_size(condition.reso)

                if info.image is None:
                    info.image = executor.submit(load_image, info.absolute_path, condition.alpha_mask)

                batch.append(info)
                current_condition = condition

                if len(batch) >= current_cache_batch_size:
                    submit_batch(batch, current_condition)
                    batch = []

            if len(batch) > 0:
                submit_batch(batch, current_condition)
                flush_pending_disk_cache()

        finally:
            executor.shutdown()
            finalize_caching_strategy()

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True, file_suffix=".npz"):
        # マルチGPUには対応していないので、そちらはtools/cache_latents.pyを使うこと
        logger.info("caching latents.")

        image_infos = list(self.image_data.values())

        # sort by resolution
        image_infos.sort(key=lambda info: info.bucket_reso[0] * info.bucket_reso[1])

        # split by resolution and some conditions
        class Condition:
            def __init__(self, reso, flip_aug, alpha_mask, random_crop):
                self.reso = reso
                self.flip_aug = flip_aug
                self.alpha_mask = alpha_mask
                self.random_crop = random_crop

            def __eq__(self, other):
                return (
                    self.reso == other.reso
                    and self.flip_aug == other.flip_aug
                    and self.alpha_mask == other.alpha_mask
                    and self.random_crop == other.random_crop
                )

        batches: List[Tuple[Condition, List[ImageInfo]]] = []
        batch: List[ImageInfo] = []
        current_condition = None

        logger.info("checking cache validity...")
        for info in tqdm(image_infos):
            subset = self.image_to_subset[info.image_key]

            if info.latents_npz is not None:  # fine tuning dataset
                continue

            # check disk cache exists and size of latents
            if cache_to_disk:
                info.latents_npz = os.path.splitext(info.absolute_path)[0] + file_suffix
                if not is_main_process:  # store to info only
                    continue

                cache_available = is_disk_cached_latents_is_expected(
                    info.bucket_reso, info.latents_npz, subset.flip_aug, subset.alpha_mask
                )

                if cache_available:  # do not add to batch
                    continue

            # if batch is not empty and condition is changed, flush the batch. Note that current_condition is not None if batch is not empty
            condition = Condition(info.bucket_reso, subset.flip_aug, subset.alpha_mask, subset.random_crop)
            if len(batch) > 0 and current_condition != condition:
                batches.append((current_condition, batch))
                batch = []

            batch.append(info)
            current_condition = condition

            # if number of data in batch is enough, flush the batch
            if len(batch) >= vae_batch_size:
                batches.append((current_condition, batch))
                batch = []
                current_condition = None

        if len(batch) > 0:
            batches.append((current_condition, batch))

        if cache_to_disk and not is_main_process:  # if cache to disk, don't cache latents in non-main process, set to info only
            return

        # iterate batches: batch doesn't have image, image will be loaded in cache_batch_latents and discarded
        logger.info("caching latents...")
        for condition, batch in tqdm(batches, smoothing=1, total=len(batches)):
            cache_batch_latents(vae, cache_to_disk, batch, condition.flip_aug, condition.alpha_mask, condition.random_crop)

    def new_cache_text_encoder_outputs(self, models: List[Any], accelerator: Accelerator):
        r"""
        a brand new method to cache text encoder outputs. This method caches text encoder outputs with caching strategy.
        """
        tokenize_strategy = TokenizeStrategy.get_strategy()
        text_encoding_strategy = TextEncodingStrategy.get_strategy()
        caching_strategy = TextEncoderOutputsCachingStrategy.get_strategy()
        batch_size = caching_strategy.batch_size or self.batch_size

        logger.info("caching Text Encoder outputs with caching strategy.")
        image_infos = list(self.image_data.values())

        # split by resolution
        batches = []
        batch = []

        # support multiple-gpus
        num_processes = accelerator.num_processes
        process_index = accelerator.process_index

        logger.info("checking cache validity...")
        for i, info in enumerate(tqdm(image_infos)):
            # check disk cache exists and size of text encoder outputs
            if caching_strategy.cache_to_disk:
                te_out_npz = caching_strategy.get_outputs_npz_path(info.absolute_path)
                info.text_encoder_outputs_npz = te_out_npz  # set npz filename regardless of cache availability

                # if the modulo of num_processes is not equal to process_index, skip caching
                # this makes each process cache different text encoder outputs
                if i % num_processes != process_index:
                    continue

                cache_available = caching_strategy.is_disk_cached_outputs_expected(te_out_npz)
                if cache_available:  # do not add to batch
                    continue

            batch.append(info)

            # if number of data in batch is enough, flush the batch
            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        if len(batches) == 0:
            logger.info("no Text Encoder outputs to cache")
            return

        # iterate batches
        logger.info("caching Text Encoder outputs...")
        for batch in tqdm(batches, smoothing=1, total=len(batches)):
            # cache_batch_latents(vae, cache_to_disk, batch, subset.flip_aug, subset.alpha_mask, subset.random_crop)
            caching_strategy.cache_batch_outputs(tokenize_strategy, models, text_encoding_strategy, batch)

    # if weight_dtype is specified, Text Encoder itself and output will be converted to the dtype
    # this method is only for SDXL, but it should be implemented here because it needs to be a method of dataset
    # to support SD1/2, it needs a flag for v2, but it is postponed
    def cache_text_encoder_outputs(
        self, tokenizers, text_encoders, device, output_dtype, cache_to_disk=False, is_main_process=True
    ):
        assert len(tokenizers) == 2, "only support SDXL"
        return self.cache_text_encoder_outputs_common(
            tokenizers, text_encoders, [device, device], output_dtype, [output_dtype], cache_to_disk, is_main_process
        )

    # same as above, but for SD3
    def cache_text_encoder_outputs_sd3(
        self, tokenizer, text_encoders, devices, output_dtype, te_dtypes, cache_to_disk=False, is_main_process=True, batch_size=None
    ):
        return self.cache_text_encoder_outputs_common(
            [tokenizer],
            text_encoders,
            devices,
            output_dtype,
            te_dtypes,
            cache_to_disk,
            is_main_process,
            TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX_SD3,
            batch_size,
        )

    def cache_text_encoder_outputs_common(
        self,
        tokenizers,
        text_encoders,
        devices,
        output_dtype,
        te_dtypes,
        cache_to_disk=False,
        is_main_process=True,
        file_suffix=TEXT_ENCODER_OUTPUTS_CACHE_SUFFIX,
        batch_size=None,
    ):
        # latentsのキャッシュと同様に、ディスクへのキャッシュに対応する
        # またマルチGPUには対応していないので、そちらはtools/cache_latents.pyを使うこと
        logger.info("caching text encoder outputs.")

        tokenize_strategy = TokenizeStrategy.get_strategy()

        if batch_size is None:
            batch_size = self.batch_size

        image_infos = list(self.image_data.values())

        logger.info("checking cache existence...")
        image_infos_to_cache = []
        for info in tqdm(image_infos):
            # subset = self.image_to_subset[info.image_key]
            if cache_to_disk:
                te_out_npz = os.path.splitext(info.absolute_path)[0] + file_suffix
                info.text_encoder_outputs_npz = te_out_npz

                if not is_main_process:  # store to info only
                    continue

                if os.path.exists(te_out_npz):
                    # TODO check varidity of cache here
                    continue

            image_infos_to_cache.append(info)

        if cache_to_disk and not is_main_process:  # if cache to disk, don't cache latents in non-main process, set to info only
            return

        # prepare tokenizers and text encoders
        for text_encoder, device, te_dtype in zip(text_encoders, devices, te_dtypes):
            text_encoder.to(device)
            if te_dtype is not None:
                text_encoder.to(dtype=te_dtype)

        # create batch
        is_sd3 = len(tokenizers) == 1
        batch = []
        batches = []
        for info in image_infos_to_cache:
            if not is_sd3:
                input_ids1 = self.get_input_ids(info.caption, tokenizers[0])
                input_ids2 = self.get_input_ids(info.caption, tokenizers[1])
                batch.append((info, input_ids1, input_ids2))
            else:
                l_tokens, g_tokens, t5_tokens = tokenize_strategy.tokenize(info.caption)
                batch.append((info, l_tokens, g_tokens, t5_tokens))

            if len(batch) >= batch_size:
                batches.append(batch)
                batch = []

        if len(batch) > 0:
            batches.append(batch)

        # iterate batches: call text encoder and cache outputs for memory or disk
        logger.info("caching text encoder outputs...")
        if not is_sd3:
            for batch in tqdm(batches):
                infos, input_ids1, input_ids2 = zip(*batch)
                input_ids1 = torch.stack(input_ids1, dim=0)
                input_ids2 = torch.stack(input_ids2, dim=0)
                cache_batch_text_encoder_outputs(
                    infos, tokenizers, text_encoders, self.max_token_length, cache_to_disk, input_ids1, input_ids2, output_dtype
                )
        else:
            for batch in tqdm(batches):
                infos, l_tokens, g_tokens, t5_tokens = zip(*batch)

                # stack tokens
                # l_tokens = [tokens[0] for tokens in l_tokens]
                # g_tokens = [tokens[0] for tokens in g_tokens]
                # t5_tokens = [tokens[0] for tokens in t5_tokens]

                cache_batch_text_encoder_outputs_sd3(
                    infos,
                    tokenizers[0],
                    text_encoders,
                    self.max_token_length,
                    cache_to_disk,
                    (l_tokens, g_tokens, t5_tokens),
                    output_dtype,
                )

    def get_image_size(self, image_path):
        if image_path.endswith(".jxl") or image_path.endswith(".JXL"):
            return get_jxl_size(image_path)
        # return imagesize.get(image_path)
        image_size = imagesize.get(image_path)
        if image_size[0] <= 0:
            # imagesize doesn't work for some images, so use PIL as a fallback
            try:
                with Image.open(image_path) as img:
                    image_size = img.size
            except Exception as e:
                logger.warning(f"failed to get image size: {image_path}, error: {e}")
                image_size = (0, 0)
        return image_size

    def load_image_with_face_info(self, subset: BaseSubset, image_path: str, alpha_mask=False):
        img = load_image(image_path, alpha_mask)

        face_cx = face_cy = face_w = face_h = 0
        if subset.face_crop_aug_range is not None:
            tokens = os.path.splitext(os.path.basename(image_path))[0].split("_")
            if len(tokens) >= 5:
                face_cx = int(tokens[-4])
                face_cy = int(tokens[-3])
                face_w = int(tokens[-2])
                face_h = int(tokens[-1])

        return img, face_cx, face_cy, face_w, face_h

    # いい感じに切り出す
    def crop_target(self, subset: BaseSubset, image, face_cx, face_cy, face_w, face_h):
        height, width = image.shape[0:2]
        if height == self.height and width == self.width:
            return image

        # 画像サイズはsizeより大きいのでリサイズする
        face_size = max(face_w, face_h)
        size = min(self.height, self.width)  # 短いほう
        min_scale = max(self.height / height, self.width / width)  # 画像がモデル入力サイズぴったりになる倍率（最小の倍率）
        min_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[1])))  # 指定した顔最小サイズ
        max_scale = min(1.0, max(min_scale, size / (face_size * subset.face_crop_aug_range[0])))  # 指定した顔最大サイズ
        if min_scale >= max_scale:  # range指定がmin==max
            scale = min_scale
        else:
            scale = random.uniform(min_scale, max_scale)

        nh = int(height * scale + 0.5)
        nw = int(width * scale + 0.5)
        assert nh >= self.height and nw >= self.width, f"internal error. small scale {scale}, {width}*{height}"
        image = resize_image(image, width, height, nw, nh, subset.resize_interpolation)
        face_cx = int(face_cx * scale + 0.5)
        face_cy = int(face_cy * scale + 0.5)
        height, width = nh, nw

        # 顔を中心として448*640とかへ切り出す
        for axis, (target_size, length, face_p) in enumerate(zip((self.height, self.width), (height, width), (face_cy, face_cx))):
            p1 = face_p - target_size // 2  # 顔を中心に持ってくるための切り出し位置

            if subset.random_crop:
                # 背景も含めるために顔を中心に置く確率を高めつつずらす
                range = max(length - face_p, face_p)  # 画像の端から顔中心までの距離の長いほう
                p1 = p1 + (random.randint(0, range) + random.randint(0, range)) - range  # -range ~ +range までのいい感じの乱数
            else:
                # range指定があるときのみ、すこしだけランダムに（わりと適当）
                if subset.face_crop_aug_range[0] != subset.face_crop_aug_range[1]:
                    if face_size > size // 10 and face_size >= 40:
                        p1 = p1 + random.randint(-face_size // 20, +face_size // 20)

            p1 = max(0, min(p1, length - target_size))

            if axis == 0:
                image = image[p1 : p1 + target_size, :]
            else:
                image = image[:, p1 : p1 + target_size]

        return image

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        bucket = self.bucket_manager.buckets[self.buckets_indices[index].bucket_index]
        bucket_batch_size = self.buckets_indices[index].bucket_batch_size
        image_index = self.buckets_indices[index].batch_index * bucket_batch_size

        if self.caching_mode is not None:  # return batch for latents/text encoder outputs caching
            return self.get_item_for_caching(bucket, bucket_batch_size, image_index)

        loss_weights = []
        captions = []
        input_ids_list = []
        latents_list = []
        alpha_mask_list = []
        images = []
        image_paths = []
        original_sizes_hw = []
        crop_top_lefts = []
        target_sizes_hw = []
        flippeds = []  # 変数名が微妙
        text_encoder_outputs_list = []
        custom_attributes = []

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            custom_attributes.append(subset.custom_attributes)
            image_paths.append(image_info.absolute_path)

            # in case of fine tuning, is_reg is always False
            loss_weights.append(self.prior_loss_weight if image_info.is_reg else 1.0)

            flipped = subset.flip_aug and random.random() < 0.5  # not flipped or flipped with 50% chance

            # image/latentsを処理する
            if image_info.latents is not None:  # cache_latents=Trueの場合
                original_size = image_info.latents_original_size
                crop_ltrb = image_info.latents_crop_ltrb  # calc values later if flipped
                if not flipped:
                    latents = image_info.latents
                    alpha_mask = image_info.alpha_mask
                else:
                    latents = image_info.latents_flipped
                    alpha_mask = None if image_info.alpha_mask is None else torch.flip(image_info.alpha_mask, [1])

                image = None
            elif image_info.latents_disk_cache_ref is not None:
                latents, original_size, crop_ltrb, flipped_latents, alpha_mask = self.latents_caching_strategy.load_latents_from_disk(
                    image_info.latents_disk_cache_ref, image_info.bucket_reso
                )
                if flipped:
                    latents = flipped_latents
                    alpha_mask = None if alpha_mask is None else torch.flip(alpha_mask, [1])
                    del flipped_latents
                if isinstance(latents, np.ndarray):
                    latents = torch.from_numpy(latents)
                else:
                    latents = latents.detach().cpu()
                if alpha_mask is not None:
                    if isinstance(alpha_mask, np.ndarray):
                        alpha_mask = torch.from_numpy(alpha_mask)
                    else:
                        alpha_mask = alpha_mask.detach().cpu()

                image = None
            elif image_info.latents_npz is not None:  # FineTuningDatasetまたはcache_latents_to_disk=Trueの場合
                latents, original_size, crop_ltrb, flipped_latents, alpha_mask = (
                    self.latents_caching_strategy.load_latents_from_disk(image_info.latents_npz, image_info.bucket_reso)
                )
                if flipped:
                    latents = flipped_latents
                    alpha_mask = None if alpha_mask is None else alpha_mask[:, ::-1].copy()  # copy to avoid negative stride problem
                    del flipped_latents
                latents = torch.FloatTensor(latents)
                if alpha_mask is not None:
                    alpha_mask = torch.FloatTensor(alpha_mask)

                image = None
            else:
                # 画像を読み込み、必要ならcropする
                img, face_cx, face_cy, face_w, face_h = self.load_image_with_face_info(
                    subset, image_info.absolute_path, subset.alpha_mask
                )
                im_h, im_w = img.shape[0:2]

                if self.enable_bucket:
                    img, original_size, crop_ltrb = trim_and_resize_if_required(
                        subset.random_crop,
                        img,
                        image_info.bucket_reso,
                        image_info.resized_size,
                        resize_interpolation=image_info.resize_interpolation,
                    )
                else:
                    if face_cx > 0:  # 顔位置情報あり
                        img = self.crop_target(subset, img, face_cx, face_cy, face_w, face_h)
                    elif im_h > self.height or im_w > self.width:
                        assert (
                            subset.random_crop
                        ), f"image too large, but cropping and bucketing are disabled / 画像サイズが大きいのでface_crop_aug_rangeかrandom_crop、またはbucketを有効にしてください: {image_info.absolute_path} / 图像尺寸过大且未启用裁剪或分桶，请启用 face_crop_aug_range、random_crop 或 bucket: {image_info.absolute_path}"
                        if im_h > self.height:
                            p = random.randint(0, im_h - self.height)
                            img = img[p : p + self.height]
                        if im_w > self.width:
                            p = random.randint(0, im_w - self.width)
                            img = img[:, p : p + self.width]

                    im_h, im_w = img.shape[0:2]
                    assert (
                        im_h == self.height and im_w == self.width
                    ), f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path} / 图像尺寸过小: {image_info.absolute_path}"

                    original_size = [im_w, im_h]
                    crop_ltrb = (0, 0, 0, 0)

                # augmentation
                aug = self.aug_helper.get_augmentor(subset.color_aug)
                if aug is not None:
                    # augment RGB channels only
                    img_rgb = img[:, :, :3]
                    img_rgb = aug(image=img_rgb)["image"]
                    img[:, :, :3] = img_rgb

                if flipped:
                    img = img[:, ::-1, :].copy()  # copy to avoid negative stride problem

                if subset.alpha_mask:
                    if img.shape[2] == 4:
                        alpha_mask = img[:, :, 3]  # [H,W]
                        alpha_mask = alpha_mask.astype(np.float32) / 255.0  # 0.0~1.0
                        alpha_mask = torch.FloatTensor(alpha_mask)
                    else:
                        alpha_mask = torch.ones((img.shape[0], img.shape[1]), dtype=torch.float32)
                else:
                    alpha_mask = None

                img = img[:, :, :3]  # remove alpha channel

                latents = None
                image = self.image_transforms(img)  # -1.0~1.0のtorch.Tensorになる
                del img

            images.append(image)
            latents_list.append(latents)
            alpha_mask_list.append(alpha_mask)

            target_size = (image.shape[2], image.shape[1]) if image is not None else (latents.shape[2] * 8, latents.shape[1] * 8)

            if not flipped:
                crop_left_top = (crop_ltrb[0], crop_ltrb[1])
            else:
                # crop_ltrb[2] is right, so target_size[0] - crop_ltrb[2] is left in flipped image
                crop_left_top = (target_size[0] - crop_ltrb[2], crop_ltrb[1])

            original_sizes_hw.append((int(original_size[1]), int(original_size[0])))
            crop_top_lefts.append((int(crop_left_top[1]), int(crop_left_top[0])))
            target_sizes_hw.append((int(target_size[1]), int(target_size[0])))
            flippeds.append(flipped)

            # captionとtext encoder outputを処理する
            caption = image_info.caption  # default

            tokenization_required = (
                self.text_encoder_output_caching_strategy is None or self.text_encoder_output_caching_strategy.is_partial
            )
            text_encoder_outputs = None
            input_ids = None

            if image_info.text_encoder_outputs is not None:
                # cached
                text_encoder_outputs = image_info.text_encoder_outputs
            elif image_info.text_encoder_outputs_npz is not None:
                # on disk
                text_encoder_outputs = self.text_encoder_output_caching_strategy.load_outputs_npz(
                    image_info.text_encoder_outputs_npz
                )
            else:
                tokenization_required = True
            text_encoder_outputs_list.append(text_encoder_outputs)

            if tokenization_required:
                caption = self.process_caption(subset, image_info.caption)
                input_ids = [ids[0] for ids in self.tokenize_strategy.tokenize(caption)]  # remove batch dimension
                # if self.XTI_layers:
                #     caption_layer = []
                #     for layer in self.XTI_layers:
                #         token_strings_from = " ".join(self.token_strings)
                #         token_strings_to = " ".join([f"{x}_{layer}" for x in self.token_strings])
                #         caption_ = caption.replace(token_strings_from, token_strings_to)
                #         caption_layer.append(caption_)
                #     captions.append(caption_layer)
                # else:
                #     captions.append(caption)

                # if not self.token_padding_disabled:  # this option might be omitted in future
                #     # TODO get_input_ids must support SD3
                #     if self.XTI_layers:
                #         token_caption = self.get_input_ids(caption_layer, self.tokenizers[0])
                #     else:
                #         token_caption = self.get_input_ids(caption, self.tokenizers[0])
                #     input_ids_list.append(token_caption)

                #     if len(self.tokenizers) > 1:
                #         if self.XTI_layers:
                #             token_caption2 = self.get_input_ids(caption_layer, self.tokenizers[1])
                #         else:
                #             token_caption2 = self.get_input_ids(caption, self.tokenizers[1])
                #         input_ids2_list.append(token_caption2)

            input_ids_list.append(input_ids)
            captions.append(caption)

        def none_or_stack_elements(tensors_list, converter):
            # [[clip_l, clip_g, t5xxl], [clip_l, clip_g, t5xxl], ...] -> [torch.stack(clip_l), torch.stack(clip_g), torch.stack(t5xxl)]
            if len(tensors_list) == 0 or tensors_list[0] == None or len(tensors_list[0]) == 0 or tensors_list[0][0] is None:
                return None

            # old implementation without padding: all elements must have same length
            # return [torch.stack([converter(x[i]) for x in tensors_list]) for i in range(len(tensors_list[0]))]

            # new implementation with padding support
            result = []
            for i in range(len(tensors_list[0])):
                tensors = [x[i] for x in tensors_list]
                if tensors[0].ndim == 0:
                    # scalar value: e.g. ocr mask
                    result.append(torch.stack([converter(x[i]) for x in tensors_list]))
                    continue

                min_len = min([len(x) for x in tensors])
                max_len = max([len(x) for x in tensors])

                if min_len == max_len:
                    # no padding
                    result.append(torch.stack([converter(x) for x in tensors]))
                else:
                    # padding
                    tensors = [converter(x) for x in tensors]
                    if tensors[0].ndim == 1:
                        # input_ids or mask
                        result.append(torch.stack([(torch.nn.functional.pad(x, (0, max_len - x.shape[0]))) for x in tensors]))
                    else:
                        # text encoder outputs
                        result.append(torch.stack([(torch.nn.functional.pad(x, (0, 0, 0, max_len - x.shape[0]))) for x in tensors]))
            return result

        # set example
        example = {}
        example["custom_attributes"] = custom_attributes  # may be list of empty dict
        example["loss_weights"] = torch.FloatTensor(loss_weights)
        example["text_encoder_outputs_list"] = none_or_stack_elements(text_encoder_outputs_list, torch.as_tensor)
        example["input_ids_list"] = none_or_stack_elements(input_ids_list, lambda x: x)

        # if one of alpha_masks is not None, we need to replace None with ones
        none_or_not = [x is None for x in alpha_mask_list]
        if all(none_or_not):
            example["alpha_masks"] = None
        elif any(none_or_not):
            for i in range(len(alpha_mask_list)):
                if alpha_mask_list[i] is None:
                    if images[i] is not None:
                        alpha_mask_list[i] = torch.ones((images[i].shape[1], images[i].shape[2]), dtype=torch.float32)
                    else:
                        alpha_mask_list[i] = torch.ones(
                            (latents_list[i].shape[1] * 8, latents_list[i].shape[2] * 8), dtype=torch.float32
                        )
            example["alpha_masks"] = torch.stack(alpha_mask_list)
        else:
            example["alpha_masks"] = torch.stack(alpha_mask_list)

        if images[0] is not None:
            images = torch.stack(images)
            images = images.to(memory_format=torch.contiguous_format).float()
        else:
            images = None
        example["images"] = images

        if latents_list[0] is not None:
            latent_shapes = [tuple(latents.shape) for latents in latents_list]
            first_latent_shape = latent_shapes[0]
            mismatched_latents = [
                f"{path} => {shape}" for path, shape in zip(image_paths, latent_shapes) if shape != first_latent_shape
            ]
            if mismatched_latents:
                batch_summary = "\n".join([f"{path} => {shape}" for path, shape in zip(image_paths, latent_shapes)])
                raise RuntimeError(
                    "Latents in the same batch have different shapes. This usually means stale latent cache files were reused "
                    "(for example after changing resolution/bucket settings or replacing images without clearing cache). "
                    "Delete the dataset latent cache files (*.safetensors or *.npz such as *_sd.npz / *_sdxl.npz / legacy .npz) and any "
                    "metadata_cache.json in the dataset folder, then run again.\n"
                    "同一个 batch 中的 latent 形状不一致。通常表示复用了过期的 latent 缓存文件，"
                    "例如修改了分辨率 / bucket 设置，或替换了图片但没有清理缓存。"
                    "请删除数据集目录中的 latent 缓存文件（*.safetensors 或 *.npz，例如 *_sd.npz / *_sdxl.npz / 旧版 .npz）"
                    "以及 metadata_cache.json 后再重试。\n"
                    f"Batch latents:\n{batch_summary}"
                )
            example["latents"] = torch.stack(latents_list)
        else:
            example["latents"] = None
        example["captions"] = captions

        example["original_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in original_sizes_hw])
        example["crop_top_lefts"] = torch.stack([torch.LongTensor(x) for x in crop_top_lefts])
        example["target_sizes_hw"] = torch.stack([torch.LongTensor(x) for x in target_sizes_hw])
        example["flippeds"] = flippeds

        example["network_multipliers"] = torch.FloatTensor([self.network_multiplier] * len(captions))

        if self.debug_dataset:
            example["image_keys"] = bucket[image_index : image_index + self.batch_size]
        return example

    def get_item_for_caching(self, bucket, bucket_batch_size, image_index):
        captions = []
        images = []
        input_ids1_list = []
        input_ids2_list = []
        absolute_paths = []
        resized_sizes = []
        bucket_reso = None
        flip_aug = None
        alpha_mask = None
        random_crop = None

        for image_key in bucket[image_index : image_index + bucket_batch_size]:
            image_info = self.image_data[image_key]
            subset = self.image_to_subset[image_key]

            if flip_aug is None:
                flip_aug = subset.flip_aug
                alpha_mask = subset.alpha_mask
                random_crop = subset.random_crop
                bucket_reso = image_info.bucket_reso
            else:
                # TODO そもそも混在してても動くようにしたほうがいい
                assert flip_aug == subset.flip_aug, "flip_aug must be same in a batch"
                assert alpha_mask == subset.alpha_mask, "alpha_mask must be same in a batch"
                assert random_crop == subset.random_crop, "random_crop must be same in a batch"
                assert bucket_reso == image_info.bucket_reso, "bucket_reso must be same in a batch"

            caption = image_info.caption  # TODO cache some patterns of dropping, shuffling, etc.

            if self.caching_mode == "latents":
                image = load_image(image_info.absolute_path)
            else:
                image = None

            if self.caching_mode == "text":
                input_ids1 = self.get_input_ids(caption, self.tokenizers[0])
                input_ids2 = self.get_input_ids(caption, self.tokenizers[1])
            else:
                input_ids1 = None
                input_ids2 = None

            captions.append(caption)
            images.append(image)
            input_ids1_list.append(input_ids1)
            input_ids2_list.append(input_ids2)
            absolute_paths.append(image_info.absolute_path)
            resized_sizes.append(image_info.resized_size)

        example = {}

        if images[0] is None:
            images = None
        example["images"] = images

        example["captions"] = captions
        example["input_ids1_list"] = input_ids1_list
        example["input_ids2_list"] = input_ids2_list
        example["absolute_paths"] = absolute_paths
        example["resized_sizes"] = resized_sizes
        example["flip_aug"] = flip_aug
        example["alpha_mask"] = alpha_mask
        example["random_crop"] = random_crop
        example["bucket_reso"] = bucket_reso
        return example


class DreamBoothDataset(BaseDataset):
    IMAGE_INFO_CACHE_FILE = "metadata_cache.json"

    # The is_training_dataset defines the type of dataset, training or validation
    # if is_training_dataset is True -> training dataset
    # if is_training_dataset is False -> validation dataset
    def __init__(
        self,
        subsets: Sequence[DreamBoothSubset],
        is_training_dataset: bool,
        batch_size: int,
        resolution,
        skip_image_resolution: Optional[Tuple[int, int]],
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        bucket_selection_mode: str,
        bucket_custom_resos: Optional[str],
        prior_loss_weight: float,
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str],
    ) -> None:
        super().__init__(resolution, skip_image_resolution, network_multiplier, debug_dataset, resize_interpolation)

        assert resolution is not None, f"resolution is required / resolution（解像度）指定は必須です / 必须指定 resolution（分辨率）"

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # 短いほう
        self.prior_loss_weight = prior_loss_weight
        self.latents_cache = None
        self.is_training_dataset = is_training_dataset
        self.validation_seed = validation_seed
        self.validation_split = validation_split

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            min_bucket_reso, max_bucket_reso = self.adjust_min_max_bucket_reso_by_steps(
                resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps
            )
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
            self.bucket_selection_mode = str(bucket_selection_mode or "legacy").strip().lower()
            self.bucket_custom_resos = bucket_custom_resos
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # この情報は使われない
            self.bucket_no_upscale = False
            self.bucket_selection_mode = "legacy"
            self.bucket_custom_resos = None

        def read_caption(img_path, subset: DreamBoothSubset):
            # captionの候補ファイル名を作る
            base_name = os.path.splitext(img_path)[0]
            base_name_face_det = base_name
            tokens = base_name.split("_")
            if len(tokens) >= 5:
                base_name_face_det = "_".join(tokens[:-4])

            custom_attributes = subset.custom_attributes if isinstance(subset.custom_attributes, dict) else {}
            prefer_json_caption = custom_attributes.get("prefer_json_caption", custom_attributes.get("prefer_json", False))
            if isinstance(prefer_json_caption, str):
                prefer_json_caption = prefer_json_caption.strip().lower() in {"1", "true", "yes", "on"}
            else:
                prefer_json_caption = bool(prefer_json_caption)

            caption_extension = subset.caption_extension
            cap_paths = [base_name + caption_extension, base_name_face_det + caption_extension]

            if prefer_json_caption or str(caption_extension).lower() == ".json":
                json_paths = [base_name + ".json", base_name_face_det + ".json"]
                for json_path in json_paths:
                    if not os.path.isfile(json_path):
                        continue
                    special_caption = anima_caption_util.load_special_caption_from_json_path(json_path)
                    if special_caption is not None:
                        return special_caption

            caption = None
            for cap_path in cap_paths:
                if os.path.isfile(cap_path):
                    with open(cap_path, "rt", encoding="utf-8") as f:
                        try:
                            lines = f.readlines()
                        except UnicodeDecodeError as e:
                            logger.error(
                                f"illegal char in file (not UTF-8) / ファイルにUTF-8以外の文字があります: {cap_path} / 文件中包含非法字符（非 UTF-8）: {cap_path}"
                            )
                            raise e
                        assert len(lines) > 0, f"caption file is empty / キャプションファイルが空です: {cap_path} / caption 文件为空: {cap_path}"
                        if subset.enable_wildcard:
                            caption = "\n".join([line.strip() for line in lines if line.strip() != ""])  # 空行を除く、改行で連結
                        else:
                            caption = lines[0].strip()
                    break
            return caption

        def load_dreambooth_dir(subset: DreamBoothSubset):
            if not os.path.isdir(subset.image_dir):
                logger.warning(f"not directory: {subset.image_dir}")
                return [], [], []

            info_cache_file = os.path.join(subset.image_dir, self.IMAGE_INFO_CACHE_FILE)
            use_cached_info_for_subset = subset.cache_info
            if use_cached_info_for_subset:
                logger.info(
                    f"using cached image info for this subset / このサブセットで、キャッシュされた画像情報を使います / 该子集将使用缓存的图像信息: {info_cache_file}"
                )
                if not os.path.isfile(info_cache_file):
                    logger.warning(
                        f"image info file not found. You can ignore this warning if this is the first time to use this subset / "
                        f"キャッシュファイルが見つかりませんでした。初回実行時はこの警告を無視してください / "
                        f"未找到图像信息缓存文件。如果这是首次使用该子集，可忽略此警告: {info_cache_file}"
                    )
                    use_cached_info_for_subset = False

            if use_cached_info_for_subset:
                # json: {`img_path`:{"caption": "caption...", "resolution": [width, height]}, ...}
                with open(info_cache_file, "r", encoding="utf-8") as f:
                    metas = json.load(f)
                img_paths = list(metas.keys())
                sizes: List[Optional[Tuple[int, int]]] = [meta["resolution"] for meta in metas.values()]

                # we may need to check image size and existence of image files, but it takes time, so user should check it before training
            else:
                img_paths = glob_images(subset.image_dir, "*")
                sizes: List[Optional[Tuple[int, int]]] = [None] * len(img_paths)

                # new caching: get image size from cache files
                strategy = LatentsCachingStrategy.get_strategy()
                if strategy is not None:
                    logger.info("get image size from name of cache files")

                    # make image path to npz path mapping
                    npz_paths = glob.glob(os.path.join(subset.image_dir, "*" + strategy.cache_suffix))
                    npz_paths.sort(
                        key=lambda item: item.rsplit("_", maxsplit=2)[0]
                    )  # sort by name excluding resolution and cache_suffix
                    npz_path_index = 0

                    size_set_count = 0
                    for i, img_path in enumerate(tqdm(img_paths)):
                        l = len(os.path.splitext(img_path)[0])  # remove extension
                        found = False
                        while npz_path_index < len(npz_paths):  # until found or end of npz_paths
                            # npz_paths are sorted, so if npz_path > img_path, img_path is not found
                            if npz_paths[npz_path_index][:l] > img_path[:l]:
                                break
                            if npz_paths[npz_path_index][:l] == img_path[:l]:  # found
                                found = True
                                break
                            npz_path_index += 1  # next npz_path

                        if found:
                            w, h = strategy.get_image_size_from_disk_cache_path(img_path, npz_paths[npz_path_index])
                        else:
                            w, h = None, None

                        if w is not None and h is not None:
                            sizes[i] = (w, h)
                            size_set_count += 1
                    logger.info(f"set image size from cache files: {size_set_count}/{len(img_paths)}")

            # We want to create a training and validation split. This should be improved in the future
            # to allow a clearer distinction between training and validation. This can be seen as a
            # short-term solution to limit what is necessary to implement validation datasets
            #
            # We split the dataset for the subset based on if we are doing a validation split
            # The self.is_training_dataset defines the type of dataset, training or validation
            # if self.is_training_dataset is True -> training dataset
            # if self.is_training_dataset is False -> validation dataset
            if self.validation_split > 0.0:
                # For regularization images we do not want to split this dataset.
                if subset.is_reg is True:
                    # Skip any validation dataset for regularization images
                    if self.is_training_dataset is False:
                        img_paths = []
                        sizes = []
                    # Otherwise the img_paths remain as original img_paths and no split
                    # required for training images dataset of regularization images
                else:
                    img_paths, sizes = split_train_val(
                        img_paths, sizes, self.is_training_dataset, self.validation_split, self.validation_seed
                    )

            logger.info(f"found directory {subset.image_dir} contains {len(img_paths)} image files")

            if use_cached_info_for_subset:
                captions = [meta["caption"] for meta in metas.values()]
                missing_captions = [img_path for img_path, caption in zip(img_paths, captions) if caption is None or caption == ""]
            else:
                # 画像ファイルごとにプロンプトを読み込み、もしあればそちらを使う
                captions = []
                missing_captions = []
                for img_path in tqdm(img_paths, desc="read caption"):
                    cap_for_img = read_caption(img_path, subset)
                    if cap_for_img is None and subset.class_tokens is None:
                        logger.warning(
                            f"neither caption file nor class tokens are found. use empty caption for {img_path} / キャプションファイルもclass tokenも見つかりませんでした。空のキャプションを使用します: {img_path} / 未找到 caption 文件和 class token，将使用空 caption: {img_path}"
                        )
                        captions.append("")
                        missing_captions.append(img_path)
                    else:
                        if cap_for_img is None:
                            captions.append(subset.class_tokens)
                            missing_captions.append(img_path)
                        else:
                            captions.append(cap_for_img)

            self.set_tag_frequency(os.path.basename(subset.image_dir), captions)  # タグ頻度を記録

            if missing_captions:
                number_of_missing_captions = len(missing_captions)
                number_of_missing_captions_to_show = 5
                remaining_missing_captions = number_of_missing_captions - number_of_missing_captions_to_show

                logger.warning(
                    f"No caption file found for {number_of_missing_captions} images. Training will continue without captions for these images. If class token exists, it will be used. / {number_of_missing_captions}枚の画像にキャプションファイルが見つかりませんでした。これらの画像についてはキャプションなしで学習を続行します。class tokenが存在する場合はそれを使います。 / 有 {number_of_missing_captions} 张图像未找到 caption 文件，将继续无 caption 训练；若存在 class token 则会使用。"
                )
                for i, missing_caption in enumerate(missing_captions):
                    if i >= number_of_missing_captions_to_show:
                        logger.warning(missing_caption + f"... and {remaining_missing_captions} more")
                        break
                    logger.warning(missing_caption)

            if not use_cached_info_for_subset and subset.cache_info:
                logger.info(f"cache image info for / 画像情報をキャッシュします / 开始缓存图像信息: {info_cache_file}")
                sizes = [self.get_image_size(img_path) for img_path in tqdm(img_paths, desc="get image size")]
                matas = {}
                for img_path, caption, size in zip(img_paths, captions, sizes):
                    matas[img_path] = {"caption": caption, "resolution": list(size)}
                with open(info_cache_file, "w", encoding="utf-8") as f:
                    json.dump(matas, f, ensure_ascii=False, indent=2)
                logger.info(f"cache image info done for / 画像情報を出力しました / 图像信息缓存完成: {info_cache_file}")

            # if sizes are not set, image size will be read in make_buckets
            return img_paths, captions, sizes

        logger.info("prepare images.")
        num_train_images = 0
        num_reg_images = 0
        reg_infos: List[Tuple[ImageInfo, DreamBoothSubset]] = []
        for subset in subsets:
            num_repeats = subset.num_repeats if self.is_training_dataset else 1
            if num_repeats < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': num_repeats is less than 1 / num_repeatsが1を下回っているためサブセットを無視します / num_repeats 小于 1，已忽略该子集: {num_repeats}"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with image_dir='{subset.image_dir}': use the first one / 既にサブセットが登録されているため、重複した後発のサブセットを無視します / 检测到重复子集，保留第一个并忽略后续重复项"
                )
                continue

            img_paths, captions, sizes = load_dreambooth_dir(subset)
            if len(img_paths) < 1:
                logger.warning(
                    f"ignore subset with image_dir='{subset.image_dir}': no images found / 画像が見つからないためサブセットを無視します / 未找到图像，已忽略该子集"
                )
                continue

            if subset.is_reg:
                num_reg_images += num_repeats * len(img_paths)
            else:
                num_train_images += num_repeats * len(img_paths)

            for img_path, caption, size in zip(img_paths, captions, sizes):
                info = ImageInfo(img_path, num_repeats, caption, subset.is_reg, img_path, subset.caption_dropout_rate)
                info.resize_interpolation = (
                    subset.resize_interpolation if subset.resize_interpolation is not None else self.resize_interpolation
                )
                if size is not None:
                    info.image_size = size
                if subset.is_reg:
                    reg_infos.append((info, subset))
                else:
                    self.register_image(info, subset)

            subset.img_count = len(img_paths)
            self.subsets.append(subset)

        images_split_name = "train" if self.is_training_dataset else "validation"
        logger.info(f"{num_train_images} {images_split_name} images with repeats.")

        self.num_train_images = num_train_images

        logger.info(f"{num_reg_images} reg images with repeats.")
        if num_train_images < num_reg_images:
            logger.warning("some of reg images are not used / 正則化画像の数が多いので、一部使用されない正則化画像があります / 正则化图像数量过多，部分图像不会被使用")

        if num_reg_images == 0:
            logger.warning("no regularization images / 正則化画像が見つかりませんでした / 未找到正则化图像")
        else:
            # num_repeatsを計算する：どうせ大した数ではないのでループで処理する
            n = 0
            first_loop = True
            while n < num_train_images:
                for info, subset in reg_infos:
                    if first_loop:
                        self.register_image(info, subset)
                        n += info.num_repeats
                    else:
                        info.num_repeats += 1  # rewrite registered info
                        n += 1
                    if n >= num_train_images:
                        break
                first_loop = False

        self.num_reg_images = num_reg_images


class FineTuningDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[FineTuningSubset],
        batch_size: int,
        resolution,
        skip_image_resolution: Optional[Tuple[int, int]],
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        bucket_selection_mode: str,
        bucket_custom_resos: Optional[str],
        debug_dataset: bool,
        validation_seed: int,
        validation_split: float,
        resize_interpolation: Optional[str],
    ) -> None:
        super().__init__(resolution, skip_image_resolution, network_multiplier, debug_dataset, resize_interpolation)

        self.batch_size = batch_size
        self.size = min(self.width, self.height)  # 短いほう
        self.latents_cache = None

        self.enable_bucket = enable_bucket
        if self.enable_bucket:
            min_bucket_reso, max_bucket_reso = self.adjust_min_max_bucket_reso_by_steps(
                resolution, min_bucket_reso, max_bucket_reso, bucket_reso_steps
            )
            self.min_bucket_reso = min_bucket_reso
            self.max_bucket_reso = max_bucket_reso
            self.bucket_reso_steps = bucket_reso_steps
            self.bucket_no_upscale = bucket_no_upscale
            self.bucket_selection_mode = str(bucket_selection_mode or "legacy").strip().lower()
            self.bucket_custom_resos = bucket_custom_resos
        else:
            self.min_bucket_reso = None
            self.max_bucket_reso = None
            self.bucket_reso_steps = None  # この情報は使われない
            self.bucket_no_upscale = False
            self.bucket_selection_mode = "legacy"
            self.bucket_custom_resos = None

        self.num_train_images = 0
        self.num_reg_images = 0

        for subset in subsets:
            if subset.num_repeats < 1:
                logger.warning(
                    f"ignore subset with metadata_file='{subset.metadata_file}': num_repeats is less than 1 / num_repeatsが1を下回っているためサブセットを無視します / num_repeats 小于 1，已忽略该子集: {subset.num_repeats}"
                )
                continue

            if subset in self.subsets:
                logger.warning(
                    f"ignore duplicated subset with metadata_file='{subset.metadata_file}': use the first one / 既にサブセットが登録されているため、重複した後発のサブセットを無視します / 检测到重复子集，保留第一个并忽略后续重复项"
                )
                continue

            # メタデータを読み込む
            if os.path.exists(subset.metadata_file):
                if subset.metadata_file.endswith(".jsonl"):
                    logger.info(f"loading existing JSOL metadata: {subset.metadata_file}")
                    # optional JSONL format
                    # {"image_path": "/path/to/image1.jpg", "caption": "A caption for image1", "image_size": [width, height]}
                    metadata_entries = []
                    with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                        for line in f:
                            if not line.strip():
                                continue
                            line_md = json.loads(line)
                            image_md = {"image_key": line_md["image_path"], "caption": line_md.get("caption", "")}
                            if "image_size" in line_md:
                                image_md["image_size"] = line_md["image_size"]
                            if "tags" in line_md:
                                image_md["tags"] = line_md["tags"]
                            metadata_entries.append(image_md)
                else:
                    # standard JSON format
                    logger.info(f"loading existing metadata: {subset.metadata_file}")
                    with open(subset.metadata_file, "rt", encoding="utf-8") as f:
                        metadata = json.load(f)
                    metadata_entries = []
                    for image_key, image_md in metadata.items():
                        normalized_md = dict(image_md or {})
                        normalized_md["image_key"] = image_key
                        metadata_entries.append(normalized_md)
            else:
                raise ValueError(f"no metadata / メタデータファイルがありません / 未找到元数据文件: {subset.metadata_file}")

            if len(metadata_entries) < 1:
                logger.warning(
                    f"ignore subset with '{subset.metadata_file}': no image entries found / 画像に関するデータが見つからないためサブセットを無視します / 元数据中未找到图像条目，已忽略该子集"
                )
                continue

            # Add full path for image
            image_dirs = set()
            if subset.image_dir is not None:
                image_dirs.add(subset.image_dir)
            for metadata_entry in metadata_entries:
                image_key = metadata_entry["image_key"]
                if not os.path.isabs(image_key):
                    assert (
                        subset.image_dir is not None
                    ), f"image_dir is required when image paths are relative / 画像パスが相対パスの場合、image_dirの指定が必要です: {image_key} / 当图像路径为相对路径时，必须指定 image_dir: {image_key}"
                    abs_path = os.path.join(subset.image_dir, image_key)
                else:
                    abs_path = image_key
                    image_dirs.add(os.path.dirname(abs_path))
                metadata_entry["abs_path"] = abs_path

            # Enumerate existing npz files
            strategy = LatentsCachingStrategy.get_strategy()
            npz_paths = []
            if strategy is not None:
                for image_dir in image_dirs:
                    npz_paths.extend(glob.glob(os.path.join(image_dir, "*" + strategy.cache_suffix)))
                npz_paths = sorted(npz_paths, key=lambda item: len(os.path.basename(item)), reverse=True)  # longer paths first

            # Match image filename longer to shorter because some images share same prefix
            metadata_entries_sorted = sorted(metadata_entries, key=lambda item: len(item["image_key"]), reverse=True)

            # Collect tags and sizes
            tags_list = []
            size_set_from_metadata = 0
            size_set_from_cache_filename = 0
            for img_md in metadata_entries_sorted:
                image_key = img_md["image_key"]
                caption = img_md.get("caption")
                tags = img_md.get("tags")
                image_size = img_md.get("image_size")
                abs_path = img_md.get("abs_path")

                # search npz if image_size is not given
                npz_path = None
                if image_size is None:
                    image_without_ext = os.path.splitext(image_key)[0]
                    for candidate in npz_paths:
                        if candidate.startswith(image_without_ext):
                            npz_path = candidate
                            break
                    if npz_path is not None:
                        npz_paths.remove(npz_path)  # remove to avoid matching same file (share prefix)
                        abs_path = npz_path

                if caption is None:
                    caption = ""

                if subset.enable_wildcard:
                    # tags must be single line (split by caption separator)
                    if tags is not None:
                        tags = tags.replace("\n", subset.caption_separator)

                    # add tags to each line of caption
                    if tags is not None:
                        caption = "\n".join(
                            [f"{line}{subset.caption_separator}{tags}" for line in caption.split("\n") if line.strip() != ""]
                        )
                        tags_list.append(tags)
                else:
                    # use as is
                    if tags is not None and len(tags) > 0:
                        if len(caption) > 0:
                            caption = caption + subset.caption_separator
                        caption = caption + tags
                        tags_list.append(tags)

                if caption is None:
                    caption = ""

                image_info = ImageInfo(image_key, subset.num_repeats, caption, False, abs_path, subset.caption_dropout_rate)
                image_info.resize_interpolation = (
                    subset.resize_interpolation if subset.resize_interpolation is not None else self.resize_interpolation
                )

                if image_size is not None:
                    image_info.image_size = tuple(image_size)  # width, height
                    size_set_from_metadata += 1
                elif npz_path is not None:
                    # get image size from npz filename
                    w, h = strategy.get_image_size_from_disk_cache_path(abs_path, npz_path)
                    image_info.image_size = (w, h)
                    size_set_from_cache_filename += 1

                self.register_image(image_info, subset)

            if size_set_from_cache_filename > 0:
                logger.info(
                    f"set image size from cache files: {size_set_from_cache_filename}/{len(metadata_entries_sorted)}"
                )
            if size_set_from_metadata > 0:
                logger.info(f"set image size from metadata: {size_set_from_metadata}/{len(metadata_entries_sorted)}")
            self.num_train_images += len(metadata_entries) * subset.num_repeats

            # TODO do not record tag freq when no tag
            self.set_tag_frequency(os.path.basename(subset.metadata_file), tags_list)
            subset.img_count = len(metadata_entries)
            self.subsets.append(subset)


class ControlNetDataset(BaseDataset):
    def __init__(
        self,
        subsets: Sequence[ControlNetSubset],
        batch_size: int,
        resolution,
        skip_image_resolution: Optional[Tuple[int, int]],
        network_multiplier: float,
        enable_bucket: bool,
        min_bucket_reso: int,
        max_bucket_reso: int,
        bucket_reso_steps: int,
        bucket_no_upscale: bool,
        bucket_selection_mode: str,
        bucket_custom_resos: Optional[str],
        debug_dataset: bool,
        validation_split: float,
        validation_seed: Optional[int],
        resize_interpolation: Optional[str] = None,
    ) -> None:
        super().__init__(resolution, skip_image_resolution, network_multiplier, debug_dataset, resize_interpolation)

        db_subsets = []
        for subset in subsets:
            assert (
                not subset.random_crop
            ), "random_crop is not supported in ControlNetDataset / random_cropはControlNetDatasetではサポートされていません / ControlNetDataset 不支持 random_crop"
            db_subset = DreamBoothSubset(
                subset.image_dir,
                False,
                None,
                subset.caption_extension,
                subset.cache_info,
                False,
                subset.num_repeats,
                subset.shuffle_caption,
                subset.caption_separator,
                subset.keep_tokens,
                subset.keep_tokens_separator,
                subset.secondary_separator,
                subset.enable_wildcard,
                subset.color_aug,
                subset.flip_aug,
                subset.face_crop_aug_range,
                subset.random_crop,
                subset.caption_dropout_rate,
                subset.caption_dropout_every_n_epochs,
                subset.caption_tag_dropout_rate,
                subset.caption_tag_dropout_targets,
                subset.caption_tag_dropout_target_mode,
                subset.caption_tag_dropout_target_count,
                subset.caption_prefix,
                subset.caption_suffix,
                subset.token_warmup_min,
                subset.token_warmup_step,
                resize_interpolation=subset.resize_interpolation,
            )
            db_subsets.append(db_subset)

        self.dreambooth_dataset_delegate = DreamBoothDataset(
            db_subsets,
            True,
            batch_size,
            resolution,
            skip_image_resolution,
            network_multiplier,
            enable_bucket,
            min_bucket_reso,
            max_bucket_reso,
            bucket_reso_steps,
            bucket_no_upscale,
            bucket_selection_mode,
            bucket_custom_resos,
            1.0,
            debug_dataset,
            validation_split,
            validation_seed,
            resize_interpolation,
        )

        # config_util等から参照される値をいれておく（若干微妙なのでなんとかしたい）
        self.image_data = self.dreambooth_dataset_delegate.image_data
        self.batch_size = batch_size
        self.num_train_images = self.dreambooth_dataset_delegate.num_train_images
        self.num_reg_images = self.dreambooth_dataset_delegate.num_reg_images
        self.validation_split = validation_split
        self.validation_seed = validation_seed
        self.resize_interpolation = resize_interpolation

        # assert all conditioning data exists
        missing_imgs = []
        cond_imgs_with_pair = set()
        for image_key, info in self.dreambooth_dataset_delegate.image_data.items():
            db_subset = self.dreambooth_dataset_delegate.image_to_subset[image_key]
            subset = None
            for s in subsets:
                if s.image_dir == db_subset.image_dir:
                    subset = s
                    break
            assert subset is not None, "internal error: subset not found"

            if not os.path.isdir(subset.conditioning_data_dir):
                logger.warning(f"not directory: {subset.conditioning_data_dir}")
                continue

            img_basename = os.path.splitext(os.path.basename(info.absolute_path))[0]
            ctrl_img_path = glob_images(subset.conditioning_data_dir, img_basename)
            if len(ctrl_img_path) < 1:
                missing_imgs.append(img_basename)
                continue
            ctrl_img_path = ctrl_img_path[0]
            ctrl_img_path = os.path.abspath(ctrl_img_path)  # normalize path

            info.cond_img_path = ctrl_img_path
            cond_imgs_with_pair.add(os.path.splitext(ctrl_img_path)[0])  # remove extension because Windows is case insensitive

        extra_imgs = []
        for subset in subsets:
            conditioning_img_paths = glob_images(subset.conditioning_data_dir, "*")
            conditioning_img_paths = [os.path.abspath(p) for p in conditioning_img_paths]  # normalize path
            extra_imgs.extend([p for p in conditioning_img_paths if os.path.splitext(p)[0] not in cond_imgs_with_pair])

        assert (
            len(missing_imgs) == 0
        ), f"missing conditioning data for {len(missing_imgs)} images / 制御用画像が見つかりませんでした: {missing_imgs} / 有 {len(missing_imgs)} 张图像缺少条件图: {missing_imgs}"
        assert (
            len(extra_imgs) == 0
        ), f"extra conditioning data for {len(extra_imgs)} images / 余分な制御用画像があります: {extra_imgs} / 存在 {len(extra_imgs)} 张多余的条件图: {extra_imgs}"

        self.conditioning_image_transforms = IMAGE_TRANSFORMS

    def set_current_strategies(self):
        return self.dreambooth_dataset_delegate.set_current_strategies()

    def make_buckets(self):
        self.dreambooth_dataset_delegate.make_buckets()
        self.image_data = self.dreambooth_dataset_delegate.image_data
        self.num_train_images = self.dreambooth_dataset_delegate.num_train_images
        self.num_reg_images = self.dreambooth_dataset_delegate.num_reg_images
        self.bucket_manager = self.dreambooth_dataset_delegate.bucket_manager
        self.buckets_indices = self.dreambooth_dataset_delegate.buckets_indices

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True):
        return self.dreambooth_dataset_delegate.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process)

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        return self.dreambooth_dataset_delegate.new_cache_latents(model, accelerator)

    def new_cache_text_encoder_outputs(self, models: List[Any], is_main_process: bool):
        return self.dreambooth_dataset_delegate.new_cache_text_encoder_outputs(models, is_main_process)

    def __len__(self):
        return self.dreambooth_dataset_delegate.__len__()

    def __getitem__(self, index):
        example = self.dreambooth_dataset_delegate[index]

        bucket = self.dreambooth_dataset_delegate.bucket_manager.buckets[
            self.dreambooth_dataset_delegate.buckets_indices[index].bucket_index
        ]
        bucket_batch_size = self.dreambooth_dataset_delegate.buckets_indices[index].bucket_batch_size
        image_index = self.dreambooth_dataset_delegate.buckets_indices[index].batch_index * bucket_batch_size

        conditioning_images = []

        for i, image_key in enumerate(bucket[image_index : image_index + bucket_batch_size]):
            image_info = self.dreambooth_dataset_delegate.image_data[image_key]

            target_size_hw = example["target_sizes_hw"][i]
            original_size_hw = example["original_sizes_hw"][i]
            crop_top_left = example["crop_top_lefts"][i]
            flipped = example["flippeds"][i]
            cond_img = load_image(image_info.cond_img_path)

            if self.dreambooth_dataset_delegate.enable_bucket:
                assert (
                    cond_img.shape[0] == original_size_hw[0] and cond_img.shape[1] == original_size_hw[1]
                ), f"size of conditioning image is not match / 画像サイズが合いません: {image_info.absolute_path} / 条件图尺寸不匹配: {image_info.absolute_path}"

                cond_img = resize_image(
                    cond_img,
                    original_size_hw[1],
                    original_size_hw[0],
                    target_size_hw[1],
                    target_size_hw[0],
                    self.resize_interpolation,
                )

                # TODO support random crop
                # 現在サポートしているcropはrandomではなく中央のみ
                h, w = target_size_hw
                ct = (cond_img.shape[0] - h) // 2
                cl = (cond_img.shape[1] - w) // 2
                cond_img = cond_img[ct : ct + h, cl : cl + w]
            else:
                # assert (
                #     cond_img.shape[0] == self.height and cond_img.shape[1] == self.width
                # ), f"image size is small / 画像サイズが小さいようです: {image_info.absolute_path}"
                # resize to target
                if cond_img.shape[0] != target_size_hw[0] or cond_img.shape[1] != target_size_hw[1]:
                    cond_img = resize_image(
                        cond_img,
                        cond_img.shape[0],
                        cond_img.shape[1],
                        target_size_hw[1],
                        target_size_hw[0],
                        self.resize_interpolation,
                    )

            if flipped:
                cond_img = cond_img[:, ::-1, :].copy()  # copy to avoid negative stride

            cond_img = self.conditioning_image_transforms(cond_img)
            conditioning_images.append(cond_img)

        example["conditioning_images"] = torch.stack(conditioning_images).to(memory_format=torch.contiguous_format).float()

        return example


# behave as Dataset mock
class DatasetGroup(torch.utils.data.ConcatDataset):
    def __init__(self, datasets: Sequence[Union[DreamBoothDataset, FineTuningDataset]]):
        self.datasets: List[Union[DreamBoothDataset, FineTuningDataset]]

        super().__init__(datasets)

        self.image_data = {}
        self.num_train_images = 0
        self.num_reg_images = 0

        # simply concat together
        # TODO: handling image_data key duplication among dataset
        #   In practical, this is not the big issue because image_data is accessed from outside of dataset only for debug_dataset.
        for dataset in datasets:
            self.image_data.update(dataset.image_data)
            self.num_train_images += dataset.num_train_images
            self.num_reg_images += dataset.num_reg_images

    def add_replacement(self, str_from, str_to):
        for dataset in self.datasets:
            dataset.add_replacement(str_from, str_to)

    # def make_buckets(self):
    #   for dataset in self.datasets:
    #     dataset.make_buckets()

    def set_text_encoder_output_caching_strategy(self, strategy: TextEncoderOutputsCachingStrategy):
        """
        DataLoader is run in multiple processes, so we need to set the strategy manually.
        """
        for dataset in self.datasets:
            dataset.set_text_encoder_output_caching_strategy(strategy)

    def enable_XTI(self, *args, **kwargs):
        for dataset in self.datasets:
            dataset.enable_XTI(*args, **kwargs)

    def cache_latents(self, vae, vae_batch_size=1, cache_to_disk=False, is_main_process=True, file_suffix=".npz"):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_latents(vae, vae_batch_size, cache_to_disk, is_main_process, file_suffix)

    def new_cache_latents(self, model: Any, accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_latents(model, accelerator)
        accelerator.wait_for_everyone()

    def cache_text_encoder_outputs(
        self, tokenizers, text_encoders, device, weight_dtype, cache_to_disk=False, is_main_process=True
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs(tokenizers, text_encoders, device, weight_dtype, cache_to_disk, is_main_process)

    def cache_text_encoder_outputs_sd3(
        self, tokenizer, text_encoders, device, output_dtype, te_dtypes, cache_to_disk=False, is_main_process=True, batch_size=None
    ):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.cache_text_encoder_outputs_sd3(
                tokenizer, text_encoders, device, output_dtype, te_dtypes, cache_to_disk, is_main_process, batch_size
            )

    def new_cache_text_encoder_outputs(self, models: List[Any], accelerator: Accelerator):
        for i, dataset in enumerate(self.datasets):
            logger.info(f"[Dataset {i}]")
            dataset.new_cache_text_encoder_outputs(models, accelerator)
        accelerator.wait_for_everyone()

    def set_caching_mode(self, caching_mode):
        for dataset in self.datasets:
            dataset.set_caching_mode(caching_mode)

    def verify_bucket_reso_steps(self, min_steps: int):
        for dataset in self.datasets:
            dataset.verify_bucket_reso_steps(min_steps)

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return [(dataset.width, dataset.height) for dataset in self.datasets]

    def is_latent_cacheable(self) -> bool:
        return all([dataset.is_latent_cacheable() for dataset in self.datasets])

    def is_text_encoder_output_cacheable(self, cache_supports_dropout: bool = False) -> bool:
        return all([dataset.is_text_encoder_output_cacheable(cache_supports_dropout) for dataset in self.datasets])

    def set_current_strategies(self):
        for dataset in self.datasets:
            dataset.set_current_strategies()

    def set_current_epoch(self, epoch):
        for dataset in self.datasets:
            dataset.set_current_epoch(epoch)

    def set_current_step(self, step):
        for dataset in self.datasets:
            dataset.set_current_step(step)

    def set_max_train_steps(self, max_train_steps):
        for dataset in self.datasets:
            dataset.set_max_train_steps(max_train_steps)

    def disable_token_padding(self):
        for dataset in self.datasets:
            dataset.disable_token_padding()


def is_disk_cached_latents_is_expected(reso, npz_path: str, flip_aug: bool, alpha_mask: bool):
    expected_latents_size = (reso[1] // 8, reso[0] // 8)  # bucket_resoはWxHなので注意

    if not os.path.exists(npz_path):
        return False

    try:
        with np.load(npz_path, allow_pickle=False) as npz:
            if "latents" not in npz or "original_size" not in npz or "crop_ltrb" not in npz:  # old ver?
                return False
            if npz["latents"].shape[1:3] != expected_latents_size:
                return False

            if flip_aug:
                if "latents_flipped" not in npz:
                    return False
                if npz["latents_flipped"].shape[1:3] != expected_latents_size:
                    return False

            if alpha_mask:
                if "alpha_mask" not in npz:
                    return False
                if (npz["alpha_mask"].shape[1], npz["alpha_mask"].shape[0]) != reso:  # HxW => WxH != reso
                    return False
            else:
                if "alpha_mask" in npz:
                    return False
    except Exception as e:
        logger.error(f"Error loading file: {npz_path}")
        raise e

    return True


# 戻り値は、latents_tensor, (original_size width, original_size height), (crop left, crop top)
# TODO update to use CachingStrategy
# def load_latents_from_disk(
#     npz_path,
# ) -> Tuple[Optional[np.ndarray], Optional[List[int]], Optional[List[int]], Optional[np.ndarray], Optional[np.ndarray]]:
#     npz = np.load(npz_path)
#     if "latents" not in npz:
#         raise ValueError(f"error: npz is old format. please re-generate {npz_path}")

#     latents = npz["latents"]
#     original_size = npz["original_size"].tolist()
#     crop_ltrb = npz["crop_ltrb"].tolist()
#     flipped_latents = npz["latents_flipped"] if "latents_flipped" in npz else None
#     alpha_mask = npz["alpha_mask"] if "alpha_mask" in npz else None
#     return latents, original_size, crop_ltrb, flipped_latents, alpha_mask


# def save_latents_to_disk(npz_path, latents_tensor, original_size, crop_ltrb, flipped_latents_tensor=None, alpha_mask=None):
#     kwargs = {}
#     if flipped_latents_tensor is not None:
#         kwargs["latents_flipped"] = flipped_latents_tensor.float().cpu().numpy()
#     if alpha_mask is not None:
#         kwargs["alpha_mask"] = alpha_mask.float().cpu().numpy()
#     np.savez(
#         npz_path,
#         latents=latents_tensor.float().cpu().numpy(),
#         original_size=np.array(original_size),
#         crop_ltrb=np.array(crop_ltrb),
#         **kwargs,
#     )


def debug_dataset(train_dataset, show_input_ids=False):
    logger.info(f"Total dataset length (steps) / データセットの長さ（ステップ数） / 数据集总长度（步数）: {len(train_dataset)}")
    logger.info(
        "`S` for next step, `E` for next epoch no. , Escape for exit. / Sキーで次のステップ、Eキーで次のエポック、Escキーで中断、終了します / 按 `S` 进入下一步，按 `E` 进入下一轮 epoch，按 Esc 退出。"
    )

    epoch = 1
    while True:
        logger.info(f"")
        logger.info(f"epoch: {epoch}")

        steps = (epoch - 1) * len(train_dataset) + 1
        indices = list(range(len(train_dataset)))
        random.shuffle(indices)

        k = 0
        for i, idx in enumerate(indices):
            train_dataset.set_current_epoch(epoch)
            train_dataset.set_current_step(steps)
            logger.info(f"steps: {steps} ({i + 1}/{len(train_dataset)})")

            example = train_dataset[idx]
            if example["latents"] is not None:
                logger.info(f"sample has latents from npz file: {example['latents'].size()}")
            for j, (ik, cap, lw, orgsz, crptl, trgsz, flpdz) in enumerate(
                zip(
                    example["image_keys"],
                    example["captions"],
                    example["loss_weights"],
                    # example["input_ids"],
                    example["original_sizes_hw"],
                    example["crop_top_lefts"],
                    example["target_sizes_hw"],
                    example["flippeds"],
                )
            ):
                logger.info(
                    f'{ik}, size: {train_dataset.image_data[ik].image_size}, loss weight: {lw}, caption: "{cap}", original size: {orgsz}, crop top left: {crptl}, target size: {trgsz}, flipped: {flpdz}'
                )
                if "network_multipliers" in example:
                    logger.info(f"network multiplier: {example['network_multipliers'][j]}")
                if "custom_attributes" in example:
                    logger.info(f"custom attributes: {example['custom_attributes'][j]}")

                # if show_input_ids:
                #     logger.info(f"input ids: {iid}")
                #     if "input_ids2" in example:
                #         logger.info(f"input ids2: {example['input_ids2'][j]}")
                if example["images"] is not None:
                    im = example["images"][j]
                    logger.info(f"image size: {im.size()}")
                    im = ((im.numpy() + 1.0) * 127.5).astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))  # c,H,W -> H,W,c
                    im = im[:, :, ::-1]  # RGB -> BGR (OpenCV)

                    if "conditioning_images" in example:
                        cond_img = example["conditioning_images"][j]
                        logger.info(f"conditioning image size: {cond_img.size()}")
                        cond_img = ((cond_img.numpy() + 1.0) * 127.5).astype(np.uint8)
                        cond_img = np.transpose(cond_img, (1, 2, 0))
                        cond_img = cond_img[:, :, ::-1]
                        if os.name == "nt":
                            cv2.imshow("cond_img", cond_img)

                    if "alpha_masks" in example and example["alpha_masks"] is not None:
                        alpha_mask = example["alpha_masks"][j]
                        logger.info(f"alpha mask size: {alpha_mask.size()}")
                        alpha_mask = (alpha_mask.numpy() * 255.0).astype(np.uint8)
                        if os.name == "nt":
                            cv2.imshow("alpha_mask", alpha_mask)

                    if os.name == "nt":  # only windows
                        cv2.imshow("img", im)
                        k = cv2.waitKey()
                        cv2.destroyAllWindows()
                    if k == 27 or k == ord("s") or k == ord("e"):
                        break
            steps += 1

            if k == ord("e"):
                break
            if k == 27 or (example["images"] is None and i >= 8):
                k = 27
                break
        if k == 27:
            break

        epoch += 1


def glob_images(directory, base="*"):
    img_paths = []
    for ext in IMAGE_EXTENSIONS:
        if base == "*":
            img_paths.extend(glob.glob(os.path.join(glob.escape(directory), base + ext)))
        else:
            img_paths.extend(glob.glob(glob.escape(os.path.join(directory, base + ext))))
    img_paths = list(set(img_paths))  # 重複を排除
    img_paths.sort()
    return img_paths


def glob_images_pathlib(dir_path, recursive):
    image_paths = []
    if recursive:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.rglob("*" + ext))
    else:
        for ext in IMAGE_EXTENSIONS:
            image_paths += list(dir_path.glob("*" + ext))
    image_paths = list(set(image_paths))  # 重複を排除
    image_paths.sort()
    return image_paths


class MinimalDataset(BaseDataset):
    def __init__(self, resolution, network_multiplier, debug_dataset=False):
        super().__init__(resolution, network_multiplier, debug_dataset)

        self.num_train_images = 0  # update in subclass
        self.num_reg_images = 0  # update in subclass
        self.datasets = [self]
        self.batch_size = 1  # update in subclass

        self.subsets = [self]
        self.num_repeats = 1  # update in subclass if needed
        self.img_count = 1  # update in subclass if needed
        self.bucket_info = {}
        self.is_reg = False
        self.image_dir = "dummy"  # for metadata

    def verify_bucket_reso_steps(self, min_steps: int):
        pass

    def is_latent_cacheable(self) -> bool:
        return False

    def __len__(self):
        raise NotImplementedError

    # override to avoid shuffling buckets
    def set_current_epoch(self, epoch):
        self.current_epoch = epoch

    def __getitem__(self, idx):
        r"""
        The subclass may have image_data for debug_dataset, which is a dict of ImageInfo objects.

        Returns: example like this:

            for i in range(batch_size):
                image_key = ...  # whatever hashable
                image_keys.append(image_key)

                image = ...  # PIL Image
                img_tensor = self.image_transforms(img)
                images.append(img_tensor)

                caption = ...  # str
                input_ids = self.get_input_ids(caption)
                input_ids_list.append(input_ids)

                captions.append(caption)

            images = torch.stack(images, dim=0)
            input_ids_list = torch.stack(input_ids_list, dim=0)
            example = {
                "images": images,
                "input_ids": input_ids_list,
                "captions": captions,   # for debug_dataset
                "latents": None,
                "image_keys": image_keys,   # for debug_dataset
                "loss_weights": torch.ones(batch_size, dtype=torch.float32),
            }
            return example
        """
        raise NotImplementedError

    def get_resolutions(self) -> List[Tuple[int, int]]:
        return []


def load_arbitrary_dataset(args, tokenizer=None) -> MinimalDataset:
    module = ".".join(args.dataset_class.split(".")[:-1])
    dataset_class = args.dataset_class.split(".")[-1]
    module = importlib.import_module(module)
    dataset_class = getattr(module, dataset_class)
    train_dataset_group: MinimalDataset = dataset_class(tokenizer, args.max_token_length, args.resolution, args.debug_dataset)
    return train_dataset_group


def load_image(image_path, alpha=False):
    try:
        with Image.open(image_path) as image:
            if alpha:
                if not image.mode == "RGBA":
                    image = image.convert("RGBA")
            else:
                if not image.mode == "RGB":
                    image = image.convert("RGB")
            img = np.array(image, np.uint8)
            return img
    except (IOError, OSError) as e:
        logger.error(f"Error loading file: {image_path}")
        raise e


# 画像を読み込む。戻り値はnumpy.ndarray,(original width, original height),(crop left, crop top, crop right, crop bottom)
def trim_and_resize_if_required(
    random_crop: bool, image: np.ndarray, reso, resized_size: Tuple[int, int], resize_interpolation: Optional[str] = None
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int, int, int]]:
    image_height, image_width = image.shape[0:2]
    original_size = (image_width, image_height)  # size before resize

    if image_width != resized_size[0] or image_height != resized_size[1]:
        image = resize_image(image, image_width, image_height, resized_size[0], resized_size[1], resize_interpolation)

    image_height, image_width = image.shape[0:2]
    crop_left = 0
    crop_top = 0
    crop_right = image_width
    crop_bottom = image_height

    if image_width > reso[0]:
        trim_size = image_width - reso[0]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"w {trim_size} {p}")
        crop_left = int(p)
        crop_right = int(p + reso[0])
        image = image[:, p : p + reso[0]]
    if image_height > reso[1]:
        trim_size = image_height - reso[1]
        p = trim_size // 2 if not random_crop else random.randint(0, trim_size)
        # logger.info(f"h {trim_size} {p})
        crop_top = int(p)
        crop_bottom = int(p + reso[1])
        image = image[p : p + reso[1]]

    crop_ltrb = (
        int(crop_left),
        int(crop_top),
        int(crop_right),
        int(crop_bottom),
    )

    assert image.shape[0] == reso[1] and image.shape[1] == reso[0], f"internal error, illegal trimmed size: {image.shape}, {reso}"
    return image, original_size, crop_ltrb


def _resolve_image_for_caching(info: "ImageInfo", use_alpha_mask: bool) -> np.ndarray:
    if info.image is None:
        return load_image(info.absolute_path, use_alpha_mask)
    if isinstance(info.image, np.ndarray):
        return info.image if info.image.dtype == np.uint8 else info.image.astype(np.uint8, copy=False)
    return np.asarray(info.image, dtype=np.uint8)


def _image_to_tensor_for_caching(image: np.ndarray) -> torch.Tensor:
    image = np.ascontiguousarray(image[:, :, :3])
    tensor = torch.from_numpy(image).permute(2, 0, 1).to(dtype=torch.float32)
    tensor.div_(127.5).sub_(1.0)
    return tensor


# for new_cache_latents
def load_images_and_masks_for_caching(
    image_infos: List[ImageInfo], use_alpha_mask: bool, random_crop: bool
) -> Tuple[torch.Tensor, List[np.ndarray], List[Tuple[int, int]], List[Tuple[int, int, int, int]]]:
    r"""
    requires image_infos to have: [absolute_path or image], bucket_reso, resized_size

    returns: image_tensor, alpha_masks, original_sizes, crop_ltrbs

    image_tensor: torch.Tensor = torch.Size([B, 3, H, W]), ...], normalized to [-1, 1]
    alpha_masks: List[np.ndarray] = [np.ndarray([H, W]), ...], normalized to [0, 1]
    original_sizes: List[Tuple[int, int]] = [(W, H), ...]
    crop_ltrbs: List[Tuple[int, int, int, int]] = [(L, T, R, B), ...]
    """
    images: List[torch.Tensor] = []
    alpha_masks: List[np.ndarray] = []
    original_sizes: List[Tuple[int, int]] = []
    crop_ltrbs: List[Tuple[int, int, int, int]] = []
    for info in image_infos:
        image = _resolve_image_for_caching(info, use_alpha_mask)
        # TODO 画像のメタデータが壊れていて、メタデータから割り当てたbucketと実際の画像サイズが一致しない場合があるのでチェック追加要
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation
        )

        original_sizes.append(original_size)
        crop_ltrbs.append(crop_ltrb)

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.from_numpy(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        images.append(_image_to_tensor_for_caching(image))

    img_tensor = torch.stack(images, dim=0)
    return img_tensor, alpha_masks, original_sizes, crop_ltrbs


def cache_batch_latents(
    vae: AutoencoderKL, cache_to_disk: bool, image_infos: List[ImageInfo], flip_aug: bool, use_alpha_mask: bool, random_crop: bool
) -> None:
    r"""
    requires image_infos to have: absolute_path, bucket_reso, resized_size, latents_npz
    optionally requires image_infos to have: image
    if cache_to_disk is True, set info.latents_npz
        flipped latents is also saved if flip_aug is True
    if cache_to_disk is False, set info.latents
        latents_flipped is also set if flip_aug is True
    latents_original_size and latents_crop_ltrb are also set
    """
    images = []
    alpha_masks: List[np.ndarray] = []
    for info in image_infos:
        image = _resolve_image_for_caching(info, use_alpha_mask)
        # TODO 画像のメタデータが壊れていて、メタデータから割り当てたbucketと実際の画像サイズが一致しない場合があるのでチェック追加要
        image, original_size, crop_ltrb = trim_and_resize_if_required(
            random_crop, image, info.bucket_reso, info.resized_size, resize_interpolation=info.resize_interpolation
        )

        info.latents_original_size = original_size
        info.latents_crop_ltrb = crop_ltrb

        if use_alpha_mask:
            if image.shape[2] == 4:
                alpha_mask = image[:, :, 3]  # [H,W]
                alpha_mask = alpha_mask.astype(np.float32) / 255.0
                alpha_mask = torch.from_numpy(alpha_mask)  # [H,W]
            else:
                alpha_mask = torch.ones_like(image[:, :, 0], dtype=torch.float32)  # [H,W]
        else:
            alpha_mask = None
        alpha_masks.append(alpha_mask)

        images.append(_image_to_tensor_for_caching(image))

    img_tensors = torch.stack(images, dim=0)
    img_tensors = img_tensors.to(device=vae.device, dtype=vae.dtype)

    with torch.no_grad():
        latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")

    if flip_aug:
        img_tensors = torch.flip(img_tensors, dims=[3])
        with torch.no_grad():
            flipped_latents = vae.encode(img_tensors).latent_dist.sample().to("cpu")
    else:
        flipped_latents = [None] * len(latents)

    for info, latent, flipped_latent, alpha_mask in zip(image_infos, latents, flipped_latents, alpha_masks):
        # check NaN
        if torch.isnan(latents).any() or (flipped_latent is not None and torch.isnan(flipped_latent).any()):
            raise RuntimeError(f"NaN detected in latents: {info.absolute_path}")

        if cache_to_disk:
            # save_latents_to_disk(
            #     info.latents_npz,
            #     latent,
            #     info.latents_original_size,
            #     info.latents_crop_ltrb,
            #     flipped_latent,
            #     alpha_mask,
            # )
            pass
        else:
            info.latents = latent
            if flip_aug:
                info.latents_flipped = flipped_latent
            info.alpha_mask = alpha_mask

    if not HIGH_VRAM:
        clean_memory_on_device(vae.device)


def cache_batch_text_encoder_outputs(
    image_infos, tokenizers, text_encoders, max_token_length, cache_to_disk, input_ids1, input_ids2, dtype
):
    from library.train_prepare_util import get_hidden_states_sdxl

    input_ids1 = input_ids1.to(text_encoders[0].device)
    input_ids2 = input_ids2.to(text_encoders[1].device)

    with torch.no_grad():
        b_hidden_state1, b_hidden_state2, b_pool2 = get_hidden_states_sdxl(
            max_token_length,
            input_ids1,
            input_ids2,
            tokenizers[0],
            tokenizers[1],
            text_encoders[0],
            text_encoders[1],
            dtype,
        )

        # ここでcpuに移動しておかないと、上書きされてしまう
        b_hidden_state1 = b_hidden_state1.detach().to("cpu")  # b,n*75+2,768
        b_hidden_state2 = b_hidden_state2.detach().to("cpu")  # b,n*75+2,1280
        b_pool2 = b_pool2.detach().to("cpu")  # b,1280

    for info, hidden_state1, hidden_state2, pool2 in zip(image_infos, b_hidden_state1, b_hidden_state2, b_pool2):
        if cache_to_disk:
            save_text_encoder_outputs_to_disk(info.text_encoder_outputs_npz, hidden_state1, hidden_state2, pool2)
        else:
            info.text_encoder_outputs1 = hidden_state1
            info.text_encoder_outputs2 = hidden_state2
            info.text_encoder_pool2 = pool2


def cache_batch_text_encoder_outputs_sd3(
    image_infos, tokenizer, text_encoders, max_token_length, cache_to_disk, input_ids, output_dtype
):
    # make input_ids for each text encoder
    l_tokens, g_tokens, t5_tokens = input_ids

    clip_l, clip_g, t5xxl = text_encoders
    with torch.no_grad():
        b_lg_out, b_t5_out, b_pool = sd3_utils.get_cond_from_tokens(
            l_tokens, g_tokens, t5_tokens, clip_l, clip_g, t5xxl, "cpu", output_dtype
        )
        b_lg_out = b_lg_out.detach()
        b_t5_out = b_t5_out.detach()
        b_pool = b_pool.detach()

    for info, lg_out, t5_out, pool in zip(image_infos, b_lg_out, b_t5_out, b_pool):
        # debug: NaN check
        if torch.isnan(lg_out).any() or torch.isnan(t5_out).any() or torch.isnan(pool).any():
            raise RuntimeError(f"NaN detected in text encoder outputs: {info.absolute_path}")

        if cache_to_disk:
            save_text_encoder_outputs_to_disk(info.text_encoder_outputs_npz, lg_out, t5_out, pool)
        else:
            info.text_encoder_outputs1 = lg_out
            info.text_encoder_outputs2 = t5_out
            info.text_encoder_pool2 = pool


def save_text_encoder_outputs_to_disk(npz_path, hidden_state1, hidden_state2, pool2):
    np.savez(
        npz_path,
        hidden_state1=hidden_state1.cpu().float().numpy(),
        hidden_state2=hidden_state2.cpu().float().numpy(),
        pool2=pool2.cpu().float().numpy(),
    )


def load_text_encoder_outputs_from_disk(npz_path):
    with np.load(npz_path, allow_pickle=False) as f:
        hidden_state1 = torch.from_numpy(f["hidden_state1"])
        hidden_state2 = torch.from_numpy(f["hidden_state2"]) if "hidden_state2" in f else None
        pool2 = torch.from_numpy(f["pool2"]) if "pool2" in f else None
    return hidden_state1, hidden_state2, pool2


# endregion

__all__ = [
    'set_high_vram',
    'configure_bucket_runtime_policy',
    'get_bucket_runtime_policy',
    'split_train_val',
    'ImageInfo',
    'parse_tag_text_list',
    'normalize_tag_token',
    'parse_bucket_resolution_list',
    'BucketManager',
    'BucketBatchIndex',
    'AugHelper',
    'BaseSubset',
    'DreamBoothSubset',
    'FineTuningSubset',
    'ControlNetSubset',
    'BaseDataset',
    'DreamBoothDataset',
    'FineTuningDataset',
    'ControlNetDataset',
    'DatasetGroup',
    'is_disk_cached_latents_is_expected',
    'debug_dataset',
    'glob_images',
    'glob_images_pathlib',
    'MinimalDataset',
    'load_arbitrary_dataset',
    'load_image',
    'trim_and_resize_if_required',
    'load_images_and_masks_for_caching',
    'cache_batch_latents',
    'cache_batch_text_encoder_outputs',
    'cache_batch_text_encoder_outputs_sd3',
    'save_text_encoder_outputs_to_disk',
    'load_text_encoder_outputs_from_disk',
]
