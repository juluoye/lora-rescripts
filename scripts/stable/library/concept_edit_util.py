from __future__ import annotations

import argparse
import inspect
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from library.argument_help_util import build_add
from library.device_utils import clean_memory_on_device
from library import train_network_batch_util, train_util
from library.train_dataset_util import IMAGE_TRANSFORMS, MinimalDataset
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)

CONCEPT_EDIT_DATASET_CLASS = "library.concept_edit_util.ConceptEditDataset"
CONCEPT_EDIT_TRAINING_TYPES = {
    "sd-ileco",
    "sd-addift",
    "sd-multi-addift",
    "sdxl-ileco",
    "sdxl-addift",
    "sdxl-multi-addift",
}

_MODE_ALIASES = {
    "ileco": "ileco",
    "leco": "ileco",
    "addift": "addift",
    "multi-addift": "multi-addift",
    "multi_addift": "multi-addift",
    "multiaddift": "multi-addift",
}


def infer_concept_edit_mode_from_training_type(training_type: str) -> str:
    normalized = str(training_type or "").strip().lower()
    if normalized.endswith("-ileco"):
        return "ileco"
    if normalized.endswith("-multi-addift"):
        return "multi-addift"
    if normalized.endswith("-addift"):
        return "addift"
    raise ValueError(f"Unsupported concept edit training type: {training_type}")


def normalize_concept_edit_mode(mode: Any, training_type: str = "") -> str:
    raw = str(mode or "").strip().lower()
    if raw:
        normalized = _MODE_ALIASES.get(raw)
        if normalized is None:
            raise ValueError(
                "Unsupported concept_edit_mode. Expected one of: ileco, addift, multi-addift "
                f"(received: {mode})"
            )
        return normalized
    return infer_concept_edit_mode_from_training_type(training_type)


def apply_concept_edit_runtime_defaults(args, log: logging.Logger) -> None:
    training_type = str(getattr(args, "model_train_type", "") or "").strip().lower()
    args.concept_edit_mode = normalize_concept_edit_mode(getattr(args, "concept_edit_mode", None), training_type)

    if not str(getattr(args, "dataset_class", "") or "").strip():
        args.dataset_class = CONCEPT_EDIT_DATASET_CLASS

    if getattr(args, "max_train_epochs", None) is not None:
        log.warning(
            "Concept edit routes currently use step-first scheduling. "
            "Ignoring max_train_epochs and keeping max_train_steps instead."
        )
        args.max_train_epochs = None

    if bool(getattr(args, "cache_latents", False)) or bool(getattr(args, "cache_latents_to_disk", False)):
        log.warning(
            "Concept edit routes currently use their own in-memory latent reuse path. "
            "Disabling cache_latents / cache_latents_to_disk for this run."
        )
        args.cache_latents = False
        if hasattr(args, "cache_latents_to_disk"):
            args.cache_latents_to_disk = False

    if bool(getattr(args, "cache_text_encoder_outputs", False)):
        log.warning(
            "Concept edit routes currently do not use the generic text-encoder output cache. "
            "Disabling cache_text_encoder_outputs for this run."
        )
        args.cache_text_encoder_outputs = False
        if hasattr(args, "cache_text_encoder_outputs_to_disk"):
            args.cache_text_encoder_outputs_to_disk = False


def add_concept_edit_arguments(parser: argparse.ArgumentParser) -> None:
    add = build_add(parser)
    add(
        "--concept_edit_mode",
        type=str,
        default=None,
        choices=["ileco", "addift", "multi-addift"],
        help="concept edit route type / 概念编辑训练模式",
    )
    add("--original_prompt", type=str, default="", help="source prompt for concept edit / 原始概念提示词")
    add("--target_prompt", type=str, default="", help="target prompt for concept edit / 目标概念提示词")
    add("--original_image_path", type=str, default=None, help="source image for ADDifT / ADDifT 原始图像")
    add("--target_image_path", type=str, default=None, help="target image for ADDifT / ADDifT 目标图像")
    add("--concept_edit_data_dir", type=str, default=None, help="paired image directory for Multi-ADDifT / Multi-ADDifT 配对图像目录")
    add("--diff_target_name", type=str, default="_target", help="target filename suffix for Multi-ADDifT / Multi-ADDifT 目标图后缀")
    add(
        "--concept_edit_fixed_timestep_per_batch",
        action="store_true",
        help="share one timestep inside the concept-edit batch / 概念编辑 batch 内共享同一个 timestep",
    )
    add(
        "--concept_edit_diff_alt_ratio",
        type=float,
        default=1.0,
        help="alternate ratio for ADDifT reverse phase / ADDifT 交替差分倍率",
    )
    add(
        "--concept_edit_use_diff_mask",
        action="store_true",
        help="apply pixel-difference mask to ADDifT loss / 对 ADDifT 损失应用像素差分 mask",
    )


def _resolve_project_path(raw_path: Any) -> Path:
    resolved = Path(str(raw_path or "").strip()).expanduser()
    if not resolved.is_absolute():
        resolved = (Path.cwd() / resolved).resolve()
    else:
        resolved = resolved.resolve()
    return resolved


def _parse_boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _resize_and_center_crop(image: Image.Image, resolution: tuple[int, int], interpolation: Optional[str]) -> np.ndarray:
    target_w, target_h = int(resolution[0]), int(resolution[1])
    src_w, src_h = image.size
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"Invalid image size: {image.size}")

    resize_method = Image.BICUBIC
    if interpolation:
        interp = interpolation.strip().lower()
        if interp == "nearest":
            resize_method = Image.NEAREST
        elif interp == "bilinear":
            resize_method = Image.BILINEAR
        elif interp == "bicubic":
            resize_method = Image.BICUBIC
        elif interp == "lanczos":
            resize_method = Image.LANCZOS

    scale = max(target_w / float(src_w), target_h / float(src_h))
    resized_w = max(target_w, int(round(src_w * scale)))
    resized_h = max(target_h, int(round(src_h * scale)))
    image = image.resize((resized_w, resized_h), resize_method)

    left = max(0, (resized_w - target_w) // 2)
    top = max(0, (resized_h - target_h) // 2)
    image = image.crop((left, top, left + target_w, top + target_h))
    image = image.convert("RGB")
    return np.array(image, dtype=np.uint8)


def _build_diff_mask(orig_rgb: np.ndarray, targ_rgb: np.ndarray, resolution: tuple[int, int]) -> torch.Tensor:
    diff = np.abs(orig_rgb.astype(np.int16) - targ_rgb.astype(np.int16)).sum(axis=-1)
    mask = (diff > 10).astype(np.float32)
    latent_h = max(1, int(resolution[1]) // 8)
    latent_w = max(1, int(resolution[0]) // 8)
    mask_tensor = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    mask_tensor = F.interpolate(mask_tensor, size=(latent_h, latent_w), mode="nearest")
    return mask_tensor.repeat(1, 4, 1, 1).squeeze(0).contiguous()


def _normalize_resolution_tuple(raw_resolution: Any) -> tuple[int, int]:
    if isinstance(raw_resolution, (tuple, list)) and len(raw_resolution) >= 2:
        return int(raw_resolution[0]), int(raw_resolution[1])

    raw_text = str(raw_resolution or "").strip()
    if "," in raw_text:
        width_text, height_text = raw_text.split(",", 1)
        return int(width_text.strip()), int(height_text.strip())

    value = int(raw_text)
    return value, value


def _load_preprocessed_rgb_from_path(
    image_path: Path,
    resolution: Any,
    interpolation: Optional[str],
) -> np.ndarray:
    normalized_resolution = _normalize_resolution_tuple(resolution)
    with Image.open(image_path) as image:
        return _resize_and_center_crop(image, normalized_resolution, interpolation)


@dataclass(frozen=True)
class ConceptEditEntry:
    key: str
    original_path: str
    target_path: str
    mask: Optional[torch.Tensor]


class ConceptEditDataset(MinimalDataset):
    def __init__(self, _tokenizer, max_token_length, resolution, debug_dataset=False, *, args):
        resize_interpolation = getattr(args, "resize_interpolation", None)
        super().__init__(
            resolution=resolution,
            debug_dataset=debug_dataset,
            resize_interpolation=resize_interpolation,
        )

        self.mode = normalize_concept_edit_mode(getattr(args, "concept_edit_mode", None), getattr(args, "model_train_type", ""))
        self.batch_size = max(1, int(getattr(args, "train_batch_size", 1) or 1))
        self._length = max(1, int(getattr(args, "max_train_steps", 1) or 1))
        self.original_prompt = str(getattr(args, "original_prompt", "") or "").strip()
        self.target_prompt = str(getattr(args, "target_prompt", "") or "").strip()
        self.tokenizer_max_length = int(max_token_length or 75) + 2
        self.tokenizers: list[Any] = []
        self.class_tokens = f"concept_edit:{self.mode}"
        self.keep_tokens = 0
        self.keep_tokens_separator = None
        self.secondary_separator = None
        self.enable_wildcard = False
        self.caption_prefix = None
        self.caption_suffix = None
        self.shuffle_caption = False
        self.color_aug = False
        self.flip_aug = False
        self.random_crop = False
        self.caption_dropout_rate = 0.0
        self.caption_dropout_every_n_epochs = 0
        self.caption_tag_dropout_rate = 0.0
        self.caption_tag_dropout_targets = None
        self.caption_tag_dropout_target_mode = "drop_all"
        self.caption_tag_dropout_target_count = 1
        self.token_warmup_step = 0
        self.enable_bucket = False
        self.bucket_no_upscale = False
        self.min_bucket_reso = min(self.width, self.height)
        self.max_bucket_reso = max(self.width, self.height)
        self.bucket_reso_steps = None
        self.bucket_info = {"concept_edit_mode": self.mode, "resolution": (self.width, self.height)}
        self.is_reg = False
        self.metadata_file = None
        self.tag_frequency = {}
        self.image_data = {}
        self.image_to_subset = {}

        self._resolution_hw = torch.tensor([self.height, self.width], dtype=torch.long)
        self._crop_top_left = torch.tensor([0, 0], dtype=torch.long)

        self.entries = self._build_entries(args)
        actual_image_count = max(1, len(self.entries))
        self.num_train_images = actual_image_count
        self.num_reg_images = 0
        self.img_count = actual_image_count

        if self.mode == "multi-addift":
            self.image_dir = str(_resolve_project_path(getattr(args, "concept_edit_data_dir", "")))
        elif self.mode == "addift":
            self.image_dir = str(_resolve_project_path(getattr(args, "original_image_path", "")).parent)
        else:
            self.image_dir = "concept-edit"

    def set_current_strategies(self):
        super().set_current_strategies()
        tokenize_strategy = self.tokenize_strategy
        if tokenize_strategy is None:
            return
        if hasattr(tokenize_strategy, "tokenizer1") and hasattr(tokenize_strategy, "tokenizer2"):
            self.tokenizers = [tokenize_strategy.tokenizer1, tokenize_strategy.tokenizer2]
        elif hasattr(tokenize_strategy, "tokenizer"):
            self.tokenizers = [tokenize_strategy.tokenizer]
        else:
            self.tokenizers = []
        self.tokenizer_max_length = int(getattr(tokenize_strategy, "max_length", self.tokenizer_max_length))

    def set_max_train_steps(self, max_train_steps):
        super().set_max_train_steps(max_train_steps)
        self._length = max(1, int(max_train_steps or self._length))

    def verify_bucket_reso_steps(self, min_steps: int):
        return

    def is_latent_cacheable(self) -> bool:
        return False

    def is_text_encoder_output_cacheable(self, cache_supports_dropout: bool = False) -> bool:
        return False

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        batch_size = self.batch_size
        loss_weights = torch.ones(batch_size, dtype=torch.float32)
        original_sizes_hw = self._resolution_hw.unsqueeze(0).repeat(batch_size, 1)
        crop_top_lefts = self._crop_top_left.unsqueeze(0).repeat(batch_size, 1)
        target_sizes_hw = self._resolution_hw.unsqueeze(0).repeat(batch_size, 1)

        if self.mode == "ileco":
            return {
                "concept_edit_type": self.mode,
                "concept_edit_original_captions": [self.original_prompt] * batch_size,
                "concept_edit_target_captions": [self.target_prompt] * batch_size,
                "loss_weights": loss_weights,
                "original_sizes_hw": original_sizes_hw,
                "crop_top_lefts": crop_top_lefts,
                "target_sizes_hw": target_sizes_hw,
                "concept_edit_pair_keys": [f"ileco:{idx}:{i}" for i in range(batch_size)],
            }

        entry_count = len(self.entries)
        assert entry_count > 0, "concept edit dataset has no prepared entries"
        selected_entries = [self.entries[(idx * batch_size + i) % entry_count] for i in range(batch_size)]

        batch = {
            "concept_edit_type": self.mode,
            "concept_edit_original_captions": [self.original_prompt] * batch_size,
            "concept_edit_target_captions": [self.target_prompt] * batch_size,
            "concept_edit_original_paths": [entry.original_path for entry in selected_entries],
            "concept_edit_target_paths": [entry.target_path for entry in selected_entries],
            "concept_edit_pair_keys": [entry.key for entry in selected_entries],
            "loss_weights": loss_weights,
            "original_sizes_hw": original_sizes_hw,
            "crop_top_lefts": crop_top_lefts,
            "target_sizes_hw": target_sizes_hw,
        }
        if any(entry.mask is not None for entry in selected_entries):
            mask_template = selected_entries[0].mask
            assert mask_template is not None
            masks = [entry.mask if entry.mask is not None else torch.ones_like(mask_template) for entry in selected_entries]
            batch["concept_edit_masks"] = torch.stack(masks, dim=0)
        return batch

    def _build_entries(self, args) -> list[ConceptEditEntry]:
        if self.mode == "ileco":
            return []
        if self.mode == "addift":
            return [self._build_single_entry(args.original_image_path, args.target_image_path, key="addift:0", use_mask=_parse_boolish(getattr(args, "concept_edit_use_diff_mask", False)))]
        return self._build_multi_entries(args)

    def _build_single_entry(self, original_image_path: str, target_image_path: str, *, key: str, use_mask: bool) -> ConceptEditEntry:
        original_path = _resolve_project_path(original_image_path)
        target_path = _resolve_project_path(target_image_path)
        mask = None
        if use_mask:
            original_rgb = self._load_preprocessed_rgb(original_path)
            target_rgb = self._load_preprocessed_rgb(target_path)
            mask = _build_diff_mask(original_rgb, target_rgb, (self.width, self.height))
        return ConceptEditEntry(
            key=key,
            original_path=str(original_path),
            target_path=str(target_path),
            mask=mask,
        )

    def _build_multi_entries(self, args) -> list[ConceptEditEntry]:
        root = _resolve_project_path(getattr(args, "concept_edit_data_dir", ""))
        suffix = str(getattr(args, "diff_target_name", "_target") or "_target").strip()
        use_mask = _parse_boolish(getattr(args, "concept_edit_use_diff_mask", False))

        path_map: dict[tuple[Path, str, str], Path] = {}
        for file_path in root.rglob("*"):
            if not file_path.is_file():
                continue
            suffix_lower = file_path.suffix.lower()
            if suffix_lower not in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
                continue
            path_map[(file_path.parent, file_path.stem, suffix_lower)] = file_path

        entries: list[ConceptEditEntry] = []
        for (parent, stem, suffix_lower), original_path in sorted(path_map.items(), key=lambda item: str(item[1]).lower()):
            if stem.endswith(suffix):
                continue
            target_path = path_map.get((parent, f"{stem}{suffix}", suffix_lower))
            if target_path is None:
                continue
            entries.append(
                self._build_single_entry(
                    str(original_path),
                    str(target_path),
                    key=f"{parent.as_posix()}::{stem}",
                    use_mask=use_mask,
                )
            )
        return entries

    def _load_preprocessed_rgb(self, image_path: Path) -> np.ndarray:
        return _load_preprocessed_rgb_from_path(
            image_path,
            (self.width, self.height),
            self.resize_interpolation,
        )


@contextmanager
def _temporary_network_multiplier(network, multiplier: float):
    if network is not None and hasattr(network, "set_multiplier"):
        network.set_multiplier(multiplier)
        try:
            yield
        finally:
            network.set_multiplier(1.0)
        return

    if multiplier != 1.0:
        raise ValueError(
            "The selected network module does not expose set_multiplier, so concept-edit routes that need a base-vs-adapter "
            "comparison cannot run on this network module."
        )
    yield


class ConceptEditTrainerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._concept_edit_latent_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._concept_edit_text_cond_cache: dict[tuple[Any, ...], list[torch.Tensor]] = {}

    def update_metadata(self, metadata, args):
        metadata["ss_training_task"] = "concept_edit"
        metadata["ss_concept_edit_mode"] = str(getattr(args, "concept_edit_mode", "") or "")
        metadata["ss_concept_edit_fixed_timestep_per_batch"] = bool(
            getattr(args, "concept_edit_fixed_timestep_per_batch", False)
        )
        metadata["ss_concept_edit_diff_alt_ratio"] = getattr(args, "concept_edit_diff_alt_ratio", 1.0)
        metadata["ss_concept_edit_use_diff_mask"] = bool(getattr(args, "concept_edit_use_diff_mask", False))
        if getattr(args, "diff_target_name", None):
            metadata["ss_concept_edit_target_suffix"] = str(args.diff_target_name)

    def _load_preprocessed_rgb(self, image_path: Path, args) -> np.ndarray:
        return _load_preprocessed_rgb_from_path(
            image_path,
            getattr(args, "resolution", (1024, 1024)),
            getattr(args, "resize_interpolation", None),
        )

    def _is_gradient_checkpointing_active(self, unet) -> bool:
        if unet is None:
            return False
        if hasattr(unet, "is_gradient_checkpointing"):
            try:
                return bool(unet.is_gradient_checkpointing())
            except Exception:
                pass
        return bool(getattr(unet, "gradient_checkpointing", False))

    def process_batch(
        self,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        is_train=True,
        train_text_encoder=True,
        train_unet=True,
        return_per_sample_loss: bool = False,
    ):
        concept_edit_type = str(batch.get("concept_edit_type", "") or "").strip().lower()
        if concept_edit_type not in {"ileco", "addift", "multi-addift"}:
            return super().process_batch(
                batch,
                text_encoders,
                unet,
                network,
                vae,
                noise_scheduler,
                vae_dtype,
                weight_dtype,
                accelerator,
                args,
                text_encoding_strategy,
                tokenize_strategy,
                is_train=is_train,
                train_text_encoder=train_text_encoder,
                train_unet=train_unet,
                return_per_sample_loss=return_per_sample_loss,
            )

        component_cpu_residency = bool(
            getattr(self, "should_use_component_cpu_residency", lambda _args: False)(args)
        )
        moved_text_encoders = False
        if component_cpu_residency and not train_text_encoder and any(t_enc.device.type == "cpu" for t_enc in text_encoders):
            moved_text_encoders = True
            for t_enc in text_encoders:
                t_enc.to(accelerator.device, dtype=weight_dtype)

        try:
            if concept_edit_type == "ileco":
                result = self._process_ileco_batch(
                    batch=batch,
                    text_encoders=text_encoders,
                    unet=unet,
                    network=network,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                    accelerator=accelerator,
                    args=args,
                    text_encoding_strategy=text_encoding_strategy,
                    tokenize_strategy=tokenize_strategy,
                    is_train=is_train,
                    train_text_encoder=train_text_encoder,
                    return_per_sample_loss=return_per_sample_loss,
                )
            else:
                result = self._process_diff_batch(
                    batch=batch,
                    text_encoders=text_encoders,
                    unet=unet,
                    network=network,
                    vae=vae,
                    noise_scheduler=noise_scheduler,
                    vae_dtype=vae_dtype,
                    weight_dtype=weight_dtype,
                    accelerator=accelerator,
                    args=args,
                    text_encoding_strategy=text_encoding_strategy,
                    tokenize_strategy=tokenize_strategy,
                    is_train=is_train,
                    train_text_encoder=train_text_encoder,
                    return_per_sample_loss=return_per_sample_loss,
                )
        finally:
            if moved_text_encoders:
                for t_enc in text_encoders:
                    t_enc.to("cpu", dtype=torch.float32)
                clean_memory_on_device(accelerator.device)
        return result

    def _process_ileco_batch(
        self,
        *,
        batch,
        text_encoders,
        unet,
        network,
        noise_scheduler,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        is_train: bool,
        train_text_encoder: bool,
        return_per_sample_loss: bool,
    ):
        batch_size = int(batch["loss_weights"].shape[0])
        height = int(batch["target_sizes_hw"][0, 0].item())
        width = int(batch["target_sizes_hw"][0, 1].item())
        latent_h = max(1, height // 8)
        latent_w = max(1, width // 8)
        latents = torch.randn((batch_size, 4, latent_h, latent_w), device=accelerator.device, dtype=weight_dtype)
        timesteps = self._sample_concept_edit_timesteps(args, batch_size, accelerator.device)

        target_text_conds = self._encode_concept_edit_prompts(
            args,
            accelerator,
            batch["concept_edit_target_captions"],
            text_encoders,
            text_encoding_strategy,
            tokenize_strategy,
            weight_dtype,
            train_text_encoder=False,
            enable_grad=False,
        )
        original_text_conds = self._encode_concept_edit_prompts(
            args,
            accelerator,
            batch["concept_edit_original_captions"],
            text_encoders,
            text_encoding_strategy,
            tokenize_strategy,
            weight_dtype,
            train_text_encoder=train_text_encoder,
            enable_grad=is_train and train_text_encoder,
        )

        target_pred = self._run_concept_edit_unet(
            args=args,
            accelerator=accelerator,
            unet=unet,
            latents=latents,
            timesteps=timesteps,
            text_conds=target_text_conds,
            batch=batch,
            weight_dtype=weight_dtype,
            network=network,
            multiplier=0.0,
            enable_grad=False,
        )
        original_pred = self._run_concept_edit_unet(
            args=args,
            accelerator=accelerator,
            unet=unet,
            latents=latents,
            timesteps=timesteps,
            text_conds=original_text_conds,
            batch=batch,
            weight_dtype=weight_dtype,
            network=network,
            multiplier=1.0,
            enable_grad=is_train,
        )
        return self._finalize_concept_edit_loss(
            original_pred,
            target_pred,
            batch,
            args,
            timesteps,
            noise_scheduler,
            return_per_sample_loss=return_per_sample_loss,
        )

    def _process_diff_batch(
        self,
        *,
        batch,
        text_encoders,
        unet,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        is_train: bool,
        train_text_encoder: bool,
        return_per_sample_loss: bool,
    ):
        batch_size = int(batch["loss_weights"].shape[0])
        original_latents, target_latents = self._get_concept_edit_latents(
            batch=batch,
            accelerator=accelerator,
            vae=vae,
            vae_dtype=vae_dtype,
            args=args,
        )

        original_text_conds = self._encode_concept_edit_prompts(
            args,
            accelerator,
            batch["concept_edit_original_captions"],
            text_encoders,
            text_encoding_strategy,
            tokenize_strategy,
            weight_dtype,
            train_text_encoder=False,
            enable_grad=False,
        )
        target_text_conds = self._encode_concept_edit_prompts(
            args,
            accelerator,
            batch["concept_edit_target_captions"],
            text_encoders,
            text_encoding_strategy,
            tokenize_strategy,
            weight_dtype,
            train_text_encoder=train_text_encoder,
            enable_grad=is_train and train_text_encoder,
        )

        timesteps = self._sample_concept_edit_timesteps(args, batch_size, accelerator.device)
        noise = torch.randn_like(original_latents)
        alt_ratio = float(getattr(args, "concept_edit_diff_alt_ratio", 1.0) or 1.0)
        global_step = int(getattr(args, "_peak_vram_runtime_global_step", 0) or 0)
        positive_turn = global_step % 2 == 0

        if positive_turn:
            baseline_source = original_latents
            train_source = target_latents
            multiplier = 0.25
        else:
            baseline_source = target_latents
            train_source = original_latents
            multiplier = -0.25 * abs(alt_ratio)
            if alt_ratio < 0:
                baseline_source = noise
                train_source = noise

        baseline_noisy = noise_scheduler.add_noise(baseline_source, noise, timesteps)
        train_noisy = noise_scheduler.add_noise(train_source, noise, timesteps)

        baseline_pred = self._run_concept_edit_unet(
            args=args,
            accelerator=accelerator,
            unet=unet,
            latents=baseline_noisy,
            timesteps=timesteps,
            text_conds=original_text_conds,
            batch=batch,
            weight_dtype=weight_dtype,
            network=network,
            multiplier=0.0,
            enable_grad=False,
        )
        train_pred = self._run_concept_edit_unet(
            args=args,
            accelerator=accelerator,
            unet=unet,
            latents=train_noisy,
            timesteps=timesteps,
            text_conds=target_text_conds,
            batch=batch,
            weight_dtype=weight_dtype,
            network=network,
            multiplier=multiplier,
            enable_grad=is_train,
        )

        if _parse_boolish(getattr(args, "concept_edit_use_diff_mask", False)) and batch.get("concept_edit_masks") is not None:
            mask = batch["concept_edit_masks"].to(accelerator.device, dtype=train_pred.dtype)
            baseline_pred = baseline_pred * mask
            train_pred = train_pred * mask

        return self._finalize_concept_edit_loss(
            train_pred,
            baseline_pred,
            batch,
            args,
            timesteps,
            noise_scheduler,
            return_per_sample_loss=return_per_sample_loss,
        )

    def _encode_concept_edit_prompts(
        self,
        args,
        accelerator,
        prompts: Iterable[str],
        text_encoders,
        text_encoding_strategy,
        tokenize_strategy,
        weight_dtype,
        *,
        train_text_encoder: bool,
        enable_grad: bool,
    ) -> list[torch.Tensor]:
        prompt_list = [str(prompt or "") for prompt in prompts]
        cache_key = None
        if not train_text_encoder:
            cache_key = (
                tuple(prompt_list),
                bool(getattr(args, "weighted_captions", False)),
                str(getattr(args, "model_train_type", "") or ""),
                bool(self.is_sdxl),
                getattr(args, "clip_skip", None),
            )
            cached = self._concept_edit_text_cond_cache.get(cache_key)
            if cached is not None:
                return [tensor.to(accelerator.device) for tensor in cached]

        with torch.set_grad_enabled(enable_grad), accelerator.autocast():
            if getattr(args, "weighted_captions", False):
                input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(prompt_list)
                encoded = text_encoding_strategy.encode_tokens_with_weights(
                    tokenize_strategy,
                    self.get_models_for_text_encoding(args, accelerator, text_encoders),
                    input_ids_list,
                    weights_list,
                )
            else:
                tokens = tokenize_strategy.tokenize(prompt_list)
                encoded = text_encoding_strategy.encode_tokens(
                    tokenize_strategy,
                    self.get_models_for_text_encoding(args, accelerator, text_encoders),
                    tokens,
                )
            if getattr(args, "full_fp16", False):
                encoded = [cond.to(weight_dtype) for cond in encoded]

        if cache_key is not None:
            self._concept_edit_text_cond_cache[cache_key] = [cond.detach().to("cpu") for cond in encoded]
        return encoded

    def _get_concept_edit_latents(self, *, batch, accelerator, vae, vae_dtype, args):
        cache = self._concept_edit_latent_cache
        pair_keys = list(batch["concept_edit_pair_keys"])
        original_results: list[torch.Tensor] = []
        target_results: list[torch.Tensor] = []
        missing_indices: list[int] = []

        for index, pair_key in enumerate(pair_keys):
            cached = cache.get(pair_key)
            if cached is None:
                missing_indices.append(index)
                original_results.append(torch.empty(0))
                target_results.append(torch.empty(0))
            else:
                original_results.append(cached[0])
                target_results.append(cached[1])

        moved_vae = False
        if missing_indices:
            if vae.device.type == "cpu":
                vae.to(accelerator.device, dtype=vae_dtype)
                moved_vae = True

            original_images = torch.stack(
                [
                    IMAGE_TRANSFORMS(
                        self._load_preprocessed_rgb(
                            Path(batch["concept_edit_original_paths"][batch_index]),
                            args,
                        )
                    )
                    for batch_index in missing_indices
                ],
                dim=0,
            ).to(accelerator.device, dtype=vae_dtype)
            target_images = torch.stack(
                [
                    IMAGE_TRANSFORMS(
                        self._load_preprocessed_rgb(
                            Path(batch["concept_edit_target_paths"][batch_index]),
                            args,
                        )
                    )
                    for batch_index in missing_indices
                ],
                dim=0,
            ).to(accelerator.device, dtype=vae_dtype)
            with torch.no_grad():
                missing_original_latents = self.shift_scale_latents(
                    args,
                    self.encode_images_to_latents(args, vae, original_images),
                )
                missing_target_latents = self.shift_scale_latents(
                    args,
                    self.encode_images_to_latents(args, vae, target_images),
                )
            if torch.any(torch.isnan(missing_original_latents)):
                missing_original_latents = torch.nan_to_num(missing_original_latents, 0, out=missing_original_latents)
            if torch.any(torch.isnan(missing_target_latents)):
                missing_target_latents = torch.nan_to_num(missing_target_latents, 0, out=missing_target_latents)

            for local_index, batch_index in enumerate(missing_indices):
                pair_key = pair_keys[batch_index]
                original_latent = missing_original_latents[local_index].detach().to("cpu")
                target_latent = missing_target_latents[local_index].detach().to("cpu")
                cache[pair_key] = (original_latent, target_latent)
                original_results[batch_index] = original_latent
                target_results[batch_index] = target_latent

            if moved_vae:
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

        stacked_original = torch.stack([tensor.to(accelerator.device) for tensor in original_results], dim=0)
        stacked_target = torch.stack([tensor.to(accelerator.device) for tensor in target_results], dim=0)
        return stacked_original, stacked_target

    def _sample_concept_edit_timesteps(self, args, batch_size: int, device: torch.device) -> torch.Tensor:
        min_timestep = int(getattr(args, "min_timestep", 0) or 0)
        max_timestep = int(getattr(args, "max_timestep", 1000) or 1000)
        min_timestep = min(999, max(0, min_timestep))
        max_timestep = min(1000, max(min_timestep + 1, max_timestep))
        fixed = bool(getattr(args, "concept_edit_fixed_timestep_per_batch", False))
        sample_count = 1 if fixed else batch_size
        timesteps = torch.randint(min_timestep, max_timestep, (sample_count,), device=device, dtype=torch.long)
        if fixed:
            timesteps = timesteps.repeat(batch_size)
        return timesteps

    def _run_concept_edit_unet(
        self,
        *,
        args,
        accelerator,
        unet,
        latents,
        timesteps,
        text_conds,
        batch,
        weight_dtype,
        network,
        multiplier: float,
        enable_grad: bool,
    ):
        with _temporary_network_multiplier(network, multiplier):
            checkpointing_temporarily_disabled = False
            try:
                if network is not None and hasattr(network, "set_current_timestep"):
                    network.set_current_timestep(timesteps)
                if (
                    not enable_grad
                    and getattr(args, "gradient_checkpointing", False)
                    and self._is_gradient_checkpointing_active(unet)
                    and hasattr(unet, "disable_gradient_checkpointing")
                ):
                    unet.disable_gradient_checkpointing()
                    checkpointing_temporarily_disabled = True
                if getattr(args, "gradient_checkpointing", False):
                    latents = latents.requires_grad_(True)
                    for cond in text_conds:
                        if isinstance(cond, torch.Tensor):
                            cond.requires_grad_(True)
                with torch.set_grad_enabled(enable_grad), accelerator.autocast():
                    return self.call_unet(
                        args,
                        accelerator,
                        unet,
                        latents,
                        timesteps,
                        text_conds,
                        batch,
                        weight_dtype,
                    )
            finally:
                if checkpointing_temporarily_disabled and hasattr(unet, "enable_gradient_checkpointing"):
                    try:
                        enable_signature = inspect.signature(unet.enable_gradient_checkpointing)
                        if "cpu_offload" in enable_signature.parameters:
                            unet.enable_gradient_checkpointing(
                                cpu_offload=bool(getattr(args, "cpu_offload_checkpointing", False))
                            )
                        else:
                            unet.enable_gradient_checkpointing()
                    except (TypeError, ValueError):
                        unet.enable_gradient_checkpointing()
                if network is not None and hasattr(network, "clear_current_timestep"):
                    network.clear_current_timestep()

    def _finalize_concept_edit_loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        batch,
        args,
        timesteps: torch.Tensor,
        noise_scheduler,
        *,
        return_per_sample_loss: bool,
    ):
        huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
        loss = train_util.conditional_loss(prediction.float(), target.float(), args.loss_type, "none", huber_c)
        loss = train_util.apply_wavelet_loss(
            loss,
            prediction,
            target,
            enabled=bool(getattr(args, "wavelet_loss_enabled", False)),
            weight=float(getattr(args, "wavelet_loss_weight", 0.0) or 0.0),
            levels=max(1, int(getattr(args, "wavelet_loss_levels", 1) or 1)),
            approx_weight=float(getattr(args, "wavelet_loss_approx_weight", 0.0) or 0.0),
        )
        loss = loss.mean(dim=list(range(1, loss.ndim)))
        loss = loss * batch["loss_weights"].to(loss.device)
        loss = train_network_batch_util.post_process_loss(loss, args, timesteps, noise_scheduler)
        mean_loss = loss.mean()
        if return_per_sample_loss:
            return mean_loss, loss
        return mean_loss
