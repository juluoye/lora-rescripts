from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

from library import anima_train_utils, flux_train_utils, train_util
from library.device_utils import clean_memory_on_device
from library.qwen_image_autoencoder_kl import SCALE_FACTOR as ANIMA_VAE_SCALE_FACTOR
from library.utils import setup_logging
from library.concept_edit_util import (
    ConceptEditDataset,
    _load_preprocessed_rgb_from_path,
    _parse_boolish,
    add_concept_edit_arguments,
    normalize_concept_edit_mode,
)
from library.train_dataset_util import IMAGE_TRANSFORMS

setup_logging()
import logging

logger = logging.getLogger(__name__)


ANIMA_CONCEPT_EDIT_DATASET_CLASS = "library.anima_concept_edit_util.AnimaConceptEditDataset"
ANIMA_CONCEPT_EDIT_TRAINING_TYPES = {
    "anima-ileco",
    "anima-addift",
    "anima-multi-addift",
}


def infer_anima_concept_edit_mode_from_training_type(training_type: str) -> str:
    normalized = str(training_type or "").strip().lower()
    if normalized.endswith("-ileco"):
        return "ileco"
    if normalized.endswith("-multi-addift"):
        return "multi-addift"
    if normalized.endswith("-addift"):
        return "addift"
    raise ValueError(f"Unsupported Anima concept edit training type: {training_type}")


def apply_anima_concept_edit_runtime_defaults(args, log: logging.Logger) -> None:
    training_type = str(getattr(args, "model_train_type", "") or "").strip().lower()
    args.concept_edit_mode = normalize_concept_edit_mode(getattr(args, "concept_edit_mode", None), training_type)

    if not str(getattr(args, "dataset_class", "") or "").strip():
        args.dataset_class = ANIMA_CONCEPT_EDIT_DATASET_CLASS

    if getattr(args, "max_train_epochs", None) is not None:
        log.warning(
            "Anima concept edit routes currently use step-first scheduling. "
            "Ignoring max_train_epochs and keeping max_train_steps instead."
        )
        args.max_train_epochs = None

    if bool(getattr(args, "cache_latents", False)) or bool(getattr(args, "cache_latents_to_disk", False)):
        log.warning(
            "Anima concept edit routes currently use their own in-memory latent reuse path. "
            "Disabling cache_latents / cache_latents_to_disk for this run."
        )
        args.cache_latents = False
        if hasattr(args, "cache_latents_to_disk"):
            args.cache_latents_to_disk = False

    if bool(getattr(args, "cache_text_encoder_outputs", False)):
        log.warning(
            "Anima concept edit routes currently do not use the generic text-encoder output cache. "
            "Disabling cache_text_encoder_outputs for this run."
        )
        args.cache_text_encoder_outputs = False
        if hasattr(args, "cache_text_encoder_outputs_to_disk"):
            args.cache_text_encoder_outputs_to_disk = False

    args.network_train_unet_only = True
    args.network_train_text_encoder_only = False


class AnimaConceptEditDataset(ConceptEditDataset):
    def __getitem__(self, idx):
        batch = super().__getitem__(idx)
        if "concept_edit_masks" in batch:
            masks = batch["concept_edit_masks"]
            if masks.shape[1] != 16:
                batch["concept_edit_masks"] = masks[:, :1].repeat(1, 16, 1, 1)
        return batch


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
            "The selected network module does not expose set_multiplier, so the Anima concept-edit route "
            "cannot compare base-vs-adapter predictions."
        )
    yield


class AnimaConceptEditTrainerMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._concept_edit_latent_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        self._concept_edit_text_cond_cache: dict[tuple[Any, ...], list[torch.Tensor]] = {}

    def build_metadata(self, args, session_id, training_started_at, optimizer_name, optimizer_args):
        metadata, minimum_metadata = super().build_metadata(args, session_id, training_started_at, optimizer_name, optimizer_args)
        metadata["ss_training_task"] = "concept_edit"
        metadata["ss_concept_edit_mode"] = str(getattr(args, "concept_edit_mode", "") or "")
        metadata["ss_concept_edit_fixed_timestep_per_batch"] = str(
            bool(getattr(args, "concept_edit_fixed_timestep_per_batch", False))
        )
        metadata["ss_concept_edit_diff_alt_ratio"] = str(getattr(args, "concept_edit_diff_alt_ratio", 1.0))
        metadata["ss_concept_edit_use_diff_mask"] = str(bool(getattr(args, "concept_edit_use_diff_mask", False)))
        if getattr(args, "diff_target_name", None):
            metadata["ss_concept_edit_target_suffix"] = str(args.diff_target_name)

        for key in train_util.SS_METADATA_MINIMUM_KEYS:
            if key in metadata:
                minimum_metadata[key] = metadata[key]
        return metadata, minimum_metadata

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
        train_text_encoder=True,
        profiler=None,
        use_non_blocking: bool = False,
        run_nan_check: bool = True,
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
                train_text_encoder=train_text_encoder,
                profiler=profiler,
                use_non_blocking=use_non_blocking,
                run_nan_check=run_nan_check,
                return_per_sample_loss=return_per_sample_loss,
            )

        if train_text_encoder:
            raise ValueError("Anima concept edit currently supports DiT-only training. Please keep network_train_unet_only=true.")

        component_cpu_offload = anima_train_utils.should_use_anima_component_cpu_offload(args)
        moved_text_encoders = False
        if component_cpu_offload and any(getattr(encoder, "device", torch.device("cpu")).type == "cpu" for encoder in text_encoders):
            moved_text_encoders = True
            for encoder in text_encoders:
                anima_train_utils.move_anima_module(
                    encoder,
                    accelerator.device,
                    dtype=weight_dtype,
                    non_blocking=use_non_blocking,
                )

        try:
            if concept_edit_type == "ileco":
                return self._process_anima_ileco_batch(
                    batch=batch,
                    text_encoders=text_encoders,
                    anima=unet,
                    network=network,
                    noise_scheduler=noise_scheduler,
                    weight_dtype=weight_dtype,
                    accelerator=accelerator,
                    args=args,
                    text_encoding_strategy=text_encoding_strategy,
                    tokenize_strategy=tokenize_strategy,
                    profiler=profiler,
                    use_non_blocking=use_non_blocking,
                    run_nan_check=run_nan_check,
                    return_per_sample_loss=return_per_sample_loss,
                )

            return self._process_anima_diff_batch(
                batch=batch,
                text_encoders=text_encoders,
                anima=unet,
                network=network,
                vae=vae,
                noise_scheduler=noise_scheduler,
                vae_dtype=vae_dtype,
                weight_dtype=weight_dtype,
                accelerator=accelerator,
                args=args,
                text_encoding_strategy=text_encoding_strategy,
                tokenize_strategy=tokenize_strategy,
                profiler=profiler,
                use_non_blocking=use_non_blocking,
                run_nan_check=run_nan_check,
                return_per_sample_loss=return_per_sample_loss,
            )
        finally:
            if moved_text_encoders:
                for encoder in text_encoders:
                    anima_train_utils.move_anima_module(encoder, "cpu", dtype=weight_dtype)
                clean_memory_on_device(accelerator.device)

    def _process_anima_ileco_batch(
        self,
        *,
        batch,
        text_encoders,
        anima,
        network,
        noise_scheduler,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        profiler,
        use_non_blocking: bool,
        run_nan_check: bool,
        return_per_sample_loss: bool,
    ):
        batch_size = int(batch["loss_weights"].shape[0])
        height = int(batch["target_sizes_hw"][0, 0].item())
        width = int(batch["target_sizes_hw"][0, 1].item())
        latent_h = max(1, height // ANIMA_VAE_SCALE_FACTOR)
        latent_w = max(1, width // ANIMA_VAE_SCALE_FACTOR)
        latents = torch.randn((batch_size, 16, latent_h, latent_w), device=accelerator.device, dtype=weight_dtype)
        latents = anima_train_utils.maybe_apply_anima_channels_last(args, latents)
        raw_timesteps = self._sample_concept_edit_timesteps(args, noise_scheduler, batch_size, accelerator.device)
        model_timesteps = (raw_timesteps / 1000.0).to(dtype=weight_dtype)

        target_text_conds = self._encode_anima_concept_edit_prompts(
            args,
            accelerator,
            batch["concept_edit_target_captions"],
            text_encoders,
            text_encoding_strategy,
            tokenize_strategy,
            weight_dtype,
            use_non_blocking=use_non_blocking,
        )
        original_text_conds = self._encode_anima_concept_edit_prompts(
            args,
            accelerator,
            batch["concept_edit_original_captions"],
            text_encoders,
            text_encoding_strategy,
            tokenize_strategy,
            weight_dtype,
            use_non_blocking=use_non_blocking,
        )

        target_pred = self._run_anima_concept_edit_dit(
            args=args,
            accelerator=accelerator,
            anima=anima,
            latents=latents,
            timesteps=model_timesteps,
            text_conds=target_text_conds,
            weight_dtype=weight_dtype,
            network=network,
            multiplier=0.0,
            enable_grad=False,
            profiler=profiler,
            run_nan_check=run_nan_check,
        )
        original_pred = self._run_anima_concept_edit_dit(
            args=args,
            accelerator=accelerator,
            anima=anima,
            latents=latents,
            timesteps=model_timesteps,
            text_conds=original_text_conds,
            weight_dtype=weight_dtype,
            network=network,
            multiplier=1.0,
            enable_grad=True,
            profiler=profiler,
            run_nan_check=run_nan_check,
        )
        return self._finalize_anima_concept_edit_loss(
            prediction=original_pred,
            target=target_pred,
            batch=batch,
            args=args,
            timesteps=model_timesteps,
            return_per_sample_loss=return_per_sample_loss,
        )

    def _process_anima_diff_batch(
        self,
        *,
        batch,
        text_encoders,
        anima,
        network,
        vae,
        noise_scheduler,
        vae_dtype,
        weight_dtype,
        accelerator,
        args,
        text_encoding_strategy,
        tokenize_strategy,
        profiler,
        use_non_blocking: bool,
        run_nan_check: bool,
        return_per_sample_loss: bool,
    ):
        batch_size = int(batch["loss_weights"].shape[0])
        original_latents, target_latents = self._get_anima_concept_edit_latents(
            batch=batch,
            accelerator=accelerator,
            vae=vae,
            vae_dtype=vae_dtype,
            weight_dtype=weight_dtype,
            args=args,
            use_non_blocking=use_non_blocking,
            run_nan_check=run_nan_check,
        )

        original_text_conds = self._encode_anima_concept_edit_prompts(
            args,
            accelerator,
            batch["concept_edit_original_captions"],
            text_encoders,
            text_encoding_strategy,
            tokenize_strategy,
            weight_dtype,
            use_non_blocking=use_non_blocking,
        )
        target_text_conds = self._encode_anima_concept_edit_prompts(
            args,
            accelerator,
            batch["concept_edit_target_captions"],
            text_encoders,
            text_encoding_strategy,
            tokenize_strategy,
            weight_dtype,
            use_non_blocking=use_non_blocking,
        )

        raw_timesteps = self._sample_concept_edit_timesteps(args, noise_scheduler, batch_size, accelerator.device)
        model_timesteps = (raw_timesteps / 1000.0).to(dtype=weight_dtype)
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

        baseline_noisy = self._build_anima_noisy_latents(
            args=args,
            noise_scheduler=noise_scheduler,
            latents=baseline_source,
            noise=noise,
            timesteps=raw_timesteps,
            device=accelerator.device,
            dtype=weight_dtype,
        )
        train_noisy = self._build_anima_noisy_latents(
            args=args,
            noise_scheduler=noise_scheduler,
            latents=train_source,
            noise=noise,
            timesteps=raw_timesteps,
            device=accelerator.device,
            dtype=weight_dtype,
        )

        baseline_pred = self._run_anima_concept_edit_dit(
            args=args,
            accelerator=accelerator,
            anima=anima,
            latents=baseline_noisy,
            timesteps=model_timesteps,
            text_conds=original_text_conds,
            weight_dtype=weight_dtype,
            network=network,
            multiplier=0.0,
            enable_grad=False,
            profiler=profiler,
            run_nan_check=run_nan_check,
        )
        train_pred = self._run_anima_concept_edit_dit(
            args=args,
            accelerator=accelerator,
            anima=anima,
            latents=train_noisy,
            timesteps=model_timesteps,
            text_conds=target_text_conds,
            weight_dtype=weight_dtype,
            network=network,
            multiplier=multiplier,
            enable_grad=True,
            profiler=profiler,
            run_nan_check=run_nan_check,
        )

        if _parse_boolish(getattr(args, "concept_edit_use_diff_mask", False)) and batch.get("concept_edit_masks") is not None:
            mask = batch["concept_edit_masks"].to(accelerator.device, dtype=train_pred.dtype)
            baseline_pred = baseline_pred * mask
            train_pred = train_pred * mask

        return self._finalize_anima_concept_edit_loss(
            prediction=train_pred,
            target=baseline_pred,
            batch=batch,
            args=args,
            timesteps=model_timesteps,
            return_per_sample_loss=return_per_sample_loss,
        )

    def _encode_anima_concept_edit_prompts(
        self,
        args,
        accelerator,
        prompts: Iterable[str],
        text_encoders,
        text_encoding_strategy,
        tokenize_strategy,
        weight_dtype,
        *,
        use_non_blocking: bool,
    ) -> list[torch.Tensor]:
        prompt_list = [str(prompt or "") for prompt in prompts]
        cache_key = (
            tuple(prompt_list),
            str(getattr(args, "model_train_type", "") or ""),
            int(getattr(args, "qwen3_max_token_length", 512) or 512),
            int(getattr(args, "t5_max_token_length", 512) or 512),
        )
        cached = self._concept_edit_text_cond_cache.get(cache_key)
        if cached is not None:
            return [
                anima_train_utils.move_anima_tensor(
                    tensor,
                    accelerator.device,
                    dtype=(weight_dtype if tensor.dtype.is_floating_point else None),
                    non_blocking=use_non_blocking,
                )
                if tensor is not None
                else None
                for tensor in cached
            ]

        with torch.no_grad(), accelerator.autocast():
            tokens = tokenize_strategy.tokenize(prompt_list)
            encoded = text_encoding_strategy.encode_tokens(
                tokenize_strategy,
                text_encoders,
                tokens,
            )
            if getattr(args, "full_fp16", False):
                encoded = [tensor.to(weight_dtype) if tensor is not None and tensor.dtype.is_floating_point else tensor for tensor in encoded]

        encoded = [
            anima_train_utils.move_anima_tensor(
                tensor,
                accelerator.device,
                dtype=(weight_dtype if tensor is not None and tensor.dtype.is_floating_point else None),
                non_blocking=use_non_blocking,
            )
            if tensor is not None
            else None
            for tensor in encoded
        ]
        self._concept_edit_text_cond_cache[cache_key] = [
            tensor.detach().to("cpu") if tensor is not None else None
            for tensor in encoded
        ]
        return encoded

    def _get_anima_concept_edit_latents(
        self,
        *,
        batch,
        accelerator,
        vae,
        vae_dtype,
        weight_dtype,
        args,
        use_non_blocking: bool,
        run_nan_check: bool,
    ):
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
                anima_train_utils.move_anima_module(vae, accelerator.device, dtype=vae_dtype, non_blocking=use_non_blocking)
                moved_vae = True

            original_images = torch.stack(
                [
                    IMAGE_TRANSFORMS(
                        _load_preprocessed_rgb_from_path(
                            Path(batch["concept_edit_original_paths"][batch_index]),
                            getattr(args, "resolution", (1024, 1024)),
                            getattr(args, "resize_interpolation", None),
                        )
                    )
                    for batch_index in missing_indices
                ],
                dim=0,
            ).to(accelerator.device, dtype=vae_dtype)
            target_images = torch.stack(
                [
                    IMAGE_TRANSFORMS(
                        _load_preprocessed_rgb_from_path(
                            Path(batch["concept_edit_target_paths"][batch_index]),
                            getattr(args, "resolution", (1024, 1024)),
                            getattr(args, "resize_interpolation", None),
                        )
                    )
                    for batch_index in missing_indices
                ],
                dim=0,
            ).to(accelerator.device, dtype=vae_dtype)
            original_images = anima_train_utils.maybe_apply_anima_channels_last(args, original_images)
            target_images = anima_train_utils.maybe_apply_anima_channels_last(args, target_images)

            with torch.no_grad():
                missing_original_latents = vae.encode_pixels_to_latents(original_images).to(accelerator.device, dtype=weight_dtype)
                missing_target_latents = vae.encode_pixels_to_latents(target_images).to(accelerator.device, dtype=weight_dtype)

            if run_nan_check and torch.any(torch.isnan(missing_original_latents)):
                missing_original_latents = torch.nan_to_num(missing_original_latents, 0, out=missing_original_latents)
            if run_nan_check and torch.any(torch.isnan(missing_target_latents)):
                missing_target_latents = torch.nan_to_num(missing_target_latents, 0, out=missing_target_latents)

            for local_index, batch_index in enumerate(missing_indices):
                pair_key = pair_keys[batch_index]
                original_latent = missing_original_latents[local_index].detach().to("cpu")
                target_latent = missing_target_latents[local_index].detach().to("cpu")
                cache[pair_key] = (original_latent, target_latent)
                original_results[batch_index] = original_latent
                target_results[batch_index] = target_latent

            if moved_vae:
                anima_train_utils.move_anima_module(vae, "cpu", dtype=vae_dtype)
                clean_memory_on_device(accelerator.device)

        stacked_original = torch.stack([tensor.to(accelerator.device, dtype=weight_dtype) for tensor in original_results], dim=0)
        stacked_target = torch.stack([tensor.to(accelerator.device, dtype=weight_dtype) for tensor in target_results], dim=0)
        stacked_original = anima_train_utils.maybe_apply_anima_channels_last(args, stacked_original)
        stacked_target = anima_train_utils.maybe_apply_anima_channels_last(args, stacked_target)
        return stacked_original, stacked_target

    def _sample_concept_edit_timesteps(
        self,
        args,
        noise_scheduler,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        min_timestep = int(getattr(args, "min_timestep", 0) or 0)
        max_timestep = int(getattr(args, "max_timestep", 1000) or 1000)
        min_timestep = min(999, max(0, min_timestep))
        max_timestep = min(1000, max(min_timestep + 1, max_timestep))
        fixed = bool(getattr(args, "concept_edit_fixed_timestep_per_batch", False))
        sample_count = 1 if fixed else batch_size
        timestep_values = torch.randint(min_timestep, max_timestep, (sample_count,), device=device, dtype=torch.long)
        if fixed:
            timestep_values = timestep_values.repeat(batch_size)
        scheduler_timesteps = noise_scheduler.timesteps
        selected_timesteps = scheduler_timesteps.index_select(0, timestep_values.to(device=scheduler_timesteps.device))
        return selected_timesteps.to(device=device, dtype=torch.float32)

    def _build_anima_noisy_latents(
        self,
        *,
        args,
        noise_scheduler,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        device,
        dtype,
    ) -> torch.Tensor:
        sigmas = flux_train_utils.get_sigmas(noise_scheduler, timesteps, device, n_dim=latents.ndim, dtype=dtype)
        sigmas = sigmas.view(-1, 1, 1, 1) if latents.ndim == 4 else sigmas.view(-1, 1, 1, 1, 1)
        if getattr(args, "ip_noise_gamma", 0):
            xi = torch.randn_like(latents, device=latents.device, dtype=dtype)
            if getattr(args, "ip_noise_gamma_random_strength", False):
                ip_noise_gamma = torch.rand(1, device=latents.device, dtype=dtype) * args.ip_noise_gamma
            else:
                ip_noise_gamma = args.ip_noise_gamma
            noisy = (1.0 - sigmas) * latents + sigmas * (noise + ip_noise_gamma * xi)
        else:
            noisy = (1.0 - sigmas) * latents + sigmas * noise
        return noisy.to(dtype)

    def _run_anima_concept_edit_dit(
        self,
        *,
        args,
        accelerator,
        anima,
        latents,
        timesteps,
        text_conds,
        weight_dtype,
        network,
        multiplier: float,
        enable_grad: bool,
        profiler,
        run_nan_check: bool,
    ):
        with _temporary_network_multiplier(network, multiplier):
            checkpointing_temporarily_disabled = False
            try:
                if (
                    not enable_grad
                    and getattr(args, "gradient_checkpointing", False)
                    and bool(getattr(anima, "gradient_checkpointing", False))
                    and hasattr(anima, "disable_gradient_checkpointing")
                ):
                    anima.disable_gradient_checkpointing()
                    checkpointing_temporarily_disabled = True

                if latents.ndim == 5:
                    latents = latents.squeeze(2)
                if run_nan_check and torch.any(torch.isnan(latents)):
                    latents = torch.nan_to_num(latents, 0, out=latents)

                prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = text_conds
                if enable_grad and getattr(args, "gradient_checkpointing", False):
                    latents.requires_grad_(True)
                    for tensor in text_conds:
                        if tensor is not None and tensor.dtype.is_floating_point:
                            tensor.requires_grad_(True)

                batch_size = latents.shape[0]
                latent_h = latents.shape[-2]
                latent_w = latents.shape[-1]
                padding_mask = anima_train_utils.get_cached_anima_padding_mask(
                    batch_size,
                    latent_h,
                    latent_w,
                    device=accelerator.device,
                    dtype=weight_dtype,
                    use_channels_last=bool(getattr(args, "opt_channels_last", False)),
                )
                noisy_model_input = anima_train_utils.maybe_apply_anima_channels_last(args, latents.unsqueeze(2))

                with torch.set_grad_enabled(enable_grad), accelerator.autocast():
                    try:
                        if network is not None and hasattr(network, "set_current_timestep"):
                            network.set_current_timestep(timesteps)
                        model_pred = anima(
                            noisy_model_input,
                            timesteps,
                            prompt_embeds,
                            padding_mask=padding_mask,
                            target_input_ids=t5_input_ids,
                            target_attention_mask=t5_attn_mask,
                            source_attention_mask=attn_mask,
                        )
                    finally:
                        if network is not None and hasattr(network, "clear_current_timestep"):
                            network.clear_current_timestep()
                return model_pred.squeeze(2)
            finally:
                if checkpointing_temporarily_disabled and hasattr(anima, "enable_gradient_checkpointing"):
                    anima.enable_gradient_checkpointing(
                        cpu_offload=bool(getattr(args, "cpu_offload_checkpointing", False)),
                        unsloth_offload=bool(getattr(args, "unsloth_offload_checkpointing", False)),
                    )

    def _finalize_anima_concept_edit_loss(
        self,
        *,
        prediction: torch.Tensor,
        target: torch.Tensor,
        batch,
        args,
        timesteps: torch.Tensor,
        return_per_sample_loss: bool,
    ):
        huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, None)
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
        mean_loss = loss.mean()
        if return_per_sample_loss:
            return mean_loss, loss
        return mean_loss
