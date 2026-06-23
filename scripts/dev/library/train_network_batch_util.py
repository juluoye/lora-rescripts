from __future__ import annotations

from typing import Optional

import torch

from library.custom_train_functions import (
    add_v_prediction_like_loss,
    apply_debiased_estimation,
    apply_masked_loss,
    apply_snr_weight,
    scale_v_prediction_loss_like_noise_prediction,
)
import library.train_util as train_util


def normalize_conflicting_network_target_flags(args, logger) -> None:
    train_unet_only = bool(getattr(args, "network_train_unet_only", False))
    train_text_encoder_only = bool(getattr(args, "network_train_text_encoder_only", False))
    if not train_unet_only or not train_text_encoder_only:
        return

    args.network_train_unet_only = False
    args.network_train_text_encoder_only = False
    logger.warning(
        "Both network_train_unet_only and network_train_text_encoder_only were enabled. "
        "Automatically switching to train both targets."
    )

    if bool(getattr(args, "cache_text_encoder_outputs", False)):
        args.cache_text_encoder_outputs = False
        if hasattr(args, "cache_text_encoder_outputs_to_disk"):
            args.cache_text_encoder_outputs_to_disk = False
        logger.warning(
            "Disabled cache_text_encoder_outputs automatically because text encoder training is now active."
        )


def get_noise_pred_and_target(
    trainer,
    args,
    accelerator,
    noise_scheduler,
    latents,
    batch,
    text_encoder_conds,
    unet,
    network,
    weight_dtype,
    train_unet,
    is_train=True,
):
    flow_pixel_counts = trainer.get_flow_pixel_counts(args, batch, latents)
    noise, noisy_latents, timesteps = train_util.get_noise_noisy_latents_and_timesteps(
        args, noise_scheduler, latents, pixel_counts=flow_pixel_counts
    )
    noisy_latents = train_util.maybe_apply_channels_last_to_tensor(args, noisy_latents)

    if args.gradient_checkpointing:
        for x in noisy_latents:
            x.requires_grad_(True)
        for t in text_encoder_conds:
            t.requires_grad_(True)

    with torch.set_grad_enabled(is_train), accelerator.autocast():
        try:
            if network is not None and hasattr(network, "set_current_timestep"):
                network.set_current_timestep(timesteps)
            noise_pred = trainer.call_unet(
                args,
                accelerator,
                unet,
                noisy_latents.requires_grad_(train_unet),
                timesteps,
                text_encoder_conds,
                batch,
                weight_dtype,
            )
        finally:
            if network is not None and hasattr(network, "clear_current_timestep"):
                network.clear_current_timestep()

    if bool(getattr(args, "flow_model", False)):
        target = noise - latents
    elif args.v_parameterization:
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        target = noise

    if "custom_attributes" in batch:
        diff_output_pr_indices = []
        for i, custom_attributes in enumerate(batch["custom_attributes"]):
            if "diff_output_preservation" in custom_attributes and custom_attributes["diff_output_preservation"]:
                diff_output_pr_indices.append(i)

        if len(diff_output_pr_indices) > 0:
            network.set_multiplier(0.0)
            try:
                with torch.no_grad(), accelerator.autocast():
                    try:
                        if network is not None and hasattr(network, "set_current_timestep"):
                            network.set_current_timestep(timesteps[diff_output_pr_indices])
                        noise_pred_prior = trainer.call_unet(
                            args,
                            accelerator,
                            unet,
                            noisy_latents,
                            timesteps,
                            text_encoder_conds,
                            batch,
                            weight_dtype,
                            indices=diff_output_pr_indices,
                        )
                    finally:
                        if network is not None and hasattr(network, "clear_current_timestep"):
                            network.clear_current_timestep()
            finally:
                network.set_multiplier(1.0)
            target[diff_output_pr_indices] = noise_pred_prior.to(target.dtype)

    return noise_pred, target, timesteps, None


def apply_contrastive_flow_matching_loss(args, noise_pred, target, loss):
    if not bool(getattr(args, "contrastive_flow_matching", False)):
        return loss
    if not bool(getattr(args, "flow_model", False)):
        return loss
    if noise_pred.shape[0] <= 1:
        return loss

    negative_target = target.roll(1, 0)
    contrastive = torch.nn.functional.mse_loss(noise_pred.float(), negative_target.float(), reduction="none")
    cfm_lambda = float(getattr(args, "cfm_lambda", 0.05) or 0.05)
    return loss - (cfm_lambda * contrastive)


def post_process_loss(loss, args, timesteps: torch.IntTensor, noise_scheduler) -> torch.FloatTensor:
    if bool(getattr(args, "flow_model", False)):
        return loss

    if args.min_snr_gamma:
        loss = apply_snr_weight(loss, timesteps, noise_scheduler, args.min_snr_gamma, args.v_parameterization)
    if args.scale_v_pred_loss_like_noise_pred:
        loss = scale_v_prediction_loss_like_noise_prediction(loss, timesteps, noise_scheduler)
    if args.v_pred_like_loss:
        loss = add_v_prediction_like_loss(loss, timesteps, noise_scheduler, args.v_pred_like_loss)
    if args.debiased_estimation_loss:
        loss = apply_debiased_estimation(loss, timesteps, noise_scheduler, args.v_parameterization)
    if str(getattr(args, "timestep_loss_weighting", "none") or "none").strip().lower() not in {"none", "uniform"}:
        loss = train_util.apply_timestep_loss_weighting(loss, timesteps, args, noise_scheduler)
    return loss


def process_batch(
    trainer,
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
    with torch.no_grad():
        if "latents" in batch and batch["latents"] is not None:
            latents = batch["latents"].to(accelerator.device)
        else:
            if args.vae_batch_size is None or len(batch["images"]) <= args.vae_batch_size:
                latents = trainer.encode_images_to_latents(args, vae, batch["images"].to(accelerator.device, dtype=vae_dtype))
            else:
                chunks = [
                    batch["images"][i : i + args.vae_batch_size] for i in range(0, len(batch["images"]), args.vae_batch_size)
                ]
                list_latents = []
                for chunk in chunks:
                    with torch.no_grad():
                        chunk = trainer.encode_images_to_latents(args, vae, chunk.to(accelerator.device, dtype=vae_dtype))
                        list_latents.append(chunk)
                latents = torch.cat(list_latents, dim=0)

            if torch.any(torch.isnan(latents)):
                accelerator.print("NaN found in latents, replacing with zeros")
                latents = torch.nan_to_num(latents, 0, out=latents)

        latents = trainer.shift_scale_latents(args, latents)

    text_encoder_conds = []
    text_encoder_outputs_list = batch.get("text_encoder_outputs_list", None)
    if text_encoder_outputs_list is not None:
        text_encoder_conds = text_encoder_outputs_list

    if len(text_encoder_conds) == 0 or text_encoder_conds[0] is None or train_text_encoder:
        with torch.set_grad_enabled(is_train and train_text_encoder), accelerator.autocast():
            if args.weighted_captions:
                input_ids_list, weights_list = tokenize_strategy.tokenize_with_weights(batch["captions"])
                encoded_text_encoder_conds = text_encoding_strategy.encode_tokens_with_weights(
                    tokenize_strategy,
                    trainer.get_models_for_text_encoding(args, accelerator, text_encoders),
                    input_ids_list,
                    weights_list,
                )
            else:
                input_ids = [ids.to(accelerator.device) for ids in batch["input_ids_list"]]
                encoded_text_encoder_conds = text_encoding_strategy.encode_tokens(
                    tokenize_strategy,
                    trainer.get_models_for_text_encoding(args, accelerator, text_encoders),
                    input_ids,
                )
            if args.full_fp16:
                encoded_text_encoder_conds = [c.to(weight_dtype) for c in encoded_text_encoder_conds]

        if len(text_encoder_conds) == 0:
            text_encoder_conds = encoded_text_encoder_conds
        else:
            for i in range(len(encoded_text_encoder_conds)):
                if encoded_text_encoder_conds[i] is not None:
                    text_encoder_conds[i] = encoded_text_encoder_conds[i]

    noise_pred, target, timesteps, weighting = get_noise_pred_and_target(
        trainer,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet,
        network,
        weight_dtype,
        train_unet,
        is_train=is_train,
    )

    huber_c = train_util.get_huber_threshold_if_needed(args, timesteps, noise_scheduler)
    loss = train_util.conditional_loss(noise_pred.float(), target.float(), args.loss_type, "none", huber_c)
    loss = apply_contrastive_flow_matching_loss(args, noise_pred, target, loss)
    if weighting is not None:
        loss = loss * weighting
    if args.masked_loss or ("alpha_masks" in batch and batch["alpha_masks"] is not None):
        loss = apply_masked_loss(loss, batch)
    loss = loss.mean(dim=list(range(1, loss.ndim)))

    loss_weights = batch["loss_weights"]
    loss = loss * loss_weights

    loss = post_process_loss(loss, args, timesteps, noise_scheduler)

    mean_loss = loss.mean()
    if return_per_sample_loss:
        return mean_loss, loss
    return mean_loss


__all__ = [
    "apply_contrastive_flow_matching_loss",
    "get_noise_pred_and_target",
    "normalize_conflicting_network_target_flags",
    "post_process_loss",
    "process_batch",
]
