from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from library import custom_train_functions


def _sample_uniform_timesteps(min_timestep: int, max_timestep: int, b_size: int, device: torch.device) -> torch.Tensor:
    if min_timestep < max_timestep:
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
    else:
        timesteps = torch.full((b_size,), max_timestep, device="cpu")
    timesteps = timesteps.long().to(device)
    return timesteps


def _sample_sigmoid_density(b_size: int, scale: float) -> torch.Tensor:
    if scale == 0:
        return torch.full((b_size,), 0.5, device="cpu")
    return torch.sigmoid(torch.randn((b_size,), device="cpu") * scale)


def _apply_shift_density(u: torch.Tensor, shift: float) -> torch.Tensor:
    if shift <= 0:
        raise ValueError("timestep_shift must be greater than 0")
    if shift == 1.0:
        return u
    return (u * shift) / (1 + (shift - 1) * u)


def _resolve_timestep_window(args, noise_scheduler) -> tuple[int, int]:
    min_timestep = 0 if args.min_timestep is None else int(args.min_timestep)
    max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else int(args.max_timestep)
    if max_timestep <= min_timestep:
        max_timestep = min_timestep + 1
    return min_timestep, max_timestep


def _normalize_timestep_positions(timesteps: torch.Tensor, min_timestep: int, max_timestep: int) -> torch.Tensor:
    max_index = max_timestep - 1
    if max_index <= min_timestep:
        return torch.zeros_like(timesteps, dtype=torch.float32)

    normalized = (timesteps.to(torch.float32) - float(min_timestep)) / float(max_index - min_timestep)
    return normalized.clamp(0.0, 1.0)


def _compute_timestep_loss_weight_curve(
    normalized_timesteps: torch.Tensor,
    mode: str,
    sigmoid_scale: float,
    shift: float,
) -> torch.Tensor:
    if mode in {"none", "uniform"}:
        return torch.ones_like(normalized_timesteps, dtype=torch.float32)

    if mode == "linear":
        curve = 1.0 - normalized_timesteps
    elif mode == "cosine":
        curve = 0.5 + 0.5 * torch.cos(normalized_timesteps * math.pi)
    else:
        if sigmoid_scale == 0:
            noise_side = torch.full_like(normalized_timesteps, 0.5, dtype=torch.float32)
        else:
            noise_side = torch.sigmoid((normalized_timesteps - 0.5) * float(sigmoid_scale))

        if mode == "shift":
            noise_side = _apply_shift_density(noise_side, float(shift))
        elif mode != "sigmoid":
            raise ValueError(f"Unsupported timestep loss weighting method: {mode}")

        curve = 1.0 - noise_side

    return curve.clamp(min=1e-3)


def get_timestep_loss_weights(loss: torch.Tensor, timesteps: torch.IntTensor, args, noise_scheduler) -> torch.Tensor:
    mode = str(getattr(args, "timestep_loss_weighting", "none") or "none").strip().lower()
    if mode in {"none", "uniform"}:
        return torch.ones_like(loss, dtype=torch.float32, device=loss.device)

    min_timestep, max_timestep = _resolve_timestep_window(args, noise_scheduler)
    normalized_timesteps = _normalize_timestep_positions(timesteps, min_timestep, max_timestep).to(loss.device)

    weights = _compute_timestep_loss_weight_curve(
        normalized_timesteps,
        mode,
        float(getattr(args, "timestep_loss_weight_sigmoid_scale", 1.0) or 0.0),
        float(getattr(args, "timestep_loss_weight_shift", 1.0) or 1.0),
    ).to(loss.device)

    reference_timesteps = torch.arange(min_timestep, max_timestep, device=loss.device, dtype=torch.float32)
    reference_normalized = _normalize_timestep_positions(reference_timesteps, min_timestep, max_timestep)
    reference_weights = _compute_timestep_loss_weight_curve(
        reference_normalized,
        mode,
        float(getattr(args, "timestep_loss_weight_sigmoid_scale", 1.0) or 0.0),
        float(getattr(args, "timestep_loss_weight_shift", 1.0) or 1.0),
    ).to(loss.device)
    normalization = reference_weights.mean().clamp(min=1e-6)
    return weights / normalization


def apply_timestep_loss_weighting(loss: torch.Tensor, timesteps: torch.IntTensor, args, noise_scheduler) -> torch.Tensor:
    weights = get_timestep_loss_weights(loss, timesteps, args, noise_scheduler)
    return loss * weights


def get_timesteps(
    min_timestep: int,
    max_timestep: int,
    b_size: int,
    device: torch.device,
    timestep_sampling: str = "uniform",
    timestep_sigmoid_scale: float = 1.0,
    timestep_shift: float = 1.0,
) -> torch.Tensor:
    if min_timestep >= max_timestep:
        return torch.full((b_size,), max_timestep, device="cpu").long().to(device)

    if timestep_sampling == "uniform":
        return _sample_uniform_timesteps(min_timestep, max_timestep, b_size, device)

    u = _sample_sigmoid_density(b_size, timestep_sigmoid_scale)
    if timestep_sampling == "shift":
        u = _apply_shift_density(u, timestep_shift)
    elif timestep_sampling != "sigmoid":
        raise ValueError(f"Unsupported timestep sampling method: {timestep_sampling}")

    t_range = max_timestep - min_timestep
    timesteps = torch.floor(min_timestep + u * t_range).long()
    timesteps = torch.clamp(timesteps, min=min_timestep, max=max_timestep - 1)
    return timesteps.to(device)


def get_noise_noisy_latents_and_timesteps(
    args, noise_scheduler, latents: torch.FloatTensor
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
    noise = torch.randn_like(latents, device=latents.device)
    if args.noise_offset:
        if args.noise_offset_random_strength:
            noise_offset = torch.rand(1, device=latents.device) * args.noise_offset
        else:
            noise_offset = args.noise_offset
        noise = custom_train_functions.apply_noise_offset(latents, noise, noise_offset, args.adaptive_noise_scale)
    if args.multires_noise_iterations:
        noise = custom_train_functions.pyramid_noise_like(
            noise, latents.device, args.multires_noise_iterations, args.multires_noise_discount
        )

    b_size = latents.shape[0]
    min_timestep = 0 if args.min_timestep is None else args.min_timestep
    max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep
    timesteps = get_timesteps(
        min_timestep,
        max_timestep,
        b_size,
        latents.device,
        getattr(args, "timestep_sampling", "uniform"),
        getattr(args, "timestep_sigmoid_scale", 1.0),
        getattr(args, "timestep_shift", 1.0),
    )

    if args.ip_noise_gamma:
        if args.ip_noise_gamma_random_strength:
            strength = torch.rand(1, device=latents.device) * args.ip_noise_gamma
        else:
            strength = args.ip_noise_gamma
        noisy_latents = noise_scheduler.add_noise(latents, noise + strength * torch.randn_like(latents), timesteps)
    else:
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    noise_scheduler.alphas_cumprod = noise_scheduler.alphas_cumprod.cpu()

    return noise, noisy_latents, timesteps


def get_huber_threshold_if_needed(args, timesteps: torch.Tensor, noise_scheduler) -> Optional[torch.Tensor]:
    if not (args.loss_type == "huber" or args.loss_type == "smooth_l1"):
        return None

    b_size = timesteps.shape[0]
    if args.huber_schedule == "exponential":
        alpha = -math.log(args.huber_c) / noise_scheduler.config.num_train_timesteps
        result = torch.exp(-alpha * timesteps) * args.huber_scale
    elif args.huber_schedule == "snr":
        if not hasattr(noise_scheduler, "alphas_cumprod"):
            raise NotImplementedError("Huber schedule 'snr' is not supported with the current model.")
        alphas_cumprod = torch.index_select(noise_scheduler.alphas_cumprod, 0, timesteps.cpu())
        sigmas = ((1.0 - alphas_cumprod) / alphas_cumprod) ** 0.5
        result = (1 - args.huber_c) / (1 + sigmas) ** 2 + args.huber_c
        result = result.to(timesteps.device)
    elif args.huber_schedule == "constant":
        result = torch.full((b_size,), args.huber_c * args.huber_scale, device=timesteps.device)
    else:
        raise NotImplementedError(f"Unknown Huber loss schedule {args.huber_schedule}!")

    return result


def conditional_loss(
    model_pred: torch.Tensor, target: torch.Tensor, loss_type: str, reduction: str, huber_c: Optional[torch.Tensor] = None
):
    if loss_type == "l2":
        loss = torch.nn.functional.mse_loss(model_pred, target, reduction=reduction)
    elif loss_type == "l1":
        loss = torch.nn.functional.l1_loss(model_pred, target, reduction=reduction)
    elif loss_type == "huber":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = 2 * huber_c * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    elif loss_type == "smooth_l1":
        if huber_c is None:
            raise NotImplementedError("huber_c not implemented correctly")
        huber_c = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
        loss = 2 * (torch.sqrt((model_pred - target) ** 2 + huber_c**2) - huber_c)
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")
    return loss


def append_step_loss_to_logs(logs, *, current_loss=None, average_loss=None):
    if current_loss is not None:
        current_loss = float(current_loss)
        logs.setdefault("loss", current_loss)
        logs["loss/current"] = current_loss

    if average_loss is not None:
        logs["loss/average"] = float(average_loss)


__all__ = [
    "apply_timestep_loss_weighting",
    "append_step_loss_to_logs",
    "conditional_loss",
    "get_huber_threshold_if_needed",
    "get_noise_noisy_latents_and_timesteps",
    "get_timestep_loss_weights",
    "get_timesteps",
]
