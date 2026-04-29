from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import torch

from library import custom_train_functions
from library.utils import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


def cosine_optimal_transport(
    source: torch.Tensor, target: torch.Tensor, backend: str = "auto"
) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    source_norm = source / torch.clamp(torch.norm(source, dim=1, keepdim=True), min=1e-8)
    target_norm = target / torch.clamp(torch.norm(target, dim=1, keepdim=True), min=1e-8)
    cost = -torch.mm(source_norm, target_norm.t())

    if backend == "cuda":
        return _cuda_assignment(cost)
    if backend == "scipy":
        return _scipy_assignment(cost)

    try:
        return _cuda_assignment(cost)
    except Exception:
        return _scipy_assignment(cost)


def _cuda_assignment(cost: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    from torch_linear_assignment import assignment_to_indices, batch_linear_assignment  # type: ignore

    assignment = batch_linear_assignment(cost.unsqueeze(0))
    row_idx, col_idx = assignment_to_indices(assignment)
    return cost, (row_idx, col_idx)


def _scipy_assignment(cost: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    from scipy.optimize import linear_sum_assignment  # type: ignore

    cost_np = cost.to(torch.float32).detach().cpu().numpy()
    row_ind, col_ind = linear_sum_assignment(cost_np)
    row = torch.from_numpy(row_ind).to(cost.device, torch.long)
    col = torch.from_numpy(col_ind).to(cost.device, torch.long)
    return cost, (row, col)


def get_timesteps(min_timestep: int, max_timestep: int, b_size: int, device: torch.device) -> torch.Tensor:
    if min_timestep < max_timestep:
        timesteps = torch.randint(min_timestep, max_timestep, (b_size,), device="cpu")
    else:
        timesteps = torch.full((b_size,), max_timestep, device="cpu")
    timesteps = timesteps.long().to(device)
    return timesteps


def get_noise_noisy_latents_and_timesteps(
    args,
    noise_scheduler,
    latents: torch.FloatTensor,
    pre_sampled_timesteps: Optional[torch.Tensor] = None,
    pixel_counts: Optional[torch.Tensor] = None,
) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.IntTensor]:
    flow_model_enabled = bool(getattr(args, "flow_model", False))

    noise = torch.randn_like(latents, device=latents.device)
    if flow_model_enabled and bool(getattr(args, "flow_use_ot", False)) and latents.shape[0] > 1:
        try:
            flat_latents = latents.detach().to(torch.float32).flatten(start_dim=1)
            flat_noise = noise.detach().to(torch.float32).flatten(start_dim=1)
            _, (_, col_idx) = cosine_optimal_transport(flat_latents, flat_noise)
            noise = noise[col_idx]
        except Exception as exc:
            logger.warning(
                f"Rectified Flow optimal transport pairing failed, continuing with random pairing. reason={exc}"
            )
            try:
                setattr(args, "flow_use_ot", False)
            except Exception:
                pass

    if not flow_model_enabled:
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
    if flow_model_enabled:
        timestep_count = int(noise_scheduler.config.num_train_timesteps)
        distribution = str(getattr(args, "flow_timestep_distribution", "logit_normal") or "logit_normal").strip().lower()
        if distribution == "uniform":
            sigmas = torch.rand((b_size,), device=latents.device, dtype=torch.float32)
        else:
            flow_logit_mean = float(getattr(args, "flow_logit_mean", 0.0) or 0.0)
            flow_logit_std = float(getattr(args, "flow_logit_std", 1.0) or 1.0)
            flow_logit_std = flow_logit_std if flow_logit_std > 0 else 1.0
            logits = torch.normal(
                mean=flow_logit_mean,
                std=flow_logit_std,
                size=(b_size,),
                device=latents.device,
                dtype=torch.float32,
            )
            sigmas = torch.sigmoid(logits)

        static_shift_ratio = getattr(args, "flow_uniform_static_ratio", None)
        if static_shift_ratio is not None and str(static_shift_ratio).strip() != "":
            try:
                ratio = float(static_shift_ratio)
            except (TypeError, ValueError):
                ratio = 0.0
            if ratio > 0:
                sigmas = (sigmas * ratio) / (1 + (ratio - 1) * sigmas)
        elif bool(getattr(args, "flow_uniform_shift", False)) and pixel_counts is not None:
            try:
                base_pixels = float(getattr(args, "flow_uniform_base_pixels", 1024.0 * 1024.0) or 1024.0 * 1024.0)
            except (TypeError, ValueError):
                base_pixels = 1024.0 * 1024.0
            base_pixels = base_pixels if base_pixels > 0 else 1024.0 * 1024.0
            ratios = torch.sqrt(torch.clamp(pixel_counts.to(latents.device, dtype=torch.float32), min=1.0) / base_pixels)
            sigmas = (sigmas * ratios) / (1 + (ratios - 1) * sigmas)

        sigmas = torch.clamp(sigmas, min=0.0, max=1.0)
        max_timestep_index = max(timestep_count - 1, 0)
        timesteps = torch.clamp((sigmas * timestep_count).long(), min=0, max=max_timestep_index)

        sigma_shape = [b_size] + [1] * (latents.ndim - 1)
        sigma_tensor = sigmas.view(*sigma_shape).to(dtype=latents.dtype)
        noisy_latents = (1.0 - sigma_tensor) * latents + sigma_tensor * noise
    else:
        min_timestep = 0 if args.min_timestep is None else args.min_timestep
        max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep
        if pre_sampled_timesteps is not None:
            timesteps = pre_sampled_timesteps.to(device=latents.device, dtype=torch.long)
        else:
            timesteps = get_timesteps(min_timestep, max_timestep, b_size, latents.device)

        if args.ip_noise_gamma:
            if args.ip_noise_gamma_random_strength:
                strength = torch.rand(1, device=latents.device) * args.ip_noise_gamma
            else:
                strength = args.ip_noise_gamma
            noisy_latents = noise_scheduler.add_noise(latents, noise + strength * torch.randn_like(latents), timesteps)
        else:
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        if hasattr(noise_scheduler, "alphas_cumprod") and noise_scheduler.alphas_cumprod is not None:
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
    "append_step_loss_to_logs",
    "conditional_loss",
    "cosine_optimal_transport",
    "get_huber_threshold_if_needed",
    "get_noise_noisy_latents_and_timesteps",
    "get_timesteps",
]
