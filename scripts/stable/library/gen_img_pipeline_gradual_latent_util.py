from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


class GradualLatentState:
    def __init__(self, *, enabled, latents, height, width, step_elapsed, current_ratio):
        self.enabled = enabled
        self.latents = latents
        self.height = height
        self.width = width
        self.step_elapsed = step_elapsed
        self.current_ratio = current_ratio


def prepare_gradual_latent_state(*, scheduler, gradual_latent, latents):
    enabled = False
    step_elapsed = 0
    current_ratio = 1.0
    height, width = latents.shape[-2:]

    if gradual_latent:
        if not hasattr(scheduler, "set_gradual_latent_params"):
            logger.warning("gradual_latent is not supported for this scheduler. Ignoring.")
            logger.warning(f"{scheduler.__class__.__name__}")
        else:
            enabled = True
            step_elapsed = 1000
            current_ratio = gradual_latent.ratio

            org_dtype = latents.dtype
            if org_dtype == torch.bfloat16:
                latents = latents.float()
            latents = torch.nn.functional.interpolate(
                latents, scale_factor=current_ratio, mode="bicubic", align_corners=False
            ).to(org_dtype)

            if gradual_latent.gaussian_blur_ksize:
                latents = gradual_latent.apply_unshark_mask(latents)

    return GradualLatentState(
        enabled=enabled,
        latents=latents,
        height=height,
        width=width,
        step_elapsed=step_elapsed,
        current_ratio=current_ratio,
    )


def update_gradual_latent_for_step(*, scheduler, gradual_latent, state, timestep):
    resized_size = None
    if state.enabled:
        if (
            timestep < gradual_latent.start_timesteps
            and state.current_ratio < 1.0
            and state.step_elapsed >= gradual_latent.every_n_steps
        ):
            state.current_ratio = min(state.current_ratio + gradual_latent.ratio_step, 1.0)
            h = int(state.height * state.current_ratio) // 8 * 8
            w = int(state.width * state.current_ratio) // 8 * 8
            resized_size = (h, w)
            scheduler.set_gradual_latent_params(resized_size, gradual_latent)
            state.step_elapsed = 0
        else:
            scheduler.set_gradual_latent_params(None, None)
        state.step_elapsed += 1

    return resized_size, state


__all__ = ["GradualLatentState", "prepare_gradual_latent_state", "update_gradual_latent_for_step"]
