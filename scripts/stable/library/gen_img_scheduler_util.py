from __future__ import annotations

from typing import NamedTuple

import diffusers
import torch
from diffusers import (
    DDIMScheduler,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    KDPM2DiscreteScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
)

from library.utils import EulerAncestralDiscreteSchedulerGL


class NoiseManager:
    def __init__(self):
        self.sampler_noises = None
        self.sampler_noise_index = 0

    def reset_sampler_noises(self, noises):
        self.sampler_noise_index = 0
        self.sampler_noises = noises

    def randn(self, shape, device=None, dtype=None, layout=None, generator=None):
        if self.sampler_noises is not None and self.sampler_noise_index < len(self.sampler_noises):
            noise = self.sampler_noises[self.sampler_noise_index]
            if shape != noise.shape:
                noise = None
        else:
            noise = None

        if noise is None:
            noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)

        self.sampler_noise_index += 1
        return noise


class TorchRandReplacer:
    def __init__(self, noise_manager):
        self.noise_manager = noise_manager

    def __getattr__(self, item):
        if item == "randn":
            return self.noise_manager.randn
        if hasattr(torch, item):
            return getattr(torch, item)
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))


class PreparedSchedulerRuntime(NamedTuple):
    scheduler: any
    scheduler_num_noises_per_step: int
    noise_manager: NoiseManager


def prepare_scheduler_runtime(
    args,
    *,
    scheduler_linear_start: float,
    scheduler_linear_end: float,
    scheduler_timesteps: int,
    scheduler_schedule: str,
):
    sched_init_args = {}
    has_steps_offset = True
    has_clip_sample = True
    scheduler_num_noises_per_step = 1

    if args.sampler == "ddim":
        scheduler_cls = DDIMScheduler
        scheduler_module = diffusers.schedulers.scheduling_ddim
    elif args.sampler == "ddpm":
        scheduler_cls = DDPMScheduler
        scheduler_module = diffusers.schedulers.scheduling_ddpm
    elif args.sampler == "pndm":
        scheduler_cls = PNDMScheduler
        scheduler_module = diffusers.schedulers.scheduling_pndm
        has_clip_sample = False
    elif args.sampler == "lms" or args.sampler == "k_lms":
        scheduler_cls = LMSDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_lms_discrete
        has_clip_sample = False
    elif args.sampler == "euler" or args.sampler == "k_euler":
        scheduler_cls = EulerDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_euler_discrete
        has_clip_sample = False
    elif args.sampler == "euler_a" or args.sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteSchedulerGL
        scheduler_module = diffusers.schedulers.scheduling_euler_ancestral_discrete
        has_clip_sample = False
    elif args.sampler == "dpmsolver" or args.sampler == "dpmsolver++":
        scheduler_cls = DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = args.sampler
        scheduler_module = diffusers.schedulers.scheduling_dpmsolver_multistep
        has_clip_sample = False
    elif args.sampler == "dpmsingle":
        scheduler_cls = DPMSolverSinglestepScheduler
        scheduler_module = diffusers.schedulers.scheduling_dpmsolver_singlestep
        has_clip_sample = False
        has_steps_offset = False
    elif args.sampler == "heun":
        scheduler_cls = HeunDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_heun_discrete
        has_clip_sample = False
    elif args.sampler == "dpm_2" or args.sampler == "k_dpm_2":
        scheduler_cls = KDPM2DiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_discrete
        has_clip_sample = False
    elif args.sampler == "dpm_2_a" or args.sampler == "k_dpm_2_a":
        scheduler_cls = KDPM2AncestralDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete
        scheduler_num_noises_per_step = 2
        has_clip_sample = False
    else:
        raise ValueError(f"Unsupported sampler: {args.sampler}")

    if args.v_parameterization:
        sched_init_args["prediction_type"] = "v_prediction"
    if has_steps_offset:
        sched_init_args["steps_offset"] = 1
    if has_clip_sample:
        sched_init_args["clip_sample"] = False

    noise_manager = NoiseManager()
    if scheduler_module is not None:
        scheduler_module.torch = TorchRandReplacer(noise_manager)

    if args.zero_terminal_snr:
        sched_init_args["rescale_betas_zero_snr"] = True

    scheduler = scheduler_cls(
        num_train_timesteps=scheduler_timesteps,
        beta_start=scheduler_linear_start,
        beta_end=scheduler_linear_end,
        beta_schedule=scheduler_schedule,
        **sched_init_args,
    )

    return PreparedSchedulerRuntime(
        scheduler=scheduler,
        scheduler_num_noises_per_step=scheduler_num_noises_per_step,
        noise_manager=noise_manager,
    )


__all__ = [
    "NoiseManager",
    "PreparedSchedulerRuntime",
    "TorchRandReplacer",
    "prepare_scheduler_runtime",
]
