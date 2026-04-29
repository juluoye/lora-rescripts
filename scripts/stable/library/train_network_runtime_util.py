from __future__ import annotations

import gc
import random
from typing import Any, NamedTuple, Optional

import numpy as np
import torch

import library.train_runtime_support_util as train_runtime_support_util
from library.device_utils import clean_memory_on_device


class ValidationRuntime(NamedTuple):
    validation_steps: int
    validation_timesteps: np.ndarray
    validation_total_steps: int
    original_args_min_timestep: Any
    original_args_max_timestep: Any


def resolve_on_step_start_callback(accelerator, network):
    unwrapped_network = accelerator.unwrap_model(network)
    if hasattr(unwrapped_network, "on_step_start"):
        return unwrapped_network.on_step_start
    return lambda *args, **kwargs: None


def make_checkpoint_handlers(
    args,
    accelerator,
    metadata,
    minimum_metadata,
    save_dtype,
    get_sai_model_spec,
    upload_fn,
    *,
    ema_model=None,
):
    def save_model(ckpt_name, unwrapped_nw, steps, epoch_no, force_sync_upload=False):
        train_runtime_support_util.save_network_checkpoint(
            args,
            accelerator,
            metadata,
            minimum_metadata,
            save_dtype,
            get_sai_model_spec,
            unwrapped_nw,
            ckpt_name,
            steps,
            epoch_no,
            ema_model=ema_model,
            upload_fn=upload_fn,
            force_sync_upload=force_sync_upload,
        )

    def remove_model(old_ckpt_name):
        train_runtime_support_util.remove_checkpoint(args, accelerator, old_ckpt_name)

    return save_model, remove_model


def drop_unused_text_encoders_if_needed(trainer, args, accelerator, logger, text_encoders, text_encoder):
    if trainer.is_text_encoder_not_needed_for_training(args):
        logger.info("text_encoder is not needed for training. deleting to save memory.")
        for t_enc in text_encoders:
            del t_enc
        text_encoders = []
        text_encoder = None
        gc.collect()
        clean_memory_on_device(accelerator.device)

    return text_encoders, text_encoder


def log_runtime_model_state(logger, unet_weight_dtype, unet, text_encoders):
    logger.info(f"unet dtype: {unet_weight_dtype}, device: {unet.device}")
    for i, t_enc in enumerate(text_encoders):
        params_itr = t_enc.parameters()
        params_itr.__next__()
        params_itr.__next__()
        param_3rd = params_itr.__next__()
        logger.info(f"text_encoder [{i}] dtype: {param_3rd.dtype}, device: {t_enc.device}")


def prepare_validation_runtime(args, noise_scheduler, val_dataloader):
    validation_steps = (
        min(args.max_validation_steps, len(val_dataloader)) if args.max_validation_steps is not None else len(val_dataloader)
    )
    num_validation_timesteps = 4
    min_timestep = 0 if args.min_timestep is None else args.min_timestep
    max_timestep = noise_scheduler.config.num_train_timesteps if args.max_timestep is None else args.max_timestep
    validation_timesteps = np.linspace(min_timestep, max_timestep, (num_validation_timesteps + 2), dtype=int)[1:-1]
    validation_total_steps = validation_steps * len(validation_timesteps)

    return ValidationRuntime(
        validation_steps=validation_steps,
        validation_timesteps=validation_timesteps,
        validation_total_steps=validation_total_steps,
        original_args_min_timestep=args.min_timestep,
        original_args_max_timestep=args.max_timestep,
    )


def switch_rng_state(accelerator, seed: int) -> tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]:
    cpu_rng_state = torch.get_rng_state()
    if accelerator.device.type == "cuda":
        gpu_rng_state = torch.cuda.get_rng_state()
    elif accelerator.device.type == "xpu":
        gpu_rng_state = torch.xpu.get_rng_state()
    elif accelerator.device.type == "mps":
        gpu_rng_state = torch.cuda.get_rng_state()
    else:
        gpu_rng_state = None
    python_rng_state = random.getstate()

    torch.manual_seed(seed)
    random.seed(seed)

    return (cpu_rng_state, gpu_rng_state, python_rng_state)


def restore_rng_state(accelerator, rng_states: tuple[torch.ByteTensor, Optional[torch.ByteTensor], tuple]):
    cpu_rng_state, gpu_rng_state, python_rng_state = rng_states
    torch.set_rng_state(cpu_rng_state)
    if gpu_rng_state is not None:
        if accelerator.device.type == "cuda":
            torch.cuda.set_rng_state(gpu_rng_state)
        elif accelerator.device.type == "xpu":
            torch.xpu.set_rng_state(gpu_rng_state)
        elif accelerator.device.type == "mps":
            torch.cuda.set_rng_state(gpu_rng_state)
    random.setstate(python_rng_state)


__all__ = [
    "ValidationRuntime",
    "drop_unused_text_encoders_if_needed",
    "log_runtime_model_state",
    "make_checkpoint_handlers",
    "prepare_validation_runtime",
    "resolve_on_step_start_callback",
    "restore_rng_state",
    "switch_rng_state",
]
