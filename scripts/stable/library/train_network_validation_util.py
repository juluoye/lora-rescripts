from __future__ import annotations

from typing import Callable, Optional

from tqdm import tqdm

import library.train_network_runtime_util as train_network_runtime_util


def run_validation_pass(
    trainer,
    *,
    args,
    accelerator,
    network,
    text_encoders,
    unet,
    vae,
    noise_scheduler,
    vae_dtype,
    weight_dtype,
    text_encoding_strategy,
    tokenize_strategy,
    train_text_encoder: bool,
    train_unet: bool,
    optimizer_eval_fn,
    optimizer_train_fn,
    val_dataloader,
    validation_steps: int,
    validation_timesteps,
    validation_total_steps: int,
    validation_loss_recorder,
    training_loss_recorder,
    is_tracking: bool,
    global_step: int,
    recorder_epoch: int,
    epoch_display: int,
    progress_bar,
    progress_desc: str,
    progress_postfix_key: str,
    log_callback: Callable[[dict], None],
    log_prefix: str,
    original_args_min_timestep,
    original_args_max_timestep,
    should_stop_fn: Optional[Callable[[], bool]] = None,
    stop_reason_fn: Optional[Callable[[], str]] = None,
):
    stop_reason = None
    optimizer_eval_fn()
    accelerator.unwrap_model(network).eval()
    rng_states = train_network_runtime_util.switch_rng_state(
        accelerator, args.validation_seed if args.validation_seed is not None else args.seed
    )

    val_progress_bar = tqdm(
        range(validation_total_steps),
        smoothing=0,
        disable=not accelerator.is_local_main_process,
        desc=progress_desc,
    )
    val_timesteps_step = 0
    for val_step, batch in enumerate(val_dataloader):
        if should_stop_fn is not None and should_stop_fn():
            stop_reason = stop_reason_fn() if stop_reason_fn is not None else "Validation interrupted."
            break
        if val_step >= validation_steps:
            break

        for timestep in validation_timesteps:
            if should_stop_fn is not None and should_stop_fn():
                stop_reason = stop_reason_fn() if stop_reason_fn is not None else "Validation interrupted."
                break
            trainer.on_step_start(args, accelerator, network, text_encoders, unet, batch, weight_dtype, is_train=False)

            args.min_timestep = args.max_timestep = timestep

            loss = trainer.process_batch(
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
                is_train=False,
                train_text_encoder=train_text_encoder,
                train_unet=train_unet,
            )

            current_loss = loss.detach().item()
            validation_loss_recorder.add(epoch=recorder_epoch, step=val_timesteps_step, loss=current_loss)
            val_progress_bar.update(1)
            val_progress_bar.set_postfix(
                {progress_postfix_key: validation_loss_recorder.moving_average, "timestep": timestep},
                refresh=False,
            )

            trainer.on_validation_step_end(args, accelerator, network, text_encoders, unet, batch, weight_dtype)
            val_timesteps_step += 1

        if stop_reason is not None:
            break

    if is_tracking:
        average_loss = validation_loss_recorder.moving_average
        divergence = average_loss - training_loss_recorder.moving_average
        logs = {
            f"loss/validation/{log_prefix}_average": average_loss,
            f"loss/validation/{log_prefix}_divergence": divergence,
        }
        log_callback(logs)

    train_network_runtime_util.restore_rng_state(accelerator, rng_states)
    args.min_timestep = original_args_min_timestep
    args.max_timestep = original_args_max_timestep
    optimizer_train_fn()
    accelerator.unwrap_model(network).train()
    progress_bar.unpause()

    return stop_reason


__all__ = ["run_validation_pass"]
