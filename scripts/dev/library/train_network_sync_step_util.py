from __future__ import annotations

from typing import NamedTuple, Optional

import library.train_network_checkpoint_util as train_network_checkpoint_util


class NetworkNormMetrics(NamedTuple):
    keys_scaled: Optional[float]
    mean_norm: Optional[float]
    maximum_norm: Optional[float]
    mean_grad_norm: Optional[float]
    mean_combined_norm: Optional[float]
    max_mean_logs: dict


class SyncStepCompletionResult(NamedTuple):
    global_step: int
    peak_vram_logs: dict
    lulynx_step_logs: dict
    lulynx_stop_reason: Optional[str]


def collect_network_norm_metrics(args, accelerator, network) -> NetworkNormMetrics:
    if args.scale_weight_norms:
        keys_scaled, mean_norm, maximum_norm = accelerator.unwrap_model(network).apply_max_norm_regularization(
            args.scale_weight_norms, accelerator.device
        )
        return NetworkNormMetrics(
            keys_scaled=keys_scaled,
            mean_norm=mean_norm,
            maximum_norm=maximum_norm,
            mean_grad_norm=None,
            mean_combined_norm=None,
            max_mean_logs={"Keys Scaled": keys_scaled, "Average key norm": mean_norm},
        )

    if hasattr(network, "weight_norms"):
        weight_norms = network.weight_norms()
        grad_norms = network.grad_norms()
        combined_weight_norms = network.combined_weight_norms()
        return NetworkNormMetrics(
            keys_scaled=None,
            mean_norm=weight_norms.mean().item() if weight_norms is not None else None,
            maximum_norm=weight_norms.max().item() if weight_norms is not None else None,
            mean_grad_norm=grad_norms.mean().item() if grad_norms is not None else None,
            mean_combined_norm=combined_weight_norms.mean().item() if combined_weight_norms is not None else None,
            max_mean_logs={},
        )

    return NetworkNormMetrics(
        keys_scaled=None,
        mean_norm=None,
        maximum_norm=None,
        mean_grad_norm=None,
        mean_combined_norm=None,
        max_mean_logs={},
    )


def handle_sync_step_completion(
    trainer,
    *,
    args,
    accelerator,
    network,
    global_step: int,
    effective_epoch_no: int,
    progress_bar,
    ema_model,
    auto_vram_controller,
    training_step_wall_seconds: float,
    optimizer_eval_fn,
    optimizer_train_fn,
    vae,
    tokenizers,
    text_encoder,
    unet,
    attention_step_profiler,
    peak_vram_diagnostics,
    save_model,
    remove_model,
    loss_recorder,
    current_loss: float,
    optimizer,
    lr_scheduler,
    lulynx_core,
    lulynx_stop_reason,
):
    progress_bar.update(1)
    global_step += 1
    if ema_model is not None:
        ema_model.update(global_step)
    auto_vram_controller.observe_step_success(
        step=global_step,
        step_wall_seconds=training_step_wall_seconds,
    )

    optimizer_eval_fn()
    with attention_step_profiler.section("preview"):
        trainer.sample_images(accelerator, args, None, global_step, accelerator.device, vae, tokenizers, text_encoder, unet)
    progress_bar.unpause()

    if args.save_every_n_steps is not None and global_step % args.save_every_n_steps == 0:
        with attention_step_profiler.section("save"):
            train_network_checkpoint_util.maybe_save_step_checkpoint(
                args, accelerator, network, global_step, max(0, effective_epoch_no - 1), save_model, remove_model
            )
    optimizer_train_fn()
    attention_step_profiler.finalize_optimizer_step(global_step)
    peak_vram_logs, peak_vram_message = peak_vram_diagnostics.finish_step()
    if peak_vram_message and accelerator.is_main_process:
        print(peak_vram_message)

    lulynx_step_logs = {}
    if lulynx_core is not None:
        lulynx_decision = lulynx_core.on_optimizer_step(
            global_step=global_step,
            current_loss=current_loss,
            average_loss=loss_recorder.moving_average,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
            network=accelerator.unwrap_model(network),
        )
        lulynx_step_logs = dict(lulynx_decision.logs or {})
        if lulynx_decision.stop_training and lulynx_stop_reason is None:
            lulynx_stop_reason = lulynx_decision.reason or "Lulynx experimental core requested stop."

    return SyncStepCompletionResult(
        global_step=global_step,
        peak_vram_logs=peak_vram_logs,
        lulynx_step_logs=lulynx_step_logs,
        lulynx_stop_reason=lulynx_stop_reason,
    )


__all__ = [
    "NetworkNormMetrics",
    "SyncStepCompletionResult",
    "collect_network_norm_metrics",
    "handle_sync_step_completion",
]
