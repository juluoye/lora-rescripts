from __future__ import annotations

import time
from typing import NamedTuple

from library.device_utils import clean_memory_on_device
from library import full_bf16_stochastic_util
from lulynx.experimental_core import build_peak_vram_micro_batch_plan, iter_training_micro_batches
from mikazuki.plugins.training_hooks import (
    apply_modify_loss_event,
    emit_after_backward_event,
    emit_after_loss_event,
    emit_after_optimizer_step_event,
    emit_before_forward_event,
    emit_before_optimizer_step_event,
)


class TrainStepExecutionResult(NamedTuple):
    current_loss: float
    skip_training_step: bool
    training_step_wall_seconds: float


def execute_train_step(
    trainer,
    *,
    args,
    batch,
    accelerator,
    training_model,
    on_step_start_for_network,
    text_encoder,
    unet,
    text_encoders,
    network,
    vae,
    noise_scheduler,
    vae_dtype,
    weight_dtype,
    text_encoding_strategy,
    tokenize_strategy,
    train_text_encoder: bool,
    train_unet: bool,
    lulynx_core,
    safeguard,
    optimizer,
    lr_scheduler,
    global_step: int,
    auto_vram_controller,
    attention_step_profiler,
    peak_vram_diagnostics,
    logger,
):
    batch_retry = 0
    skip_training_step = False
    training_step_wall_seconds = 0.0
    current_loss = 0.0
    auto_vram_controller.begin_step(global_step + 1)

    while True:
        attention_step_profiler.begin_micro_step()
        try:
            attempt_started_at = time.perf_counter()
            with accelerator.accumulate(training_model):
                on_step_start_for_network(text_encoder, unet)

                return_per_sample_loss = lulynx_core is not None and lulynx_core.requires_per_sample_losses()
                micro_batch_plan = build_peak_vram_micro_batch_plan(args, batch)
                if return_per_sample_loss and micro_batch_plan.requires_split:
                    if not getattr(args, "_peak_vram_pcgrad_micro_batch_warned", False):
                        logger.warning(
                            "Peak VRAM micro-batch splitting is currently not combined with Lulynx PCGrad on this route. "
                            "Falling back to full-batch backward for the current step."
                        )
                        args._peak_vram_pcgrad_micro_batch_warned = True
                    micro_batch_plan.enabled = False
                    micro_batch_plan.micro_batch_size = micro_batch_plan.actual_batch_size
                    micro_batch_plan.split_count = 1

                peak_vram_diagnostics.start_step(
                    global_step + 1,
                    batch_size=micro_batch_plan.actual_batch_size,
                    micro_batch_size=micro_batch_plan.micro_batch_size,
                    split_count=micro_batch_plan.split_count,
                )
                weighted_loss = 0.0
                skip_training_step = False
                stop_training_reason = None

                micro_batch_count = max(1, int(micro_batch_plan.split_count or 1))
                for micro_batch_index, (micro_batch, sub_batch_size, loss_scale) in enumerate(
                    iter_training_micro_batches(batch, micro_batch_plan),
                    start=1,
                ):
                    setattr(args, "_peak_vram_runtime_global_step", global_step)
                    trainer.on_step_start(args, accelerator, network, text_encoders, unet, micro_batch, weight_dtype, is_train=True)
                    emit_before_forward_event(
                        route="network",
                        training_type=getattr(args, "model_train_type", ""),
                        global_step=global_step,
                        micro_batch_index=micro_batch_index,
                        micro_batch_count=micro_batch_count,
                        micro_batch_size=sub_batch_size,
                        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
                        sync_gradients=bool(accelerator.sync_gradients),
                        extra={
                            "train_text_encoder": bool(train_text_encoder),
                            "train_unet": bool(train_unet),
                            "return_per_sample_loss": bool(return_per_sample_loss),
                            "uses_lulynx_core": lulynx_core is not None,
                        },
                        source="train_network",
                    )

                    with attention_step_profiler.section("forward"):
                        per_sample_losses = None
                        batch_result = trainer.process_batch(
                            micro_batch,
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
                            train_text_encoder=train_text_encoder,
                            train_unet=train_unet,
                            return_per_sample_loss=return_per_sample_loss,
                        )
                        if return_per_sample_loss:
                            micro_loss, per_sample_losses = batch_result
                        else:
                            micro_loss = batch_result
                    peak_vram_diagnostics.capture("forward")

                    raw_micro_loss_value = float(micro_loss.detach().item())
                    emit_after_loss_event(
                        route="network",
                        training_type=getattr(args, "model_train_type", ""),
                        global_step=global_step,
                        micro_batch_index=micro_batch_index,
                        micro_batch_count=micro_batch_count,
                        micro_batch_size=sub_batch_size,
                        loss_value=raw_micro_loss_value,
                        loss_scale=loss_scale,
                        weighted_loss=weighted_loss + (raw_micro_loss_value * loss_scale),
                        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
                        sync_gradients=bool(accelerator.sync_gradients),
                        extra={
                            "return_per_sample_loss": bool(return_per_sample_loss),
                            "uses_lulynx_core": lulynx_core is not None,
                            "modify_loss_runtime_supported": True,
                        },
                        source="train_network",
                    )
                    loss_mutation = apply_modify_loss_event(
                        loss=micro_loss,
                        route="network",
                        training_type=getattr(args, "model_train_type", ""),
                        global_step=global_step,
                        micro_batch_index=micro_batch_index,
                        micro_batch_count=micro_batch_count,
                        micro_batch_size=sub_batch_size,
                        loss_value=raw_micro_loss_value,
                        loss_scale=loss_scale,
                        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
                        sync_gradients=bool(accelerator.sync_gradients),
                        extra={
                            "return_per_sample_loss": bool(return_per_sample_loss),
                            "uses_lulynx_core": lulynx_core is not None,
                        },
                        source="train_network",
                    )
                    micro_loss = loss_mutation.loss
                    micro_loss_value = loss_mutation.final_loss_value
                    weighted_loss += micro_loss_value * loss_scale

                    if safeguard is not None:
                        safeguard_decision = safeguard.inspect_loss(micro_loss_value, global_step + 1, optimizer)
                        if safeguard_decision.reason:
                            logger.warning(safeguard_decision.reason)
                        if safeguard_decision.stop_training:
                            optimizer.zero_grad(set_to_none=True)
                            stop_training_reason = safeguard_decision.reason
                            break
                        if safeguard_decision.skip_step:
                            optimizer.zero_grad(set_to_none=True)
                            attention_step_profiler.discard_current_step()
                            skip_training_step = True
                            break

                    scaled_loss = micro_loss * loss_scale
                    scaled_per_sample_losses = (
                        per_sample_losses * loss_scale * float(loss_mutation.scale) if per_sample_losses is not None else None
                    )
                    with attention_step_profiler.section("backward"):
                        if lulynx_core is not None:
                            lulynx_core.backward(
                                loss=scaled_loss,
                                accelerator=accelerator,
                                optimizer=optimizer,
                                network=accelerator.unwrap_model(network),
                                per_sample_losses=scaled_per_sample_losses,
                            )
                        else:
                            accelerator.backward(scaled_loss)
                    peak_vram_diagnostics.capture("backward")
                    emit_after_backward_event(
                        route="network",
                        training_type=getattr(args, "model_train_type", ""),
                        global_step=global_step,
                        micro_batch_index=micro_batch_index,
                        micro_batch_count=micro_batch_count,
                        micro_batch_size=sub_batch_size,
                        loss_value=micro_loss_value,
                        loss_scale=loss_scale,
                        backward_loss=micro_loss_value * float(loss_scale),
                        weighted_loss=weighted_loss,
                        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
                        sync_gradients=bool(accelerator.sync_gradients),
                        extra={
                            "return_per_sample_loss": bool(return_per_sample_loss),
                            "uses_lulynx_core": lulynx_core is not None,
                            "raw_loss": raw_micro_loss_value,
                            "loss_modified": bool(loss_mutation.modified),
                            "loss_modifier_scale": float(loss_mutation.scale),
                            "loss_modifier_bias": float(loss_mutation.bias),
                            "modify_loss_exclusive_conflict": bool(loss_mutation.dispatch.get("exclusive_conflict")),
                            "modify_loss_error_count": len(loss_mutation.dispatch.get("errors") or []),
                            **({"loss_modifier_reason": loss_mutation.reason} if loss_mutation.reason else {}),
                            **({"loss_modifier_metadata": loss_mutation.metadata} if loss_mutation.metadata else {}),
                        },
                        source="train_network",
                    )

                current_loss = weighted_loss
                if stop_training_reason is not None:
                    raise RuntimeError(stop_training_reason)
                if skip_training_step:
                    training_step_wall_seconds = time.perf_counter() - attempt_started_at
                    break

                if accelerator.sync_gradients:
                    full_bf16_optimizer = full_bf16_stochastic_util.unwrap_full_bf16_optimizer(optimizer)
                    if full_bf16_optimizer is not None:
                        full_bf16_optimizer.sync_model_grads_to_master()
                    trainer.all_reduce_network(accelerator, network)
                    if full_bf16_optimizer is not None:
                        full_bf16_stochastic_util.sync_master_grads_to_model_if_needed(optimizer)
                    if args.max_grad_norm != 0.0:
                        params_to_clip = full_bf16_stochastic_util.get_params_for_grad_clipping(
                            accelerator.unwrap_model(network).get_trainable_params(),
                            optimizer,
                        )
                        accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                        if full_bf16_optimizer is not None:
                            full_bf16_stochastic_util.sync_master_grads_to_model_if_needed(optimizer)

                    if hasattr(network, "update_grad_norms"):
                        network.update_grad_norms()
                    if hasattr(network, "update_norms"):
                        network.update_norms()

                with attention_step_profiler.section("optimizer"):
                    emit_before_optimizer_step_event(
                        route="network",
                        training_type=getattr(args, "model_train_type", ""),
                        global_step=global_step,
                        current_loss=current_loss,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
                        sync_gradients=bool(accelerator.sync_gradients),
                        max_grad_norm=getattr(args, "max_grad_norm", 0.0),
                        extra={
                            "train_text_encoder": bool(train_text_encoder),
                            "train_unet": bool(train_unet),
                            "uses_lulynx_core": lulynx_core is not None,
                        },
                        source="train_network",
                    )
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                peak_vram_diagnostics.capture("optimizer")
                emit_after_optimizer_step_event(
                    route="network",
                    training_type=getattr(args, "model_train_type", ""),
                    global_step=global_step,
                    current_loss=current_loss,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
                    sync_gradients=bool(accelerator.sync_gradients),
                    max_grad_norm=getattr(args, "max_grad_norm", 0.0),
                    optimizer_step_executed=True,
                    scheduler_step_executed=True,
                    zero_grad_called=True,
                    extra={
                        "train_text_encoder": bool(train_text_encoder),
                        "train_unet": bool(train_unet),
                        "uses_lulynx_core": lulynx_core is not None,
                    },
                    source="train_network",
                )
                training_step_wall_seconds = time.perf_counter() - attempt_started_at
                break
        except RuntimeError as exc:
            if auto_vram_controller.maybe_retry_after_oom(exc, retry_count=batch_retry, step=global_step + 1):
                optimizer.zero_grad(set_to_none=True)
                attention_step_profiler.discard_current_step()
                clean_memory_on_device(accelerator.device)
                batch_retry += 1
                continue
            raise
        finally:
            attention_step_profiler.end_micro_step()

    return TrainStepExecutionResult(
        current_loss=current_loss,
        skip_training_step=skip_training_step,
        training_step_wall_seconds=training_step_wall_seconds,
    )


__all__ = [
    "TrainStepExecutionResult",
    "execute_train_step",
]
