from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass
class LoadedTrainingState:
    steps_from_state: Optional[int] = None


@dataclass
class InitialTrainingPlan:
    initial_step: int
    epoch_to_start: int
    progress_start_step: int


def register_network_state_hooks(
    accelerator,
    args,
    network,
    current_epoch,
    current_step,
    logger,
    *,
    save_state_payload_builder: Optional[Callable[[], dict[str, Any]]] = None,
) -> LoadedTrainingState:
    loaded_state = LoadedTrainingState()
    network_type = type(accelerator.unwrap_model(network))

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process or args.deepspeed:
            remove_indices = []
            for index, model in enumerate(models):
                if not isinstance(model, network_type):
                    remove_indices.append(index)
            for index in reversed(remove_indices):
                if len(weights) > index:
                    weights.pop(index)

        train_state_file = os.path.join(output_dir, "train_state.json")
        state_payload = {
            "current_epoch": current_epoch.value,
            "current_step": current_step.value + 1,
        }
        if save_state_payload_builder is not None:
            extra_payload = save_state_payload_builder()
            if extra_payload:
                state_payload.update(extra_payload)

        logging_run_dir = str(getattr(args, "logging_run_dir", "") or "").strip()
        if not logging_run_dir:
            logging_run_dir = str(getattr(accelerator, "project_dir", "") or "").strip()
        if logging_run_dir:
            state_payload["logging_run_dir"] = logging_run_dir
            state_payload["logging_dir"] = logging_run_dir

        logger.info(
            f"save train state to {train_state_file} at epoch {state_payload['current_epoch']} step {state_payload['current_step']}"
        )
        with open(train_state_file, "w", encoding="utf-8") as file:
            json.dump(state_payload, file)

    def load_model_hook(models, input_dir):
        remove_indices = []
        for index, model in enumerate(models):
            if not isinstance(model, network_type):
                remove_indices.append(index)
        for index in reversed(remove_indices):
            models.pop(index)

        train_state_file = os.path.join(input_dir, "train_state.json")
        if os.path.exists(train_state_file):
            with open(train_state_file, "r", encoding="utf-8") as file:
                data = json.load(file)
            loaded_state.steps_from_state = data["current_step"]
            logger.info(f"load train state from {train_state_file}: {data}")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)
    return loaded_state


def resolve_initial_training_plan(
    args,
    logger,
    *,
    train_dataloader_len: int,
    accelerator_num_processes: int,
    steps_from_state: Optional[int],
    track_initial_progress: bool,
) -> InitialTrainingPlan:
    initial_step = 0
    if args.initial_epoch is not None or args.initial_step is not None:
        if steps_from_state is not None:
            logger.warning(
                "steps from the state is ignored because initial_step is specified / initial_stepが指定されているため、stateからのステップ数は無視されます"
            )
        if args.initial_step is not None:
            initial_step = args.initial_step
        else:
            initial_step = (args.initial_epoch - 1) * math.ceil(
                train_dataloader_len / accelerator_num_processes / args.gradient_accumulation_steps
            )
    elif steps_from_state is not None:
        initial_step = steps_from_state

    if initial_step > 0:
        assert (
            args.max_train_steps > initial_step
        ), f"max_train_steps should be greater than initial step / max_train_stepsは初期ステップより大きい必要があります: {args.max_train_steps} vs {initial_step}"

    epoch_to_start = 0
    progress_start_step = 0
    resolved_initial_step = initial_step
    steps_per_epoch_for_resume = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)

    if initial_step > 0:
        if args.skip_until_initial_step:
            if not args.resume:
                logger.info(
                    "initial_step is specified but not resuming. lr scheduler will be started from the beginning / "
                    "initial_stepが指定されていますがresumeしていないため、lr schedulerは最初から始まります"
                )
            logger.info(f"skipping {initial_step} steps / {initial_step}ステップをスキップします")
            resolved_initial_step = initial_step * args.gradient_accumulation_steps
            epoch_to_start = resolved_initial_step // steps_per_epoch_for_resume
            if track_initial_progress:
                progress_start_step = resolved_initial_step
        else:
            epoch_to_start = initial_step // steps_per_epoch_for_resume
            if track_initial_progress:
                progress_start_step = initial_step
            resolved_initial_step = 0

    return InitialTrainingPlan(
        initial_step=resolved_initial_step,
        epoch_to_start=epoch_to_start,
        progress_start_step=progress_start_step,
    )
