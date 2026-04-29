from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class TrainingSchedule:
    num_update_steps_per_epoch: int
    num_train_epochs: int
    total_batch_size: int


@dataclass
class PreparedLoopProgress:
    initial_step: int
    global_step: int
    progress_total: int


def prepare_training_schedule(args, *, train_dataloader_len: int, accelerator_num_processes: int) -> TrainingSchedule:
    num_update_steps_per_epoch = math.ceil(train_dataloader_len / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    if (args.save_n_epoch_ratio is not None) and (args.save_n_epoch_ratio > 0):
        args.save_every_n_epochs = math.floor(num_train_epochs / args.save_n_epoch_ratio) or 1

    total_batch_size = args.train_batch_size * accelerator_num_processes * args.gradient_accumulation_steps
    return TrainingSchedule(
        num_update_steps_per_epoch=num_update_steps_per_epoch,
        num_train_epochs=num_train_epochs,
        total_batch_size=total_batch_size,
    )


def log_training_start_summary(
    accelerator,
    args,
    train_dataset_group,
    val_dataset_group,
    *,
    train_dataloader_len: int,
    displayed_num_train_epochs: int,
    mixed_resolution_epoch_window: Optional[tuple[int, int]] = None,
) -> None:
    accelerator.print("running training / 学習開始")
    accelerator.print(f"  num train images * repeats / 学習画像の数×繰り返し回数: {train_dataset_group.num_train_images}")
    accelerator.print(
        f"  num validation images * repeats / 学習画像の数×繰り返し回数: "
        f"{val_dataset_group.num_train_images if val_dataset_group is not None else 0}"
    )
    accelerator.print(f"  num reg images / 正則化画像の数: {train_dataset_group.num_reg_images}")
    accelerator.print(f"  num batches per epoch / 1epochのバッチ数: {train_dataloader_len}")
    accelerator.print(f"  num epochs / epoch数: {displayed_num_train_epochs}")
    if mixed_resolution_epoch_window is not None:
        accelerator.print(
            f"  mixed-resolution epoch window / 阶段分辨率连续 epoch 区间: "
            f"{mixed_resolution_epoch_window[0]} -> {mixed_resolution_epoch_window[1]}"
        )
    accelerator.print(
        f"  batch size per device / バッチサイズ: {', '.join([str(dataset.batch_size) for dataset in train_dataset_group.datasets])}"
    )
    accelerator.print(f"  gradient accumulation steps / 勾配を合計するステップ数 = {args.gradient_accumulation_steps}")
    accelerator.print(f"  total optimization steps / 学習ステップ数: {args.max_train_steps}")


def prepare_loop_progress_state(
    logger,
    args,
    *,
    initial_step: int,
    epoch_to_start: int,
    train_dataloader_len: int,
    progress_start_step: int = 0,
    use_progress_start_step: bool = False,
) -> PreparedLoopProgress:
    adjusted_initial_step = initial_step
    if adjusted_initial_step > 0:
        for skip_epoch in range(epoch_to_start):
            logger.info(f"skipping epoch {skip_epoch+1} because initial_step (multiplied) is {adjusted_initial_step}")
            adjusted_initial_step -= train_dataloader_len

    if use_progress_start_step:
        global_step = progress_start_step
        progress_total = args.max_train_steps - progress_start_step
    else:
        global_step = adjusted_initial_step if adjusted_initial_step > 0 else 0
        progress_total = args.max_train_steps - adjusted_initial_step

    return PreparedLoopProgress(
        initial_step=adjusted_initial_step,
        global_step=global_step,
        progress_total=progress_total,
    )
