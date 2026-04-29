from __future__ import annotations

import library.train_util as train_util


def maybe_save_step_checkpoint(args, accelerator, network, global_step: int, epoch_no: int, save_model, remove_model) -> None:
    if args.save_every_n_steps is None or global_step % args.save_every_n_steps != 0:
        return

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, global_step)
        save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch_no)

        if args.save_state:
            train_util.save_and_remove_state_stepwise(args, accelerator, global_step)

        remove_step_no = train_util.get_remove_step_no(args, global_step)
        if remove_step_no is not None:
            remove_ckpt_name = train_util.get_step_ckpt_name(args, "." + args.save_model_as, remove_step_no)
            remove_model(remove_ckpt_name)


def handle_epoch_end(
    *,
    args,
    accelerator,
    is_tracking: bool,
    epoch_logs,
    epoch_log_callback,
    optimizer_eval_fn,
    optimizer_train_fn,
    is_main_process: bool,
    network,
    global_step: int,
    epoch_no: int,
    total_epochs: int,
    save_model,
    remove_model,
    progress_bar,
    sample_images_fn,
    cooldown_fn=None,
):
    if is_tracking and epoch_logs is not None:
        epoch_log_callback(epoch_logs)

    accelerator.wait_for_everyone()

    optimizer_eval_fn()
    if args.save_every_n_epochs is not None:
        saving = epoch_no % args.save_every_n_epochs == 0 and epoch_no < total_epochs
        if is_main_process and saving:
            ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, epoch_no)
            save_model(ckpt_name, accelerator.unwrap_model(network), global_step, epoch_no)

            remove_epoch_no = train_util.get_remove_epoch_no(args, epoch_no)
            if remove_epoch_no is not None:
                remove_ckpt_name = train_util.get_epoch_ckpt_name(args, "." + args.save_model_as, remove_epoch_no)
                remove_model(remove_ckpt_name)

            if args.save_state:
                train_util.save_and_remove_state_on_epoch_end(args, accelerator, epoch_no)

    sample_images_fn(epoch_no, global_step)
    progress_bar.unpause()
    optimizer_train_fn()
    if cooldown_fn is not None:
        cooldown_fn(epoch_no, total_epochs)


def finalize_training(args, accelerator, is_main_process: bool, network, global_step: int, final_epoch_no: int, save_model, optimizer_eval_fn, logger):
    if is_main_process:
        network = accelerator.unwrap_model(network)

    accelerator.end_training()
    optimizer_eval_fn()

    if is_main_process and (args.save_state or args.save_state_on_train_end):
        train_util.save_state_on_train_end(args, accelerator)

    if is_main_process:
        ckpt_name = train_util.get_last_ckpt_name(args, "." + args.save_model_as)
        save_model(ckpt_name, network, global_step, final_epoch_no, force_sync_upload=True)
        logger.info("model saved.")

    return network


__all__ = [
    "finalize_training",
    "handle_epoch_end",
    "maybe_save_step_checkpoint",
]
