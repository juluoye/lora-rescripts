from __future__ import annotations

import os
import time
from contextlib import nullcontext

import library.train_util as train_util
from mikazuki.compliance import build_lulynx_metadata_fields


def save_network_checkpoint(
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
    *,
    ema_model=None,
    upload_fn=None,
    force_sync_upload: bool = False,
):
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_file = os.path.join(args.output_dir, ckpt_name)

    accelerator.print(f"\nsaving checkpoint: {ckpt_file}")
    metadata["ss_training_finished_at"] = str(time.time())
    metadata["ss_steps"] = str(steps)
    metadata["ss_epoch"] = str(epoch_no)

    metadata_to_save = dict(minimum_metadata if args.no_metadata else metadata)
    sai_metadata = get_sai_model_spec(args)
    metadata_to_save.update(sai_metadata)

    save_context = ema_model.apply_to_models() if ema_model is not None else nullcontext()
    with save_context:
        try:
            weight_fingerprint = train_util.compute_tensor_payload_sha256(unwrapped_nw.state_dict())
        except Exception:
            weight_fingerprint = None
        metadata_to_save.update(
            build_lulynx_metadata_fields(
                metadata=metadata_to_save,
                git_commit=train_util.get_git_revision_hash(),
                model_hash=weight_fingerprint,
            )
        )
        unwrapped_nw.save_weights(ckpt_file, save_dtype, metadata_to_save)

    if args.huggingface_repo_id is not None and upload_fn is not None:
        upload_fn(args, ckpt_file, "/" + ckpt_name, force_sync_upload=force_sync_upload)


def remove_checkpoint(args, accelerator, old_ckpt_name):
    old_ckpt_file = os.path.join(args.output_dir, old_ckpt_name)
    if os.path.exists(old_ckpt_file):
        accelerator.print(f"removing old checkpoint: {old_ckpt_file}")
        os.remove(old_ckpt_file)


def run_initial_sampling(accelerator, optimizer_eval_fn, optimizer_train_fn, sample_images_fn) -> bool:
    optimizer_eval_fn()
    sample_images_fn()
    optimizer_train_fn()

    is_tracking = len(accelerator.trackers) > 0
    if is_tracking:
        accelerator.log({}, step=0)
    return is_tracking
