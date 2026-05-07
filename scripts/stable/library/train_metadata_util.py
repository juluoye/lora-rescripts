from __future__ import annotations

import json
import os
from typing import Any, Callable, Mapping, NamedTuple

import library.train_util as train_util
from library.train_util import DreamBoothDataset
from mikazuki.compliance import build_lulynx_metadata_fields


class PreparedMetadataBundle(NamedTuple):
    metadata: dict[str, str]
    minimum_metadata: dict[str, str]


def _metadata_boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def infer_training_algo(args, net_kwargs: Mapping[str, Any] | None = None) -> str | None:
    net_kwargs = net_kwargs or {}
    network_module = str(getattr(args, "network_module", "") or "").strip().lower()

    if "algo" in net_kwargs and net_kwargs["algo"] not in (None, ""):
        return str(net_kwargs["algo"]).strip()

    adapter_variant = str(net_kwargs.get("adapter_type", "") or "").strip().lower()
    if adapter_variant:
        return adapter_variant

    anima_adapter_type = str(net_kwargs.get("anima_adapter_type", "") or "").strip().lower()
    if anima_adapter_type:
        return anima_adapter_type

    if network_module == "lycoris.kohya":
        return "lycoris"
    if network_module.startswith("networks.lora_fa"):
        return "lora_fa"
    if network_module.startswith("networks.vera"):
        return "vera"
    if network_module.startswith("networks.tlora"):
        return "tlora"
    if network_module.startswith("networks.dylora"):
        return "dylora"
    if network_module.startswith("networks.oft"):
        return "oft"
    if network_module.startswith("networks.lokr"):
        return "lokr"
    if network_module.startswith("networks.lora"):
        return "lora"
    return None


def build_compatibility_metadata(args, net_kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    net_kwargs = dict(net_kwargs or {})
    network_module = str(getattr(args, "network_module", "") or "").strip()
    normalized_network_module = network_module.lower()
    training_algo = infer_training_algo(args, net_kwargs)

    metadata: dict[str, Any] = {
        "ss_training_network_module": network_module,
    }

    if net_kwargs:
        metadata["ss_training_network_args"] = json.dumps(net_kwargs, ensure_ascii=False)

    if training_algo:
        metadata["ss_training_algo"] = training_algo

    if normalized_network_module == "lycoris.kohya":
        metadata["ss_training_is_lycoris"] = True
        lycoris_algo = str(net_kwargs.get("algo", "") or "").strip()
        if lycoris_algo:
            metadata["ss_lycoris_algo"] = lycoris_algo
            metadata["ss_training_lycoris_algo"] = lycoris_algo

    if "dora_wd" in net_kwargs:
        metadata["ss_dora_enabled"] = _metadata_boolish(net_kwargs.get("dora_wd"))

    if "train_norm" in net_kwargs:
        metadata["ss_train_norm_enabled"] = _metadata_boolish(net_kwargs.get("train_norm"))

    if training_algo in {"loha", "lokr", "glora", "diag-oft", "boft", "dylora", "ia3"}:
        metadata["ss_network_type"] = training_algo
    elif normalized_network_module == "lycoris.kohya":
        metadata["ss_network_type"] = str(net_kwargs.get("algo", "") or "lycoris")
    elif training_algo:
        metadata["ss_network_type"] = training_algo

    return metadata


def build_base_metadata(
    args,
    *,
    session_id,
    training_started_at,
    text_encoder_lr,
    train_dataset_group,
    val_dataset_group,
    train_dataloader,
    num_train_epochs,
    model_version,
    optimizer_name,
    optimizer_args,
    update_metadata: Callable[[dict[str, Any], Any], None],
    include_attention_backend: bool = False,
):
    metadata = {
        "ss_session_id": session_id,
        "ss_training_started_at": training_started_at,
        "ss_output_name": args.output_name,
        "ss_learning_rate": args.learning_rate,
        "ss_text_encoder_lr": text_encoder_lr,
        "ss_unet_lr": args.unet_lr,
        "ss_num_train_images": train_dataset_group.num_train_images,
        "ss_num_validation_images": val_dataset_group.num_train_images if val_dataset_group is not None else 0,
        "ss_num_reg_images": train_dataset_group.num_reg_images,
        "ss_num_batches_per_epoch": len(train_dataloader),
        "ss_num_epochs": num_train_epochs,
        "ss_gradient_checkpointing": args.gradient_checkpointing,
        "ss_gradient_accumulation_steps": args.gradient_accumulation_steps,
        "ss_max_train_steps": args.max_train_steps,
        "ss_lr_warmup_steps": args.lr_warmup_steps,
        "ss_lr_scheduler": args.lr_scheduler,
        "ss_network_module": args.network_module,
        "ss_network_dim": args.network_dim,
        "ss_network_alpha": args.network_alpha,
        "ss_network_dropout": args.network_dropout,
        "ss_mixed_precision": args.mixed_precision,
        "ss_full_fp16": bool(args.full_fp16),
        "ss_v2": bool(args.v2),
        "ss_base_model_version": model_version,
        "ss_clip_skip": args.clip_skip,
        "ss_max_token_length": args.max_token_length,
        "ss_cache_latents": bool(args.cache_latents),
        "ss_seed": args.seed,
        "ss_lowram": args.lowram,
        "ss_noise_offset": args.noise_offset,
        "ss_multires_noise_iterations": args.multires_noise_iterations,
        "ss_multires_noise_discount": args.multires_noise_discount,
        "ss_adaptive_noise_scale": args.adaptive_noise_scale,
        "ss_zero_terminal_snr": args.zero_terminal_snr,
        "ss_training_comment": args.training_comment,
        "ss_sd_scripts_commit_hash": train_util.get_git_revision_hash(),
        "ss_optimizer": optimizer_name + (f"({optimizer_args})" if len(optimizer_args) > 0 else ""),
        "ss_max_grad_norm": args.max_grad_norm,
        "ss_caption_dropout_rate": args.caption_dropout_rate,
        "ss_caption_dropout_every_n_epochs": args.caption_dropout_every_n_epochs,
        "ss_caption_tag_dropout_rate": args.caption_tag_dropout_rate,
        "ss_face_crop_aug_range": args.face_crop_aug_range,
        "ss_prior_loss_weight": args.prior_loss_weight,
        "ss_min_snr_gamma": args.min_snr_gamma,
        "ss_scale_weight_norms": args.scale_weight_norms,
        "ss_ip_noise_gamma": args.ip_noise_gamma,
        "ss_debiased_estimation": bool(args.debiased_estimation_loss),
        "ss_noise_offset_random_strength": args.noise_offset_random_strength,
        "ss_ip_noise_gamma_random_strength": args.ip_noise_gamma_random_strength,
        "ss_loss_type": args.loss_type,
        "ss_huber_schedule": args.huber_schedule,
        "ss_huber_scale": args.huber_scale,
        "ss_huber_c": args.huber_c,
        "ss_wavelet_loss_enabled": bool(getattr(args, "wavelet_loss_enabled", False)),
        "ss_wavelet_loss_weight": getattr(args, "wavelet_loss_weight", 0.0),
        "ss_wavelet_loss_levels": getattr(args, "wavelet_loss_levels", 1),
        "ss_wavelet_loss_approx_weight": getattr(args, "wavelet_loss_approx_weight", 0.0),
        "ss_fp8_base": bool(args.fp8_base),
        "ss_fp8_base_unet": bool(args.fp8_base_unet),
        "ss_validation_seed": args.validation_seed,
        "ss_validation_split": args.validation_split,
        "ss_max_validation_steps": args.max_validation_steps,
        "ss_validate_every_n_epochs": args.validate_every_n_epochs,
        "ss_validate_every_n_steps": args.validate_every_n_steps,
        "ss_resize_interpolation": args.resize_interpolation,
    }
    if include_attention_backend:
        metadata["ss_attention_backend"] = train_util.resolve_attention_backend(args)

    update_metadata(metadata, args)
    return metadata


def build_user_config_dataset_metadata(train_dataset_group, *, include_skip_image_resolution: bool = False):
    datasets_metadata = []
    tag_frequency = {}
    dataset_dirs_info = {}

    for dataset in train_dataset_group.datasets:
        is_dreambooth_dataset = isinstance(dataset, DreamBoothDataset)
        dataset_metadata = {
            "is_dreambooth": is_dreambooth_dataset,
            "batch_size_per_device": dataset.batch_size,
            "num_train_images": dataset.num_train_images,
            "num_reg_images": dataset.num_reg_images,
            "resolution": (dataset.width, dataset.height),
            "enable_bucket": bool(dataset.enable_bucket),
            "min_bucket_reso": dataset.min_bucket_reso,
            "max_bucket_reso": dataset.max_bucket_reso,
            "tag_frequency": dataset.tag_frequency,
            "bucket_info": dataset.bucket_info,
            "resize_interpolation": dataset.resize_interpolation,
        }
        if include_skip_image_resolution:
            dataset_metadata["skip_image_resolution"] = getattr(dataset, "skip_image_resolution", None)

        subsets_metadata = []
        for subset in dataset.subsets:
            subset_metadata = {
                "img_count": subset.img_count,
                "num_repeats": subset.num_repeats,
                "color_aug": bool(subset.color_aug),
                "flip_aug": bool(subset.flip_aug),
                "random_crop": bool(subset.random_crop),
                "shuffle_caption": bool(subset.shuffle_caption),
                "keep_tokens": subset.keep_tokens,
                "keep_tokens_separator": subset.keep_tokens_separator,
                "secondary_separator": subset.secondary_separator,
                "enable_wildcard": bool(subset.enable_wildcard),
                "caption_prefix": subset.caption_prefix,
                "caption_suffix": subset.caption_suffix,
                "resize_interpolation": subset.resize_interpolation,
            }

            image_dir_or_metadata_file = None
            if subset.image_dir:
                image_dir = os.path.basename(subset.image_dir)
                subset_metadata["image_dir"] = image_dir
                image_dir_or_metadata_file = image_dir

            if is_dreambooth_dataset:
                subset_metadata["class_tokens"] = subset.class_tokens
                subset_metadata["is_reg"] = subset.is_reg
                if subset.is_reg:
                    image_dir_or_metadata_file = None
            else:
                metadata_file = os.path.basename(subset.metadata_file)
                subset_metadata["metadata_file"] = metadata_file
                image_dir_or_metadata_file = metadata_file

            subsets_metadata.append(subset_metadata)

            if image_dir_or_metadata_file is not None:
                merged_name = image_dir_or_metadata_file
                index = 2
                while merged_name in dataset_dirs_info:
                    merged_name = image_dir_or_metadata_file + f" ({index})"
                    index += 1

                dataset_dirs_info[merged_name] = {
                    "n_repeats": subset.num_repeats,
                    "img_count": subset.img_count,
                }

        dataset_metadata["subsets"] = subsets_metadata
        datasets_metadata.append(dataset_metadata)

        for ds_dir_name, ds_freq_for_dir in dataset.tag_frequency.items():
            if ds_dir_name in tag_frequency:
                continue
            tag_frequency[ds_dir_name] = ds_freq_for_dir

    return {
        "ss_datasets": json.dumps(datasets_metadata),
        "ss_tag_frequency": json.dumps(tag_frequency),
        "ss_dataset_dirs": json.dumps(dataset_dirs_info),
    }


def build_legacy_dataset_metadata(
    args,
    train_dataset_group,
    use_dreambooth_method,
    total_batch_size,
    *,
    include_skip_image_resolution: bool = False,
):
    assert (
        len(train_dataset_group.datasets) == 1
    ), f"There should be a single dataset but {len(train_dataset_group.datasets)} found. This seems to be a bug. / データセットは1個だけ存在するはずですが、実際には{len(train_dataset_group.datasets)}個でした。プログラムのバグかもしれません。"

    dataset = train_dataset_group.datasets[0]
    dataset_dirs_info = {}
    reg_dataset_dirs_info = {}

    if use_dreambooth_method:
        for subset in dataset.subsets:
            info = reg_dataset_dirs_info if subset.is_reg else dataset_dirs_info
            info[os.path.basename(subset.image_dir)] = {
                "n_repeats": subset.num_repeats,
                "img_count": subset.img_count,
            }
    else:
        for subset in dataset.subsets:
            dataset_dirs_info[os.path.basename(subset.metadata_file)] = {
                "n_repeats": subset.num_repeats,
                "img_count": subset.img_count,
            }

    metadata = {
        "ss_batch_size_per_device": args.train_batch_size,
        "ss_total_batch_size": total_batch_size,
        "ss_resolution": args.resolution,
        "ss_color_aug": bool(args.color_aug),
        "ss_flip_aug": bool(args.flip_aug),
        "ss_random_crop": bool(args.random_crop),
        "ss_shuffle_caption": bool(args.shuffle_caption),
        "ss_enable_bucket": bool(dataset.enable_bucket),
        "ss_bucket_no_upscale": bool(dataset.bucket_no_upscale),
        "ss_min_bucket_reso": dataset.min_bucket_reso,
        "ss_max_bucket_reso": dataset.max_bucket_reso,
        "ss_keep_tokens": args.keep_tokens,
        "ss_dataset_dirs": json.dumps(dataset_dirs_info),
        "ss_reg_dataset_dirs": json.dumps(reg_dataset_dirs_info),
        "ss_tag_frequency": json.dumps(dataset.tag_frequency),
        "ss_bucket_info": json.dumps(dataset.bucket_info),
    }
    if include_skip_image_resolution:
        metadata["ss_skip_image_resolution"] = getattr(args, "skip_image_resolution", None)
    return metadata


def add_runtime_metadata(metadata, args, net_kwargs, extra_metadata: Mapping[str, Any] | None = None):
    if args.network_args:
        metadata["ss_network_args"] = json.dumps(net_kwargs)
    metadata.update(build_compatibility_metadata(args, net_kwargs))
    if extra_metadata is not None:
        metadata.update(extra_metadata)


def add_model_reference_metadata(metadata, args):
    if args.pretrained_model_name_or_path is not None:
        sd_model_name = args.pretrained_model_name_or_path
        if os.path.exists(sd_model_name):
            metadata["ss_sd_model_hash"] = train_util.model_hash(sd_model_name)
            metadata["ss_new_sd_model_hash"] = train_util.calculate_sha256(sd_model_name)
            sd_model_name = os.path.basename(sd_model_name)
        metadata["ss_sd_model_name"] = sd_model_name

    if args.vae is not None:
        vae_name = args.vae
        if os.path.exists(vae_name):
            metadata["ss_vae_hash"] = train_util.model_hash(vae_name)
            metadata["ss_new_vae_hash"] = train_util.calculate_sha256(vae_name)
            vae_name = os.path.basename(vae_name)
        metadata["ss_vae_name"] = vae_name


def finalize_metadata(metadata):
    metadata = {k: str(v) for k, v in metadata.items()}
    metadata.update(
        build_lulynx_metadata_fields(
            metadata=metadata,
            git_commit=metadata.get("ss_sd_scripts_commit_hash", ""),
        )
    )
    minimum_metadata = {}
    for key in train_util.SS_METADATA_MINIMUM_KEYS:
        if key in metadata:
            minimum_metadata[key] = metadata[key]
    return PreparedMetadataBundle(metadata, minimum_metadata)


def build_metadata_bundle(
    args,
    *,
    session_id,
    training_started_at,
    text_encoder_lr,
    train_dataset_group,
    val_dataset_group,
    train_dataloader,
    num_train_epochs,
    model_version,
    optimizer_name,
    optimizer_args,
    use_user_config,
    use_dreambooth_method,
    total_batch_size,
    net_kwargs,
    update_metadata: Callable[[dict[str, Any], Any], None],
    extra_metadata: Mapping[str, Any] | None = None,
    include_attention_backend: bool = False,
    include_dataset_skip_image_resolution: bool = False,
    include_legacy_skip_image_resolution: bool = False,
):
    metadata = build_base_metadata(
        args,
        session_id=session_id,
        training_started_at=training_started_at,
        text_encoder_lr=text_encoder_lr,
        train_dataset_group=train_dataset_group,
        val_dataset_group=val_dataset_group,
        train_dataloader=train_dataloader,
        num_train_epochs=num_train_epochs,
        model_version=model_version,
        optimizer_name=optimizer_name,
        optimizer_args=optimizer_args,
        update_metadata=update_metadata,
        include_attention_backend=include_attention_backend,
    )

    if use_user_config:
        metadata.update(
            build_user_config_dataset_metadata(
                train_dataset_group,
                include_skip_image_resolution=include_dataset_skip_image_resolution,
            )
        )
    else:
        metadata.update(
            build_legacy_dataset_metadata(
                args,
                train_dataset_group,
                use_dreambooth_method,
                total_batch_size,
                include_skip_image_resolution=include_legacy_skip_image_resolution,
            )
        )

    add_runtime_metadata(metadata, args, net_kwargs, extra_metadata)
    add_model_reference_metadata(metadata, args)
    return finalize_metadata(metadata)
