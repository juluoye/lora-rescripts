from __future__ import annotations

import math
import os
from multiprocessing import Value
from typing import Optional

import torch

import library.config_util as config_util
import library.strategy_base as strategy_base
import library.train_util as train_util
from library.config_util import BlueprintGenerator, ConfigSanitizer
from library.device_utils import clean_memory_on_device


def prepare_dataset_setup(
    trainer,
    args,
    *,
    use_user_config: bool,
    use_dreambooth_method: bool,
    cache_latents: bool,
    prepared_cls,
    logger,
):
    if args.dataset_class is None:
        blueprint_generator = BlueprintGenerator(ConfigSanitizer(True, True, args.masked_loss, True))
        if use_user_config:
            logger.info(f"Loading dataset config from {args.dataset_config}")
            user_config = config_util.load_user_config(args.dataset_config)
            ignored = ["train_data_dir", "reg_data_dir", "in_json"]
            if any(getattr(args, attr) is not None for attr in ignored):
                logger.warning(
                    "ignoring the following options because config file is found: {0} / 設定ファイルが利用されるため以下のオプションは無視されます: {0}".format(
                        ", ".join(ignored)
                    )
                )
        else:
            if use_dreambooth_method:
                logger.info("Using DreamBooth method.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": config_util.generate_dreambooth_subsets_config_by_subdirs(
                                args.train_data_dir, args.reg_data_dir
                            )
                        }
                    ]
                }
            else:
                logger.info("Training with captions.")
                user_config = {
                    "datasets": [
                        {
                            "subsets": [
                                {
                                    "image_dir": args.train_data_dir,
                                    "metadata_file": args.in_json,
                                }
                            ]
                        }
                    ]
                }

        blueprint = blueprint_generator.generate(user_config, args)
        train_dataset_group, val_dataset_group = config_util.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    else:
        train_dataset_group = train_util.load_arbitrary_dataset(args)
        val_dataset_group = None

    current_epoch = Value("i", 0)
    current_step = Value("i", 0)
    ds_for_collator = train_dataset_group if args.max_data_loader_n_workers == 0 else None
    collator = train_util.collator_class(current_epoch, current_step, ds_for_collator)

    if args.debug_dataset:
        train_dataset_group.set_current_strategies()
        train_util.debug_dataset(train_dataset_group)

        if val_dataset_group is not None:
            val_dataset_group.set_current_strategies()
            train_util.debug_dataset(val_dataset_group)
        return None

    if len(train_dataset_group) == 0:
        logger.error(
            "No data found. Please verify arguments (train_data_dir must be the parent of folders with images) / "
            "画像がありません。引数指定を確認してください（train_data_dirには画像があるフォルダではなく、画像があるフォルダの親フォルダを指定する必要があります）"
        )
        return None

    if cache_latents:
        assert (
            train_dataset_group.is_latent_cacheable()
        ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"
        if val_dataset_group is not None:
            assert (
                val_dataset_group.is_latent_cacheable()
            ), "when caching latents, either color_aug or random_crop cannot be used / latentをキャッシュするときはcolor_augとrandom_cropは使えません"

    trainer.assert_extra_args(args, train_dataset_group, val_dataset_group)

    return prepared_cls(
        train_dataset_group=train_dataset_group,
        val_dataset_group=val_dataset_group,
        current_epoch=current_epoch,
        current_step=current_step,
        collator=collator,
    )


def prepare_cached_model_inputs(
    trainer,
    args,
    accelerator,
    unet,
    vae,
    text_encoders,
    train_dataset_group,
    val_dataset_group,
    weight_dtype,
    vae_dtype,
    cache_latents: bool,
    prepared_cls,
):
    if cache_latents:
        vae.to(accelerator.device, dtype=vae_dtype)
        vae.requires_grad_(False)
        vae.eval()

        train_dataset_group.new_cache_latents(vae, accelerator)
        if val_dataset_group is not None:
            val_dataset_group.new_cache_latents(vae, accelerator)

        vae.to("cpu")
        clean_memory_on_device(accelerator.device)
        accelerator.wait_for_everyone()

    text_encoding_strategy = trainer.get_text_encoding_strategy(args)
    strategy_base.TextEncodingStrategy.set_strategy(text_encoding_strategy)

    text_encoder_outputs_caching_strategy = trainer.get_text_encoder_outputs_caching_strategy(args)
    if text_encoder_outputs_caching_strategy is not None:
        strategy_base.TextEncoderOutputsCachingStrategy.set_strategy(text_encoder_outputs_caching_strategy)

    trainer.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, train_dataset_group, weight_dtype)
    if val_dataset_group is not None:
        trainer.cache_text_encoder_outputs_if_needed(args, accelerator, unet, vae, text_encoders, val_dataset_group, weight_dtype)

    if unet is None:
        unet, text_encoders = trainer.load_unet_lazily(args, weight_dtype, accelerator, text_encoders)

    return prepared_cls(
        text_encoding_strategy=text_encoding_strategy,
        text_encoders=text_encoders,
        unet=unet,
    )


def prepare_runtime_models(trainer, args, prepared_cls, logger):
    logger.info("preparing accelerator")
    accelerator = train_util.prepare_accelerator(args)
    is_main_process = accelerator.is_main_process

    weight_dtype, save_dtype = train_util.prepare_dtype(args)
    vae_dtype = (torch.float32 if args.no_half_vae else weight_dtype) if trainer.cast_vae(args) else None

    model_version, text_encoder, vae, unet = trainer.load_target_model(args, weight_dtype, accelerator)
    if vae_dtype is None:
        vae_dtype = vae.dtype
        logger.info(f"vae_dtype is set to {vae_dtype} by the model since cast_vae() is false")

    text_encoders = text_encoder if isinstance(text_encoder, list) else [text_encoder]

    return prepared_cls(
        accelerator=accelerator,
        is_main_process=is_main_process,
        weight_dtype=weight_dtype,
        save_dtype=save_dtype,
        vae_dtype=vae_dtype,
        model_version=model_version,
        text_encoder=text_encoder,
        vae=vae,
        unet=unet,
        text_encoders=text_encoders,
    )


def resolve_text_encoder_lr(args, network):
    support_multiple_lrs = hasattr(network, "prepare_optimizer_params_with_multiple_te_lrs")
    if support_multiple_lrs:
        return args.text_encoder_lr
    if args.text_encoder_lr is None or isinstance(args.text_encoder_lr, float) or isinstance(args.text_encoder_lr, int):
        return args.text_encoder_lr
    return None if len(args.text_encoder_lr) == 0 else args.text_encoder_lr[0]


def prepare_network_trainable_params(args, network):
    text_encoder_lr = resolve_text_encoder_lr(args, network)
    support_multiple_lrs = hasattr(network, "prepare_optimizer_params_with_multiple_te_lrs")
    try:
        if support_multiple_lrs:
            results = network.prepare_optimizer_params_with_multiple_te_lrs(text_encoder_lr, args.unet_lr, args.learning_rate)
        else:
            results = network.prepare_optimizer_params(text_encoder_lr, args.unet_lr, args.learning_rate)
        if type(results) is tuple:
            trainable_params = results[0]
            lr_descriptions = results[1]
        else:
            trainable_params = results
            lr_descriptions = None
    except TypeError:
        trainable_params = network.prepare_optimizer_params(text_encoder_lr, args.unet_lr)
        lr_descriptions = None

    return text_encoder_lr, trainable_params, lr_descriptions


def prepare_training_components(
    args,
    accelerator,
    network,
    train_dataset_group,
    val_dataset_group,
    collator,
    prepared_cls,
):
    text_encoder_lr, trainable_params, lr_descriptions = prepare_network_trainable_params(args, network)
    optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(args, trainable_params)
    optimizer_train_fn, optimizer_eval_fn = train_util.get_optimizer_train_eval_fn(optimizer, args)

    train_dataset_group.set_current_strategies()
    if val_dataset_group is not None:
        val_dataset_group.set_current_strategies()

    n_workers = min(args.max_data_loader_n_workers, os.cpu_count())
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_group,
        batch_size=1,
        shuffle=True,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    val_dataloader = torch.utils.data.DataLoader(
        val_dataset_group if val_dataset_group is not None else [],
        shuffle=False,
        batch_size=1,
        collate_fn=collator,
        num_workers=n_workers,
        persistent_workers=args.persistent_data_loader_workers,
    )

    if args.max_train_epochs is not None:
        args.max_train_steps = args.max_train_epochs * math.ceil(
            len(train_dataloader) / accelerator.num_processes / args.gradient_accumulation_steps
        )
        accelerator.print(
            f"override steps. steps for {args.max_train_epochs} epochs is / 指定エポックまでのステップ数: {args.max_train_steps}"
        )

    train_dataset_group.set_max_train_steps(args.max_train_steps)
    lr_scheduler = train_util.get_scheduler_fix(args, optimizer, accelerator.num_processes)

    return prepared_cls(
        text_encoder_lr=text_encoder_lr,
        optimizer_name=optimizer_name,
        optimizer_args=optimizer_args,
        optimizer=optimizer,
        optimizer_train_fn=optimizer_train_fn,
        optimizer_eval_fn=optimizer_eval_fn,
        lr_descriptions=lr_descriptions,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr_scheduler=lr_scheduler,
    )


__all__ = [
    "prepare_cached_model_inputs",
    "prepare_dataset_setup",
    "prepare_network_trainable_params",
    "prepare_runtime_models",
    "prepare_training_components",
    "resolve_text_encoder_lr",
]
