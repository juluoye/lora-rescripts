from __future__ import annotations

import inspect
import importlib
import os
import sys

import torch

from library import deepspeed_utils
from library import full_bf16_stochastic_util
from lulynx.experimental_core import create_lulynx_core
import library.train_util as train_util


def import_network_module(args, accelerator):
    sys.path.append(os.path.dirname(__file__))
    accelerator.print("import network module:", args.network_module)
    return importlib.import_module(args.network_module)


def merge_base_network_weights(args, accelerator, network_module, vae, text_encoder, unet, weight_dtype) -> None:
    if args.base_weights is None:
        return

    for i, weight_path in enumerate(args.base_weights):
        if args.base_weights_multiplier is None or len(args.base_weights_multiplier) <= i:
            multiplier = 1.0
        else:
            multiplier = args.base_weights_multiplier[i]

        accelerator.print(f"merging module: {weight_path} with multiplier {multiplier}")
        module, weights_sd = network_module.create_network_from_weights(
            multiplier, weight_path, vae, text_encoder, unet, for_inference=True
        )
        module.merge_to(text_encoder, unet, weights_sd, weight_dtype, accelerator.device if args.lowram else "cpu")

    accelerator.print(f"all weights merged: {', '.join(args.base_weights)}")


def parse_network_kwargs(args) -> dict[str, str]:
    net_kwargs = {}
    if args.network_args is not None:
        for net_arg in args.network_args:
            key, value = net_arg.split("=", 1)
            net_kwargs[key] = value
    return net_kwargs


def configure_network_gradient_checkpointing(trainer, args, accelerator, network, text_encoders, unet) -> None:
    if not args.gradient_checkpointing:
        return

    supports_cpu_offload_checkpointing = False
    try:
        supports_cpu_offload_checkpointing = "cpu_offload" in inspect.signature(unet.enable_gradient_checkpointing).parameters
    except (TypeError, ValueError):
        supports_cpu_offload_checkpointing = False

    if args.cpu_offload_checkpointing and supports_cpu_offload_checkpointing:
        unet.enable_gradient_checkpointing(cpu_offload=True)
    else:
        if args.cpu_offload_checkpointing and not supports_cpu_offload_checkpointing:
            accelerator.print(
                "WARNING: cpu_offload_checkpointing is not supported by the current U-Net/DiT route. "
                "Falling back to standard gradient checkpointing. "
                "/ 当前训练路由不支持 cpu_offload_checkpointing，已自动回退为普通梯度检查点。"
            )
        unet.enable_gradient_checkpointing()

    for t_enc, flag in zip(text_encoders, trainer.get_text_encoders_train_flags(args, text_encoders)):
        if flag and t_enc.supports_gradient_checkpointing:
            t_enc.gradient_checkpointing_enable()

    network.enable_gradient_checkpointing()


def prepare_network_setup(
    trainer,
    args,
    accelerator,
    vae,
    text_encoder,
    unet,
    text_encoders,
    weight_dtype,
    prepared_cls,
    logger,
):
    network_module = import_network_module(args, accelerator)
    merge_base_network_weights(args, accelerator, network_module, vae, text_encoder, unet, weight_dtype)

    net_kwargs = parse_network_kwargs(args)
    if args.dim_from_weights:
        network, _ = network_module.create_network_from_weights(1, args.network_weights, vae, text_encoder, unet, **net_kwargs)
    else:
        if "dropout" not in net_kwargs:
            net_kwargs["dropout"] = args.network_dropout

        network = network_module.create_network(
            1.0,
            args.network_dim,
            args.network_alpha,
            vae,
            text_encoder,
            unet,
            neuron_dropout=args.network_dropout,
            **net_kwargs,
        )

    if network is None:
        raise ValueError("Network module returned None during create_network().")

    if hasattr(network, "prepare_network"):
        network.prepare_network(args)
    if args.scale_weight_norms and not hasattr(network, "apply_max_norm_regularization"):
        logger.warning(
            "warning: scale_weight_norms is specified but the network does not support it / scale_weight_normsが指定されていますが、ネットワークが対応していません"
        )
        args.scale_weight_norms = False

    trainer.post_process_network(args, accelerator, network, text_encoders, unet)

    lulynx_core = create_lulynx_core(
        args,
        route_kind="sdxl" if trainer.is_sdxl else "stable",
        route_label="SDXL LoRA" if trainer.is_sdxl else "Stable LoRA",
    )
    if lulynx_core is not None:
        lulynx_core.apply_pre_optimizer_settings(network)

    train_unet = not args.network_train_text_encoder_only
    train_text_encoder = trainer.is_train_text_encoder(args)
    if not train_unet and not train_text_encoder:
        raise ValueError(
            "No training target is enabled for this network route. "
            "Please enable DiT/U-Net training or text encoder training before starting. "
            "/ 当前没有任何训练目标，请至少启用 DiT/U-Net 或文本编码器中的一个。"
        )
    network.apply_to(text_encoder, unet, train_text_encoder, train_unet)
    trainer.validate_network_target_modules(args, network, train_text_encoder, train_unet)

    if args.network_weights is not None:
        info = network.load_weights(args.network_weights)
        accelerator.print(f"load network weights from {args.network_weights}: {info}")

    configure_network_gradient_checkpointing(trainer, args, accelerator, network, text_encoders, unet)

    return prepared_cls(
        network=network,
        net_kwargs=net_kwargs,
        train_unet=train_unet,
        train_text_encoder=train_text_encoder,
        lulynx_core=lulynx_core,
    )


def prepare_execution_runtime(
    trainer,
    args,
    accelerator,
    network,
    optimizer,
    train_dataloader,
    val_dataloader,
    lr_scheduler,
    text_encoder,
    text_encoders,
    unet,
    weight_dtype,
    vae,
    vae_dtype,
    cache_latents: bool,
    train_unet: bool,
    train_text_encoder: bool,
    prepared_cls,
    logger,
):
    if args.full_fp16:
        assert args.mixed_precision == "fp16", "full_fp16 requires mixed precision='fp16' / full_fp16を使う場合はmixed_precision='fp16'を指定してください。"
        accelerator.print("enable full fp16 training.")
        network.to(weight_dtype)
    elif args.full_bf16:
        assert args.mixed_precision == "bf16", "full_bf16 requires mixed precision='bf16' / full_bf16を使う場合はmixed_precision='bf16'を指定してください。"
        accelerator.print("enable full bf16 training.")
        network.to(weight_dtype)

    unet_weight_dtype = te_weight_dtype = weight_dtype
    if args.fp8_base or args.fp8_base_unet:
        assert torch.__version__ >= "2.1.0", "fp8_base requires torch>=2.1.0 / fp8を使う場合はtorch>=2.1.0が必要です。"
        assert args.mixed_precision != "no", "fp8_base requires mixed precision='fp16' or 'bf16' / fp8を使う場合はmixed_precision='fp16'または'bf16'が必要です。"
        accelerator.print("enable fp8 training for U-Net.")
        unet_weight_dtype = torch.float8_e4m3fn

        if not args.fp8_base_unet:
            accelerator.print("enable fp8 training for Text Encoder.")
        te_weight_dtype = weight_dtype if args.fp8_base_unet else torch.float8_e4m3fn

        logger.info(f"set U-Net weight dtype to {unet_weight_dtype}")
        unet.to(dtype=unet_weight_dtype)

    unet.requires_grad_(False)
    if trainer.cast_unet(args):
        unet.to(dtype=unet_weight_dtype)
    for i, t_enc in enumerate(text_encoders):
        t_enc.requires_grad_(False)
        if t_enc.device.type != "cpu" and trainer.cast_text_encoder(args):
            t_enc.to(dtype=te_weight_dtype)
            if te_weight_dtype != weight_dtype:
                trainer.prepare_text_encoder_fp8(i, t_enc, te_weight_dtype, weight_dtype)

    if args.deepspeed:
        flags = trainer.get_text_encoders_train_flags(args, text_encoders)
        ds_model = deepspeed_utils.prepare_deepspeed_model(
            args,
            unet=unet if train_unet else None,
            text_encoder1=text_encoders[0] if flags[0] else None,
            text_encoder2=(text_encoders[1] if flags[1] else None) if len(text_encoders) > 1 else None,
            network=network,
        )
        ds_model, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            ds_model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
        training_model = ds_model
    else:
        if train_unet:
            unet = trainer.prepare_unet_with_accelerator(args, accelerator, unet)
        else:
            unet.to(accelerator.device, dtype=unet_weight_dtype if trainer.cast_unet(args) else None)
        if train_text_encoder:
            text_encoders = [
                (accelerator.prepare(t_enc) if flag else t_enc)
                for t_enc, flag in zip(text_encoders, trainer.get_text_encoders_train_flags(args, text_encoders))
            ]
            if len(text_encoders) > 1:
                text_encoder = text_encoders
            else:
                text_encoder = text_encoders[0]

        network, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            network, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
        training_model = network

    if args.gradient_checkpointing:
        unet.train()
        for i, (t_enc, frag) in enumerate(zip(text_encoders, trainer.get_text_encoders_train_flags(args, text_encoders))):
            t_enc.train()
            if frag:
                trainer.prepare_text_encoder_grad_ckpt_workaround(i, t_enc)
    else:
        unet.eval()
        for t_enc in text_encoders:
            t_enc.eval()

    accelerator.unwrap_model(network).prepare_grad_etc(text_encoder, unet)
    trainer.configure_model_runtime(args, accelerator, network, text_encoders, unet)

    if not cache_latents:
        vae.requires_grad_(False)
        vae.eval()
        vae.to(accelerator.device, dtype=vae_dtype)

    if args.full_fp16:
        train_util.patch_accelerator_for_fp16_training(accelerator)
    elif args.full_bf16:
        full_bf16_stochastic_util.activate_training_model_grads_if_needed(
            args,
            optimizer=optimizer,
        )

    return prepared_cls(
        network=network,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        lr_scheduler=lr_scheduler,
        text_encoder=text_encoder,
        text_encoders=text_encoders,
        unet=unet,
        training_model=training_model,
        unet_weight_dtype=unet_weight_dtype,
    )


__all__ = [
    "configure_network_gradient_checkpointing",
    "import_network_module",
    "merge_base_network_weights",
    "parse_network_kwargs",
    "prepare_execution_runtime",
    "prepare_network_setup",
]
