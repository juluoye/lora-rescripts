from __future__ import annotations

import importlib
from typing import Any, NamedTuple

from accelerate import init_empty_weights

import library.model_util as model_util
from library.sdxl_original_control_net import SdxlControlNet
from library.utils import GradualLatent
from networks.control_net_lllite import ControlNetLLLite
from tools.original_control_net import ControlNetInfo
import tools.original_control_net as original_control_net


class PreparedGenImgPipelineSetup(NamedTuple):
    networks: list[Any]
    network_default_muls: list[float]
    network_pre_calc: bool
    upscaler: Any
    control_nets: list[Any]
    control_net_lllites: list[Any]
    pipe: Any


def _load_networks(args, *, dtype, device, vae, text_encoders, unet, logger):
    if not args.network_module:
        return [], [], args.network_pre_calc

    networks = []
    network_default_muls = []
    network_pre_calc = args.network_pre_calc

    if args.network_merge:
        network_merge = len(args.network_module)
    elif args.network_merge_n_models:
        network_merge = args.network_merge_n_models
    else:
        network_merge = 0
    logger.info(f"network_merge: {network_merge}")

    for i, network_module in enumerate(args.network_module):
        logger.info("import network module: {network_module}")
        imported_module = importlib.import_module(network_module)

        network_mul = 1.0 if args.network_mul is None or len(args.network_mul) <= i else args.network_mul[i]
        net_kwargs = {}
        if args.network_args and i < len(args.network_args):
            for net_arg in args.network_args[i].split(";"):
                key, value = net_arg.split("=")
                net_kwargs[key] = value

        if args.network_weights is None or len(args.network_weights) <= i:
            raise ValueError("No weight. Weight is required.")

        network_weight = args.network_weights[i]
        logger.info(f"load network weights from: {network_weight}")

        if model_util.is_safetensors(network_weight) and args.network_show_meta:
            from safetensors.torch import safe_open

            with safe_open(network_weight, framework="pt") as f:
                metadata = f.metadata()
            if metadata is not None:
                logger.info(f"metadata for: {network_weight}: {metadata}")

        network, weights_sd = imported_module.create_network_from_weights(
            network_mul, network_weight, vae, text_encoders, unet, for_inference=True, **net_kwargs
        )
        if network is None:
            raise ValueError("Network module returned None during create_network_from_weights().")

        mergeable = network.is_mergeable()
        if network_merge and not mergeable:
            logger.warning("network is not mergiable. ignore merge option.")

        if not mergeable or i >= network_merge:
            network.apply_to(text_encoders, unet)
            info = network.load_state_dict(weights_sd, False)
            logger.info(f"weights are loaded: {info}")

            if args.opt_channels_last:
                import torch

                network.to(memory_format=torch.channels_last)
            network.to(dtype).to(device)

            if network_pre_calc:
                logger.info("backup original weights")
                network.backup_weights()

            networks.append(network)
            network_default_muls.append(network_mul)
        else:
            network.merge_to(text_encoders, unet, weights_sd, dtype, device)

    return networks, network_default_muls, network_pre_calc


def _load_upscaler(args, *, dtype, device, logger):
    if not args.highres_fix_upscaler:
        return None

    logger.info("import upscaler module: {args.highres_fix_upscaler}")
    imported_module = importlib.import_module(args.highres_fix_upscaler)

    us_kwargs = {}
    if args.highres_fix_upscaler_args:
        for net_arg in args.highres_fix_upscaler_args.split(";"):
            key, value = net_arg.split("=")
            us_kwargs[key] = value

    logger.info("create upscaler")
    upscaler = imported_module.create_upscaler(**us_kwargs)
    upscaler.to(dtype).to(device)
    return upscaler


def _load_control_nets(args, *, is_sdxl: bool, dtype, device, unet, logger):
    control_nets = []
    if args.control_net_models:
        if not is_sdxl:
            for i, model in enumerate(args.control_net_models):
                prep_type = None if not args.control_net_preps or len(args.control_net_preps) <= i else args.control_net_preps[i]
                weight = 1.0 if not args.control_net_multipliers or len(args.control_net_multipliers) <= i else args.control_net_multipliers[i]
                ratio = 1.0 if not args.control_net_ratios or len(args.control_net_ratios) <= i else args.control_net_ratios[i]

                ctrl_unet, ctrl_net = original_control_net.load_control_net(args.v2, unet, model)
                prep = original_control_net.load_preprocess(prep_type)
                control_nets.append(ControlNetInfo(ctrl_unet, ctrl_net, prep, weight, ratio))
        else:
            for i, model_file in enumerate(args.control_net_models):
                multiplier = 1.0 if not args.control_net_multipliers or len(args.control_net_multipliers) <= i else args.control_net_multipliers[i]
                ratio = 1.0 if not args.control_net_ratios or len(args.control_net_ratios) <= i else args.control_net_ratios[i]

                logger.info(f"loading SDXL ControlNet: {model_file}")
                from safetensors.torch import load_file

                state_dict = load_file(model_file)
                logger.info(f"Initializing SDXL ControlNet with multiplier: {multiplier}")
                with init_empty_weights():
                    control_net = SdxlControlNet(multiplier=multiplier)
                control_net.load_state_dict(state_dict)
                control_net.to(dtype).to(device)
                control_nets.append((control_net, ratio))

    control_net_lllites = []
    if args.control_net_lllite_models:
        for i, model_file in enumerate(args.control_net_lllite_models):
            logger.info(f"loading ControlNet-LLLite: {model_file}")
            from safetensors.torch import load_file

            state_dict = load_file(model_file)
            mlp_dim = None
            cond_emb_dim = None
            for key, value in state_dict.items():
                if mlp_dim is None and "down.0.weight" in key:
                    mlp_dim = value.shape[0]
                elif cond_emb_dim is None and "conditioning1.0" in key:
                    cond_emb_dim = value.shape[0] * 2
                if mlp_dim is not None and cond_emb_dim is not None:
                    break
            assert mlp_dim is not None and cond_emb_dim is not None, f"invalid control net: {model_file}"

            multiplier = 1.0 if not args.control_net_multipliers or len(args.control_net_multipliers) <= i else args.control_net_multipliers[i]
            ratio = 1.0 if not args.control_net_ratios or len(args.control_net_ratios) <= i else args.control_net_ratios[i]

            control_net_lllite = ControlNetLLLite(unet, cond_emb_dim, mlp_dim, multiplier=multiplier)
            control_net_lllite.apply_to()
            control_net_lllite.load_state_dict(state_dict)
            control_net_lllite.to(dtype).to(device)
            control_net_lllite.set_batch_cond_only(False, False)
            control_net_lllites.append((control_net_lllite, ratio))

    assert len(control_nets) == 0 or len(control_net_lllites) == 0, "ControlNet and ControlNet-LLLite cannot be used at the same time"
    return control_nets, control_net_lllites


def _apply_channels_last(args, *, text_encoders, vae, unet, networks, control_nets, control_net_lllites, logger):
    if not args.opt_channels_last:
        return

    import torch

    logger.info("set optimizing: channels last")
    for text_encoder in text_encoders:
        text_encoder.to(memory_format=torch.channels_last)
    vae.to(memory_format=torch.channels_last)
    unet.to(memory_format=torch.channels_last)
    for network in networks:
        network.to(memory_format=torch.channels_last)
    for control_net in control_nets:
        control_net.to(memory_format=torch.channels_last)
    for control_net_lllite in control_net_lllites:
        control_net_lllite.to(memory_format=torch.channels_last)


def _configure_pipeline(args, *, is_sdxl: bool, device, vae, text_encoders, tokenizers, unet, scheduler, control_nets, control_net_lllites, pipeline_like_cls, logger):
    pipe = pipeline_like_cls(
        is_sdxl,
        device,
        vae,
        text_encoders,
        tokenizers,
        unet,
        scheduler,
        args.clip_skip,
    )
    pipe.set_control_nets(control_nets)
    pipe.set_control_net_lllites(control_net_lllites)
    logger.info("pipeline is ready.")

    if args.diffusers_xformers:
        pipe.enable_xformers_memory_efficient_attention()

    if args.ds_depth_1 is not None:
        unet.set_deep_shrink(args.ds_depth_1, args.ds_timesteps_1, args.ds_depth_2, args.ds_timesteps_2, args.ds_ratio)

    if args.gradual_latent_timesteps is not None:
        if args.gradual_latent_unsharp_params:
            us_params = args.gradual_latent_unsharp_params.split(",")
            us_ksize, us_sigma, us_strength = [float(v) for v in us_params[:3]]
            us_target_x = True if len(us_params) <= 3 else bool(int(us_params[3]))
            us_ksize = int(us_ksize)
        else:
            us_ksize, us_sigma, us_strength, us_target_x = None, None, None, None

        gradual_latent = GradualLatent(
            args.gradual_latent_ratio,
            args.gradual_latent_timesteps,
            args.gradual_latent_every_n_steps,
            args.gradual_latent_ratio_step,
            args.gradual_latent_s_noise,
            us_ksize,
            us_sigma,
            us_strength,
            us_target_x,
        )
        pipe.set_gradual_latent(gradual_latent)

    return pipe


def prepare_gen_img_pipeline_setup(
    args,
    *,
    dtype,
    device,
    vae,
    text_encoders,
    tokenizers,
    unet,
    scheduler,
    is_sdxl: bool,
    pipeline_like_cls,
    logger,
):
    networks, network_default_muls, network_pre_calc = _load_networks(
        args,
        dtype=dtype,
        device=device,
        vae=vae,
        text_encoders=text_encoders,
        unet=unet,
        logger=logger,
    )
    upscaler = _load_upscaler(args, dtype=dtype, device=device, logger=logger)
    control_nets, control_net_lllites = _load_control_nets(
        args,
        is_sdxl=is_sdxl,
        dtype=dtype,
        device=device,
        unet=unet,
        logger=logger,
    )
    _apply_channels_last(
        args,
        text_encoders=text_encoders,
        vae=vae,
        unet=unet,
        networks=networks,
        control_nets=control_nets,
        control_net_lllites=control_net_lllites,
        logger=logger,
    )
    pipe = _configure_pipeline(
        args,
        is_sdxl=is_sdxl,
        device=device,
        vae=vae,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        unet=unet,
        scheduler=scheduler,
        control_nets=control_nets,
        control_net_lllites=control_net_lllites,
        pipeline_like_cls=pipeline_like_cls,
        logger=logger,
    )

    return PreparedGenImgPipelineSetup(
        networks=networks,
        network_default_muls=network_default_muls,
        network_pre_calc=network_pre_calc,
        upscaler=upscaler,
        control_nets=control_nets,
        control_net_lllites=control_net_lllites,
        pipe=pipe,
    )


__all__ = [
    "PreparedGenImgPipelineSetup",
    "prepare_gen_img_pipeline_setup",
]
