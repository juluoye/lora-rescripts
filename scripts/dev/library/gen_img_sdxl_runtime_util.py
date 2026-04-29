from __future__ import annotations

import glob
import importlib
import os
from types import SimpleNamespace

import diffusers
import torch

import library.model_util as model_util
import library.sdxl_model_util as sdxl_model_util
import library.sdxl_train_util as sdxl_train_util
from library.device_utils import get_preferred_device
from library.sdxl_original_unet import InferSdxlUNet2DConditionModel
from library.utils import EulerAncestralDiscreteSchedulerGL
from networks.control_net_lllite import ControlNetLLLite


class NoiseManager:
    def __init__(self, logger):
        self.logger = logger
        self.sampler_noises = None
        self.sampler_noise_index = 0

    def reset_sampler_noises(self, noises):
        self.sampler_noise_index = 0
        self.sampler_noises = noises

    def randn(self, shape, device=None, dtype=None, layout=None, generator=None):
        if self.sampler_noises is not None and self.sampler_noise_index < len(self.sampler_noises):
            noise = self.sampler_noises[self.sampler_noise_index]
            if shape != noise.shape:
                noise = None
        else:
            noise = None

        if noise is None:
            self.logger.warning(f"unexpected noise request: {self.sampler_noise_index}, {shape}")
            noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)

        self.sampler_noise_index += 1
        return noise


class TorchRandReplacer:
    def __init__(self, noise_manager):
        self.noise_manager = noise_manager

    def __getattr__(self, item):
        if item == "randn":
            return self.noise_manager.randn
        if hasattr(torch, item):
            return getattr(torch, item)
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))


def prepare_sdxl_runtime(
    *,
    args,
    dtype,
    logger,
    replace_unet_modules_fn,
    replace_vae_modules_fn,
    pipeline_like_cls,
    scheduler_linear_start,
    scheduler_linear_end,
    scheduler_timesteps,
    scheduler_schedule,
):
    if not os.path.isfile(args.ckpt):
        files = glob.glob(args.ckpt)
        if len(files) == 1:
            args.ckpt = files[0]

    (_, text_encoder1, text_encoder2, vae, unet, _, _) = sdxl_train_util._load_target_model(
        args.ckpt, args.vae, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, dtype
    )
    unet = InferSdxlUNet2DConditionModel(unet)

    if not args.diffusers_xformers:
        mem_eff = not (args.xformers or args.sdpa)
        replace_unet_modules_fn(unet, mem_eff, args.xformers, args.sdpa)
        replace_vae_modules_fn(vae, mem_eff, args.xformers, args.sdpa)

    logger.info("loading tokenizer")
    tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)

    sched_init_args = {}
    has_steps_offset = True
    has_clip_sample = True
    scheduler_num_noises_per_step = 1

    if args.sampler == "ddim":
        scheduler_cls = diffusers.DDIMScheduler
        scheduler_module = diffusers.schedulers.scheduling_ddim
    elif args.sampler == "ddpm":
        scheduler_cls = diffusers.DDPMScheduler
        scheduler_module = diffusers.schedulers.scheduling_ddpm
    elif args.sampler == "pndm":
        scheduler_cls = diffusers.PNDMScheduler
        scheduler_module = diffusers.schedulers.scheduling_pndm
        has_clip_sample = False
    elif args.sampler == "lms" or args.sampler == "k_lms":
        scheduler_cls = diffusers.LMSDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_lms_discrete
        has_clip_sample = False
    elif args.sampler == "euler" or args.sampler == "k_euler":
        scheduler_cls = diffusers.EulerDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_euler_discrete
        has_clip_sample = False
    elif args.sampler == "euler_a" or args.sampler == "k_euler_a":
        scheduler_cls = EulerAncestralDiscreteSchedulerGL
        scheduler_module = diffusers.schedulers.scheduling_euler_ancestral_discrete
        has_clip_sample = False
    elif args.sampler == "dpmsolver" or args.sampler == "dpmsolver++":
        scheduler_cls = diffusers.DPMSolverMultistepScheduler
        sched_init_args["algorithm_type"] = args.sampler
        scheduler_module = diffusers.schedulers.scheduling_dpmsolver_multistep
        has_clip_sample = False
    elif args.sampler == "dpmsingle":
        scheduler_cls = diffusers.DPMSolverSinglestepScheduler
        scheduler_module = diffusers.schedulers.scheduling_dpmsolver_singlestep
        has_clip_sample = False
        has_steps_offset = False
    elif args.sampler == "heun":
        scheduler_cls = diffusers.HeunDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_heun_discrete
        has_clip_sample = False
    elif args.sampler == "dpm_2" or args.sampler == "k_dpm_2":
        scheduler_cls = diffusers.KDPM2DiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_discrete
        has_clip_sample = False
    elif args.sampler == "dpm_2_a" or args.sampler == "k_dpm_2_a":
        scheduler_cls = diffusers.KDPM2AncestralDiscreteScheduler
        scheduler_module = diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete
        scheduler_num_noises_per_step = 2
        has_clip_sample = False
    else:
        raise ValueError(f"Unknown sampler: {args.sampler}")

    if has_steps_offset:
        sched_init_args["steps_offset"] = 1
    if has_clip_sample:
        sched_init_args["clip_sample"] = False

    noise_manager = NoiseManager(logger)
    if scheduler_module is not None:
        scheduler_module.torch = TorchRandReplacer(noise_manager)

    scheduler = scheduler_cls(
        num_train_timesteps=scheduler_timesteps,
        beta_start=scheduler_linear_start,
        beta_end=scheduler_linear_end,
        beta_schedule=scheduler_schedule,
        **sched_init_args,
    )

    device = get_preferred_device()

    if args.vae_slices:
        from library.slicing_vae import SlicingAutoencoderKL

        sli_vae = SlicingAutoencoderKL(
            act_fn="silu",
            block_out_channels=(128, 256, 512, 512),
            down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"],
            in_channels=3,
            latent_channels=4,
            layers_per_block=2,
            norm_num_groups=32,
            out_channels=3,
            sample_size=512,
            up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"],
            num_slices=args.vae_slices,
        )
        sli_vae.load_state_dict(vae.state_dict())
        vae = sli_vae
        del sli_vae

    vae_dtype = dtype
    if args.no_half_vae:
        logger.info("set vae_dtype to float32")
        vae_dtype = torch.float32
    vae.to(vae_dtype).to(device)
    vae.eval()

    text_encoder1.to(dtype).to(device)
    text_encoder2.to(dtype).to(device)
    unet.to(dtype).to(device)
    text_encoder1.eval()
    text_encoder2.eval()
    unet.eval()

    if args.network_module:
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
            logger.info(f"import network module: {network_module}")
            imported_module = importlib.import_module(network_module)

            network_mul = 1.0 if args.network_mul is None or len(args.network_mul) <= i else args.network_mul[i]

            net_kwargs = {}
            if args.network_args and i < len(args.network_args):
                network_args = args.network_args[i].split(";")
                for net_arg in network_args:
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
                network_mul, network_weight, vae, [text_encoder1, text_encoder2], unet, for_inference=True, **net_kwargs
            )
            if network is None:
                return None

            mergeable = network.is_mergeable()
            if network_merge and not mergeable:
                logger.warning("network is not mergiable. ignore merge option.")

            if not mergeable or i >= network_merge:
                network.apply_to([text_encoder1, text_encoder2], unet)
                info = network.load_state_dict(weights_sd, False)
                logger.info(f"weights are loaded: {info}")

                if args.opt_channels_last:
                    network.to(memory_format=torch.channels_last)
                network.to(dtype).to(device)

                if network_pre_calc:
                    logger.info("backup original weights")
                    network.backup_weights()

                networks.append(network)
                network_default_muls.append(network_mul)
            else:
                network.merge_to([text_encoder1, text_encoder2], unet, weights_sd, dtype, device)
    else:
        networks = []
        network_default_muls = []
        network_pre_calc = False

    upscaler = None
    if args.highres_fix_upscaler:
        logger.info(f"import upscaler module: {args.highres_fix_upscaler}")
        imported_module = importlib.import_module(args.highres_fix_upscaler)

        us_kwargs = {}
        if args.highres_fix_upscaler_args:
            for net_arg in args.highres_fix_upscaler_args.split(";"):
                key, value = net_arg.split("=")
                us_kwargs[key] = value

        logger.info("create upscaler")
        upscaler = imported_module.create_upscaler(**us_kwargs)
        upscaler.to(dtype).to(device)

    control_nets = []
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

            control_net = ControlNetLLLite(unet, cond_emb_dim, mlp_dim, multiplier=multiplier)
            control_net.apply_to()
            control_net.load_state_dict(state_dict)
            control_net.to(dtype).to(device)
            control_net.set_batch_cond_only(False, False)
            control_nets.append((control_net, ratio))

    if args.opt_channels_last:
        logger.info("set optimizing: channels last")
        text_encoder1.to(memory_format=torch.channels_last)
        text_encoder2.to(memory_format=torch.channels_last)
        vae.to(memory_format=torch.channels_last)
        unet.to(memory_format=torch.channels_last)
        if networks:
            for network in networks:
                network.to(memory_format=torch.channels_last)
        for control_net, _ratio in control_nets:
            control_net.to(memory_format=torch.channels_last)

    pipe = pipeline_like_cls(
        device,
        vae,
        [text_encoder1, text_encoder2],
        [tokenizer1, tokenizer2],
        unet,
        scheduler,
        args.clip_skip,
    )
    pipe.set_control_nets(control_nets)

    return SimpleNamespace(
        text_encoder1=text_encoder1,
        text_encoder2=text_encoder2,
        vae=vae,
        vae_dtype=vae_dtype,
        unet=unet,
        tokenizer1=tokenizer1,
        tokenizer2=tokenizer2,
        scheduler=scheduler,
        scheduler_num_noises_per_step=scheduler_num_noises_per_step,
        noise_manager=noise_manager,
        device=device,
        networks=networks,
        network_default_muls=network_default_muls,
        network_pre_calc=network_pre_calc,
        upscaler=upscaler,
        control_nets=control_nets,
        pipe=pipe,
    )


__all__ = ["prepare_sdxl_runtime"]
