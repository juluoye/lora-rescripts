from __future__ import annotations

import glob
import os
from typing import Any, NamedTuple

import torch
from diffusers import StableDiffusionPipeline

import library.model_util as model_util
import library.sdxl_model_util as sdxl_model_util
import library.sdxl_train_util as sdxl_train_util
from library.original_unet import InferUNet2DConditionModel, UNet2DConditionModel
from library.sdxl_original_unet import InferSdxlUNet2DConditionModel
from library.strategy_sd import SdTokenizeStrategy


class PreparedGenImgModelSetup(NamedTuple):
    is_sdxl: bool
    device: Any
    vae_dtype: Any
    text_encoders: list[Any]
    tokenizers: list[Any]
    vae: Any
    unet: Any


def _resolve_checkpoint(args):
    if not os.path.exists(args.ckpt):
        files = glob.glob(args.ckpt)
        if len(files) == 1:
            args.ckpt = files[0]

    name_or_path = os.readlink(args.ckpt) if os.path.islink(args.ckpt) else args.ckpt
    use_stable_diffusion_format = os.path.isfile(name_or_path)
    return name_or_path, use_stable_diffusion_format


def _detect_sdxl(args, name_or_path: str, use_stable_diffusion_format: bool):
    is_sdxl = args.sdxl
    if not is_sdxl and not args.v1 and not args.v2:
        if use_stable_diffusion_format:
            is_sdxl = os.path.getsize(name_or_path) > 5.5 * 1024**3
        else:
            is_sdxl = os.path.isdir(os.path.join(name_or_path, "text_encoder_2"))
    return is_sdxl


def _load_base_models(args, *, dtype, is_sdxl: bool, use_stable_diffusion_format: bool, logger):
    tokenizer = None
    if is_sdxl:
        if args.clip_skip is None:
            args.clip_skip = 2

        (_, text_encoder1, text_encoder2, vae, unet, _, _) = sdxl_train_util._load_target_model(
            args.ckpt, args.vae, sdxl_model_util.MODEL_VERSION_SDXL_BASE_V1_0, dtype
        )
        unet = InferSdxlUNet2DConditionModel(unet)
        text_encoders = [text_encoder1, text_encoder2]
    else:
        if args.clip_skip is None:
            args.clip_skip = 2 if args.v2 else 1

        if use_stable_diffusion_format:
            logger.info("load StableDiffusion checkpoint")
            text_encoder, vae, unet = model_util.load_models_from_stable_diffusion_checkpoint(args.v2, args.ckpt)
        else:
            logger.info("load Diffusers pretrained models")
            loading_pipe = StableDiffusionPipeline.from_pretrained(args.ckpt, safety_checker=None, torch_dtype=dtype)
            text_encoder = loading_pipe.text_encoder
            vae = loading_pipe.vae
            unet = loading_pipe.unet
            tokenizer = loading_pipe.tokenizer
            del loading_pipe

            original_unet = UNet2DConditionModel(
                unet.config.sample_size,
                unet.config.attention_head_dim,
                unet.config.cross_attention_dim,
                unet.config.use_linear_projection,
                unet.config.upcast_attention,
            )
            original_unet.load_state_dict(unet.state_dict())
            unet = original_unet

        unet = InferUNet2DConditionModel(unet)
        text_encoders = [text_encoder]

        if args.vae is not None:
            vae = model_util.load_vae(args.vae, dtype)
            logger.info("additional VAE loaded")

    return text_encoders, vae, unet, tokenizer


def _load_tokenizers(args, *, is_sdxl: bool, use_stable_diffusion_format: bool, tokenizer, logger):
    logger.info("loading tokenizer")
    if is_sdxl:
        tokenizer1, tokenizer2 = sdxl_train_util.load_tokenizers(args)
        return [tokenizer1, tokenizer2]

    if use_stable_diffusion_format:
        tokenize_strategy = SdTokenizeStrategy(args.v2, max_length=None, tokenizer_cache_dir=args.tokenizer_cache_dir)
        tokenizer = tokenize_strategy.tokenizer
    return [tokenizer]


def _prepare_runtime_modules(args, *, dtype, device, vae, text_encoders, unet, logger):
    if args.vae_slices:
        from library.slicing_vae import SlicingAutoencoderKL

        slicing_vae = SlicingAutoencoderKL(
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
        slicing_vae.load_state_dict(vae.state_dict())
        vae = slicing_vae

    vae_dtype = dtype
    if args.no_half_vae:
        logger.info("set vae_dtype to float32")
        vae_dtype = torch.float32
    vae.to(vae_dtype).to(device)
    vae.eval()

    for text_encoder in text_encoders:
        text_encoder.to(dtype).to(device)
        text_encoder.eval()
    unet.to(dtype).to(device)
    unet.eval()
    return vae, text_encoders, unet, vae_dtype


def prepare_gen_img_model_setup(
    args,
    *,
    dtype,
    replace_unet_modules_fn,
    replace_vae_modules_fn,
    logger,
):
    name_or_path, use_stable_diffusion_format = _resolve_checkpoint(args)
    is_sdxl = _detect_sdxl(args, name_or_path, use_stable_diffusion_format)
    logger.info(f"SDXL: {is_sdxl}")

    text_encoders, vae, unet, tokenizer = _load_base_models(
        args,
        dtype=dtype,
        is_sdxl=is_sdxl,
        use_stable_diffusion_format=use_stable_diffusion_format,
        logger=logger,
    )

    if not args.diffusers_xformers:
        mem_eff = not (args.xformers or args.sdpa)
        replace_unet_modules_fn(unet, mem_eff, args.xformers, args.sdpa)
        replace_vae_modules_fn(vae, mem_eff, args.xformers, args.sdpa)

    tokenizers = _load_tokenizers(
        args,
        is_sdxl=is_sdxl,
        use_stable_diffusion_format=use_stable_diffusion_format,
        tokenizer=tokenizer,
        logger=logger,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae, text_encoders, unet, vae_dtype = _prepare_runtime_modules(
        args,
        dtype=dtype,
        device=device,
        vae=vae,
        text_encoders=text_encoders,
        unet=unet,
        logger=logger,
    )

    return PreparedGenImgModelSetup(
        is_sdxl=is_sdxl,
        device=device,
        vae_dtype=vae_dtype,
        text_encoders=text_encoders,
        tokenizers=tokenizers,
        vae=vae,
        unet=unet,
    )


__all__ = [
    "PreparedGenImgModelSetup",
    "prepare_gen_img_model_setup",
]
