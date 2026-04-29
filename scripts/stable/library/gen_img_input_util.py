from __future__ import annotations

import glob
import importlib.util
import os
import random
import sys
from types import SimpleNamespace
from typing import Any, NamedTuple, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection


class PreparedGenImgInputs(NamedTuple):
    prompt_list: Optional[list[str]]
    prompter: Any
    init_images: Any
    mask_images: Any
    regional_network: bool
    guide_images: Any
    seed_random: Any


def load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Module '{module_name}' cannot be loaded from '{file_path}'")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_images(path: str, logger):
    if os.path.isfile(path):
        paths = [path]
    else:
        paths = (
            glob.glob(os.path.join(path, "*.png"))
            + glob.glob(os.path.join(path, "*.jpg"))
            + glob.glob(os.path.join(path, "*.jpeg"))
            + glob.glob(os.path.join(path, "*.webp"))
        )
        paths.sort()

    images = []
    for image_path in paths:
        image = Image.open(image_path)
        if image.mode != "RGB":
            logger.info(f"convert image to RGB from {image.mode}: {image_path}")
            image = image.convert("RGB")
        images.append(image)

    return images


def resize_images(images, size):
    resized = []
    for image in images:
        resized_image = image.resize(size, Image.Resampling.LANCZOS)
        if hasattr(image, "filename"):
            resized_image.filename = image.filename
        resized.append(resized_image)
    return resized


def prepare_gen_img_inputs(
    args,
    *,
    pipe,
    networks,
    highres_fix: bool,
    is_sdxl: bool,
    device,
    dtype,
    clip_vision_model_name: str,
    list_prompter_cls,
    logger,
):
    prompt_list = None
    if args.from_file is not None:
        logger.info(f"reading prompts from {args.from_file}")
        with open(args.from_file, "r", encoding="utf-8") as f:
            prompt_list = f.read().splitlines()
            prompt_list = [line for line in prompt_list if len(line.strip()) > 0 and line[0] != "#"]
        prompter = list_prompter_cls(prompt_list)
    elif args.from_module is not None:
        logger.info(f"reading prompts from module: {args.from_module}")
        prompt_module = load_module_from_path("prompt_module", args.from_module)
        prompter = prompt_module.get_prompter(args, pipe, networks)
    elif args.prompt is not None:
        prompter = list_prompter_cls([args.prompt])
    else:
        prompter = None

    if args.interactive:
        args.n_iter = 1

    if args.image_path is not None:
        logger.info(f"load image for img2img: {args.image_path}")
        init_images = load_images(args.image_path, logger)
        assert len(init_images) > 0, f"No image / 画像がありません: {args.image_path}"
        logger.info(f"loaded {len(init_images)} images for img2img")

        if args.clip_vision_strength is not None:
            logger.info(f"load CLIP Vision model: {clip_vision_model_name}")
            vision_model = CLIPVisionModelWithProjection.from_pretrained(clip_vision_model_name, projection_dim=1280)
            vision_model.to(device, dtype)
            processor = CLIPImageProcessor.from_pretrained(clip_vision_model_name)

            pipe.clip_vision_model = vision_model
            pipe.clip_vision_processor = processor
            pipe.clip_vision_strength = args.clip_vision_strength
            logger.info("CLIP Vision model loaded.")
    else:
        init_images = None

    if args.mask_path is not None:
        logger.info(f"load mask for inpainting: {args.mask_path}")
        mask_images = load_images(args.mask_path, logger)
        assert len(mask_images) > 0, f"No mask image / マスク画像がありません: {args.image_path}"
        logger.info(f"loaded {len(mask_images)} mask images for inpainting")
    else:
        mask_images = None

    if init_images is not None and prompter is None and not args.interactive:
        logger.info("get prompts from images' metadata")
        prompt_list = []
        for image in init_images:
            if "prompt" in image.text:
                prompt = image.text["prompt"]
                if "negative-prompt" in image.text:
                    prompt += " --n " + image.text["negative-prompt"]
                prompt_list.append(prompt)
        prompter = list_prompter_cls(prompt_list)

        expanded_init_images = []
        for image in init_images:
            expanded_init_images.extend([image] * args.images_per_prompt)
        init_images = expanded_init_images

        if mask_images is not None:
            expanded_mask_images = []
            for image in mask_images:
                expanded_mask_images.extend([image] * args.images_per_prompt)
            mask_images = expanded_mask_images

    if args.W is not None and args.H is not None:
        width, height = args.W, args.H
        if highres_fix:
            width = int(width * args.highres_fix_scale + 0.5)
            height = int(height * args.highres_fix_scale + 0.5)

        if init_images is not None:
            logger.info(f"resize img2img source images to {width}*{height}")
            init_images = resize_images(init_images, (width, height))
        if mask_images is not None:
            logger.info(f"resize img2img mask images to {width}*{height}")
            mask_images = resize_images(mask_images, (width, height))

    regional_network = False
    if networks and mask_images:
        regional_network = True
        logger.info("use mask as region")

        size = None
        for i, network in enumerate(networks):
            if (i < 3 and args.network_regional_mask_max_color_codes is None) or i < args.network_regional_mask_max_color_codes:
                np_mask = np.array(mask_images[0])

                if args.network_regional_mask_max_color_codes:
                    ch0 = (i + 1) & 1
                    ch1 = ((i + 1) >> 1) & 1
                    ch2 = ((i + 1) >> 2) & 1
                    np_mask = np.all(np_mask == np.array([ch0, ch1, ch2]) * 255, axis=2)
                    np_mask = np_mask.astype(np.uint8) * 255
                else:
                    np_mask = np_mask[:, :, i]
                size = np_mask.shape
            else:
                np_mask = np.full(size, 255, dtype=np.uint8)
            mask = torch.from_numpy(np_mask.astype(np.float32) / 255.0)
            network.set_region(i, i == len(networks) - 1, mask)
        mask_images = None

    if args.guide_image_path is not None:
        logger.info(f"load image for ControlNet guidance: {args.guide_image_path}")
        guide_images = []
        for path in args.guide_image_path:
            guide_images.extend(load_images(path, logger))

        logger.info(f"loaded {len(guide_images)} guide images for guidance")
        if len(guide_images) == 0:
            logger.warning(
                f"No guide image, use previous generated image. / ガイド画像がありません。直前に生成した画像を使います: {args.image_path}"
            )
            guide_images = None
    else:
        guide_images = None

    if args.seed is not None:
        if prompt_list and len(prompt_list) == 1 and args.images_per_prompt == 1:

            def fixed_seed(*_args, **_kwargs):
                return args.seed

            seed_random = SimpleNamespace(randint=fixed_seed)
        else:
            seed_random = random.Random(args.seed)
    else:
        seed_random = random.Random()

    if args.W is None:
        args.W = 1024 if is_sdxl else 512
    if args.H is None:
        args.H = 1024 if is_sdxl else 512

    return PreparedGenImgInputs(
        prompt_list=prompt_list,
        prompter=prompter,
        init_images=init_images,
        mask_images=mask_images,
        regional_network=regional_network,
        guide_images=guide_images,
        seed_random=seed_random,
    )


__all__ = [
    "PreparedGenImgInputs",
    "load_images",
    "prepare_gen_img_inputs",
    "resize_images",
]
