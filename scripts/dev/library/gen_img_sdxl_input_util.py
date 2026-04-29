from __future__ import annotations

import glob
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

import library.model_util as model_util


def load_sdxl_textual_inversion_inputs(
    *,
    args,
    pipe,
    tokenizer1,
    tokenizer2,
    text_encoder1,
    text_encoder2,
    logger,
):
    if not args.textual_inversion_embeddings:
        return

    token_ids_embeds1 = []
    token_ids_embeds2 = []
    for embeds_file in args.textual_inversion_embeddings:
        if model_util.is_safetensors(embeds_file):
            from safetensors.torch import load_file

            data = load_file(embeds_file)
        else:
            data = torch.load(embeds_file, map_location="cpu")

        if "string_to_param" in data:
            data = data["string_to_param"]

        embeds1 = data["clip_l"]
        embeds2 = data["clip_g"]

        num_vectors_per_token = embeds1.size()[0]
        token_string = os.path.splitext(os.path.basename(embeds_file))[0]
        token_strings = [token_string] + [f"{token_string}{i+1}" for i in range(num_vectors_per_token - 1)]

        num_added_tokens1 = tokenizer1.add_tokens(token_strings)
        num_added_tokens2 = tokenizer2.add_tokens(token_strings)
        assert num_added_tokens1 == num_vectors_per_token and num_added_tokens2 == num_vectors_per_token, (
            f"tokenizer has same word to token string (filename): {embeds_file}"
            + f" / 指定した名前（ファイル名）のトークンが既に存在します: {embeds_file}"
        )

        token_ids1 = tokenizer1.convert_tokens_to_ids(token_strings)
        token_ids2 = tokenizer2.convert_tokens_to_ids(token_strings)
        logger.info(f"Textual Inversion embeddings `{token_string}` loaded. Tokens are added: {token_ids1} and {token_ids2}")
        assert min(token_ids1) == token_ids1[0] and token_ids1[-1] == token_ids1[0] + len(token_ids1) - 1, "token ids1 is not ordered"
        assert min(token_ids2) == token_ids2[0] and token_ids2[-1] == token_ids2[0] + len(token_ids2) - 1, "token ids2 is not ordered"
        assert len(tokenizer1) - 1 == token_ids1[-1], f"token ids 1 is not end of tokenize: {len(tokenizer1)}"
        assert len(tokenizer2) - 1 == token_ids2[-1], f"token ids 2 is not end of tokenize: {len(tokenizer2)}"

        if num_vectors_per_token > 1:
            pipe.add_token_replacement(0, token_ids1[0], token_ids1)
            pipe.add_token_replacement(1, token_ids2[0], token_ids2)

        token_ids_embeds1.append((token_ids1, embeds1))
        token_ids_embeds2.append((token_ids2, embeds2))

    text_encoder1.resize_token_embeddings(len(tokenizer1))
    text_encoder2.resize_token_embeddings(len(tokenizer2))
    token_embeds1 = text_encoder1.get_input_embeddings().weight.data
    token_embeds2 = text_encoder2.get_input_embeddings().weight.data
    for token_ids, embeds in token_ids_embeds1:
        for token_id, embed in zip(token_ids, embeds):
            token_embeds1[token_id] = embed
    for token_ids, embeds in token_ids_embeds2:
        for token_id, embed in zip(token_ids, embeds):
            token_embeds2[token_id] = embed


def prepare_sdxl_generation_inputs(
    *,
    args,
    highres_fix,
    networks,
    pipe,
    device,
    dtype,
    clip_vision_model_name,
    logger,
):
    if args.from_file is not None:
        logger.info(f"reading prompts from {args.from_file}")
        with open(args.from_file, "r", encoding="utf-8") as f:
            prompt_list = f.read().splitlines()
            prompt_list = [line for line in prompt_list if len(line.strip()) > 0 and line[0] != "#"]
    elif args.prompt is not None:
        prompt_list = [args.prompt]
    else:
        prompt_list = []

    if args.interactive:
        args.n_iter = 1

    def load_images(path):
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

    if args.image_path is not None:
        logger.info(f"load image for img2img: {args.image_path}")
        init_images = load_images(args.image_path)
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
        mask_images = load_images(args.mask_path)
        assert len(mask_images) > 0, f"No mask image / マスク画像がありません: {args.image_path}"
        logger.info(f"loaded {len(mask_images)} mask images for inpainting")
    else:
        mask_images = None

    if init_images is not None and len(prompt_list) == 0 and not args.interactive:
        logger.info("get prompts from images' metadata")
        for image in init_images:
            if "prompt" in image.text:
                prompt = image.text["prompt"]
                if "negative-prompt" in image.text:
                    prompt += " --n " + image.text["negative-prompt"]
                prompt_list.append(prompt)

        repeated_init_images = []
        for image in init_images:
            repeated_init_images.extend([image] * args.images_per_prompt)
        init_images = repeated_init_images

        if mask_images is not None:
            repeated_mask_images = []
            for image in mask_images:
                repeated_mask_images.extend([image] * args.images_per_prompt)
            mask_images = repeated_mask_images

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
        for index, network in enumerate(networks):
            if (index < 3 and args.network_regional_mask_max_color_codes is None) or index < args.network_regional_mask_max_color_codes:
                np_mask = np.array(mask_images[0])

                if args.network_regional_mask_max_color_codes:
                    ch0 = (index + 1) & 1
                    ch1 = ((index + 1) >> 1) & 1
                    ch2 = ((index + 1) >> 2) & 1
                    np_mask = np.all(np_mask == np.array([ch0, ch1, ch2]) * 255, axis=2)
                    np_mask = np_mask.astype(np.uint8) * 255
                else:
                    np_mask = np_mask[:, :, index]
                size = np_mask.shape
            else:
                np_mask = np.full(size, 255, dtype=np.uint8)
            mask = torch.from_numpy(np_mask.astype(np.float32) / 255.0)
            network.set_region(index, index == len(networks) - 1, mask)
        mask_images = None

    if args.guide_image_path is not None:
        logger.info(f"load image for ControlNet guidance: {args.guide_image_path}")
        guide_images = []
        for guide_path in args.guide_image_path:
            guide_images.extend(load_images(guide_path))

        logger.info(f"loaded {len(guide_images)} guide images for guidance")
        if len(guide_images) == 0:
            logger.warning(
                f"No guide image, use previous generated image. / ガイド画像がありません。直前に生成した画像を使います: {args.image_path}"
            )
            guide_images = None
    else:
        guide_images = None

    if args.seed is not None:
        random.seed(args.seed)
        predefined_seeds = [random.randint(0, 0x7FFFFFFF) for _ in range(args.n_iter * len(prompt_list) * args.images_per_prompt)]
        if len(predefined_seeds) == 1:
            predefined_seeds[0] = args.seed
    else:
        predefined_seeds = None

    return SimpleNamespace(
        prompt_list=prompt_list,
        init_images=init_images,
        mask_images=mask_images,
        guide_images=guide_images,
        predefined_seeds=predefined_seeds,
        regional_network=regional_network,
    )


__all__ = [
    "load_sdxl_textual_inversion_inputs",
    "prepare_sdxl_generation_inputs",
]
