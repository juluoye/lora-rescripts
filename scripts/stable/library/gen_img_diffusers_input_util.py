from __future__ import annotations

import glob
import os
from types import SimpleNamespace

import diffusers
import numpy as np
import torch
from PIL import Image

import library.model_util as model_util
from XTI_hijack import downblock_forward_XTI, unet_forward_XTI, upblock_forward_XTI


XTI_LAYERS = [
    "IN01",
    "IN02",
    "IN04",
    "IN05",
    "IN07",
    "IN08",
    "MID",
    "OUT03",
    "OUT04",
    "OUT05",
    "OUT06",
    "OUT07",
    "OUT08",
    "OUT09",
    "OUT10",
    "OUT11",
]


def load_diffusers_textual_inversion_inputs(*, args, pipe, tokenizer, text_encoder, logger):
    if args.XTI_embeddings:
        diffusers.models.UNet2DConditionModel.forward = unet_forward_XTI
        diffusers.models.unet_2d_blocks.CrossAttnDownBlock2D.forward = downblock_forward_XTI
        diffusers.models.unet_2d_blocks.CrossAttnUpBlock2D.forward = upblock_forward_XTI

    if args.textual_inversion_embeddings:
        token_ids_embeds = []
        for embeds_file in args.textual_inversion_embeddings:
            if model_util.is_safetensors(embeds_file):
                from safetensors.torch import load_file

                data = load_file(embeds_file)
            else:
                data = torch.load(embeds_file, map_location="cpu")

            if "string_to_param" in data:
                data = data["string_to_param"]
            embeds = next(iter(data.values()))

            if type(embeds) != torch.Tensor:
                raise ValueError(
                    f"weight file does not contains Tensor / 重みファイルのデータがTensorではありません: {embeds_file}"
                )

            num_vectors_per_token = embeds.size()[0]
            token_string = os.path.splitext(os.path.basename(embeds_file))[0]
            token_strings = [token_string] + [f"{token_string}{i+1}" for i in range(num_vectors_per_token - 1)]

            num_added_tokens = tokenizer.add_tokens(token_strings)
            assert (
                num_added_tokens == num_vectors_per_token
            ), f"tokenizer has same word to token string (filename). please rename the file / 指定した名前（ファイル名）のトークンが既に存在します。ファイルをリネームしてください: {embeds_file}"

            token_ids = tokenizer.convert_tokens_to_ids(token_strings)
            logger.info(f"Textual Inversion embeddings `{token_string}` loaded. Tokens are added: {token_ids}")
            assert (
                min(token_ids) == token_ids[0] and token_ids[-1] == token_ids[0] + len(token_ids) - 1
            ), f"token ids is not ordered"
            assert len(tokenizer) - 1 == token_ids[-1], f"token ids is not end of tokenize: {len(tokenizer)}"

            if num_vectors_per_token > 1:
                pipe.add_token_replacement(token_ids[0], token_ids)

            token_ids_embeds.append((token_ids, embeds))

        text_encoder.resize_token_embeddings(len(tokenizer))
        token_embeds = text_encoder.get_input_embeddings().weight.data
        for token_ids, embeds in token_ids_embeds:
            for token_id, embed in zip(token_ids, embeds):
                token_embeds[token_id] = embed

    if args.XTI_embeddings:
        token_ids_embeds_xti = []
        for embeds_file in args.XTI_embeddings:
            if model_util.is_safetensors(embeds_file):
                from safetensors.torch import load_file

                data = load_file(embeds_file)
            else:
                data = torch.load(embeds_file, map_location="cpu")
            if set(data.keys()) != set(XTI_LAYERS):
                raise ValueError("NOT XTI")
            embeds = torch.concat(list(data.values()))
            num_vectors_per_token = data["MID"].size()[0]

            token_string = os.path.splitext(os.path.basename(embeds_file))[0]
            token_strings = [token_string] + [f"{token_string}{i+1}" for i in range(num_vectors_per_token - 1)]

            num_added_tokens = tokenizer.add_tokens(token_strings)
            assert (
                num_added_tokens == num_vectors_per_token
            ), f"tokenizer has same word to token string (filename). please rename the file / 指定した名前（ファイル名）のトークンが既に存在します。ファイルをリネームしてください: {embeds_file}"

            token_ids = tokenizer.convert_tokens_to_ids(token_strings)
            logger.info(f"XTI embeddings `{token_string}` loaded. Tokens are added: {token_ids}")

            pipe.add_token_replacement(token_ids[0], token_ids)

            token_strings_xti = []
            for layer_name in XTI_LAYERS:
                token_strings_xti += [f"{t}_{layer_name}" for t in token_strings]
            tokenizer.add_tokens(token_strings_xti)
            token_ids_xti = tokenizer.convert_tokens_to_ids(token_strings_xti)
            token_ids_embeds_xti.append((token_ids_xti, embeds))
            for token_id in token_ids:
                token_xti_map = {}
                for i, layer_name in enumerate(XTI_LAYERS):
                    token_xti_map[layer_name] = token_id + (i + 1) * num_added_tokens
                pipe.add_token_replacement_XTI(token_id, token_xti_map)

            text_encoder.resize_token_embeddings(len(tokenizer))
            token_embeds = text_encoder.get_input_embeddings().weight.data
            for token_ids, embeds in token_ids_embeds_xti:
                for token_id, embed in zip(token_ids, embeds):
                    token_embeds[token_id] = embed


def prepare_diffusers_generation_inputs(*, args, highres_fix, networks, logger):
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
        logger.info("get prompts from images' meta data")
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

    prev_image = None
    if args.guide_image_path is not None:
        logger.info(f"load image for CLIP/VGG16/ControlNet guidance: {args.guide_image_path}")
        guide_images = []
        for guide_path in args.guide_image_path:
            guide_images.extend(load_images(guide_path))

        logger.info(f"loaded {len(guide_images)} guide images for guidance")
        if len(guide_images) == 0:
            logger.info(
                f"No guide image, use previous generated image. / ガイド画像がありません。直前に生成した画像を使います: {args.image_path}"
            )
            guide_images = None
    else:
        guide_images = None

    return SimpleNamespace(
        prompt_list=prompt_list,
        init_images=init_images,
        mask_images=mask_images,
        regional_network=regional_network,
        prev_image=prev_image,
        guide_images=guide_images,
    )


__all__ = [
    "load_diffusers_textual_inversion_inputs",
    "prepare_diffusers_generation_inputs",
]
