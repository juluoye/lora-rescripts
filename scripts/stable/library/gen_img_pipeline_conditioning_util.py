from __future__ import annotations

import PIL
import torch

import library.sdxl_train_util as sdxl_train_util
from library.gen_img_preprocess_util import preprocess_image


def prepare_clip_guidance_images(
    clip_guide_images,
    *,
    has_control_net_lllite,
    has_sdxl_control_net,
    device,
    dtype,
):
    if not (has_control_net_lllite or has_sdxl_control_net):
        return clip_guide_images

    if isinstance(clip_guide_images, PIL.Image.Image):
        clip_guide_images = [clip_guide_images]
    if isinstance(clip_guide_images[0], PIL.Image.Image):
        clip_guide_images = [preprocess_image(im) for im in clip_guide_images]
        clip_guide_images = torch.cat(clip_guide_images)
    if isinstance(clip_guide_images, list):
        clip_guide_images = torch.stack(clip_guide_images)

    return clip_guide_images.to(device, dtype=dtype)


def prepare_sdxl_vector_embeddings(
    *,
    device,
    dtype,
    batch_size,
    height,
    width,
    original_height,
    original_width,
    original_height_negative,
    original_width_negative,
    crop_top,
    crop_left,
    regional_network,
    text_pool,
    uncond_pool,
    init_image,
    clip_vision_model,
    clip_vision_processor,
    clip_vision_strength,
    do_classifier_free_guidance,
):
    if original_height is None:
        original_height = height
    if original_width is None:
        original_width = width
    if original_height_negative is None:
        original_height_negative = original_height
    if original_width_negative is None:
        original_width_negative = original_width
    if crop_top is None:
        crop_top = 0
    if crop_left is None:
        crop_left = 0

    emb1 = sdxl_train_util.get_timestep_embedding(torch.FloatTensor([original_height, original_width]).unsqueeze(0), 256)
    uc_emb1 = sdxl_train_util.get_timestep_embedding(
        torch.FloatTensor([original_height_negative, original_width_negative]).unsqueeze(0), 256
    )
    emb2 = sdxl_train_util.get_timestep_embedding(torch.FloatTensor([crop_top, crop_left]).unsqueeze(0), 256)
    emb3 = sdxl_train_util.get_timestep_embedding(torch.FloatTensor([height, width]).unsqueeze(0), 256)
    c_vector = torch.cat([emb1, emb2, emb3], dim=1).to(device, dtype=dtype).repeat(batch_size, 1)
    uc_vector = torch.cat([uc_emb1, emb2, emb3], dim=1).to(device, dtype=dtype).repeat(batch_size, 1)

    if regional_network:
        num_sub_prompts = len(text_pool) // batch_size
        text_pool = text_pool[num_sub_prompts - 1 :: num_sub_prompts]

    if init_image is not None and clip_vision_model is not None:
        vision_input = clip_vision_processor(init_image, return_tensors="pt", device=device)
        pixel_values = vision_input["pixel_values"].to(device, dtype=dtype)

        clip_vision_embeddings = clip_vision_model(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
        clip_vision_embeddings = clip_vision_embeddings.image_embeds

        if len(clip_vision_embeddings) == 1 and batch_size > 1:
            clip_vision_embeddings = clip_vision_embeddings.repeat((batch_size, 1))

        clip_vision_embeddings = clip_vision_embeddings * clip_vision_strength
        assert clip_vision_embeddings.shape == text_pool.shape, f"{clip_vision_embeddings.shape} != {text_pool.shape}"
        text_pool = clip_vision_embeddings

    c_vector = torch.cat([text_pool, c_vector], dim=1)
    if do_classifier_free_guidance:
        uc_vector = torch.cat([uncond_pool, uc_vector], dim=1)
        vector_embeddings = torch.cat([uc_vector, c_vector])
    else:
        vector_embeddings = c_vector

    return vector_embeddings, text_pool, uncond_pool


__all__ = ["prepare_clip_guidance_images", "prepare_sdxl_vector_embeddings"]
