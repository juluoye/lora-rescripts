from __future__ import annotations

import PIL
import torch


def _scale_and_round(value, scale: float):
    if value is None:
        return None
    return int(value * scale + 0.5)


def build_highres_first_stage_batch(batch, *, args, upscaler, batch_data_cls, batch_data_ext_cls):
    is_1st_latent = upscaler.support_latents() if upscaler else args.highres_fix_latents_upscaling
    batch_1st = []
    for _, base, ext in batch:
        width_1st = _scale_and_round(ext.width, args.highres_fix_scale)
        height_1st = _scale_and_round(ext.height, args.highres_fix_scale)
        width_1st = width_1st - width_1st % 32
        height_1st = height_1st - height_1st % 32

        ext_1st = batch_data_ext_cls(
            width_1st,
            height_1st,
            _scale_and_round(ext.original_width, args.highres_fix_scale),
            _scale_and_round(ext.original_height, args.highres_fix_scale),
            _scale_and_round(ext.original_width_negative, args.highres_fix_scale),
            _scale_and_round(ext.original_height_negative, args.highres_fix_scale),
            _scale_and_round(ext.crop_left, args.highres_fix_scale),
            _scale_and_round(ext.crop_top, args.highres_fix_scale),
            args.highres_fix_steps,
            ext.scale,
            ext.negative_scale,
            ext.strength if args.highres_fix_strength is None else args.highres_fix_strength,
            ext.network_muls,
            ext.num_sub_prompts,
        )
        batch_1st.append(batch_data_cls(is_1st_latent, base, ext_1st))
    return is_1st_latent, batch_1st


def upscale_first_stage_outputs(images_1st, *, batch, args, upscaler, vae, dtype):
    width_2nd, height_2nd = batch[0].ext.width, batch[0].ext.height

    if upscaler:
        lowreso_imgs = None if batch[0].return_latents else images_1st
        lowreso_latents = None if not batch[0].return_latents else images_1st

        batch_size = len(images_1st)
        vae_batch_size = (
            batch_size
            if args.vae_batch_size is None
            else (max(1, int(batch_size * args.vae_batch_size)) if args.vae_batch_size < 1 else args.vae_batch_size)
        )
        vae_batch_size = int(vae_batch_size)
        return upscaler.upscale(vae, lowreso_imgs, lowreso_latents, dtype, width_2nd, height_2nd, batch_size, vae_batch_size)

    if args.highres_fix_latents_upscaling:
        original_dtype = images_1st.dtype
        if images_1st.dtype == torch.bfloat16:
            images_1st = images_1st.to(torch.float)
        images_1st = torch.nn.functional.interpolate(
            images_1st,
            (batch[0].ext.height // 8, batch[0].ext.width // 8),
            mode="bicubic",
        )
        return images_1st.to(original_dtype)

    return [image.resize((width_2nd, height_2nd), resample=PIL.Image.LANCZOS) for image in images_1st]


def build_highres_second_stage_batch(batch, images_1st, *, batch_data_cls, batch_data_base_cls):
    batch_2nd = []
    for batch_data, image in zip(batch, images_1st):
        batch_data_2nd = batch_data_cls(
            False,
            batch_data_base_cls(*batch_data.base[0:3], batch_data.base.seed + 1, image, None, *batch_data.base[6:]),
            batch_data.ext,
        )
        batch_2nd.append(batch_data_2nd)
    return batch_2nd


__all__ = [
    "build_highres_first_stage_batch",
    "build_highres_second_stage_batch",
    "upscale_first_stage_outputs",
]
