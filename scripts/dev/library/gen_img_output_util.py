from __future__ import annotations

import os
import time

import numpy as np
from PIL.PngImagePlugin import PngInfo


def save_generated_images(
    args,
    *,
    images,
    prompts,
    negative_prompts,
    seeds,
    clip_prompts,
    raw_prompts,
    filenames,
    highres_fix: bool,
    highres_1st: bool,
    steps,
    scale,
    negative_scale,
    is_sdxl: bool,
    original_height,
    original_width,
    original_height_negative,
    original_width_negative,
    crop_top,
    crop_left,
    init_images,
    step_first: int,
    logger,
):
    highres_prefix = ("0" if highres_1st else "1") if highres_fix else ""
    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())

    for i, (image, prompt, negative_prompt, seed, clip_prompt, raw_prompt, filename) in enumerate(
        zip(images, prompts, negative_prompts, seeds, clip_prompts, raw_prompts, filenames)
    ):
        if highres_fix:
            seed -= 1
        metadata = PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("seed", str(seed))
        metadata.add_text("sampler", args.sampler)
        metadata.add_text("steps", str(steps))
        metadata.add_text("scale", str(scale))
        if negative_prompt is not None:
            metadata.add_text("negative-prompt", negative_prompt)
        if negative_scale is not None:
            metadata.add_text("negative-scale", str(negative_scale))
        if clip_prompt is not None:
            metadata.add_text("clip-prompt", clip_prompt)
        if raw_prompt is not None:
            metadata.add_text("raw-prompt", raw_prompt)
        if is_sdxl:
            metadata.add_text("original-height", str(original_height))
            metadata.add_text("original-width", str(original_width))
            metadata.add_text("original-height-negative", str(original_height_negative))
            metadata.add_text("original-width-negative", str(original_width_negative))
            metadata.add_text("crop-top", str(crop_top))
            metadata.add_text("crop-left", str(crop_left))

        if filename is not None:
            output_filename = filename
        else:
            if args.use_original_file_name and init_images is not None:
                if type(init_images) is list:
                    output_filename = os.path.splitext(os.path.basename(init_images[i % len(init_images)].filename))[0] + ".png"
                else:
                    output_filename = os.path.splitext(os.path.basename(init_images.filename))[0] + ".png"
            elif args.sequential_file_name:
                output_filename = f"im_{highres_prefix}{step_first + i + 1:06d}.png"
            else:
                output_filename = f"im_{ts_str}_{highres_prefix}{i:03d}_{seed}.png"

        output_path = os.path.join(args.outdir, output_filename)
        if output_filename.endswith(".webp"):
            image.save(output_path, pnginfo=metadata, quality=100)
        else:
            image.save(output_path, pnginfo=metadata)


def preview_generated_images(args, *, highres_1st: bool, prompts, images, logger):
    if args.no_preview or highres_1st or not args.interactive:
        return

    try:
        import cv2

        for prompt, image in zip(prompts, images):
            cv2.imshow(prompt[:128], np.array(image)[:, :, ::-1])
            cv2.waitKey()
            cv2.destroyAllWindows()
    except ImportError:
        logger.warning("opencv-python is not installed, cannot preview / opencv-pythonがインストールされていないためプレビューできません")


__all__ = [
    "preview_generated_images",
    "save_generated_images",
]
