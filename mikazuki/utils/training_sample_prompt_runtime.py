from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

from mikazuki.log import log
from mikazuki.utils import train_utils


def prepare_training_sample_prompt_config(
    config: dict,
    *,
    autosave_dir: Path,
    timestamp: str,
    direct_python_training: bool,
    skip_preview_prompt_prep: bool,
    get_sample_prompts: Callable[[dict], tuple[Optional[str], str]],
    should_use_inline_sample_prompts: Callable[[str, dict], bool],
    enrich_inline_sample_prompts: Callable[[str, dict], str],
    build_sample_prompt_file_name: Callable[[dict], str],
) -> Optional[str]:
    model_train_type = str(config.get("model_train_type", "") or "").strip().lower()
    preview_capable_direct_python_types = {"newbie-lora"}
    if skip_preview_prompt_prep or (
        direct_python_training and model_train_type not in preview_capable_direct_python_types
    ):
        for key in (
            "prompt_file",
            "sample_prompts",
            "positive_prompts",
            "negative_prompts",
            "randomly_choice_prompt",
            "random_prompt_include_subdirs",
        ):
            config.pop(key, None)
        return None

    prompt_file = str(config.get("prompt_file", "") or "").strip()
    inline_sample_prompts = str(config.get("sample_prompts", "") or "").strip()

    if prompt_file:
        if not os.path.exists(prompt_file):
            return f"Prompt 文件 {prompt_file} 不存在，请检查路径。"
        config["sample_prompts"] = prompt_file
        config.pop("prompt_file", None)
        return None

    if inline_sample_prompts and should_use_inline_sample_prompts(inline_sample_prompts, config):
        if os.path.isfile(inline_sample_prompts):
            config["sample_prompts"] = inline_sample_prompts
        else:
            sample_prompts_file = str(autosave_dir / build_sample_prompt_file_name(config))
            with open(sample_prompts_file, "w", encoding="utf-8") as f:
                normalized = enrich_inline_sample_prompts(inline_sample_prompts, config)
                f.write(normalized)
            config["sample_prompts"] = sample_prompts_file
            log.info(f"Wrote inline sample_prompts to file {sample_prompts_file}")
        config.pop("prompt_file", None)
        return None

    try:
        positive_prompt, sample_prompts_arg = get_sample_prompts(config=config)
        if positive_prompt is not None and train_utils.is_promopt_like(sample_prompts_arg):
            sample_prompts_file = str(autosave_dir / f"{timestamp}-prompt.txt")
            with open(sample_prompts_file, "w", encoding="utf-8") as f:
                f.write(sample_prompts_arg)
            config["sample_prompts"] = sample_prompts_file
            log.info(f"Wrote prompts to file {sample_prompts_file}")
    except ValueError as exc:
        log.error(f"Error while processing prompts: {exc}")
        return str(exc)

    config.pop("prompt_file", None)
    return None
