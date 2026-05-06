from __future__ import annotations

import os
import random
import re
from glob import glob
from pathlib import Path
from typing import Optional, Tuple

from mikazuki.log import log
from mikazuki.utils.devices import get_xformers_status
from mikazuki.utils.train_utils import parse_boolish


def read_prompt_text_file(path: str) -> str:
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def get_sample_prompts(config: dict) -> Tuple[Optional[str], str]:
    if "sample_prompts" in config and "positive_prompts" not in config:
        return None, config["sample_prompts"]

    config_view = dict(config)
    train_data_dir = config_view["train_data_dir"]
    sub_dirs = [dir for dir in glob(os.path.join(train_data_dir, "*")) if os.path.isdir(dir)]
    sub_dirs_with_txt = [dir for dir in sub_dirs if glob(os.path.join(dir, "*.txt"))]
    root_txt_files = glob(os.path.join(train_data_dir, "*.txt"))

    enable_preview = parse_boolish(config_view.get("enable_preview", False))
    positive_prompts = config_view.get("positive_prompts", None)
    negative_prompts = config_view.get("negative_prompts", "")
    sample_width = config_view.get("sample_width", 512)
    sample_height = config_view.get("sample_height", 512)
    sample_cfg = config_view.get("sample_cfg", 7)
    sample_seed = config_view.get("sample_seed", 2333)
    sample_steps = config_view.get("sample_steps", 24)
    randomly_choice_prompt = parse_boolish(config_view.get("randomly_choice_prompt", False))
    random_prompt_include_subdirs = parse_boolish(config_view.get("random_prompt_include_subdirs", False))

    if not enable_preview:
        randomly_choice_prompt = False

    if randomly_choice_prompt:
        txt_files = []
        if random_prompt_include_subdirs:
            txt_files.extend(root_txt_files)
            prompt_source_dirs = [dir_path for dir_path in sub_dirs if glob(os.path.join(dir_path, "*.txt"))]
            if not prompt_source_dirs:
                prompt_source_dirs = [train_data_dir]
            for dir_path in prompt_source_dirs:
                txt_files.extend(glob(os.path.join(dir_path, "*.txt")))
        else:
            if root_txt_files:
                txt_files = root_txt_files
            else:
                if len(sub_dirs_with_txt) > 1:
                    raise ValueError("训练数据集下有多个包含 txt 标注的子文件夹，请启用“从所有子目录随机选择 Prompt”。")

                prompt_source_dir = train_data_dir
                if len(sub_dirs_with_txt) == 1:
                    prompt_source_dir = sub_dirs_with_txt[0]
                txt_files = glob(os.path.join(prompt_source_dir, "*.txt"))

        if not txt_files:
            raise ValueError("训练数据集路径没有 txt 文件")
        try:
            sample_prompt_file = random.choice(txt_files)
            with open(sample_prompt_file, "r", encoding="utf-8") as f:
                positive_prompts = f.read()
        except IOError:
            log.error(f"读取 {sample_prompt_file} 文件失败")

    def _normalize_single_preview_prompt_text(value) -> str:
        text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
        lines = [" ".join(line.split()) for line in text.split("\n") if line.strip()]
        return " ".join(lines)

    positive_prompts = _normalize_single_preview_prompt_text(positive_prompts)
    negative_prompts = _normalize_single_preview_prompt_text(negative_prompts)

    sample_prompt = f"{positive_prompts} --n {negative_prompts}  --w {sample_width} --h {sample_height} --l {sample_cfg}  --s {sample_steps}"
    normalized_seed = _normalize_preview_seed_value(sample_seed)
    if normalized_seed is not None:
        sample_prompt += f"  --d {normalized_seed}"
    return positive_prompts, sample_prompt


def build_sample_prompt_file_name(config: dict) -> str:
    base_name = str(config.get("output_name", "")).strip() or str(config.get("model_train_type", "")).strip() or "sample-prompts"
    safe_name = re.sub(r"[^0-9A-Za-z._-]+", "-", base_name).strip("._-") or "sample-prompts"
    return f"{safe_name}-sample-prompts.txt"


def build_prompt_preview_text(content: str, max_lines: int = 3) -> Tuple[str, int]:
    non_empty_lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not non_empty_lines:
        return "(prompt text is empty)", 0
    return "\n".join(non_empty_lines[:max_lines]), len(non_empty_lines)


def _has_prompt_cli_arg(line: str, flag: str) -> bool:
    return re.search(rf"(?:^|\s)--{re.escape(flag)}(?:\s|$)", line, flags=re.IGNORECASE) is not None


def should_use_inline_sample_prompts(sample_prompts: str, config: dict) -> bool:
    sample_prompts = str(sample_prompts or "").strip()
    if not sample_prompts:
        return False
    if os.path.isfile(sample_prompts):
        return True

    lines = [line.strip() for line in sample_prompts.splitlines() if line.strip()]
    if not lines:
        return False
    if len(lines) > 1:
        return True

    line = lines[0]
    if " --" in line or line.startswith("--"):
        return True

    return not str(config.get("positive_prompts", "") or "").strip()


def _normalize_preview_seed_value(value):
    if value is None:
        return None
    try:
        seed = int(value)
    except (TypeError, ValueError):
        return None
    return None if seed == 0 else seed


def enrich_inline_sample_prompts(sample_prompts: str, config: dict) -> str:
    lines = [line.strip() for line in str(sample_prompts or "").splitlines() if line.strip()]
    if not lines:
        return ""

    negative_prompt = str(config.get("negative_prompts", "") or "").strip()
    sample_sampler = str(config.get("sample_sampler", "") or "").strip()
    flow_shift = config.get("discrete_flow_shift", None)

    try:
        sample_width = int(config.get("sample_width", 0) or 0)
    except (TypeError, ValueError):
        sample_width = 0
    try:
        sample_height = int(config.get("sample_height", 0) or 0)
    except (TypeError, ValueError):
        sample_height = 0
    try:
        sample_cfg = float(config.get("sample_cfg", 0) or 0)
    except (TypeError, ValueError):
        sample_cfg = 0
    try:
        sample_steps = int(config.get("sample_steps", 0) or 0)
    except (TypeError, ValueError):
        sample_steps = 0

    sample_seed = _normalize_preview_seed_value(config.get("sample_seed", None))

    normalized_lines = []
    for line in lines:
        entry = line
        if negative_prompt and not _has_prompt_cli_arg(entry, "n"):
            entry += f" --n {negative_prompt}"
        if sample_width > 0 and not _has_prompt_cli_arg(entry, "w"):
            entry += f" --w {sample_width}"
        if sample_height > 0 and not _has_prompt_cli_arg(entry, "h"):
            entry += f" --h {sample_height}"
        if sample_cfg > 0 and not _has_prompt_cli_arg(entry, "l"):
            entry += f" --l {sample_cfg:g}"
        if sample_steps > 0 and not _has_prompt_cli_arg(entry, "s"):
            entry += f" --s {sample_steps}"
        if sample_seed is not None and not _has_prompt_cli_arg(entry, "d"):
            entry += f" --d {sample_seed}"
        if sample_sampler and not _has_prompt_cli_arg(entry, "ss"):
            entry += f" --ss {sample_sampler}"
        if flow_shift not in (None, "") and not _has_prompt_cli_arg(entry, "fs"):
            entry += f" --fs {flow_shift}"
        normalized_lines.append(entry)

    return "\n".join(normalized_lines)


def build_sample_prompt_record(config: dict) -> Optional[dict]:
    enable_preview = parse_boolish(config.get("enable_preview"))
    notes: list[str] = []
    warnings: list[str] = []

    if not enable_preview:
        notes.append("Preview images are currently disabled. This prompt will only be used after enable_preview is turned on.")

    prompt_file = str(config.get("prompt_file", "")).strip()
    if prompt_file:
        if not os.path.exists(prompt_file):
            raise ValueError(f"Prompt 文件 {prompt_file} 不存在，请检查路径。")

        content = read_prompt_text_file(prompt_file)
        preview, line_count = build_prompt_preview_text(content)
        return {
            "enabled": enable_preview,
            "source": "prompt_file",
            "detail": prompt_file,
            "preview": preview,
            "content": content,
            "line_count": line_count,
            "suggested_file_name": Path(prompt_file).name or build_sample_prompt_file_name(config),
            "warnings": warnings,
            "notes": notes,
        }

    legacy_sample_prompts = str(config.get("sample_prompts", "")).strip()
    if legacy_sample_prompts and should_use_inline_sample_prompts(legacy_sample_prompts, config):
        if str(config.get("positive_prompts", "")).strip():
            notes.append("多提示词轮换已启用，单提示词输入框会被忽略。")
        if os.path.isfile(legacy_sample_prompts):
            content = read_prompt_text_file(legacy_sample_prompts)
            preview, line_count = build_prompt_preview_text(content)
            return {
                "enabled": enable_preview,
                "source": "sample_prompts_file",
                "detail": legacy_sample_prompts,
                "preview": preview,
                "content": content,
                "line_count": line_count,
                "suggested_file_name": Path(legacy_sample_prompts).name or build_sample_prompt_file_name(config),
                "warnings": warnings,
                "notes": notes + ["Using sample_prompts file."],
            }

        enriched_sample_prompts = enrich_inline_sample_prompts(legacy_sample_prompts, config)
        preview, line_count = build_prompt_preview_text(enriched_sample_prompts)
        return {
            "enabled": enable_preview,
            "source": "sample_prompts_inline",
            "detail": "Inline multi-prompt rotation",
            "preview": preview,
            "content": enriched_sample_prompts,
            "line_count": line_count,
            "suggested_file_name": build_sample_prompt_file_name(config),
            "warnings": warnings,
            "notes": notes + ["Using inline sample_prompts text with current preview defaults merged in when missing."],
        }
    elif legacy_sample_prompts:
        notes.append("检测到单行 sample_prompts 旧值；当前将优先使用下方单提示词字段，避免被残留值覆盖。")

    has_positive_prompt = bool(str(config.get("positive_prompts", "") or "").strip())
    if not has_positive_prompt and not parse_boolish(config.get("randomly_choice_prompt")):
        return None

    config_copy = dict(config)
    _, sample_prompt = get_sample_prompts(config_copy)
    if sample_prompt is None:
        return None

    source = "generated"
    detail = "Current positive / negative prompt fields"
    if parse_boolish(config.get("randomly_choice_prompt")):
        if parse_boolish(config.get("random_prompt_include_subdirs")):
            source = "random_dataset_prompt_preview_all_subdirs"
            detail = "Random caption-derived preview from all dataset subdirectories"
        else:
            source = "random_dataset_prompt_preview"
            detail = "Random caption-derived preview from dataset"

    preview, line_count = build_prompt_preview_text(sample_prompt)
    return {
        "enabled": enable_preview,
        "source": source,
        "detail": detail,
        "preview": preview,
        "content": sample_prompt,
        "line_count": line_count,
        "suggested_file_name": build_sample_prompt_file_name(config),
        "warnings": warnings,
        "notes": notes,
    }


def simulate_attention_backend_fallback_warning(config: dict, gpu_ids) -> Optional[str]:
    if config.get("mem_eff_attn", False):
        return None
    if not config.get("xformers", False):
        return None

    xformers_info = get_xformers_status(gpu_ids)
    if xformers_info.get("selected_verified", xformers_info.get("verified", False)):
        return None

    if "sdpa" in config:
        return f"Current GPU/runtime would fall back from xformers to sdpa ({xformers_info['reason']})."
    return f"Current GPU/runtime would disable xformers ({xformers_info['reason']})."


def build_sample_prompt_preview(config: dict) -> Optional[dict]:
    record = build_sample_prompt_record(config)
    if not record:
        return None
    return {
        "source": record["source"],
        "detail": record["detail"],
        "preview": record["preview"],
        "warnings": record.get("warnings", []),
        "notes": record.get("notes", []),
    }


def build_disabled_sample_prompt_record(
    config: dict,
    *,
    source: str,
    detail: str,
    warnings: list[str] | None = None,
    notes: list[str] | None = None,
) -> dict:
    normalized_detail = str(detail or "当前训练配置未提供可用的预览提示词。").strip()
    return {
        "enabled": False,
        "source": source,
        "detail": normalized_detail,
        "preview": normalized_detail,
        "content": "",
        "line_count": 0,
        "suggested_file_name": build_sample_prompt_file_name(config),
        "warnings": [str(item) for item in (warnings or []) if str(item).strip()],
        "notes": [str(item) for item in (notes or []) if str(item).strip()],
    }
