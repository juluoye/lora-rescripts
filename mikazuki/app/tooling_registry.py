from __future__ import annotations

import hashlib
import re
from pathlib import Path

import toml

from mikazuki import launch_utils
from mikazuki.utils.runtime_mode import (
    infer_attention_runtime_mode,
    is_amd_rocm_runtime,
    is_intel_xpu_runtime,
)


AVAILABLE_SCRIPTS = [
    "networks/extract_lora_from_models.py",
    "networks/extract_lora_from_dylora.py",
    "networks/merge_lora.py",
    "networks/sdxl_merge_lora.py",
    "networks/svd_merge_lora.py",
    "networks/flux_extract_lora.py",
    "networks/resize_lora.py",
    "networks/lora_interrogator.py",
    "networks/flux_merge_lora.py",
    "networks/convert_flux_lora.py",
    "networks/convert_hunyuan_image_lora_to_comfy.py",
    "networks/convert_anima_lora_to_comfy.py",
    "networks/check_lora_weights.py",
    "tools/merge_models.py",
    "tools/merge_sd3_safetensors.py",
    "tools/convert_diffusers_to_flux.py",
    "tools/convert_diffusers20_original_sd.py",
    "tools/show_metadata.py",
    "tools/resize_images_to_resolution.py",
    "tools/canny.py",
    "tools/detect_face_rotate.py",
    "tools/latent_upscaler.py",
]

SCRIPT_POSITIONAL_ARGS = {
    "networks/convert_hunyuan_image_lora_to_comfy.py": ["src_path", "dst_path"],
    "networks/convert_anima_lora_to_comfy.py": ["src_path", "dst_path"],
    "networks/check_lora_weights.py": ["file"],
    "tools/resize_images_to_resolution.py": ["src_img_folder", "dst_img_folder"],
    "tools/convert_diffusers20_original_sd.py": ["model_to_load", "model_to_save"],
}

AVAILABLE_SCHEMAS: list[dict] = []
AVAILABLE_PRESETS: list[dict] = []

EXPERIMENTAL_SAFE_OPTIMIZERS = (
    "AdamW",
    "AdaFactor",
    "Lion",
    "SGDNesterov",
)
EXPERIMENTAL_RUNTIME_SCHEMA_OPTIMIZER_PATTERN = re.compile(
    r'(?P<indent>[ \t]*)optimizer_type:\s*Schema\.union\(\[(?P<body>.*?)\]\)\.default\("(?P<default>[^"]+)"\)\.description\("优化器设置"\)',
    re.DOTALL,
)


def apply_runtime_schema_overrides(content: str) -> str:
    runtime_name = infer_attention_runtime_mode()
    if not (is_amd_rocm_runtime(runtime_name) or is_intel_xpu_runtime(runtime_name)):
        return content

    runtime_label = "AMD ROCm" if is_amd_rocm_runtime(runtime_name) else "Intel XPU"

    def replace_optimizer_block(match: re.Match[str]) -> str:
        indent = match.group("indent")
        option_indent = indent + "    "
        options = "\n".join(f'{option_indent}"{item}",' for item in EXPERIMENTAL_SAFE_OPTIMIZERS)
        return (
            f'{indent}optimizer_type: Schema.union([\n'
            f"{options}\n"
            f'{indent}]).default("AdamW").description("优化器设置（{runtime_label} 实验运行时仅显示已验证选项）")'
        )

    return EXPERIMENTAL_RUNTIME_SCHEMA_OPTIMIZER_PATTERN.sub(replace_optimizer_block, content)


async def load_schemas() -> None:
    AVAILABLE_SCHEMAS.clear()

    schema_dir = launch_utils.base_dir_path() / "mikazuki" / "schema"
    schemas = sorted(p for p in schema_dir.iterdir() if p.is_file() and p.suffix == ".ts")

    def lambda_hash(value: str) -> str:
        return hashlib.md5(value.encode()).hexdigest()

    for schema_path in schemas:
        with open(schema_path, encoding="utf-8") as f:
            content = apply_runtime_schema_overrides(f.read())
            AVAILABLE_SCHEMAS.append({
                "name": schema_path.stem,
                "schema": content,
                "hash": lambda_hash(content),
            })


async def load_presets() -> None:
    AVAILABLE_PRESETS.clear()

    preset_dir = launch_utils.base_dir_path() / "config" / "presets"
    if not preset_dir.exists():
        return
    presets = sorted(p for p in preset_dir.iterdir() if p.is_file() and p.suffix == ".toml")

    for preset_path in presets:
        with open(preset_path, encoding="utf-8") as f:
            content = f.read()
            AVAILABLE_PRESETS.append(toml.loads(content))


def resolve_script_path(script_name: str) -> Path | None:
    repo_root = launch_utils.base_dir_path()
    script_path = repo_root / "scripts" / script_name
    if script_path.exists():
        return script_path

    for candidate_root in ("stable", "dev"):
        candidate = repo_root / "scripts" / candidate_root / script_name
        if candidate.exists():
            return candidate

    return None
