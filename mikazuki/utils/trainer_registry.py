from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional

from mikazuki.utils.direct_trainers import (
    build_aesthetic_scorer_preflight_summary,
    build_aesthetic_scorer_start_warnings,
    build_newbie_start_warnings,
    build_yolo_preflight_summary,
    build_yolo_start_warnings,
    validate_aesthetic_scorer_runtime_config,
    validate_newbie_runtime_config,
    validate_yolo_runtime_config,
)


TrainerConfigValidator = Callable[[dict], Optional[str]]
TrainerWarningBuilder = Callable[[dict], list[str]]
TrainerPreflightBuilder = Callable[[dict, list[str], list[str], list[str]], Optional[dict]]


AMD_ANIMA_LORA_TRAINER_FILE = "./scripts/stable/anima_train_network_amd.py"
INTEL_ANIMA_LORA_TRAINER_FILE = "./scripts/stable/anima_train_network_intel.py"


@dataclass(frozen=True)
class TrainerDefinition:
    train_type: str
    trainer_file: str
    direct_python: bool = False
    direct_cli_args: tuple[str, ...] = ()
    direct_launch_summary: str | None = None
    skip_model_validation: bool = False
    config_validator: TrainerConfigValidator | None = None
    start_warning_builder: TrainerWarningBuilder | None = None
    preflight_builder: TrainerPreflightBuilder | None = None
    preflight_handles_resume: bool = False
    allow_dataset_config_without_train_data_dir: bool = False
    allow_dataset_class_without_train_data_dir: bool = False


TRAINER_REGISTRY = {
    "sd-lora": TrainerDefinition(
        "sd-lora",
        "./scripts/stable/train_network.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "sdxl-lora": TrainerDefinition(
        "sdxl-lora",
        "./scripts/stable/sdxl_train_network.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "yolo": TrainerDefinition(
        "yolo",
        "./scripts/stable/yolo_train.py",
        direct_python=True,
        direct_launch_summary="YOLO 训练直接由 Ultralytics 启动，不走 accelerate 分布式包装。",
        config_validator=validate_yolo_runtime_config,
        start_warning_builder=build_yolo_start_warnings,
        preflight_builder=build_yolo_preflight_summary,
        preflight_handles_resume=True,
    ),
    "aesthetic-scorer": TrainerDefinition(
        "aesthetic-scorer",
        "./scripts/stable/aesthetic_scorer_train.py",
        direct_python=True,
        direct_launch_summary="美学评分训练直接由独立 Python 训练器启动，不走 accelerate 分布式包装。",
        skip_model_validation=True,
        config_validator=validate_aesthetic_scorer_runtime_config,
        start_warning_builder=build_aesthetic_scorer_start_warnings,
        preflight_builder=build_aesthetic_scorer_preflight_summary,
        preflight_handles_resume=True,
    ),
    "newbie-lora": TrainerDefinition(
        "newbie-lora",
        "./scripts/stable/newbie_lora_train.py",
        direct_python=True,
        direct_cli_args=("--execute", "--phase", "full"),
        direct_launch_summary=(
            "Newbie 训练当前由独立 Python 训练器直接启动，默认执行 full phase："
            "缺缓存时会先补 cache，再进入正式训练。"
        ),
        config_validator=validate_newbie_runtime_config,
        start_warning_builder=build_newbie_start_warnings,
        preflight_handles_resume=True,
    ),
    "sd-dreambooth": TrainerDefinition(
        "sd-dreambooth",
        "./scripts/stable/train_db.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "sdxl-finetune": TrainerDefinition(
        "sdxl-finetune",
        "./scripts/stable/sdxl_train.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "sd-controlnet": TrainerDefinition(
        "sd-controlnet",
        "./scripts/stable/train_control_net.py",
        allow_dataset_config_without_train_data_dir=True,
    ),
    "sdxl-controlnet": TrainerDefinition(
        "sdxl-controlnet",
        "./scripts/stable/sdxl_train_control_net.py",
        allow_dataset_config_without_train_data_dir=True,
    ),
    "sdxl-controlnet-lllite": TrainerDefinition(
        "sdxl-controlnet-lllite",
        "./scripts/stable/sdxl_train_control_net_lllite.py",
        allow_dataset_config_without_train_data_dir=True,
    ),
    "flux-controlnet": TrainerDefinition(
        "flux-controlnet",
        "./scripts/stable/flux_train_control_net.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "sd-textual-inversion": TrainerDefinition(
        "sd-textual-inversion",
        "./scripts/stable/train_textual_inversion.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "sd-textual-inversion-xti": TrainerDefinition(
        "sd-textual-inversion-xti",
        "./scripts/stable/train_textual_inversion_XTI.py",
        allow_dataset_config_without_train_data_dir=True,
    ),
    "sdxl-textual-inversion": TrainerDefinition("sdxl-textual-inversion", "./scripts/stable/sdxl_train_textual_inversion.py"),
    "sd3-lora": TrainerDefinition(
        "sd3-lora",
        "./scripts/dev/sd3_train_network.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "sd3-finetune": TrainerDefinition(
        "sd3-finetune",
        "./scripts/stable/sd3_train.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "flux-lora": TrainerDefinition(
        "flux-lora",
        "./scripts/dev/flux_train_network.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "flux-finetune": TrainerDefinition(
        "flux-finetune",
        "./scripts/dev/flux_train.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "lumina-lora": TrainerDefinition(
        "lumina-lora",
        "./scripts/stable/lumina_train_network.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "lumina2-lora": TrainerDefinition(
        "lumina2-lora",
        "./scripts/stable/lumina_train_network.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "lumina-finetune": TrainerDefinition(
        "lumina-finetune",
        "./scripts/stable/lumina_train.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "hunyuan-image-lora": TrainerDefinition(
        "hunyuan-image-lora",
        "./scripts/stable/hunyuan_image_train_network.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "anima-lora": TrainerDefinition(
        "anima-lora",
        "./scripts/stable/anima_train_network.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
    "anima-finetune": TrainerDefinition(
        "anima-finetune",
        "./scripts/stable/anima_train.py",
        allow_dataset_config_without_train_data_dir=True,
        allow_dataset_class_without_train_data_dir=True,
    ),
}


def is_rocm_amd_runtime_requested() -> bool:
    preferred_runtime = str(os.environ.get("MIKAZUKI_PREFERRED_RUNTIME", "") or "").strip().lower()
    if preferred_runtime == "rocm-amd":
        return True
    if str(os.environ.get("MIKAZUKI_ROCM_AMD_STARTUP", "") or "").strip() == "1":
        return True
    if str(os.environ.get("MIKAZUKI_AMD_EXPERIMENTAL", "") or "").strip() == "1":
        return True
    return False


def is_intel_xpu_runtime_requested() -> bool:
    preferred_runtime = str(os.environ.get("MIKAZUKI_PREFERRED_RUNTIME", "") or "").strip().lower()
    if preferred_runtime in {"intel-xpu", "intel-xpu-sage"}:
        return True
    if str(os.environ.get("MIKAZUKI_INTEL_XPU_STARTUP", "") or "").strip() == "1":
        return True
    if str(os.environ.get("MIKAZUKI_INTEL_XPU_SAGE_STARTUP", "") or "").strip() == "1":
        return True
    if str(os.environ.get("MIKAZUKI_INTEL_XPU_EXPERIMENTAL", "") or "").strip() == "1":
        return True
    if str(os.environ.get("MIKAZUKI_INTEL_XPU_SAGE_EXPERIMENTAL", "") or "").strip() == "1":
        return True
    return False


def resolve_trainer_file_for_training_type(training_type: str, fallback_trainer_file: str) -> str:
    normalized = str(training_type or "").strip().lower()
    if normalized == "anima-lora":
        if is_rocm_amd_runtime_requested():
            return AMD_ANIMA_LORA_TRAINER_FILE
        if is_intel_xpu_runtime_requested():
            return INTEL_ANIMA_LORA_TRAINER_FILE
    return fallback_trainer_file


def get_trainer_definition(training_type: str) -> TrainerDefinition | None:
    normalized = str(training_type or "").strip().lower()
    if not normalized:
        return None
    definition = TRAINER_REGISTRY.get(normalized)
    if definition is None:
        return None

    resolved_trainer_file = resolve_trainer_file_for_training_type(normalized, definition.trainer_file)
    if resolved_trainer_file == definition.trainer_file:
        return definition

    return replace(definition, trainer_file=resolved_trainer_file)


def get_trainer_file_for_training_type(training_type: str, fallback_trainer_file: str | None = None) -> str | None:
    definition = get_trainer_definition(training_type)
    if definition is not None:
        return definition.trainer_file
    return fallback_trainer_file


def get_trainer_definition_by_file(trainer_file: str) -> TrainerDefinition | None:
    trainer_name = Path(str(trainer_file or "")).name.lower()
    if not trainer_name:
        return None

    if trainer_name == Path(AMD_ANIMA_LORA_TRAINER_FILE).name.lower():
        return replace(TRAINER_REGISTRY["anima-lora"], trainer_file=AMD_ANIMA_LORA_TRAINER_FILE)
    if trainer_name == Path(INTEL_ANIMA_LORA_TRAINER_FILE).name.lower():
        return replace(TRAINER_REGISTRY["anima-lora"], trainer_file=INTEL_ANIMA_LORA_TRAINER_FILE)

    for training_type in TRAINER_REGISTRY:
        definition = get_trainer_definition(training_type)
        if definition is None:
            continue
        if Path(definition.trainer_file).name.lower() == trainer_name:
            return definition
    return None


def is_direct_python_training_type(training_type: str) -> bool:
    definition = get_trainer_definition(training_type)
    return bool(definition and definition.direct_python)
