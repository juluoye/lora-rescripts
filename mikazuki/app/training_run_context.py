from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from fastapi import Request

from mikazuki import launch_utils
from mikazuki.app.models import APIResponseFail
from mikazuki.app.training_prompt_utils import parse_boolish
from mikazuki.app.training_ui_overrides import apply_training_ui_overrides
from mikazuki.plugins.runtime import plugin_runtime
from mikazuki.utils import train_utils
from mikazuki.utils.training_launch_runtime import resolve_training_launch_runtime
from mikazuki.utils.training_runtime_context import resolve_training_runtime_guard_context
from mikazuki.utils.training_start_warnings import (
    build_training_gpu_selection_warning,
    build_training_resource_warnings,
)
from mikazuki.utils.trainer_registry import get_trainer_definition


@dataclass
class TrainingRunContext:
    timestamp: str
    autosave_dir: Path
    toml_file: str
    config: dict
    start_warnings: list[str]
    gpu_ids: list
    model_train_type: str
    trainer_definition: object
    direct_python_training: bool
    distributed_runtime: dict
    skip_preview_prompt_prep: bool
    suggest_cpu_threads: int
    trainer_file: str
    mixed_resolution_payload: Optional[dict] = None


def is_yolo_training_type(training_type: str) -> bool:
    return str(training_type or "").strip().lower() == "yolo"


def _build_run_file_context() -> Tuple[str, Path, str]:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    autosave_dir = launch_utils.base_dir_path() / "config" / "autosave"
    autosave_dir.mkdir(parents=True, exist_ok=True)
    toml_file = str(autosave_dir / f"{timestamp}.toml")
    return timestamp, autosave_dir, toml_file


async def _load_run_config(request: Request) -> Tuple[Optional[dict], Optional[APIResponseFail]]:
    json_data = await request.body()
    config: dict = json.loads(json_data.decode("utf-8"))
    config.setdefault("pytorch_cuda_expandable_segments", True)
    try:
        train_utils.fix_config_types(config)
    except (TypeError, ValueError) as exc:
        return None, APIResponseFail(message=f"Invalid config value / 配置值无效: {exc}")
    return config, None


def _emit_config_loaded_event(config: dict) -> None:
    plugin_runtime.emit_event(
        "on_config_loaded",
        {
            "model_train_type": str(config.get("model_train_type", "") or "").strip().lower(),
            "config_key_count": len(config.keys()),
            "config_keys": sorted(str(item) for item in config.keys()),
        },
        source="api.run",
    )


def _apply_model_specific_launch_defaults(config: dict, model_train_type: str) -> None:
    # Windows + multiprocessing dataloader is more fragile on Anima routes.
    # If the user did not choose these explicitly, default to safer single-process loading.
    if model_train_type in {"anima-lora", "anima-finetune"}:
        config.setdefault("max_data_loader_n_workers", 0)
        config.setdefault("persistent_data_loader_workers", False)

    if model_train_type in {
        "sd-ileco",
        "sd-addift",
        "sd-multi-addift",
        "sdxl-ileco",
        "sdxl-addift",
        "sdxl-multi-addift",
        "anima-ileco",
        "anima-addift",
        "anima-multi-addift",
    }:
        if model_train_type.startswith("anima-"):
            config.setdefault("dataset_class", "library.anima_concept_edit_util.AnimaConceptEditDataset")
        else:
            config.setdefault("dataset_class", "library.concept_edit_util.ConceptEditDataset")
        config.setdefault("max_data_loader_n_workers", 0)
        config.setdefault("persistent_data_loader_workers", False)


def _build_training_run_context(
    *,
    config: dict,
    timestamp: str,
    autosave_dir: Path,
    toml_file: str,
) -> Tuple[Optional[TrainingRunContext], Optional[APIResponseFail]]:
    start_warnings = apply_training_ui_overrides(config)
    start_warnings.extend(build_training_resource_warnings(config))

    raw_gpu_ids = config.pop("gpu_ids", None)
    runtime_context = resolve_training_runtime_guard_context(config, raw_gpu_ids)
    if runtime_context["errors"]:
        return None, APIResponseFail(message="\n".join(runtime_context["errors"]))

    gpu_ids = runtime_context["gpu_ids"]
    start_warnings.extend(runtime_context["warnings"])
    start_warnings.extend(runtime_context["notes"])
    start_warnings.append(build_training_gpu_selection_warning(gpu_ids))

    requested_train_type = str(config.get("model_train_type", "sd-lora") or "sd-lora").strip().lower()
    trainer_definition = get_trainer_definition(requested_train_type)
    if trainer_definition is None:
        return None, APIResponseFail(message=f"Unsupported trainer type: {requested_train_type}")

    direct_python_training = bool(trainer_definition.direct_python)
    if direct_python_training and parse_boolish(config.get("enable_distributed_training")):
        return None, APIResponseFail(message="当前训练种类暂不走 Mikazuki 分布式启动。")

    launch_runtime = resolve_training_launch_runtime(
        config,
        gpu_ids,
        direct_python_training=direct_python_training,
        yolo_training=is_yolo_training_type(requested_train_type),
        base_dir=launch_utils.base_dir_path(),
    )
    if launch_runtime["error_message"]:
        return None, APIResponseFail(message=launch_runtime["error_message"])

    start_warnings.extend(launch_runtime["warnings"])

    model_train_type = str(config.pop("model_train_type", "sd-lora") or "sd-lora").strip().lower()
    config["model_train_type"] = model_train_type
    _apply_model_specific_launch_defaults(config, model_train_type)

    train_data_dir = str(config.get("train_data_dir", "") or "").strip()
    suggest_cpu_threads = 8 if train_data_dir and len(train_utils.get_total_images(train_data_dir)) > 200 else 2

    context = TrainingRunContext(
        timestamp=timestamp,
        autosave_dir=autosave_dir,
        toml_file=toml_file,
        config=config,
        start_warnings=start_warnings,
        gpu_ids=gpu_ids,
        model_train_type=model_train_type,
        trainer_definition=trainer_definition,
        direct_python_training=direct_python_training,
        distributed_runtime=launch_runtime["distributed_runtime"],
        skip_preview_prompt_prep=runtime_context["skip_preview_prompt_prep"],
        suggest_cpu_threads=suggest_cpu_threads,
        trainer_file=trainer_definition.trainer_file,
    )
    return context, None


async def create_training_run_context(request: Request) -> Tuple[Optional[TrainingRunContext], Optional[APIResponseFail]]:
    timestamp, autosave_dir, toml_file = _build_run_file_context()
    config, config_error = await _load_run_config(request)
    if config_error:
        return None, config_error

    _emit_config_loaded_event(config)

    return _build_training_run_context(
        config=config,
        timestamp=timestamp,
        autosave_dir=autosave_dir,
        toml_file=toml_file,
    )
