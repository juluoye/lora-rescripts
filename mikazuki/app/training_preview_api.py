from __future__ import annotations

import json
from collections.abc import Callable

from fastapi import APIRouter, Request

from mikazuki.app.models import APIResponse, APIResponseFail, APIResponseSuccess
from mikazuki.app.training_prompt_utils import (
    build_disabled_sample_prompt_record,
    build_sample_prompt_preview,
    build_sample_prompt_record,
    simulate_attention_backend_fallback_warning,
)
from mikazuki.log import log
from mikazuki.utils import train_utils
from mikazuki.utils.attention_runtime_guard import apply_startup_attention_policy
from mikazuki.utils.training_preflight import analyze_training_preflight
from mikazuki.utils.training_runtime_context import resolve_training_runtime_guard_context
from mikazuki.utils.trainer_registry import get_trainer_definition


def build_router(
    *,
    apply_training_ui_overrides: Callable[[dict], list[str]],
    parse_boolish: Callable[[object], bool],
) -> APIRouter:
    router = APIRouter()

    @router.post("/train/preflight")
    async def training_preflight(request: Request) -> APIResponse:
        json_data = await request.body()
        config: dict = json.loads(json_data.decode("utf-8"))
        try:
            train_utils.fix_config_types(config)
        except (TypeError, ValueError) as exc:
            return APIResponseFail(message=f"Invalid config value / 配置值无效: {exc}")

        override_warnings = apply_training_ui_overrides(config)

        runtime_context = resolve_training_runtime_guard_context(
            config,
            config.get("gpu_ids"),
            persist_gpu_ids=True,
        )
        gpu_ids = runtime_context["gpu_ids"]
        startup_attention_warning = apply_startup_attention_policy(config, parse_boolish)
        training_type = str(config.get("model_train_type", "sd-lora"))

        result = analyze_training_preflight(
            config,
            training_type=training_type,
            trainer_supported=get_trainer_definition(training_type) is not None,
            conditioning_required=training_type in {"sd-controlnet", "sdxl-controlnet", "flux-controlnet"},
            sample_prompt_builder=build_sample_prompt_preview,
            attention_fallback_checker=lambda payload: simulate_attention_backend_fallback_warning(payload, gpu_ids),
        )

        if runtime_context["warnings"]:
            result["warnings"] = result.get("warnings", []) + runtime_context["warnings"]
        if runtime_context["notes"]:
            result["notes"] = result.get("notes", []) + runtime_context["notes"]
        if runtime_context["errors"]:
            result["errors"] = result.get("errors", []) + runtime_context["errors"]
        if override_warnings:
            result["warnings"] = result.get("warnings", []) + override_warnings
        if startup_attention_warning:
            result["warnings"] = result.get("warnings", []) + [startup_attention_warning]

        return APIResponseSuccess(data=result)

    @router.post("/train/sample_prompt")
    async def training_sample_prompt(request: Request) -> APIResponse:
        json_data = await request.body()
        config: dict = json.loads(json_data.decode("utf-8"))
        try:
            train_utils.fix_config_types(config)
        except (TypeError, ValueError) as exc:
            return APIResponseFail(message=f"Invalid config value / 配置值无效: {exc}")

        override_warnings = apply_training_ui_overrides(config)

        runtime_context = resolve_training_runtime_guard_context(config, config.get("gpu_ids"))
        if runtime_context["errors"]:
            return APIResponseFail(message="\n".join(runtime_context["errors"]))

        amd_preview_warnings = list(runtime_context["warnings"])
        amd_preview_notes = list(runtime_context["notes"])

        if runtime_context["skip_preview_prompt_prep"]:
            result = build_disabled_sample_prompt_record(
                config,
                source="runtime_guard_disabled",
                detail="实验运行时已强制关闭训练预览图与预览提示词。",
                warnings=amd_preview_warnings,
                notes=amd_preview_notes,
            )
            if override_warnings:
                result["warnings"] = [*override_warnings, *result.get("warnings", [])]
            return APIResponseSuccess(data=result)

        try:
            result = build_sample_prompt_record(config)
            if not result:
                return APIResponseFail(message="Current config does not expose a sample prompt preview.")
        except ValueError as exc:
            return APIResponseFail(message=str(exc))
        except Exception:
            log.exception("Training sample prompt preview failed")
            return APIResponseFail(message="Sample prompt preview failed.")

        if amd_preview_warnings:
            result["warnings"] = [*amd_preview_warnings, *result.get("warnings", [])]
        if amd_preview_notes:
            result["notes"] = [*amd_preview_notes, *result.get("notes", [])]
        if override_warnings:
            result["warnings"] = [*override_warnings, *result.get("warnings", [])]

        return APIResponseSuccess(data=result)

    return router
