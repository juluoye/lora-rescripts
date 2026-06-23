from __future__ import annotations

from dataclasses import asdict

import toml

import mikazuki.process as process
from mikazuki.app.models import APIResponseFail
from mikazuki.app.training_prompt_utils import (
    build_sample_prompt_file_name,
    enrich_inline_sample_prompts,
    get_sample_prompts,
    parse_boolish,
    should_use_inline_sample_prompts,
)
from mikazuki.app.training_run_context import TrainingRunContext
from mikazuki.app.training_ui_overrides import build_sdxl_clip_skip_warning
from mikazuki.log import log
from mikazuki.plugins.runtime import plugin_runtime
from mikazuki.training_route_contract import extract_route_contract_metadata, resolve_training_route_contract
from mikazuki.utils import train_utils
from mikazuki.utils.attention_runtime_guard import (
    apply_sageattention_route_guard,
    apply_sageattention_runtime_override,
    apply_sagebwd_runtime_guard,
    apply_startup_attention_policy,
)
from mikazuki.utils.dataset_cache_preflight import analyze_dataset_cache_preflight
from mikazuki.utils.devices import get_xformers_status
from mikazuki.utils.mixed_resolution import build_mixed_resolution_plan
from mikazuki.utils.runtime_dependencies import analyze_training_runtime_dependencies
from mikazuki.utils.training_preflight import (
    build_route_contract_preflight_note,
    build_sageattention_experimental_warning,
    train_data_dir_can_be_omitted,
    validate_dataset_config_reference,
)
from mikazuki.utils.training_sample_prompt_runtime import prepare_training_sample_prompt_config
from mikazuki.utils.training_start_warnings import (
    build_runtime_dependency_failure_message,
    merge_training_result_warnings,
)


def apply_attention_backend_fallback(config: dict, gpu_ids) -> str | None:
    if config.get("mem_eff_attn", False):
        return None

    if not config.get("xformers", False):
        return None

    xformers_info = get_xformers_status(gpu_ids)
    if xformers_info.get("selected_verified", xformers_info.get("verified", False)):
        return None

    config["xformers"] = False

    if "sdpa" in config:
        config["sdpa"] = True
        message = (
            f"检测到当前显卡或环境暂不支持 xformers（{xformers_info['reason']}），"
            "已自动切换为 sdpa 训练。"
        )
    else:
        message = (
            f"检测到当前显卡或环境暂不支持 xformers（{xformers_info['reason']}），"
            "已自动禁用 xformers。"
        )

    log.warning(message)
    return message


def _validate_training_inputs(context: TrainingRunContext) -> APIResponseFail | None:
    config = context.config
    trainer_definition = context.trainer_definition

    if trainer_definition.config_validator is not None:
        config_error = trainer_definition.config_validator(config)
        if config_error:
            return APIResponseFail(message=config_error)

    if trainer_definition.start_warning_builder is not None:
        context.start_warnings.extend(trainer_definition.start_warning_builder(config))

    dataset_config_error = validate_dataset_config_reference(config, training_type=context.model_train_type)
    if dataset_config_error:
        return APIResponseFail(message=dataset_config_error)

    if not context.direct_python_training and not train_data_dir_can_be_omitted(config, context.model_train_type):
        if not train_utils.validate_data_dir(config["train_data_dir"]):
            return APIResponseFail(message="训练数据集路径不存在或没有图片，请检查目录。")

    if not context.direct_python_training and context.model_train_type in {"sd-controlnet", "sdxl-controlnet", "flux-controlnet"}:
        conditioning_data_dir = config.get("conditioning_data_dir", "")
        if not conditioning_data_dir or not train_utils.validate_data_dir(conditioning_data_dir):
            return APIResponseFail(message="条件图数据集路径不存在或没有图片，请检查目录。")

    return None


def _prepare_mixed_resolution_payload(context: TrainingRunContext) -> APIResponseFail | None:
    try:
        if context.direct_python_training:
            mixed_resolution_plan = None
        else:
            planning_config = dict(context.config)
            planning_config["num_processes"] = int(context.distributed_runtime.get("total_num_processes", 1) or 1)
            mixed_resolution_plan = build_mixed_resolution_plan(planning_config, training_type=context.model_train_type)
    except ValueError as exc:
        return APIResponseFail(message=str(exc))
    except Exception:
        log.exception("Mixed-resolution planning failed unexpectedly")
        return APIResponseFail(message="阶段分辨率训练规划失败，请查看日志。")

    if mixed_resolution_plan is not None and mixed_resolution_plan.enabled:
        context.mixed_resolution_payload = asdict(mixed_resolution_plan)
        context.start_warnings.append(
            f"已启用阶段分辨率训练：共 {len(mixed_resolution_plan.phases)} 个阶段，将按顺序自动切换分辨率与 batch。"
        )

    return None


def _run_dataset_cache_preflight(context: TrainingRunContext) -> APIResponseFail | None:
    try:
        if context.direct_python_training:
            cache_preflight = {"errors": [], "warnings": [], "notes": []}
        else:
            cache_preflight = analyze_dataset_cache_preflight(context.config, training_type=context.model_train_type)
    except Exception:
        log.exception("Dataset cache preflight failed unexpectedly")
        return APIResponseFail(message="数据集缓存预检失败，请查看日志。")

    if cache_preflight.get("errors"):
        return APIResponseFail(message="\n".join(cache_preflight["errors"]))

    context.start_warnings.extend(cache_preflight.get("warnings", []))
    return None


def _apply_runtime_training_warnings(context: TrainingRunContext) -> APIResponseFail | None:
    config = context.config

    if not context.trainer_definition.skip_model_validation:
        validated, message = train_utils.validate_model(config["pretrained_model_name_or_path"], context.model_train_type)
        if not validated:
            return APIResponseFail(message=message)

    startup_attention_message = apply_startup_attention_policy(config, parse_boolish)
    if startup_attention_message:
        context.start_warnings.append(startup_attention_message)

    sagebwd_runtime_message = apply_sagebwd_runtime_guard(config, parse_boolish)
    if sagebwd_runtime_message:
        context.start_warnings.append(sagebwd_runtime_message)

    sageattention_route_message = apply_sageattention_route_guard(config)
    if sageattention_route_message:
        context.start_warnings.append(sageattention_route_message)

    sdxl_clip_skip_warning = build_sdxl_clip_skip_warning(config)
    if sdxl_clip_skip_warning:
        context.start_warnings.append(sdxl_clip_skip_warning)

    sageattention_training_warning = build_sageattention_experimental_warning(config, context.model_train_type)
    if sageattention_training_warning:
        context.start_warnings.append(sageattention_training_warning)

    sageattention_override_message = apply_sageattention_runtime_override(config, parse_boolish)
    if sageattention_override_message:
        context.start_warnings.append(sageattention_override_message)

    attention_fallback_message = apply_attention_backend_fallback(config, context.gpu_ids)
    if attention_fallback_message:
        context.start_warnings.append(attention_fallback_message)

    dependency_report = analyze_training_runtime_dependencies(config)
    dependency_failure_message = build_runtime_dependency_failure_message(dependency_report)
    if dependency_failure_message:
        return APIResponseFail(message=dependency_failure_message)

    return None


def _prepare_training_sample_prompts(context: TrainingRunContext) -> APIResponseFail | None:
    sample_prompt_error = prepare_training_sample_prompt_config(
        context.config,
        autosave_dir=context.autosave_dir,
        timestamp=context.timestamp,
        direct_python_training=context.direct_python_training,
        skip_preview_prompt_prep=context.skip_preview_prompt_prep,
        get_sample_prompts=get_sample_prompts,
        should_use_inline_sample_prompts=should_use_inline_sample_prompts,
        enrich_inline_sample_prompts=enrich_inline_sample_prompts,
        build_sample_prompt_file_name=build_sample_prompt_file_name,
    )
    if sample_prompt_error:
        return APIResponseFail(message=sample_prompt_error)
    return None


def _emit_dataset_prepared_event(context: TrainingRunContext) -> None:
    route_contract = extract_route_contract_metadata(context.config) or resolve_training_route_contract(
        context.model_train_type,
        config=context.config,
        route_kind_override=getattr(context.trainer_definition, "route_kind", None),
        route_label_override=getattr(context.trainer_definition, "route_label", None),
    ).as_metadata_fields()
    plugin_runtime.emit_event(
        "on_dataset_prepared",
        {
            "model_train_type": context.model_train_type,
            "trainer_route_kind": route_contract.get("lulynx_route_kind", ""),
            "trainer_route_label": route_contract.get("lulynx_route_label", ""),
            "trainer_route_family": route_contract.get("lulynx_route_family", ""),
            "trainer_route_capabilities": route_contract.get("lulynx_route_capabilities", ""),
            "train_data_dir": str(context.config.get("train_data_dir", "") or ""),
            "direct_python_training": bool(context.direct_python_training),
        },
        source="api.run",
    )


def launch_training(context: TrainingRunContext):
    _emit_dataset_prepared_event(context)

    with open(context.toml_file, "w", encoding="utf-8") as f:
        f.write(toml.dumps(context.config))

    plugin_runtime.emit_event(
        "on_train_launch",
        {
            "model_train_type": context.model_train_type,
            "trainer_route_kind": str(context.config.get("_lulynx_route_kind", "") or ""),
            "trainer_route_label": str(context.config.get("_lulynx_route_label", "") or ""),
            "trainer_route_capabilities": ",".join(context.config.get("_lulynx_route_capabilities", []) or []),
            "trainer_file": context.trainer_file,
            "toml_file": context.toml_file,
            "gpu_ids": list(context.gpu_ids or []),
            "distributed_world_size": int(context.distributed_runtime.get("total_num_processes", 1) or 1),
        },
        source="api.run",
    )

    result = process.run_train(context.toml_file, context.trainer_file, context.gpu_ids, context.suggest_cpu_threads)
    merge_training_result_warnings(
        result,
        context.start_warnings,
        mixed_resolution_payload=context.mixed_resolution_payload,
    )
    return result


def prepare_training_run(context: TrainingRunContext) -> APIResponseFail | None:
    context.start_warnings.append(build_route_contract_preflight_note(context.config, context.model_train_type))

    validation_error = _validate_training_inputs(context)
    if validation_error:
        return validation_error

    mixed_resolution_error = _prepare_mixed_resolution_payload(context)
    if mixed_resolution_error:
        return mixed_resolution_error

    cache_preflight_error = _run_dataset_cache_preflight(context)
    if cache_preflight_error:
        return cache_preflight_error

    runtime_warning_error = _apply_runtime_training_warnings(context)
    if runtime_warning_error:
        return runtime_warning_error

    sample_prompt_error = _prepare_training_sample_prompts(context)
    if sample_prompt_error:
        return sample_prompt_error

    return None
