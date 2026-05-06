from __future__ import annotations

import importlib
import os
from typing import Any

from mikazuki.utils.runtime_safe_preview import apply_runtime_safe_preview_policy
from mikazuki.utils.train_utils import parse_boolish


_SAFE_OPTIMIZER_NAMES = {
    "adamw",
    "adafactor",
    "lion",
    "sgdnesterov",
}

_UNSAFE_OPTIMIZER_KEYWORDS = (
    "8bit",
    "paged",
    "bitsandbytes",
    "ademamix",
)

_PYTORCH_OPTIMIZER_PREFIX = "pytorch_optimizer."

_SUPPORTED_INTEL_ANIMA_TRAINING_TYPES = {
    "anima-lora",
    "anima-ileco",
    "anima-addift",
    "anima-multi-addift",
    "sdxl-lora",
    "sdxl-finetune",
}

_INTEL_WINDOWS_OFFICIAL_GPU_PATTERNS = (
    "arc a",
    "arc b",
    "arc graphics",
    "meteor lake",
    "arrow lake",
    "lunar lake",
    "panther lake",
    "core ultra",
)

_INTEL_WINDOWS_EXPERIMENTAL_GPU_PATTERNS = (
    "flex",
    "max",
)

_INTEL_WINDOWS_BLOCKED_GPU_PATTERNS = (
    "iris xe",
    "iris",
    "uhd",
    "hd graphics",
)


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


def get_intel_xpu_runtime_probe() -> dict[str, Any]:
    result: dict[str, Any] = {
        "runtime_requested": is_intel_xpu_runtime_requested(),
        "torch_import_ok": False,
        "torch_error": "",
        "torch_version": "",
        "xpu_available": False,
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_memory_mb": [],
        "bf16_supported": None,
    }

    if not result["runtime_requested"]:
        return result

    try:
        import torch
    except Exception as exc:
        result["torch_error"] = str(exc)
        return result

    result["torch_import_ok"] = True
    result["torch_version"] = str(getattr(torch, "__version__", "") or "")

    try:
        xpu_module = getattr(torch, "xpu", None)
        is_available = getattr(xpu_module, "is_available", None)
        result["xpu_available"] = bool(callable(is_available) and is_available())
    except Exception as exc:
        result["torch_error"] = f"torch.xpu.is_available failed: {exc}"
        return result

    try:
        if result["xpu_available"]:
            result["gpu_count"] = int(torch.xpu.device_count())
            result["gpu_names"] = [str(torch.xpu.get_device_name(index) or "") for index in range(result["gpu_count"])]
            result["gpu_memory_mb"] = [
                int(getattr(torch.xpu.get_device_properties(index), "total_memory", 0) // (1024 * 1024))
                for index in range(result["gpu_count"])
            ]
    except Exception as exc:
        result["torch_error"] = f"torch.xpu device probe failed: {exc}"

    if len(result["gpu_memory_mb"]) != len(result["gpu_names"]):
        result["gpu_memory_mb"] = [0 for _ in result["gpu_names"]]

    try:
        if hasattr(torch.xpu, "is_bf16_supported"):
            result["bf16_supported"] = bool(torch.xpu.is_bf16_supported())
    except Exception:
        result["bf16_supported"] = None

    return result


def _normalize_optimizer_type(raw_value: str) -> tuple[str, str | None]:
    normalized = str(raw_value or "").strip()
    if not normalized:
        return "AdamW", "Intel XPU 实验路线未指定 optimizer_type，已自动改用 AdamW。"

    lowered = normalized.lower()
    if lowered.startswith(_PYTORCH_OPTIMIZER_PREFIX):
        return (
            "AdamW",
            f"Intel XPU 实验路线当前暂不启用 {normalized}，已自动回退为 AdamW。",
        )

    if lowered in _SAFE_OPTIMIZER_NAMES:
        return normalized, None

    if any(keyword in lowered for keyword in _UNSAFE_OPTIMIZER_KEYWORDS):
        return "AdamW", f"Intel XPU 实验路线暂不启用 {normalized}，已自动回退为 AdamW。"

    if lowered.startswith("torch.optim."):
        return normalized, None

    return "AdamW", f"Intel XPU 实验路线暂未验证 optimizer_type={normalized}，已自动回退为 AdamW。"


def _is_anima_training_type(training_type: str) -> bool:
    return str(training_type or "").strip().lower().startswith("anima")


def _is_sdxl_training_type(training_type: str) -> bool:
    return str(training_type or "").strip().lower().startswith("sdxl")


def _is_runtime_guarded_training_type(training_type: str) -> bool:
    return _is_anima_training_type(training_type) or _is_sdxl_training_type(training_type)


def _is_supported_intel_anima_training_type(training_type: str) -> bool:
    return str(training_type or "").strip().lower() in _SUPPORTED_INTEL_ANIMA_TRAINING_TYPES


def _probe_experimental_sageattention() -> dict[str, Any]:
    result: dict[str, Any] = {
        "ready": False,
        "importable": False,
        "reason": "",
    }
    try:
        sage_module = importlib.import_module("sageattention")
    except Exception as exc:
        result["reason"] = f"sageattention import failed: {exc}"
        return result

    result["importable"] = True
    result["ready"] = callable(getattr(sage_module, "sageattn", None)) and callable(getattr(sage_module, "sageattn_varlen", None))
    if not result["ready"]:
        result["reason"] = "required SageAttention symbols are missing"
    return result


def _pick_preferred_gpu_index(probe: dict[str, Any]) -> int:
    gpu_names = [str(name or "").strip() for name in probe.get("gpu_names", [])]
    gpu_memory_mb = [int(value or 0) for value in probe.get("gpu_memory_mb", [])]
    if not gpu_names:
        return 0

    best_index = 0
    best_score = (-1, -1)
    for index, gpu_name in enumerate(gpu_names):
        lowered = gpu_name.lower()
        is_discrete = int(("arc" in lowered) or ("flex" in lowered) or ("b580" in lowered))
        memory_score = gpu_memory_mb[index] if index < len(gpu_memory_mb) else 0
        score = (is_discrete, memory_score)
        if score > best_score:
            best_score = score
            best_index = index
    return best_index


def _classify_intel_windows_support(gpu_name: str) -> str:
    lowered = str(gpu_name or "").strip().lower()
    if not lowered:
        return "unknown"

    if any(pattern in lowered for pattern in _INTEL_WINDOWS_BLOCKED_GPU_PATTERNS):
        return "blocked"

    if any(pattern in lowered for pattern in _INTEL_WINDOWS_OFFICIAL_GPU_PATTERNS):
        return "official"

    if any(pattern in lowered for pattern in _INTEL_WINDOWS_EXPERIMENTAL_GPU_PATTERNS):
        return "experimental"

    return "experimental"


def _is_intel_arc_a_series(gpu_name: str) -> bool:
    lowered = str(gpu_name or "").strip().lower()
    return "arc a" in lowered or " a3" in lowered or " a5" in lowered or " a7" in lowered


def _requested_sageattention(config: dict) -> bool:
    attn_mode = str(config.get("attn_mode", "") or "").strip().lower()
    return (
        attn_mode == "sageattn"
        or parse_boolish(config.get("sageattn"))
        or parse_boolish(config.get("use_sage_attn"))
    )


def _is_preview_requested(config: dict) -> bool:
    if parse_boolish(config.get("enable_preview")):
        return True
    if parse_boolish(config.get("sample_at_first")):
        return True

    for key in ("sample_every_n_steps", "sample_every_n_epochs"):
        raw_value = config.get(key)
        try:
            if raw_value is not None and int(raw_value) > 0:
                return True
        except (TypeError, ValueError):
            pass

    for key in ("prompt_file", "sample_prompts", "positive_prompts", "negative_prompts"):
        if str(config.get(key, "") or "").strip():
            return True

    return False


def _clear_preview_fields(config: dict) -> None:
    config["enable_preview"] = False
    config["sample_at_first"] = False
    for key in (
        "sample_every_n_steps",
        "sample_every_n_epochs",
        "prompt_file",
        "sample_prompts",
        "positive_prompts",
        "negative_prompts",
        "randomly_choice_prompt",
        "random_prompt_include_subdirs",
    ):
        config.pop(key, None)


def apply_intel_anima_topology_guard(config: dict, gpu_ids: list[str] | None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "gpu_ids": list(gpu_ids or []),
        "warnings": [],
        "notes": [],
        "errors": [],
        "probe": get_intel_xpu_runtime_probe(),
    }

    training_type = str(config.get("model_train_type", "") or "").strip().lower()
    if not result["probe"]["runtime_requested"] or not _is_runtime_guarded_training_type(training_type):
        return result

    if not _is_supported_intel_anima_training_type(training_type):
        supported = ", ".join(sorted(_SUPPORTED_INTEL_ANIMA_TRAINING_TYPES))
        result["errors"].append(
            f"当前 Intel XPU 实验运行时仅开放 {supported}。"
            f"检测到的是 {training_type}，为避免进入未完整接入 Intel XPU 诊断与保护策略的训练路径，已阻止启动。"
        )
        return result

    probe = result["probe"]
    if not probe["torch_import_ok"]:
        result["errors"].append(f"Intel XPU 实验运行时当前无法导入 torch：{probe['torch_error']}")
        return result

    if not probe["xpu_available"] or int(probe["gpu_count"] or 0) <= 0:
        result["errors"].append("当前 Intel XPU 运行时未检测到可用 Intel GPU，Intel XPU 实验训练无法启动。")
        return result

    gpu_names = [name for name in probe.get("gpu_names", []) if str(name).strip()]
    gpu_summary = ", ".join(gpu_names) if gpu_names else f"{probe['gpu_count']} visible Intel GPU(s)"
    result["notes"].append(f"Intel XPU 运行时：Torch {probe['torch_version']}；可见显卡：{gpu_summary}。")

    if parse_boolish(config.get("enable_distributed_training")):
        config["enable_distributed_training"] = False
        result["warnings"].append("Intel XPU 实验路线当前按单卡安全模式运行，已自动关闭分布式训练。")

    selected_gpu_ids = list(result["gpu_ids"])
    if len(selected_gpu_ids) > 1:
        kept_gpu_id = selected_gpu_ids[0]
        selected_gpu_ids = [kept_gpu_id]
        result["warnings"].append(f"Intel XPU 实验路线当前默认只保留单卡，已自动改为 GPU {kept_gpu_id}。")
    elif len(selected_gpu_ids) == 0:
        preferred_index = _pick_preferred_gpu_index(probe)
        selected_gpu_ids = [str(preferred_index)]
        result["warnings"].append(f"Intel XPU 实验路线未显式指定 GPU，已自动优先选择显存更大的 Intel GPU {preferred_index}。")

    result["gpu_ids"] = selected_gpu_ids
    if selected_gpu_ids:
        config["gpu_ids"] = selected_gpu_ids
        result["notes"].append(f"Intel XPU 实验路线最终使用 GPU: {', '.join(selected_gpu_ids)}。")
    else:
        config.pop("gpu_ids", None)

    selected_gpu_name = ""
    try:
        selected_index = int(selected_gpu_ids[0]) if selected_gpu_ids else 0
        gpu_names = [str(name or "").strip() for name in probe.get("gpu_names", [])]
        if 0 <= selected_index < len(gpu_names):
            selected_gpu_name = gpu_names[selected_index]
    except (TypeError, ValueError, IndexError):
        selected_gpu_name = ""

    windows_support = _classify_intel_windows_support(selected_gpu_name)
    if os.name == "nt":
        if windows_support == "official":
            result["notes"].append("当前显卡命中 PyTorch 当前公开的 Windows XPU 支持范围。")
        elif windows_support == "experimental":
            result["warnings"].append(
                f"当前 Intel GPU（{selected_gpu_name or 'unknown'}）不在 PyTorch 当前公开的 Windows XPU 主支持名单里。"
                "项目仍会按实验路线继续，但会优先启用更保守的保护与 fallback。"
            )
        elif windows_support == "blocked":
            result["errors"].append(
                f"当前 Intel GPU（{selected_gpu_name or 'unknown'}）不在项目当前的 Windows Intel XPU 实验支持范围内。"
                "目前优先面向 Arc A/B 与带 Arc Graphics 的 Core Ultra 平台；"
                "对明显非目标设备（如 UHD / Iris Xe）会直接阻止启动。"
            )
            return result

    raw_gpu_power_limit = config.get("gpu_power_limit_w")
    try:
        gpu_power_limit_w = int(round(float(raw_gpu_power_limit)))
    except (TypeError, ValueError):
        gpu_power_limit_w = 0
    if gpu_power_limit_w > 0:
        config["gpu_power_limit_w"] = 0
        result["warnings"].append("Intel XPU 实验路线不支持显卡功率墙，已自动忽略 GPU 功率限制。")

    return result


def apply_intel_anima_runtime_config_guard(config: dict, runtime_probe: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {
        "warnings": [],
        "notes": [],
        "errors": [],
        "skip_preview_prompt_prep": False,
    }

    training_type = str(config.get("model_train_type", "") or "").strip().lower()
    if not is_intel_xpu_runtime_requested() or not _is_runtime_guarded_training_type(training_type):
        return result

    if not _is_supported_intel_anima_training_type(training_type):
        supported = ", ".join(sorted(_SUPPORTED_INTEL_ANIMA_TRAINING_TYPES))
        result["errors"].append(
            f"当前 Intel XPU 实验运行时仅开放 {supported}。"
            f"检测到的是 {training_type}，已停止继续应用运行时改写。"
        )
        return result

    probe = runtime_probe if isinstance(runtime_probe, dict) else get_intel_xpu_runtime_probe()
    bf16_supported = probe.get("bf16_supported")
    if bf16_supported is True:
        result["notes"].append("当前 Intel XPU 运行时报告 bf16 可用。")
    elif bf16_supported is False:
        result["notes"].append("当前 Intel XPU 运行时未报告 bf16 可用。")
    else:
        result["notes"].append("当前 Intel XPU 运行时无法确认 bf16 支持状态。")

    wants_sageattention = _requested_sageattention(config)
    sage_probe = _probe_experimental_sageattention() if wants_sageattention else {"ready": False, "reason": ""}
    requested_attn_mode = str(config.get("attn_mode", "") or "").strip().lower()
    is_sdxl_route = _is_sdxl_training_type(training_type)
    if requested_attn_mode not in {"", "none", "null", "torch", "sdpa", "sageattn"}:
        result["warnings"].append(f"Intel XPU 实验路线暂不启用 {requested_attn_mode} attention，已自动改用 SDPA。")

    config["attn_mode"] = "torch" if not is_sdxl_route else "sdpa"
    config["sdpa"] = True

    for key, label in (
        ("xformers", "xformers"),
        ("mem_eff_attn", "mem_eff_attn"),
        ("torch_compile", "torch_compile"),
        ("full_fp16", "full_fp16"),
        ("full_bf16", "full_bf16"),
        ("use_8bit_adam", "use_8bit_adam"),
        ("fused_backward_pass", "fused_backward_pass"),
    ):
        if parse_boolish(config.get(key)):
            config[key] = False
            result["warnings"].append(f"Intel XPU 实验路线已自动禁用 {label}。")

    if is_sdxl_route:
        if parse_boolish(config.get("sageattn")) or parse_boolish(config.get("use_sage_attn")) or requested_attn_mode == "sageattn":
            config["sageattn"] = False
            config["use_sage_attn"] = False
            config["sdpa"] = True
            config["attn_mode"] = "sdpa"
            result["warnings"].append(
                "Intel XPU 实验路线中的 SDXL 当前只开放 SDPA 路径；已自动禁用 SageAttention 请求。"
            )
    elif wants_sageattention and sage_probe["ready"]:
        config["attn_mode"] = "sageattn"
        config["sdpa"] = False
        config["sageattn"] = True
        config["use_sage_attn"] = True
        result["warnings"].append("Intel XPU 实验路线将试运行实验性 SageAttention；若导入成功但内核调用失败，训练时会自动回退为 SDPA。")
    else:
        if parse_boolish(config.get("sageattn")) or parse_boolish(config.get("use_sage_attn")):
            config["sageattn"] = False
            config["use_sage_attn"] = False
        if wants_sageattention:
            result["warnings"].append(
                f"Intel XPU 实验路线当前未检测到可用的 SageAttention 构建（{sage_probe['reason'] or 'runtime probe failed'}），已自动回退为 SDPA。"
            )

    normalized_optimizer, optimizer_warning = _normalize_optimizer_type(str(config.get("optimizer_type", "") or ""))
    config["optimizer_type"] = normalized_optimizer
    if optimizer_warning:
        result["warnings"].append(optimizer_warning)

    mixed_precision = str(config.get("mixed_precision", "") or "").strip().lower()
    if not mixed_precision:
        config["mixed_precision"] = "bf16" if bf16_supported is not False else "fp16"
        result["warnings"].append(f"Intel XPU 实验路线未指定 mixed_precision，已自动改用 {config['mixed_precision']}。")
    elif mixed_precision == "bf16" and bf16_supported is False:
        config["mixed_precision"] = "fp16"
        result["warnings"].append("当前 Intel XPU 运行时未检测到 bf16 支持，已自动把 mixed_precision 从 bf16 回退为 fp16。")

    selected_gpu_name = ""
    gpu_names = [str(name or "").strip() for name in probe.get("gpu_names", [])]
    try:
        selected_index = int(config.get("gpu_ids", [0])[0]) if isinstance(config.get("gpu_ids"), list) and config.get("gpu_ids") else 0
        if 0 <= selected_index < len(gpu_names):
            selected_gpu_name = gpu_names[selected_index]
    except (TypeError, ValueError, IndexError):
        selected_gpu_name = ""

    if str(config.get("mixed_precision", "") or "").strip().lower() == "fp16" and _is_intel_arc_a_series(selected_gpu_name):
        if bf16_supported is not False:
            config["mixed_precision"] = "bf16"
            result["warnings"].append(
                "检测到 Intel Arc A 系列。根据 PyTorch XPU 文档，fp16 AMP + GradScaler 在这类设备上存在硬件限制；"
                "已自动把 mixed_precision 从 fp16 调整为 bf16。"
            )
        else:
            result["warnings"].append(
                "检测到 Intel Arc A 系列。根据 PyTorch XPU 文档，fp16 AMP + GradScaler 在这类设备上存在硬件限制；"
                "若训练中出现 AMP/GradScaler 相关异常，建议优先改用 bf16 或关闭混合精度。"
            )

    try:
        max_workers = int(config.get("max_data_loader_n_workers", 0) or 0)
    except (TypeError, ValueError):
        max_workers = 0
    if max_workers > 0:
        config["max_data_loader_n_workers"] = 0
        result["warnings"].append("Intel XPU 实验路线已自动把 max_data_loader_n_workers 改为 0。")
    else:
        config.setdefault("max_data_loader_n_workers", 0)

    if parse_boolish(config.get("persistent_data_loader_workers")):
        config["persistent_data_loader_workers"] = False
        result["warnings"].append("Intel XPU 实验路线已自动关闭 persistent_data_loader_workers。")
    else:
        config.setdefault("persistent_data_loader_workers", False)

    try:
        nan_check_interval = int(config.get("anima_nan_check_interval", 0) or 0)
    except (TypeError, ValueError):
        nan_check_interval = 0
    if nan_check_interval <= 0:
        config["anima_nan_check_interval"] = 1
        result["warnings"].append("Intel XPU 实验路线已自动把 anima_nan_check_interval 改为 1。")

    config.setdefault("ipex_sdpa_slice_trigger_rate", 0.75)
    config.setdefault("ipex_attention_slice_rate", 0.4)
    result["notes"].append(
        f"Intel XPU 实验路线将使用 IPEX attention slicing：trigger={config['ipex_sdpa_slice_trigger_rate']}, slice={config['ipex_attention_slice_rate']}。"
    )

    if apply_runtime_safe_preview_policy(
        config,
        runtime_label="Intel XPU 实验路线",
        messages=result["warnings"],
    ):
        result["skip_preview_prompt_prep"] = False

    if is_sdxl_route:
        result["notes"].append("Intel XPU 实验 SDXL 当前会强制走 SDPA，并复用主线 SDXL trainer。")
        if parse_boolish(config.get("_runtime_safe_preview_enabled", False)):
            result["notes"].append("Intel XPU 实验 SDXL 的训练预览会使用独立的安全 SDPA 预览后端。")

    return result
